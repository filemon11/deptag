import os
import logging
import pickle
import json

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp.grad_scaler import GradScaler
import tqdm
import pathlib
from . import model, dataset, evaluate, factorisation, pcgrad, config
from .. import extraction, data, settings, utils
import dataclasses
from collections import defaultdict

from typing import Mapping, Sequence, Self, Type, Literal, Iterator


# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)

import transformers.utils.output_capturing as hf_output_capturing
torch._functorch.config.donated_buffer = False

# Work around Transformers 5.9 + PyTorch 2.6 Dynamo incompatibility.
# This only changes the current Python process.
hf_output_capturing.torch = torch  # type: ignore

torch.set_float32_matmul_precision("medium")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def initialize_tag_system(
        ds: str,
        tag_vocab_path: pathlib.Path = pathlib.Path(".")
        ) -> dict[str, int]:
    with open(tag_vocab_path / (ds + '.pkl'), 'rb') as f:
        tag_vocab = pickle.load(f)

    return tag_vocab


def save_vocab(args: settings.Settings):
    data_path = pathlib.Path(args.file.data_folder)
    prefix = args.file.conllu_file

    train_reader = data.load_conllu(
        prefix, "train", dir=data_path)
    _, sup2id = extraction.prepare_train(
        train_reader,
        arguments=args.deprels.arguments,
        adjuncts=args.deprels.adjuncts,
        delete=args.deprels.delete,
        merged=args.deprels.merged,
        without_labels=not args.deprels.labelled,
        distinguish_fallback_subtypes=not args.deprels.labelled,
        merged_fallback_subtypes=args.deprels.merged_fallback_subtypes,
        distinguish_merged_fallback_subtypes=(
            args.deprels.distinguish_merged_fallback_subtypes),
        order_relations=args.deprels.order_relations,
        )

    path = pathlib.Path(args.tagging.tag_vocab_path)
    path.mkdir(parents=True, exist_ok=True)
    with (path
            / (args.file.conllu_file + '.pkl')).open("wb+", ) as f:
        pickle.dump(sup2id, f)


def prepare_training_data(
        train_data: Sequence[Sequence[extraction.Token]],
        eval_data: Sequence[Sequence[extraction.Token]],
        dataset_name: str,
        tag_system: Mapping[str, int],
        model_path: str,
        batch_size: int,
        factorised: bool = False,
        train_fraction: float = 1.0,
        eval_fraction: float = 1.0,
        ) -> tuple[
            dataset.TaggingDataset, dataset.TaggingDataset,
            DataLoader, DataLoader]:

    tokeniser = transformers.AutoTokenizer.from_pretrained(
        model_path.split("/")[-1], truncation=True, use_fast=True)

    if factorised:
        factorised_max_left_right = config.get_max_lr(tag_system)
    else:
        factorised_max_left_right = None

    train_dataset = dataset.TaggingDataset(
        "train", tokeniser, tag_system, train_data, device, dataset_name,
        factorised_max_left_right=factorised_max_left_right,
        fraction=train_fraction)
    eval_dataset = dataset.TaggingDataset(
        "eval", tokeniser, tag_system, eval_data, device, dataset_name,
        factorised_max_left_right=factorised_max_left_right,
        fraction=eval_fraction)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size,
        collate_fn=train_dataset.collate,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=eval_dataset.collate,
        pin_memory=True
    )
    return train_dataset, eval_dataset, train_dataloader, eval_dataloader


def prepare_test_data(
        test_data: Sequence[Sequence[extraction.Token]],
        dataset_name: str,
        tag_system: Mapping[str, int],
        model_path: str,
        batch_size: int,
        factorised: bool = False) -> tuple[dataset.TaggingDataset, DataLoader]:

    print(f"Evaluating {model_path}")
    tokeniser = transformers.AutoTokenizer.from_pretrained(
        model_path.split("/")[-1], truncation=True, use_fast=True)

    if factorised:
        factorised_max_left_right = config.get_max_lr(tag_system)
    else:
        factorised_max_left_right = None

    test_dataset = dataset.TaggingDataset(
        "test", tokeniser, tag_system, test_data, device,
        dataset_name, factorised_max_left_right=factorised_max_left_right,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=test_dataset.collate
    )
    return test_dataset, test_dataloader


def register_run_metrics(
        writer, run_name,
        # lr,
        epochs, tag_accuracy: float | None = None,
        pos_accuracy: None | float = None, arc_accuracy: None | float = None,
        deprel_accuracy: None | float = None,
        factorised_accuracies: dict[str, float] = dict(),
        xpos_accuracy: None | float = None,
        feats_accuracies: dict[str, float] = dict(),
        subtypes_accuracies: dict[str, float] = dict()):
    add_dict = {}
    if pos_accuracy is not None:
        add_dict['pos_accuracy'] = pos_accuracy
    if xpos_accuracy is not None:
        add_dict['xpos_accuracy'] = xpos_accuracy
    if tag_accuracy is not None:
        add_dict['tag_accuracy'] = tag_accuracy
    if arc_accuracy is not None:
        add_dict['arc_accuracy'] = arc_accuracy
    if deprel_accuracy is not None:
        add_dict['deprel_accuracy'] = deprel_accuracy
    for f_name, acc in factorised_accuracies.items():
        add_dict[f_name] = acc
    for f_name, acc in feats_accuracies.items():
        add_dict[f_name] = acc
    for s_name, acc in subtypes_accuracies.items():
        add_dict[s_name] = acc

    writer.add_hparams(
        {
            'run_name': run_name,  # 'lr': lr,
            'epochs': epochs},
        add_dict)


def get_accuracies(
        writer, n_iter, use_tensorboard,
        sup_predictions, eval_sup_labels,
        pos_predictions, eval_pos_labels,
        arc_predictions, eval_arc_labels,
        deprel_predictions, eval_deprel_labels,
        factorised_predictions, eval_factorised_labels,
        xpos_predictions, eval_xpos_labels,
        feats_predictions, eval_feats_labels,
        subtypes_predictions, eval_subtypes_labels,
        f_supertag_logps, printinfo: bool = True,
        *, k: int = 1):
    dev_sup_acc = None
    dev_pos_acc = None
    dev_xpos_acc = None
    dev_arc_acc = None
    dev_deprel_acc = None
    dev_factorised_accs = dict()
    dev_feats_accs = dict()
    dev_subtypes_accs = dict()
    if k == 1:
        func = evaluate.calc_tag_accuracy_k
    else:
        func = evaluate.calc_tag_accuracy_upto_k
    if pos_predictions is not None:
        dev_pos_acc = func(
            pos_predictions, eval_pos_labels, writer,
            use_tensorboard, n_iter,
            typ="pos", k=k, printinfo=printinfo)
    if xpos_predictions is not None:
        dev_xpos_acc = func(
            xpos_predictions, eval_xpos_labels, writer,
            use_tensorboard, n_iter,
            typ="xpos", k=k, printinfo=printinfo)
    if arc_predictions is not None:
        dev_arc_acc = func(
            arc_predictions, eval_arc_labels, writer,
            use_tensorboard, n_iter,
            typ="arc", k=k, printinfo=printinfo)
    if deprel_predictions is not None:
        dev_deprel_acc = func(
            deprel_predictions, eval_deprel_labels, writer,
            use_tensorboard, n_iter,
            typ="deprel", k=k, printinfo=printinfo)
    if sup_predictions is not None:
        dev_sup_acc = func(
            sup_predictions, eval_sup_labels, writer,
            use_tensorboard, n_iter,
            typ="sup", k=k, printinfo=printinfo)
        evaluate.calc_tag_accuracy_upto_k(
            sup_predictions, eval_sup_labels, writer,
            use_tensorboard, n_iter,
            typ="sup", k=10, printinfo=True)
    for f_name, f_predictions in factorised_predictions.items():
        dev_factorised_accs[f_name] = func(
            f_predictions, eval_factorised_labels[f_name],
            writer, use_tensorboard, n_iter,
            typ=f_name, k=k, printinfo=printinfo
        )
    for f_name, f_predictions in feats_predictions.items():
        dev_feats_accs[f_name] = func(
            f_predictions, eval_feats_labels[f_name],
            writer, use_tensorboard, n_iter,
            typ=f_name, k=k, printinfo=printinfo
        )
    for s_name, s_predictions in subtypes_predictions.items():
        dev_subtypes_accs[s_name] = func(
            s_predictions, eval_subtypes_labels[s_name],
            writer, use_tensorboard, n_iter,
            typ=s_name, k=k, printinfo=printinfo
        )

    if f_supertag_logps is not None:
        dev_sup_acc = func(
            f_supertag_logps, eval_sup_labels, writer,
            use_tensorboard, n_iter,
            typ="sup", k=k, printinfo=printinfo)

    return (
        dev_sup_acc, dev_pos_acc, dev_arc_acc,
        dev_deprel_acc, dev_factorised_accs,
        dev_xpos_acc, dev_feats_accs, dev_subtypes_accs)


@dataclasses.dataclass
class TrainState():
    n_iter: int
    best_metric: float
    last_metric: float
    tol: int
    epochs: int
    epo: int
    log_dir: str = ""

    def save(self, dir: pathlib.Path, run_name: str) -> None:
        with open(dir / (run_name + "_train_state.json"), "w") as f:
            f.write(json.dumps(dataclasses.asdict(self)))

    @classmethod
    def load(cls: Type[Self], dir: pathlib.Path, run_name: str) -> Self:
        with open(dir / (run_name + "_train_state.json"), "r") as f:
            dictionary = json.load(f)
        return cls(**dictionary)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1)[..., np.newaxis])
    return e_x / e_x.sum(axis=-1)[..., np.newaxis]


def gather_deprels_for_gold_arcs(
        deprel_predictions: None | np.ndarray,
        eval_deprel_labels: None | np.ndarray,
        eval_arc_labels: None | np.ndarray,
        ) -> np.ndarray | None:
    if (
            deprel_predictions is not None
            and eval_deprel_labels is not None
            ):
        assert eval_arc_labels is not None

        # eval_arc_labels: [B, D]
        # Values:
        #   -1 for ROOT/padding
        #    0 for ROOT as head
        #   >0 for word heads
        #
        # Replace -1 with 0 solely to obtain a legal gather index.
        safe_heads = np.maximum(eval_arc_labels, 0)
        # [B, D]

        # deprel_predictions: [B, D, H, L]
        head_indices = safe_heads[..., np.newaxis, np.newaxis]
        # [B, D, 1, 1]

        deprel_predictions_ = np.take_along_axis(
            deprel_predictions,
            head_indices,
            axis=2,
        )
        # [B, D, 1, L]

        deprel_predictions_ = np.squeeze(
            deprel_predictions_,
            axis=2,
        )
        # [B, D, L]
        return deprel_predictions_
    return deprel_predictions


def prepare_data_and_loaders(
        file_args: settings.FileSettings,
        dep_args: settings.DepSettings,
        model_path: str,
        batch_size: int,
        ) -> tuple[
            dataset.TaggingDataset, dataset.TaggingDataset,
            DataLoader, DataLoader]:
    data_path = pathlib.Path(file_args.data_folder)
    prefix: str = file_args.conllu_file

    train_reader = data.load_conllu(prefix, "train", dir=data_path)
    dev_reader = data.load_conllu(prefix, "dev", dir=data_path)
    logging.info("Preparing Data")

    train_data, sup2id = extraction.prepare_train(
        train_reader,
        arguments=dep_args.arguments,
        adjuncts=dep_args.adjuncts,
        delete=dep_args.delete,
        merged=dep_args.merged,
        without_labels=not dep_args.labelled,
        distinguish_fallback_subtypes=not dep_args.labelled,
        merged_fallback_subtypes=dep_args.merged_fallback_subtypes,
        distinguish_merged_fallback_subtypes=(
            dep_args.distinguish_merged_fallback_subtypes),
        order_relations=dep_args.order_relations,
        )
    dev_data = extraction.prepare(
        dev_reader,
        arguments=dep_args.arguments,
        adjuncts=dep_args.adjuncts,
        delete=dep_args.delete,
        merged=dep_args.merged,
        without_labels=not dep_args.labelled,
        distinguish_fallback_subtypes=not dep_args.labelled,
        merged_fallback_subtypes=dep_args.merged_fallback_subtypes,
        distinguish_merged_fallback_subtypes=(
            dep_args.distinguish_merged_fallback_subtypes),
        order_relations=dep_args.order_relations,
        )

    logging.info(f"Loaded {len(train_data)} training sentences.")
    logging.info(f"Loaded {len(dev_data)} evaluation sentences.")
    train_dataset, dev_dataset, train_dataloader, dev_dataloader = (
        prepare_training_data(
            train_data, dev_data, prefix,
            sup2id, model_path, batch_size,
            factorised=True,
            train_fraction=file_args.train_fraction,
            eval_fraction=file_args.eval_fraction,))

    return train_dataset, dev_dataset, train_dataloader, dev_dataloader


def train_command(
        args: settings.Settings | None = None,
        tagging_settings: settings.TaggingSettings | None = None,
        file_settings: settings.FileSettings | None = None,
        dep_settings: settings.DepSettings | None = None,
        data: tuple[
            dataset.TaggingDataset,
            dataset.TaggingDataset,
            DataLoader,
            DataLoader] | None = None,
        save_model: bool = True,
        final_eval: bool = True) -> Iterator[float]:
    if data is None:
        assert args is not None
        tagging_settings = args.tagging
        dep_settings = args.deprels
        file_settings = args.file
        (
            train_dataset, dev_dataset,
            train_dataloader, dev_dataloader
        ) = prepare_data_and_loaders(
            file_settings, args.deprels,
            args.tagging.model_path, args.tagging.batch_size)
    else:
        assert tagging_settings is not None
        assert file_settings is not None
        assert dep_settings is not None
        (
            train_dataset, dev_dataset,
            train_dataloader, dev_dataloader
        ) = data

    sup2id = train_dataset.sup2id
    id2sup = train_dataset.id2sup
    id2sup_relative = train_dataset.id2sup_relative

    max_l = train_dataset.max_l
    max_r = train_dataset.max_r

    id2pos = train_dataset.id2pos
    id2deprel = train_dataset.id2deprel

    logging.info("Initializing the model")
    tagging_model = config.initialise_model(
        sup2id, tagging_settings.model_path,
        train_pos=tagging_settings.train_pos,
        train_xpos=tagging_settings.train_xpos,
        train_feats=tagging_settings.train_feats,
        num_pos_tags=len(train_dataset.pos_dict),
        num_xpos_tags=len(train_dataset.xpos_dict),
        num_deprel_tags=len(train_dataset.deprel_dict),
        num_sup_deprel_tags=len(train_dataset.sup_deprel_dict),
        num_feats_tags={
            feat: len(dic) for feat, dic in train_dataset.feats_dicts.items()},
        train_deprel=tagging_settings.train_deprel,
        train_arc=tagging_settings.train_arc,
        train_sup=tagging_settings.train_sup,
        factorised=tagging_settings.factorised,
        extra_num_labels={
            subtype: len(dic) for subtype, dic
            in train_dataset.subtypes_dicts.items()},
        train_subtypes=tagging_settings.train_subtypes,
        pos_label_smoothing=tagging_settings.pos_label_smoothing,
        xpos_label_smoothing=tagging_settings.xpos_label_smoothing,
        arc_label_smoothing=tagging_settings.arc_label_smoothing,
        deprel_label_smoothing=tagging_settings.deprel_label_smoothing,
        sup_label_smoothing=tagging_settings.sup_label_smoothing,
        feats_label_smoothing=tagging_settings.feats_label_smoothing,
        subtypes_label_smoothing=tagging_settings.subtypes_label_smoothing,
        proj_drop=tagging_settings.proj_drop,
        arc_drop=tagging_settings.arc_drop,
        deprel_drop=tagging_settings.deprel_drop,
        mix_drop=tagging_settings.mix_drop,
        deprel_hidden=tagging_settings.deprel_hidden,
        arc_hidden=tagging_settings.arc_hidden,
    )
    assert tagging_model is not None
    tagging_model.to(device)

    run_name = (
        file_settings.conllu_file + "-" + tagging_settings.model_name + "-"
        # + str(tagging_settings.lr) + "-"
        + str(tagging_settings.epochs))

    train_set_size = len(train_dataloader)
    optimizer, scheduler, num_training_steps = (
        config.initialize_optimizer_and_scheduler(
            tagging_model,
            num_batches_per_epoch=train_set_size,
            num_epochs=tagging_settings.epochs,
            grad_acc=tagging_settings.grad_acc,
            encoder_lr=tagging_settings.encoder_lr,
            head_lr=tagging_settings.head_lr,
            weight_decay=tagging_settings.weight_decay,
            warmup_epochs=tagging_settings.warmup_epochs,
            # tagging_settings.encoder_lr, tagging_settings.head_lr
        )
    )

    if tagging_settings.mode != "init":
        logging.info("Loading model state dict")
        tagging_model.load_state_dict(
            torch.load(
                pathlib.Path(
                    tagging_settings.output_path) / (run_name + "_last")))
        optimizer.load_state_dict(
            torch.load(
                pathlib.Path(
                    tagging_settings.output_path
                ) / (run_name + "_opt")
            )
        )
        scheduler.load_state_dict(
                    torch.load(
                        pathlib.Path(
                            tagging_settings.output_path
                        ) / (run_name + "_sch")
                    )
                )

    scaler = GradScaler(
        "cpu" if device == torch.device("cpu") else "cuda")

    optimizer.zero_grad()

    logging.info("Starting The Training Loop")
    tagging_model.train()

    if tagging_settings.mode in ("init", "add"):
        n_iter = 0
        best_metric: float = 0
        last_metric: float = 0
        tol = tagging_settings.tol
        epochs = tagging_settings.epochs
        epo = 0
        if tagging_settings.use_tensorboard:
            writer = SummaryWriter(comment=run_name)
    elif tagging_settings.mode == "continue":
        train_state = TrainState.load(
            pathlib.Path(tagging_settings.output_path), run_name)
        n_iter = train_state.n_iter
        best_metric = train_state.best_metric
        last_metric = train_state.last_metric
        tol = train_state.tol
        epochs = train_state.epochs
        epo = train_state.epo
        if tagging_settings.use_tensorboard:
            writer = SummaryWriter(train_state.log_dir, comment=run_name)
    else:
        raise Exception(
            f"tagging_settings.mode '{tagging_settings.mode}' unknown")

    if not tagging_settings.use_tensorboard:
        writer = None

    pcgrad_params = [
        p
        for p in tagging_model.encoder.parameters()
        if p.requires_grad
    ]

    pcgrad_accumulator = pcgrad.make_accumulator(
        pcgrad_params
    )

    seen_factors = None
    valid_factors = None
    valid_supertag2id = None
    valid_id2sup = None
    valid_id2sup_relative = None
    if tagging_settings.factorised is not False:
        if tagging_settings.factorised in ("complete", "seen"):
            seen_factors = factorisation.preprocess_supertags(
                sup2id,
                train_dataset.deprel_dict,
                max_l,
                max_r,
            )

        if tagging_settings.eval_metric.startswith("a*"):
            # print(len(train_dataset.deprel_dict)); raise Exception
            if tagging_settings.factorised == "structural":
                valid_supertag2id = (
                    factorisation.generate_valid_structural_supertag2id(
                        max_l=max_l,
                        max_r=max_r,
                        mode="projective",
                    ))

                valid_factors = factorisation.preprocess_supertags(
                    valid_supertag2id,
                    {"_": 0},
                    max_l,
                    max_r,
                )
                valid_id2sup = {
                    i: sup for sup, i in valid_supertag2id.items()}
                valid_id2sup_relative = {
                    i: extraction.convert_string_to_relative_relation(tag)
                    for i, tag in valid_id2sup.items()}

    # freeze_factor = 5

    # three load methods: load / continue / init
    # state dict:
    # n_iter, best_acc, tol, tagging_settings.epochs, epo

    loss_weights = defaultdict(
        lambda: 1.0,
        tagging_settings.loss_weights if tagging_settings.loss_weights is not None
        else dict())

    # TODO: change
    k_supertag = tagging_settings.k_supertag
    k_head_scores = tagging_settings.k_head_scores
    t_sup: float = tagging_settings.t_sup
    t_arc: float = tagging_settings.t_arc

    accum_count = 0
    eval_count = 0

    for epo in tqdm.tqdm(range(epo, tagging_settings.epochs)):
        # if (epo+1) % freeze_factor == 0:
        #     for name, param in model.named_parameters():
        #         if "bert" in name:
        #             try:
        #                 if int(name.split(".")[3]) >= 12-(
        #                         (epo+1) % freeze_factor):
        #                     param.requires_grad = True
        #             except ValueError:
        #                 if (epo+1) % freeze_factor == 12:
        #                     param.requires_grad = True
        #                 else:
        #                     param.requires_grad = False

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"requires gradient: {name}")

        grad_acc = tagging_settings.grad_acc

        logging.info(f"*******************EPOCH {epo}*******************")
        t = 1

        with tqdm.tqdm(train_dataloader, disable=False) as progbar:
            for i, batch in enumerate(progbar):
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.amp.autocast(
                        "cpu" if device == torch.device("cpu") else "cuda",
                        enabled=True, dtype=torch.float16
                        ):
                    # main losses:
                    # MST las: arc + deprel
                    # MST uas: arc
                    # A* unfactorised no merge: arc + sup
                    # A* unfactorised merge las: arc + sup + pos
                    # A* unfactorised merge uas: arc + sup
                    # A* factorised all/seen deprels from A* no merge: arc + factorised
                    # A* factorised all/seen deprels from A* merge las: arc + factorised + pos
                    # A* factorised all/seen deprels from A* merge uas: arc + factorised
                    # A* factorised all/seen deprels not from A* las: arc + factorised + deprel
                    # A* factorised all/seen deprels not from A* uas: arc + factorised
                    # A* factorised structural from A* las: arc + factorised + deprel
                    # A* factorised structural from A* uas: arc + factorised

                    logits: model.TaggingLogits
                    word_mask: torch.Tensor
                    logits, word_mask = tagging_model(**batch)

                    losses: model.TaggingLosses
                    losses = tagging_model.calc_losses(
                        logits, word_mask, **batch
                    )

                    sup_loss = losses["sup"]
                    pos_loss = losses["pos"]
                    arc_loss = losses["arc"]
                    deprel_loss = losses["deprel"]
                    factorised_losses = losses["factorised"]
                    xpos_loss = losses["xpos"]
                    feats_losses = losses["feats"]
                    subtypes_losses = losses["subtypes"]

                    if factorised_losses is not None and len(
                            factorised_losses) > 0:
                        sup_loss = torch.stack(
                            list(factorised_losses.values())).mean()
                        losses["sup"] = sup_loss

                    primary_loss_names = {"arc"}
                    if "a*" in tagging_settings.eval_metric:
                        primary_loss_names.add("sup")

                    if (
                            tagging_settings.eval_metric == "mst-las"
                            or (
                                not tagging_settings.deprels_from_supertags
                                and "uas" not in tagging_settings.eval_metric)):
                        primary_loss_names.add("deprel")

                    if (
                            dep_settings.merged is not None
                            and len(dep_settings.merged) > 0
                            and "a*" in tagging_settings.eval_metric
                            and tagging_settings.deprels_from_supertags):
                        primary_loss_names.add("pos")

                    auxiliary_loss_names = set(
                        losses.keys()) - primary_loss_names

                    num_active_losses = len(primary_loss_names)
                    loss: torch.Tensor = torch.zeros(
                        (1,), device="cpu" if device == torch.device("cpu")
                        else "cuda")

                    primary_loss = torch.stack(
                        [
                            losses[name]*loss_weights[name]  # type: ignore
                            for name in primary_loss_names]).sum()
                    loss = primary_loss

                    pcgrad_aux_losses = {}
                    for name in auxiliary_loss_names:
                        aux_loss = losses[name]  # type: ignore

                        if aux_loss is not None:
                            if isinstance(aux_loss, dict):
                                if len(aux_loss) == 0:
                                    continue

                                aux_loss = torch.stack(
                                    list(aux_loss.values())).mean()
                            loss = loss + aux_loss * loss_weights[name]
                            num_active_losses += 1

                            pcgrad_aux_losses[name] = (
                                aux_loss, loss_weights[name])

                    global_loss_scale = 1.0 / num_active_losses

                    loss = loss * global_loss_scale

                    if pcgrad_aux_losses:
                        pcgrad.accumulate_task_gradients(
                            accumulator=pcgrad_accumulator,
                            primary_loss=primary_loss,
                            aux_losses=pcgrad_aux_losses,
                            shared_params=pcgrad_params,
                            scaler=scaler,
                            grad_acc=grad_acc,
                            global_loss_scale=global_loss_scale,
                        )

                scaler.scale(loss / grad_acc).backward()

                accum_count += 1

                if accum_count == tagging_settings.grad_acc:
                    pcgrad_corrections = None
                    pcgrad_stats = {}  # type: ignore

                    (
                        pcgrad_corrections,
                        pcgrad_stats,
                    ) = pcgrad.compute_corrections(
                        accumulator=pcgrad_accumulator,
                        scaler=scaler,
                    )

                    if tagging_settings.use_tensorboard:
                        for name, stats in pcgrad_stats.items():
                            if (
                                    stats["cosine"] is None
                                    or stats["norm_ratio"] is None
                                    or stats["coefficient"] is None):
                                continue

                            writer.add_scalar(
                                f"GradCosine/{name}_vs_Parse",
                                stats["cosine"].item(),
                                n_iter,
                            )

                            writer.add_scalar(
                                f"GradNorm/{name}_to_Parse",
                                stats["norm_ratio"].item(),
                                n_iter,
                            )

                            writer.add_scalar(
                                f"PCGrad/{name}_coefficient",
                                stats["coefficient"].item(),
                                n_iter,
                            )

                            if stats["projected_norm_ratio"] is not None:
                                writer.add_scalar(
                                    f"GradNorm/{name}_projected_to_Parse",
                                    stats["projected_norm_ratio"].item(),
                                    n_iter,
                                )

                    with torch.no_grad():
                        for p, correction in zip(
                                pcgrad_params,
                                pcgrad_corrections,
                                ):

                            if correction is None:
                                continue

                            if p.grad is None:
                                p.grad = correction.clone()
                            else:
                                p.grad.add_(correction)

                    if tagging_settings.use_tensorboard:
                        assert writer is not None
                        writer.add_scalar(
                            'Loss/train', loss, n_iter)
                        if sup_loss is not None:
                            writer.add_scalar(
                                'SupLoss/train', sup_loss, n_iter)
                        if arc_loss is not None:
                            writer.add_scalar(
                                'ArcLoss/train', arc_loss, n_iter)
                        if pos_loss is not None:
                            writer.add_scalar(
                                'PosLoss/train', pos_loss, n_iter)
                        if xpos_loss is not None:
                            writer.add_scalar(
                                'XposLoss/train', xpos_loss, n_iter)
                        if deprel_loss is not None:
                            writer.add_scalar(
                                'DeprelLoss/train', deprel_loss, n_iter)
                        if factorised_losses is not None:
                            for f_name, f_loss in factorised_losses.items():
                                writer.add_scalar(
                                    f'{f_name}Loss/train', f_loss, n_iter)
                        if feats_losses is not None:
                            for f_name, f_loss in feats_losses.items():
                                writer.add_scalar(
                                    f'{f_name}Loss/train', f_loss, n_iter)
                        if subtypes_losses is not None:
                            for f_name, f_loss in subtypes_losses.items():
                                writer.add_scalar(
                                    f'{f_name}Loss/train', f_loss, n_iter)
                    progbar.set_postfix(loss=loss.item())

                    scaler.unscale_(optimizer)
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        tagging_model.parameters(), 1.0)

                    writer.add_scalar(
                        "GradNorm/TotalBeforeClip",
                        total_norm.item(),
                        n_iter,
                    )
                    clip_factor = min(
                        1.0,
                        1.0 / (total_norm.item() + 1e-6),
                    )
                    writer.add_scalar(
                        "GradNorm/ClipFactor",
                        clip_factor,
                        n_iter,
                    )

                    # debug_optimizer_devices(model, optimizer)
                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()
                    optimizer.zero_grad()

                    pcgrad_accumulator = pcgrad.make_accumulator(
                        pcgrad_params
                    )

                    n_iter += 1
                    t += 1
                    accum_count = 0
                    eval_count += 1

                    for j, group in enumerate(optimizer.param_groups):
                        writer.add_scalar(
                            f"LR/group_{j}",
                            group["lr"],
                            n_iter,
                        )

                if (
                        (
                            tagging_settings.eval_steps is not None
                            and tagging_settings.eval_steps == eval_count)
                        or (
                            tagging_settings.eval_steps is None
                            and len(train_dataloader) == i+1
                        )):
                    eval_count = 0

                    # evaluation at the end of epoch or at eval steps
                    (
                        predictions, eval_labels,
                        pos_predictions, eval_pos_labels,
                        arc_predictions, eval_arc_labels,
                        deprel_predictions, eval_deprel_labels,
                        factorised_predictions, eval_factorised_labels,
                        xpos_predictions, eval_xpos_labels,
                        feats_predictions, eval_feats_labels,
                        subtypes_predictions, eval_subtypes_labels,
                        dev_loss,
                        dev_sup_loss, dev_pos_loss,
                        dev_arc_loss, dev_deprel_loss,
                        dev_factorised_losses,
                        dev_xpos_loss,
                        dev_feats_losses,
                        dev_subtypes_losses,) = (
                        evaluate.predict(
                            tagging_model, dev_dataloader, len(dev_dataset),
                            len(sup2id), tagging_settings.batch_size, device,
                            report_loss=True,
                            deprels_matrix=True)
                    )

                    if tagging_settings.use_tensorboard:
                        assert writer is not None
                        writer.add_scalar(
                            'Loss/dev', dev_loss, n_iter)
                        if dev_sup_loss is not None:
                            writer.add_scalar(
                                'SupLoss/dev', dev_sup_loss, n_iter)
                        if dev_arc_loss is not None:
                            writer.add_scalar(
                                'ArcLoss/dev', dev_arc_loss, n_iter)
                        if dev_pos_loss is not None:
                            writer.add_scalar(
                                'PosLoss/dev', dev_pos_loss, n_iter)
                        if dev_xpos_loss is not None:
                            writer.add_scalar(
                                'XposLoss/dev', dev_xpos_loss, n_iter)
                        if dev_deprel_loss is not None:
                            writer.add_scalar(
                                'DeprelLoss/dev', dev_deprel_loss, n_iter)
                        for f_name, f_dev_loss in dev_factorised_losses.items():
                            writer.add_scalar(
                                f'{f_name}Loss/dev', f_dev_loss, n_iter
                            )
                        for f_name, f_dev_loss in dev_feats_losses.items():
                            writer.add_scalar(
                                f'{f_name}Loss/dev', f_dev_loss, n_iter
                            )
                        for f_name, f_dev_loss in dev_subtypes_losses.items():
                            writer.add_scalar(
                                f'{f_name}Loss/dev', f_dev_loss, n_iter
                            )

                    deprel_predictions_ = gather_deprels_for_gold_arcs(
                        deprel_predictions, eval_deprel_labels,
                        eval_arc_labels
                    )
                    subtypes_predictions_ = {
                        s_name: gather_deprels_for_gold_arcs(
                            s_preds, eval_subtypes_labels[s_name],
                            eval_arc_labels
                        )
                        for s_name, s_preds in subtypes_predictions.items()}

                    seen_supertag_logps = None
                    if tagging_settings.factorised in ("seen", "complete"):
                        assert seen_factors is not None
                        seen_supertag_logps = (
                            factorisation.score_supertags_batch(
                                seen_factors,
                                {
                                    f_name: -utils.neg_log10_softmax(
                                        f_pred / t_sup)
                                    for f_name, f_pred
                                    in factorised_predictions.items()},
                                -utils.neg_log10_softmax(
                                    factorised_predictions["l_arg_nums"]
                                    / t_sup),
                                -utils.neg_log10_softmax(
                                    factorised_predictions["r_arg_nums"]
                                    / t_sup),
                                -utils.neg_log10_softmax(
                                    factorised_predictions["aux_positions"]
                                    / t_sup),
                                -utils.neg_log10_softmax(
                                    factorised_predictions["aux_rel_ids"]
                                    / t_sup),
                            ))
                    (
                        dev_sup_acc, dev_pos_acc, dev_arc_acc,
                        dev_deprel_acc, dev_factorised_accs,
                        dev_xpos_acc, dev_feats_accs,
                        dev_subtypes_accs,) = (
                        get_accuracies(
                            writer, n_iter, tagging_settings.use_tensorboard,
                            predictions, eval_labels,
                            pos_predictions, eval_pos_labels,
                            arc_predictions, eval_arc_labels,
                            deprel_predictions_, eval_deprel_labels,
                            factorised_predictions, eval_factorised_labels,
                            xpos_predictions, eval_xpos_labels,
                            feats_predictions, eval_feats_labels,
                            subtypes_predictions_, eval_subtypes_labels,
                            f_supertag_logps=seen_supertag_logps,
                            printinfo=False,
                            )
                    )

                    if tagging_settings.use_tensorboard:
                        assert writer is not None
                        if dev_sup_acc is not None:
                            writer.add_scalar(
                                'sup_acc/dev',
                                dev_sup_acc, n_iter)
                        if dev_pos_acc is not None:
                            writer.add_scalar(
                                'pos_acc/dev',
                                dev_pos_acc, n_iter)
                        if dev_xpos_acc is not None:
                            writer.add_scalar(
                                'xpos_acc/dev',
                                dev_xpos_acc, n_iter)
                        if dev_arc_acc is not None:
                            writer.add_scalar(
                                'arc_acc/dev',
                                dev_arc_acc, n_iter
                            )
                        if dev_deprel_acc is not None:
                            writer.add_scalar(
                                'deprel_acc/dev',
                                dev_deprel_acc, n_iter
                            )
                        for f_name, f_dev_acc in dev_factorised_accs.items():
                            writer.add_scalar(
                                f'{f_name}_acc/dev',
                                f_dev_acc, n_iter,
                            )
                        for f_name, f_dev_acc in dev_feats_accs.items():
                            writer.add_scalar(
                                f'{f_name}_acc/dev',
                                f_dev_acc, n_iter,
                            )
                        for s_name, s_dev_acc in dev_subtypes_accs.items():
                            writer.add_scalar(
                                f'{s_name}_acc/dev',
                                s_dev_acc, n_iter,
                            )
                        # add reporting of rebuilt factorised supertags

                    combined_acc = 0
                    if dev_sup_acc is not None:
                        combined_acc += dev_sup_acc
                    if dev_pos_acc is not None:
                        combined_acc += dev_pos_acc
                    if dev_xpos_acc is not None:
                        combined_acc += dev_xpos_acc
                    if dev_arc_acc is not None:
                        combined_acc += dev_arc_acc
                    if dev_deprel_acc is not None:
                        combined_acc += dev_deprel_acc
                    for f_dev_acc in dev_factorised_accs.values():
                        combined_acc += f_dev_acc
                    for f_dev_acc in dev_feats_accs.values():
                        combined_acc += f_dev_acc
                    # combined_acc /= num_losses

                    assert eval_labels is not None
                    eval_metric: float = evaluate.get_eval_metric(
                        tagging_settings.eval_metric,
                        tagging_settings.factorised,
                        tagging_settings.deprels_from_supertags,
                        combined_acc=combined_acc,
                        sup_predictions=predictions,
                        arc_predictions=arc_predictions,
                        pos_predictions=(
                            pos_predictions if dep_settings.merged is not None
                            and len(dep_settings.merged) > 0 else None),
                        deprel_predictions=deprel_predictions,
                        factorised_predictions=factorised_predictions,
                        seen_supertag_logps=seen_supertag_logps,
                        eval_sup_labels=eval_labels,
                        eval_arc_labels=eval_arc_labels,
                        eval_deprel_labels=eval_deprel_labels,
                        id2pos=id2pos,
                        id2deprel=id2deprel,
                        deprel2id=train_dataset.deprel_dict,
                        id2sup=id2sup,
                        sup2id=sup2id,
                        id2sup_relative=id2sup_relative,
                        valid_id2sup=valid_id2sup,
                        valid_id2sup_relative=valid_id2sup_relative,
                        valid_factors=valid_factors,
                        max_l=max_l,
                        max_r=max_r,
                        k_supertag=k_supertag,
                        k_head_scores=k_head_scores,
                        t_arc=t_arc,
                        t_sup=t_sup,
                        sup_score_scale=tagging_settings.sup_score_scale,
                    )

                    writer.add_scalar(
                        f'{tagging_settings.eval_metric}/dev',
                        eval_metric, n_iter)

                    if dev_pos_acc is not None:
                        logging.info("current pos acc {}".format(dev_pos_acc))
                    if dev_xpos_acc is not None:
                        logging.info("current xpos acc {}".format(
                            dev_xpos_acc))
                    if dev_arc_acc is not None:
                        logging.info("current arc acc {}".format(dev_arc_acc))
                    if dev_deprel_acc is not None:
                        logging.info("current deprel acc {}".format(
                            dev_deprel_acc))
                    if dev_sup_acc is not None:
                        logging.info("current supertag acc {}".format(
                            dev_sup_acc))
                    for f_name, f_dev_acc in dev_factorised_accs.items():
                        logging.info("current {} acc {}".format(
                            f_name, f_dev_acc))
                    for f_name, f_dev_acc in dev_feats_accs.items():
                        logging.info("current {} acc {}".format(
                            f_name, f_dev_acc))
                    for s_name, s_dev_acc in dev_subtypes_accs.items():
                        logging.info("current {} acc {}".format(
                            s_name, s_dev_acc))
                    if eval_metric is not None:
                        logging.info("eval metric {}".format(eval_metric))
                    logging.info("last metric {}".format(last_metric))
                    logging.info("best metric {}".format(best_metric))
                    logging.info("tol {}".format(tol))

                    if save_model:
                        _save_model(
                            tagging_model, pathlib.Path(
                                tagging_settings.output_path),
                            run_name + "_last")
                        _save_optimiser(
                            optimizer, pathlib.Path(
                                tagging_settings.output_path), run_name)
                        _save_scheduler(
                            scheduler, pathlib.Path(
                                tagging_settings.output_path), run_name)

                    # if dev_metrics.fscore > last_fscore or dev_loss < last...
                    last_metric = eval_metric
                    if eval_metric > best_metric:
                        tol = 99999
                        logging.info("tol refill")
                        logging.info("save the best model")
                        best_metric = eval_metric
                        logging.info("Saving The Newly Found Best Model")
                        _save_model(
                            tagging_model, pathlib.Path(
                                tagging_settings.output_path), run_name)
                    else:
                        tol -= 1

                    if tol < 0:
                        if final_eval:
                            _finish_training(
                                tagging_model, sup2id, dev_dataloader,
                                dev_dataset, run_name, writer,
                                tagging_settings,
                                n_iter,
                                tagging_settings.factorised,
                                seen_factors,
                                t_sup=t_sup)
                        return
                    # end of epoch

                    train_state = TrainState(
                        n_iter, best_metric, last_metric,
                        tol, epochs, epo+1,
                        str(writer.log_dir) if writer is not None else ""
                    )
                    train_state.save(
                        pathlib.Path(tagging_settings.output_path), run_name)

                    yield eval_metric
                    pass

    if final_eval:
        _finish_training(
            tagging_model, sup2id, dev_dataloader, dev_dataset,
            run_name, writer, tagging_settings, n_iter,
            tagging_settings.factorised, seen_factors,
            t_sup=t_sup)


def _save_model(
        model: torch.nn.Module, output_path: pathlib.Path, run_name: str):
    os.makedirs(output_path, exist_ok=True)
    to_save_file = os.path.join(output_path, run_name)
    torch.save(model.state_dict(), to_save_file)


def _save_optimiser(
        optimiser: torch.nn.Module, output_path: pathlib.Path, run_name: str):
    os.makedirs(output_path, exist_ok=True)
    to_save_file = os.path.join(output_path, run_name + "_opt")
    torch.save(optimiser.state_dict(), to_save_file)


def _save_scheduler(
        scheduler: torch.nn.Module, output_path: pathlib.Path, run_name: str):
    os.makedirs(output_path, exist_ok=True)
    to_save_file = os.path.join(output_path, run_name + "_sch")
    torch.save(scheduler.state_dict(), to_save_file)


def debug_optimizer_devices(model, optimizer):
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("current device:", torch.cuda.current_device())

    for name, p in model.named_parameters():
        if p.requires_grad:
            print(
                name,
                "param:", p.device,
                "grad:", None if p.grad is None else p.grad.device,
                "shape:", tuple(p.shape),
            )
            state = optimizer.state.get(p, {})
            for k, v in state.items():
                if torch.is_tensor(v):
                    print(f"  state[{k}]:", v.device, tuple(v.shape))
                else:
                    print(f"  state[{k}]:", type(v).__name__, v)
            print("-" * 60)


def _finish_training(
        model: torch.nn.Module,
        sup2id: Mapping[str, int],
        eval_dataloader: DataLoader,
        eval_dataset: dataset.TaggingDataset,
        run_name: str,
        writer: None | SummaryWriter,
        args: settings.TaggingSettings,
        n_iter: int,
        factorised: Literal[
            "seen", "complete", "structural", False],
        seen_factors: factorisation.SupertagFactors | None = None,
        t_sup: float = 1):

    (
        predictions, eval_labels,
        pos_predictions, eval_pos_labels,
        arc_predictions, eval_arc_labels,
        deprel_predictions, eval_deprel_labels,
        factorised_predictions, eval_factorised_labels,
        xpos_predictions, eval_xpos_labels,
        feats_predictions, eval_feats_labels,
        subtypes_predictions, eval_subtypes_labels,
        *_) = (
        evaluate.predict(
            model, eval_dataloader, len(eval_dataset),
            len(sup2id), args.batch_size,
            device))

    seen_supertag_logps = None
    if factorised in ("seen", "complete"):
        assert seen_factors is not None
        seen_supertag_logps = factorisation.score_supertags_batch(
            seen_factors,
            {
                f_name: -utils.neg_log10_softmax(f_pred / t_sup)
                for f_name, f_pred in factorised_predictions.items()},
            -utils.neg_log10_softmax(
                factorised_predictions["l_arg_nums"] / t_sup),
            -utils.neg_log10_softmax(
                factorised_predictions["r_arg_nums"] / t_sup),
            -utils.neg_log10_softmax(
                factorised_predictions["aux_positions"] / t_sup),
            -utils.neg_log10_softmax(
                factorised_predictions["aux_rel_ids"] / t_sup),
        )

    (
        sup_acc, pos_acc, arc_acc,
        deprel_acc, dev_factorised_accs,
        dev_xpos_accs, dev_feats_accs,
        dev_subtypes_accs,) = (
        get_accuracies(
            writer, n_iter, args.use_tensorboard,
            predictions, eval_labels,
            pos_predictions, eval_pos_labels,
            arc_predictions, eval_arc_labels,
            deprel_predictions, eval_deprel_labels,
            factorised_predictions, eval_factorised_labels,
            xpos_predictions, eval_xpos_labels,
            feats_predictions, eval_feats_labels,
            subtypes_predictions, eval_subtypes_labels,
            seen_supertag_logps,
            printinfo=False
            )
    )

    register_run_metrics(
        writer, run_name,  # args.lr,
        args.epochs, sup_acc, pos_acc, arc_acc, deprel_acc,
        dev_factorised_accs, dev_xpos_accs, dev_feats_accs,
        dev_subtypes_accs,)


def evaluate_command(args: settings.Settings, k: int = 1):
    data_path: pathlib.Path = pathlib.Path(
        args.file.data_folder)

    print("Evaluation Args", args)
    prefix: str = args.file.conllu_file

    test_reader = data.load_conllu(prefix, "test", dir=data_path)
    test_data = extraction.prepare(
        test_reader,
        arguments=args.deprels.arguments,
        adjuncts=args.deprels.adjuncts,
        delete=args.deprels.delete,
        merged=args.deprels.merged,
        without_labels=not args.deprels.labelled,
        distinguish_fallback_subtypes=not args.deprels.labelled,
        merged_fallback_subtypes=args.deprels.merged_fallback_subtypes,
        distinguish_merged_fallback_subtypes=(
            args.deprels.distinguish_merged_fallback_subtypes),
        order_relations=args.deprels.order_relations,
        )

    sup2id = initialize_tag_system(
        prefix, pathlib.Path(args.tagging.tag_vocab_path)
    )
    id2sup = {i: sup for sup, i in sup2id.items()}
    id2sup_relative = {
        i: extraction.convert_string_to_relative_relation(tag)
        for i, tag in id2sup.items()}

    id2relative_sup: dict[
        int, None | extraction.ProjectiveTag]
    id2relative_sup = {
        i: extraction.process_relative_tag_to_projective(
            extraction.convert_string_to_relative_relation(sup))
        for i, sup in id2sup.items()}
    lr_args = [
        extraction.get_lr_argnum(tag)
        for tag in id2relative_sup.values() if tag is not None]
    max_l = max([lr[0] for lr in lr_args])
    max_r = max([lr[1] for lr in lr_args])

    writer = SummaryWriter(comment=args.tagging.model_name)

    logging.info("Preparing Data")
    eval_dataset, eval_dataloader = prepare_test_data(
        test_data, prefix, sup2id, args.tagging.model_path,
        args.tagging.batch_size, args.tagging.factorised is not False)

    id2pos = {
        i: pos for pos, i in eval_dataset.pos_dict.items()}
    id2deprel = {
        i: deprel for deprel, i in eval_dataset.deprel_dict.items()}

    id2pos = {i: pos for pos, i in eval_dataset.pos_dict.items()}

    model = config.initialise_model(
        sup2id,
        args.tagging.model_path,
        num_pos_tags=len(eval_dataset.pos_dict),
        num_xpos_tags=len(eval_dataset.xpos_dict),
        num_deprel_tags=len(eval_dataset.deprel_dict),
        num_sup_deprel_tags=len(eval_dataset.sup_deprel_dict),
        num_feats_tags={
            feat: len(dic) for feat, dic in eval_dataset.feats_dicts.items()},
        train_deprel=args.tagging.train_deprel,
        train_arc=args.tagging.train_arc,
        train_sup=args.tagging.train_sup,
        train_pos=args.tagging.train_pos,
        train_xpos=args.tagging.train_xpos,
        train_feats=args.tagging.train_feats,
        factorised=args.tagging.factorised,
        extra_num_labels={
            subtype: len(dic)
            for subtype, dic
            in eval_dataset.subtypes_dicts.items()},
        train_subtypes=args.tagging.train_subtypes,
        pos_label_smoothing=args.tagging.pos_label_smoothing,
        xpos_label_smoothing=args.tagging.xpos_label_smoothing,
        arc_label_smoothing=args.tagging.arc_label_smoothing,
        deprel_label_smoothing=args.tagging.deprel_label_smoothing,
        sup_label_smoothing=args.tagging.sup_label_smoothing,
        feats_label_smoothing=args.tagging.feats_label_smoothing,
        subtypes_label_smoothing=args.tagging.subtypes_label_smoothing,
        proj_drop=args.tagging.proj_drop,
        arc_drop=args.tagging.arc_drop,
        deprel_drop=args.tagging.deprel_drop,
        mix_drop=args.tagging.mix_drop,
        deprel_hidden=args.tagging.deprel_hidden,
        arc_hidden=args.tagging.arc_hidden,)

    assert model is not None

    model.load_state_dict(
        torch.load(
            pathlib.Path(
                args.tagging.output_path) / args.tagging.eval_model_name),
        strict=False)
    model.to(device)

    seen_factors = None
    valid_factors = None
    valid_supertag2id = None
    valid_id2sup = None
    valid_id2sup_relative = None
    if args.tagging.factorised is not False:
        if args.tagging.factorised in ("complete", "seen"):
            seen_factors = factorisation.preprocess_supertags(
                sup2id,
                eval_dataset.deprel_dict,
                max_l,
                max_r,
            )

        if args.tagging.eval_metric.startswith("a*"):
            # print(len(train_dataset.deprel_dict)); raise Exception
            if args.tagging.factorised == "structural":
                valid_supertag2id = (
                    factorisation.generate_valid_structural_supertag2id(
                        max_l=max_l,
                        max_r=max_r,
                        mode="projective",
                    ))

                valid_factors = factorisation.preprocess_supertags(
                    valid_supertag2id,
                    {"_": 0},
                    max_l,
                    max_r,
                )
                valid_id2sup = {
                    i: sup for sup, i in valid_supertag2id.items()}
                valid_id2sup_relative = {
                    i: extraction.convert_string_to_relative_relation(tag)
                    for i, tag in valid_id2sup.items()}

    (
        predictions, eval_labels,
        pos_predictions, eval_pos_labels,
        arc_predictions, eval_arc_labels,
        deprel_predictions, eval_deprel_labels,
        factorised_predictions, eval_factorised_labels,
        xpos_predictions, eval_xpos_labels,
        feats_predictions, eval_feats_labels,
        subtypes_predictions, eval_subtypes_labels,
        *_) = (
        evaluate.predict(
            model, eval_dataloader, len(eval_dataset),
            len(sup2id), args.tagging.batch_size, device,
            deprels_matrix=True)
        )

    deprel_predictions_ = deprel_predictions

    if (
            deprel_predictions is not None
            and eval_deprel_labels is not None
            ):
        assert eval_arc_labels is not None

        deprel_predictions_ = evaluate.select_deprel_logits(
            deprel_predictions,
            eval_arc_labels,
        )
        # [B, D, L]

    subtypes_predictions_ = {
        s_name: evaluate.select_deprel_logits(
            s_preds,
            eval_arc_labels,
        )
        for s_name, s_preds
        in subtypes_predictions.items()
        if eval_arc_labels is not None
    }

    t_sup: float = args.tagging.t_sup
    t_arc: float = args.tagging.t_arc

    seen_supertag_logps = None
    if args.tagging.factorised in ("seen", "complete"):
        assert seen_factors is not None
        seen_supertag_logps = factorisation.score_supertags_batch(
            seen_factors,
            {
                f_name: -utils.neg_log10_softmax(f_pred / t_sup)
                for f_name, f_pred in factorised_predictions.items()},
            -utils.neg_log10_softmax(
                factorised_predictions["l_arg_nums"] / t_sup),
            -utils.neg_log10_softmax(
                factorised_predictions["r_arg_nums"] / t_sup),
            -utils.neg_log10_softmax(
                factorised_predictions["aux_positions"] / t_sup),
            -utils.neg_log10_softmax(
                factorised_predictions["aux_rel_ids"] / t_sup),
        )
    (
        dev_sup_accs, dev_pos_accs, dev_arc_accs,
        dev_deprel_accs, dev_factorised_accs,
        dev_xpos_accs, dev_feats_accs,
        dev_subtypes_accs,) = (
        get_accuracies(
            writer, 0, args.tagging.use_tensorboard,
            predictions, eval_labels,
            pos_predictions, eval_pos_labels,
            arc_predictions, eval_arc_labels,
            deprel_predictions_, eval_deprel_labels,
            factorised_predictions, eval_factorised_labels,
            xpos_predictions, eval_xpos_labels,
            feats_predictions, eval_feats_labels,
            subtypes_predictions_, eval_subtypes_labels,
            seen_supertag_logps,
            k=k, printinfo=False
            )
    )
    if k > 1:
        for k in range(1, k+1):
            if dev_sup_accs is not None:
                print(
                    f"sup_acc k={k}:", dev_sup_accs[k-1])
            if dev_pos_accs is not None:
                print(
                    f"pos_acc k={k}:", dev_pos_accs[k-1])
            if dev_xpos_accs is not None:
                print(
                    f"xpos_acc k={k}:", dev_xpos_accs[k-1])
            if dev_arc_accs is not None:
                print(
                    f"arc_acc k={k}:", dev_arc_accs[k-1])
            if dev_deprel_accs is not None:
                print(
                    f"deprel_acc k={k}:", dev_deprel_accs[k-1])
            for f_name, f_dev_accs in dev_factorised_accs.items():
                print(
                    f"{f_name}_acc k={k}:", f_dev_accs[k-1]
                )
            for f_name, f_dev_accs in dev_feats_accs.items():
                print(
                    f"{f_name}_acc k={k}:", f_dev_accs[k-1]
                )
            for s_name, s_dev_accs in dev_subtypes_accs.items():
                print(
                    f"{s_name}_acc k={k}:", s_dev_accs[k-1]
                )

    else:
        if dev_sup_accs is not None:
            print(
                f"sup_acc k={k}:", dev_sup_accs)
        if dev_pos_accs is not None:
            print(
                f"pos_acc k={k}:", dev_pos_accs)
        if dev_xpos_accs is not None:
            print(
                f"xpos_acc k={k}:", dev_xpos_accs)
        if dev_arc_accs is not None:
            print(
                f"arc_acc k={k}:", dev_arc_accs)
        if dev_deprel_accs is not None:
            print(
                f"deprel_acc k={k}:", dev_deprel_accs)
        for f_name, f_dev_accs in dev_factorised_accs.items():
            print(
                f"{f_name}_acc k={k}:", f_dev_accs)
        for f_name, f_dev_accs in dev_feats_accs.items():
            print(
                f"{f_name}_acc k={k}:", f_dev_accs)
        for s_name, s_dev_accs in dev_subtypes_accs.items():
            print(
                f"{s_name}_acc k={k}:", s_dev_accs)

    assert eval_labels is not None
    eval_metric: float = get_eval_metric(
        args.tagging.eval_metric,
        args.tagging.factorised,
        args.tagging.deprels_from_supertags,
        combined_acc=0,
        sup_predictions=predictions,
        arc_predictions=arc_predictions,
        pos_predictions=(
            pos_predictions if args.deprels.merged is not None
            and len(args.deprels.merged) > 0 else None),
        deprel_predictions=deprel_predictions,
        factorised_predictions=factorised_predictions,
        seen_supertag_logps=seen_supertag_logps,
        eval_sup_labels=eval_labels,
        eval_arc_labels=eval_arc_labels,
        eval_deprel_labels=eval_deprel_labels,
        id2pos=id2pos,
        id2deprel=id2deprel,
        deprel2id=eval_dataset.deprel_dict,
        id2sup=id2sup,
        sup2id=sup2id,
        id2sup_relative=id2sup_relative,
        valid_id2sup=valid_id2sup,
        valid_id2sup_relative=valid_id2sup_relative,
        valid_factors=valid_factors,
        max_l=max_l,
        max_r=max_r,
        k_supertag=args.tagging.k_supertag,
        k_head_scores=args.tagging.k_head_scores,
        t_arc=t_arc,
        t_sup=t_sup,
        sup_score_scale=args.tagging.sup_score_scale,
    )

    print(
        f"eval metric {args.tagging.eval_metric}:", eval_metric)


def predict_command(args: settings.Settings):
    data_path: pathlib.Path = pathlib.Path(
        args.file.data_folder)

    print("predict Args", args)

    prefix: str = args.file.conllu_file

    pred_reader = data.load_conllu(prefix, args.file.split, dir=data_path)
    pred_data = extraction.prepare(
        pred_reader,
        arguments=args.deprels.arguments,
        adjuncts=args.deprels.adjuncts,
        delete=args.deprels.delete,
        merged=args.deprels.merged,
        without_labels=not args.deprels.labelled,
        distinguish_fallback_subtypes=not args.deprels.labelled,
        merged_fallback_subtypes=args.deprels.merged_fallback_subtypes,
        distinguish_merged_fallback_subtypes=(
            args.deprels.distinguish_merged_fallback_subtypes),
        order_relations=args.deprels.order_relations,
        )

    logging.info("Initializing Tag System")
    sup2id = initialize_tag_system(
        prefix, pathlib.Path(args.tagging.tag_vocab_path)
    )

    logging.info("Preparing Data")
    pred_dataset, pred_dataloader = prepare_test_data(
        pred_data, prefix, sup2id, args.tagging.model_path,
        args.tagging.batch_size, args.tagging.factorised is not False)

    model = config.initialise_model(
        sup2id, args.tagging.model_path,
        num_pos_tags=len(pred_dataset.pos_dict),
        num_xpos_tags=len(pred_dataset.xpos_dict),
        num_deprel_tags=len(
            pred_dataset.deprel_dict) if args.tagging.train_deprel else None,
        num_sup_deprel_tags=len(pred_dataset.sup_deprel_dict),
        num_feats_tags={
            feat: len(dic) for feat, dic in pred_dataset.feats_dicts.items()},
        train_arc=args.tagging.train_arc, train_sup=args.tagging.train_sup,
        train_pos=args.tagging.train_pos, train_xpos=args.tagging.train_xpos,
        train_feats=args.tagging.train_feats,
        factorised=args.tagging.factorised,
        extra_num_labels={
            subtype: len(dic)
            for subtype, dic
            in pred_dataset.subtypes_dicts.items()},
        train_subtypes=args.tagging.train_subtypes,
        pos_label_smoothing=args.tagging.pos_label_smoothing,
        xpos_label_smoothing=args.tagging.xpos_label_smoothing,
        arc_label_smoothing=args.tagging.arc_label_smoothing,
        deprel_label_smoothing=args.tagging.deprel_label_smoothing,
        sup_label_smoothing=args.tagging.sup_label_smoothing,
        feats_label_smoothing=args.tagging.feats_label_smoothing,
        subtypes_label_smoothing=args.tagging.subtypes_label_smoothing,
        proj_drop=args.tagging.proj_drop,
        arc_drop=args.tagging.arc_drop,
        deprel_drop=args.tagging.deprel_drop,
        mix_drop=args.tagging.mix_drop,
        deprel_hidden=args.tagging.deprel_hidden,
        arc_hidden=args.tagging.arc_hidden,)
    assert model is not None

    model.load_state_dict(
        torch.load(
            pathlib.Path(
                args.tagging.output_path) / args.tagging.eval_model_name))
    model.to(device)

    (
        predictions, _,
        pos_predictions, _,
        arc_predictions, _,
        deprel_predictions, _,
        factorised_predictions, _,
        xpos_predictions, _,
        feats_predictions, _,
        subtypes_predictions, _,
        *_) = (
        evaluate.predict(
            model, pred_dataloader, len(pred_dataset),
            len(sup2id), args.tagging.batch_size, device,
            deprels_from_pred_head=True))

    id2sup = {i: sup for sup, i in sup2id.items()}
    pred_ids = None
    if predictions is not None:
        pred_ids = predictions.argmax(-1)

    id2pos = {i: sup for sup, i in pred_dataset.pos_dict.items()}
    pred_pos_ids = None
    if pos_predictions is not None:
        pred_pos_ids = pos_predictions.argmax(-1)

    pred_heads = None
    if arc_predictions is not None:
        pred_heads = arc_predictions.argmax(-1)+1

    id2deprel = {i: deprel for deprel, i in pred_dataset.deprel_dict.items()}
    pred_deprel_ids = None
    if deprel_predictions is not None:
        pred_deprel_ids = deprel_predictions.argmax(-1)+1

    with open(
            pathlib.Path(
                args.tagging.output_path,
                args.tagging.model_name
                + ".preds"),
            "w") as fout:
        print(
            "Saving predictions to",
            args.tagging.output_path + "/" + args.tagging.model_name
            + ".preds")
        for i in range(len(pred_dataset)):
            # for i, (pred_sen, label_sen) in enumerate(
            #   zip(pred_ids, eval_labels)):
            for j in range(len(pred_data[i])):
                results = []
                if pred_ids is not None:
                    sup = pred_ids[i][j+1]  # account for BOS token?
                    sup_out = id2sup[sup] if sup in id2sup else "UNK"
                    results.append(sup_out)
                if pred_pos_ids is not None:
                    pos = pred_pos_ids[i][j+1]
                    pos_out = id2pos[pos] if pos in id2pos else "UNK"
                    if pos not in id2pos:
                        print(pos)
                    results.append(pos_out)
                if pred_heads is not None:
                    head = pred_heads[i][j+1]
                    results.append(str(head))
                if pred_deprel_ids is not None:
                    deprel = pred_deprel_ids[i][j+1]
                    deprel_out = id2deprel[
                        deprel] if deprel in id2deprel else "UNK"
                    results.append(deprel_out)
                fout.write(
                    "\t".join(results) + "\n"
                )
            # sen = pred_sen[label_sen != -1]
            # pos_sen = None
            # if pred_pos_ids is not None:
            #     pos_sen = pred_pos_ids[i][label_sen != -1]
            # for j, sup in enumerate(sen):
            #     sup_out = id2sup[sup] if sup in id2sup else "UNK"
            #     if pred_pos_ids is not None and pos_sen is not None:
            #         fout.write(
            #             sup_out + "\t" +
            #             (id2pos[
            #                   pos_sen[j]] if pos_sen[j] in id2pos else "UNK")
            #             + "\n")
            #     else:
            #         fout.write(
            #             (sup_out) + "\n")
            fout.write("\n")

# TODO: check RoBERTa, BERT-large

# TODO: check A* LAS without deprel learning
