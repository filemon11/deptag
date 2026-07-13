import os
import logging
import pickle
import json

import numpy as np
import torch
import transformers
from bitsandbytes.optim import Adam8bit
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp.grad_scaler import GradScaler
import tqdm
import pathlib
from . import model, dataset, evaluate
from .. import extraction, data, settings, parsing
import dataclasses


from typing import Mapping, Sequence, Self, Type

BERT = ("bert-base-multilingual-cased",)

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
    print(sup2id, not args.deprels.labelled)

    path = pathlib.Path(args.tagging.tag_vocab_path)
    path.mkdir(parents=True, exist_ok=True)
    with (path
            / (args.file.conllu_file + '.pkl')).open("wb+", ) as f:
        pickle.dump(sup2id, f)


def prepare_training_data(
        train_data: Sequence[Sequence[tuple[str, str, str, int, str]]],
        eval_data: Sequence[Sequence[tuple[str, str, str, int, str]]],
        dataset_name: str,
        tag_system: Mapping[str, int],
        model_name: str,
        batch_size: int
        ) -> tuple[
            dataset.TaggingDataset, dataset.TaggingDataset,
            DataLoader, DataLoader]:

    tokeniser = transformers.AutoTokenizer.from_pretrained(
        model_name, truncation=True, use_fast=True)

    train_dataset = dataset.TaggingDataset(
        "train", tokeniser, tag_system, train_data, device, dataset_name)
    eval_dataset = dataset.TaggingDataset(
        "eval", tokeniser, tag_system, eval_data, device, dataset_name)

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
        test_data: Sequence[Sequence[tuple[str, str, str, int, str]]],
        dataset_name: str,
        tag_system: Mapping[str, int],
        model_name: str,
        batch_size: int) -> tuple[dataset.TaggingDataset, DataLoader]:

    print(f"Evaluating {model_name}")
    tokeniser = transformers.AutoTokenizer.from_pretrained(
        model_name, truncation=True, use_fast=True)
    test_dataset = dataset.TaggingDataset(
        "test", tokeniser, tag_system, test_data, device,
        dataset_name
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=test_dataset.collate
    )
    return test_dataset, test_dataloader


def generate_config(
        model_type: str, tag_system: Mapping[str, int], model_path: str,
        train_pos: bool = True,
        num_pos_tags: int = 50, num_deprel_tags: int | None = None,
        train_arc: bool = False,
        train_sup: bool = True):
    if model_type in BERT:
        config = transformers.AutoConfig.from_pretrained(
            model_path,
            num_labels=len(tag_system)+1,
        )
        config.task_specific_params = {
                'model_path': model_path,
                'pos_emb_dim': 256,
                'num_pos_tags': num_pos_tags+1,
                # 'lstm_layers': 3,
                'dropout': 0,  # 0.33,
                'use_pos': False,
                'n_heads': 12,
                'transformer_layers': 0,
                'train_pos': train_pos,
                'mlp_arc_hidden': 500 if train_arc is not None else None,
                'mlp_lab_hidden': 100 if num_deprel_tags is not None else None,
                'mlp_dropout': 0.3,
                'mlp_num_labels': (
                    num_deprel_tags+1 if num_deprel_tags is not None
                    else None),
                'train_sup': train_sup,
        }
    else:
        logging.error("Invalid model type.")
        return
    return config


def initialize_model(
        model_type: str, tag_system: Mapping[str, int], model_path: str,
        train_pos: bool = True, num_pos_tags: int = 50,
        num_deprel_tags: int | None = None, train_arc: bool = False,
        train_sup: bool = True,
        ) -> model.ModelForTagging | None:
    config = generate_config(
        model_type, tag_system, model_path, train_pos=train_pos,
        num_pos_tags=num_pos_tags, num_deprel_tags=num_deprel_tags,
        train_arc=train_arc, train_sup=train_sup
    )
    if model_type in BERT:
        m = model.ModelForTagging(config=config)
        # m = torch.compile(m)  # type: ignore
    else:
        logging.error("Invalid model type")
        return None
    return m


def initialize_optimizer_and_scheduler(
        model, dataset_size, lr=5e-5, num_epochs=4,
        num_warmup_steps=160, grad_acc: int = 1):
    num_training_steps = dataset_size // grad_acc * num_epochs
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if "bert" not in n],
            "weight_decay": 0.0,
            "lr": lr * 50, "betas": (0.9, 0.9),
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       "bert" in n and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr, "betas": (0.9, 0.999),
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       "bert" in n and not any(nd in n for nd in no_decay)],
            "weight_decay": 0.1,
            "lr": lr, "betas": (0.9, 0.999),
        },
    ]
    # Freeze all layers
    # for name, param in model.named_parameters():
    #     if "bert" in name:
    #         param.requires_grad = False
    #         try:
    #             if int(name.split(".")[3]) <= 5:
    #                 param.requires_grad = False
    #         except ValueError:
    #             param.requires_grad = False

    optimizer = Adam8bit(
        grouped_parameters, lr=lr
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler, num_training_steps


def register_run_metrics(
        writer, run_name, lr, epochs, tag_accuracy: float | None = None,
        pos_accuracy: None | float = None, arc_accuracy: None | float = None,
        deprel_accuracy: None | float = None):
    add_dict = {}
    if pos_accuracy is not None:
        add_dict['pos_accuracy'] = pos_accuracy
    if tag_accuracy is not None:
        add_dict['tag_accuracy'] = tag_accuracy
    if arc_accuracy is not None:
        add_dict['arc_accuracy'] = arc_accuracy
    if deprel_accuracy is not None:
        add_dict['deprel_accuracy'] = deprel_accuracy

    writer.add_hparams(
        {'run_name': run_name, 'lr': lr, 'epochs': epochs},
        add_dict)


def get_accuracies(
        writer, n_iter, use_tensorboard,
        sup_predictions, eval_sup_labels,
        pos_predictions, eval_pos_labels,
        arc_predictions, eval_arc_labels,
        deprel_predictions, eval_deprel_labels,
        *, k: int = 1):
    dev_sup_acc = None
    dev_pos_acc = None
    dev_arc_acc = None
    dev_deprel_acc = None
    if k == 1:
        func = evaluate.calc_tag_accuracy_k
    else:
        func = evaluate.calc_tag_accuracy_upto_k
    if pos_predictions is not None:
        dev_pos_acc = func(
            pos_predictions, eval_pos_labels, writer,
            use_tensorboard, n_iter,
            typ="pos", k=k)
    if arc_predictions is not None:
        dev_arc_acc = func(
            arc_predictions, eval_arc_labels, writer,
            use_tensorboard, n_iter,
            typ="arc", k=k)
    if deprel_predictions is not None:
        dev_deprel_acc = func(
            deprel_predictions, eval_deprel_labels, writer,
            use_tensorboard, n_iter,
            typ="deprel", k=k)
    if sup_predictions is not None:
        dev_sup_acc = func(
            sup_predictions, eval_sup_labels, writer,
            use_tensorboard, n_iter,
            typ="sup", k=k)

    return dev_sup_acc, dev_pos_acc, dev_arc_acc, dev_deprel_acc


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


def train_command(args: settings.Settings):
    data_path = pathlib.Path(args.file.data_folder)
    prefix: str = args.file.conllu_file

    train_reader = data.load_conllu(prefix, "train", dir=data_path)
    dev_reader = data.load_conllu(prefix, "dev", dir=data_path)
    logging.info("Preparing Data")

    train_data, sup2id = extraction.prepare_train(
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
    dev_data = extraction.prepare(
        dev_reader,
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

    logging.info("Preparing Data")
    train_dataset, dev_dataset, train_dataloader, dev_dataloader = (
        prepare_training_data(
            train_data, dev_data, prefix,
            sup2id, args.tagging.model_name, args.tagging.batch_size))

    logging.info("Initializing the model")
    model = initialize_model(
        args.tagging.model_name, sup2id, args.tagging.model_path,
        train_pos=args.tagging.train_pos,
        num_pos_tags=len(train_dataset.pos_dict),
        num_deprel_tags=len(
            train_dataset.deprel_dict) if args.tagging.train_deprel else None,
        train_arc=args.tagging.train_arc, train_sup=args.tagging.train_sup
    )
    assert model is not None
    model.to(device)

    run_name = (
        args.file.conllu_file + "-" + args.tagging.model_name + "-" + str(
            args.tagging.lr) + "-" + str(args.tagging.epochs))

    train_set_size = len(train_dataloader)
    optimizer, scheduler, num_training_steps = (
        initialize_optimizer_and_scheduler(
            model, train_set_size, args.tagging.lr, args.tagging.epochs,
            args.tagging.num_warmup_steps, args.tagging.grad_acc
        )
    )

    if args.tagging.mode != "init":
        logging.info("Loading model state dict")
        model.load_state_dict(
            torch.load(
                pathlib.Path(
                    args.tagging.output_path) / (run_name + "_last")))
        optimizer.load_state_dict(
            torch.load(
                pathlib.Path(
                    args.tagging.output_path
                ) / (run_name + "_opt")
            )
        )
        scheduler.load_state_dict(
                    torch.load(
                        pathlib.Path(
                            args.tagging.output_path
                        ) / (run_name + "_sch")
                    )
                )

    scaler = GradScaler(
        "cpu" if device == torch.device("cpu") else "cuda")

    optimizer.zero_grad()


    logging.info("Starting The Training Loop")
    model.train()

    if args.tagging.mode in ("init", "add"):
        n_iter = 0
        best_metric: float = 0
        last_metric: float = 0
        tol = args.tagging.tol
        epochs = args.tagging.epochs
        epo = 0
        if args.tagging.use_tensorboard:
            writer = SummaryWriter(comment=run_name)
    elif args.tagging.mode == "continue":
        train_state = TrainState.load(
            pathlib.Path(args.tagging.output_path), run_name)
        n_iter = train_state.n_iter
        best_metric = train_state.best_metric
        last_metric = train_state.last_metric
        tol = train_state.tol
        epochs = train_state.epochs
        epo = train_state.epo
        if args.tagging.use_tensorboard:
            writer = SummaryWriter(train_state.log_dir, comment=run_name)
    else:
        raise Exception(f"args.tagging.mode '{args.tagging.mode}' unknown")

    if not args.tagging.use_tensorboard:
        writer = None

    # freeze_factor = 5

    # three load methods: load / continue / init
    # state dict:
    # n_iter, best_acc, tol, args.tagging.epochs, epo

    for epo in tqdm.tqdm(range(epo, args.tagging.epochs)):
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

        logging.info(f"*******************EPOCH {epo}*******************")
        t = 1
        model.train()

        with tqdm.tqdm(train_dataloader, disable=False) as progbar:
            for i, batch in enumerate(progbar):
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.amp.autocast(
                        "cpu" if device == torch.device("cpu") else "cuda",
                        enabled=True, dtype=torch.bfloat16
                        ):
                    outputs = model(**batch)

                    sup_loss = outputs[0]
                    pos_loss = outputs[2]
                    arc_loss = outputs[4]
                    deprel_loss = outputs[6]

                    loss: torch.Tensor = torch.zeros(
                        (1,), device="cpu" if device == torch.device("cpu")
                        else "cuda")
                    num_losses: int = 0
                    if sup_loss is not None:
                        loss += sup_loss
                        num_losses += 1
                    if arc_loss is not None:
                        loss += arc_loss
                        num_losses += 1
                    if pos_loss is not None:
                        # loss = args.tagging.loss_ratio*loss + (
                        #     1-args.tagging.loss_ratio)*pos_loss
                        loss += pos_loss
                        num_losses += 1
                    if deprel_loss is not None:
                        loss += deprel_loss
                        num_losses += 1
                    loss /= num_losses

                scaler.scale(loss / args.tagging.grad_acc).backward()

                if (i + 1) % args.tagging.grad_acc == 0:
                    if args.tagging.use_tensorboard:
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
                        if deprel_loss is not None:
                            writer.add_scalar(
                                'DeprelLoss/train', deprel_loss, n_iter)
                    progbar.set_postfix(loss=loss.item())

                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    # debug_optimizer_devices(model, optimizer)
                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()
                    optimizer.zero_grad()

                    n_iter += 1
                    t += 1

        if True:  # evaluation at the end of epoch
            (
                predictions, eval_labels,
                pos_predictions, eval_pos_labels,
                arc_predictions, eval_arc_labels,
                deprel_predictions, eval_deprel_labels,
                dev_loss,
                dev_sup_loss, dev_pos_loss,
                dev_arc_loss, dev_deprel_loss) = (
                evaluate.predict(
                    model, dev_dataloader, len(dev_dataset),
                    len(sup2id), args.tagging.batch_size, device,
                    report_loss=True,
                    deprels_matrix=True)
            )
            if args.tagging.use_tensorboard:
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
                if dev_deprel_loss is not None:
                    writer.add_scalar(
                        'DeprelLoss/dev', dev_deprel_loss, n_iter)

            if (
                    deprel_predictions is not None
                    and eval_deprel_labels is not None):
                assert eval_arc_labels is not None
                hds = eval_arc_labels + (eval_arc_labels < 0)
                # [B, S]
                hds = hds[..., np.newaxis, np.newaxis].repeat(
                    deprel_predictions.shape[-1], axis=-1)
                # [B, S, 1, N]

                # deprel_predictions, [B, S, Slab, N]

                deprel_predictions = np.take_along_axis(
                    deprel_predictions, hds, axis=2)
                # [B, S, 1, N]
                deprel_predictions = np.squeeze(
                    deprel_predictions, axis=2)
                # [B, S, N]

            dev_sup_acc, dev_pos_acc, dev_arc_acc, dev_deprel_acc = (
                get_accuracies(
                    writer, n_iter, args.tagging.use_tensorboard,
                    predictions, eval_labels,
                    pos_predictions, eval_pos_labels,
                    arc_predictions, eval_arc_labels,
                    deprel_predictions, eval_deprel_labels
                    )
            )

            if args.tagging.use_tensorboard:
                assert writer is not None
                if dev_sup_acc is not None:
                    writer.add_scalar(
                        'sup_acc/dev',
                        dev_sup_acc, n_iter)
                if dev_pos_acc is not None:
                    writer.add_scalar(
                        'pos_acc/dev',
                        dev_pos_acc, n_iter)
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

            combined_acc = 0
            if dev_sup_acc is not None:
                combined_acc += dev_sup_acc
            if dev_pos_acc is not None:
                combined_acc += dev_pos_acc
            if dev_arc_acc is not None:
                combined_acc += dev_arc_acc
            if dev_deprel_acc is not None:
                combined_acc += dev_deprel_acc
            combined_acc /= num_losses

            # args.tagging.loss_ratio*dev_acc + (
            # 1-args.tagging.loss_ratio)*dev_pos_acc
            eval_metric: float
            match args.tagging.eval_metric:
                case "cacc":
                    eval_metric = combined_acc
                case "a*-las" | "a*-uas":
                    raise NotImplementedError
                    # TODO: implement return k best arcs and sups
                    # provide k best arcs and k best sups to a* alg
                    # get trees
                    # get arcs from trees
                    # (get combined deprels from trees
                    # TODO create mapping: combined deprel x pos tag -> deprel
                    # provide supertag predictions and combined deprels
                    #   to function
                    # get real deprels)
                    # compute las/uas
                case "mst-las" | "mst-uas":
                    assert arc_predictions is not None
                    assert eval_arc_labels is not None
                    mst = parsing.mst(
                        arc_predictions, eval_arc_labels)
                    if args.tagging.eval_metric == "mst_las":
                        assert deprel_predictions is not None
                        assert eval_deprel_labels is not None

                        hds = mst + (mst < 0)
                        # [B, S]
                        hds = hds[..., np.newaxis, np.newaxis].repeat(
                            deprel_predictions.shape[-1], axis=-1)
                        # [B, S, 1, N]

                        # deprel_predictions, [B, S, Slab, N]

                        deprel_predictions_mst = np.take_along_axis(
                            deprel_predictions, hds, axis=2)
                        # [B, S, 1, N]
                        deprel_predictions_mst = np.squeeze(
                            deprel_predictions_mst, axis=2)
                        # [B, S, N]

                        eval_metric = parsing.las(
                            mst, deprel_predictions_mst,
                            eval_arc_labels, eval_deprel_labels
                        )
                    else:
                        eval_metric = parsing.uas(
                            mst, eval_arc_labels)

                    # run mst, get heads
                    # (select deprels using mst heads)
                    # compute las/uas

                    # TODO: add las option for predicted deprel matrix or not
                    # then, select from matrix using predictions there
                case _:
                    raise Exception(
                        f"args.tagging.eval_metric '{args.tagging.eval_metric}"
                        "' unknown")
                # TODO: for arc scoring also supervise the root token.
                # Achieve this by including the BOS token as the
                # artificial root token
            writer.add_scalar(
                f'{args.tagging.eval_metric}/dev', eval_metric, n_iter)

            if dev_pos_acc is not None:
                logging.info("current pos acc {}".format(dev_pos_acc))
            if dev_arc_acc is not None:
                logging.info("current arc acc {}".format(dev_arc_acc))
            if dev_deprel_acc is not None:
                logging.info("current deprel acc {}".format(dev_deprel_acc))
            if dev_sup_acc is not None:
                logging.info("current supertag acc {}".format(dev_sup_acc))
            if dev_pos_acc is not None:
                logging.info("eval metric {}".format(eval_metric))
            logging.info("last metric {}".format(last_metric))
            logging.info("best metric {}".format(best_metric))
            logging.info("tol {}".format(tol))

            _save_model(
                model, pathlib.Path(
                    args.tagging.output_path), run_name + "_last")
            _save_optimiser(
                optimizer, pathlib.Path(
                    args.tagging.output_path), run_name)
            _save_scheduler(
                scheduler, pathlib.Path(
                    args.tagging.output_path), run_name)

            # if dev_metrics.fscore > last_fscore or dev_loss < last...
            last_metric = eval_metric
            if eval_metric > best_metric:
                tol = 99999
                logging.info("tol refill")
                logging.info("save the best model")
                best_metric = eval_metric
                logging.info("Saving The Newly Found Best Model")
                _save_model(
                    model, pathlib.Path(
                        args.tagging.output_path), run_name)
            else:
                tol -= 1

            if tol < 0:
                _finish_training(
                    model, sup2id, dev_dataloader,
                    dev_dataset, run_name, writer, args.tagging,
                    n_iter)
                return
            # end of epoch

            train_state = TrainState(
                n_iter, best_metric, last_metric,
                tol, epochs, epo+1,
                str(writer.log_dir) if writer is not None else ""
            )
            train_state.save(pathlib.Path(args.tagging.output_path), run_name)
            pass

    _finish_training(
        model, sup2id, dev_dataloader, dev_dataset,
        run_name, writer, args.tagging, n_iter)


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
        n_iter: int):

    (
        predictions, eval_labels,
        pos_predictions, eval_pos_labels,
        arc_predictions, eval_arc_labels,
        deprel_predictions, eval_deprel_labels,
        *_) = (
        evaluate.predict(
            model, eval_dataloader, len(eval_dataset),
            len(sup2id), args.batch_size,
            device))

    sup_acc, pos_acc, arc_acc, deprel_acc = (
        get_accuracies(
            writer, n_iter, args.use_tensorboard,
            predictions, eval_labels,
            pos_predictions, eval_pos_labels,
            arc_predictions, eval_arc_labels,
            deprel_predictions, eval_deprel_labels
            )
    )

    register_run_metrics(
        writer, run_name, args.lr,
        args.epochs, sup_acc, pos_acc, arc_acc, deprel_acc)


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

    writer = SummaryWriter(comment=args.tagging.model_name)

    logging.info("Preparing Data")
    eval_dataset, eval_dataloader = prepare_test_data(
        test_data, prefix, sup2id, args.tagging.model_name,
        args.tagging.batch_size)

    model = initialize_model(
        args.tagging.model_name, sup2id, args.tagging.model_path,
        num_pos_tags=len(eval_dataset.pos_dict),
        num_deprel_tags=len(
            eval_dataset.deprel_dict) if args.tagging.train_deprel else None,
        train_arc=args.tagging.train_arc, train_sup=args.tagging.train_sup)
    assert model is not None

    model.load_state_dict(
        torch.load(
            pathlib.Path(
                args.tagging.output_path) / args.tagging.eval_model_name))
    model.to(device)

    (
        predictions, eval_labels,
        pos_predictions, eval_pos_labels,
        arc_predictions, eval_arc_labels,
        deprel_predictions, eval_deprel_labels,
        *_) = (
        evaluate.predict(
            model, eval_dataloader, len(eval_dataset),
            len(sup2id), args.tagging.batch_size, device))

    dev_sup_accs, dev_pos_accs, dev_arc_accs, dev_deprel_accs = (
        get_accuracies(
            writer, 0, args.tagging.use_tensorboard,
            predictions, eval_labels,
            pos_predictions, eval_pos_labels,
            arc_predictions, eval_arc_labels,
            deprel_predictions, eval_deprel_labels,
            k=k
            )
    )

    for k in range(1, k+1):
        if dev_sup_accs is not None:
            print(
                f"sup_acc k={k}:", dev_sup_accs[k-1])
        if dev_pos_accs is not None:
            print(
                f"pos_acc k={k}:", dev_pos_accs[k-1])
        if dev_arc_accs is not None:
            print(
                f"arc_acc k={k}:", dev_arc_accs[k-1])
        if dev_deprel_accs is not None:
            print(
                f"deprel_acc k={k}:", dev_deprel_accs[k-1])


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
        pred_data, prefix, sup2id, args.tagging.model_name,
        args.tagging.batch_size)

    model = initialize_model(
        args.tagging.model_name, sup2id, args.tagging.model_path,
        num_pos_tags=len(pred_dataset.pos_dict),
        num_deprel_tags=len(
            pred_dataset.deprel_dict) if args.tagging.train_deprel else None,
        train_arc=args.tagging.train_arc, train_sup=args.tagging.train_sup)
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
