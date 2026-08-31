import os
import logging
import pickle
import json
import math

import numpy as np
import torch
import transformers
from bitsandbytes.optim import AdamW8bit
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.amp.grad_scaler import GradScaler
import tqdm
import pathlib
from . import model, dataset, evaluate, factorisation, pcgrad
from .. import extraction, data, settings, parsing, utils
import dataclasses
from collections import defaultdict

from typing import Mapping, Sequence, Self, Type, Literal

# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)

import transformers.utils.output_capturing as hf_output_capturing
torch._functorch.config.donated_buffer = False

# Work around Transformers 5.9 + PyTorch 2.6 Dynamo incompatibility.
# This only changes the current Python process.
hf_output_capturing.torch = torch  # type: ignore

torch.set_float32_matmul_precision("medium")

BERT = (
    "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased",
    "xlm-roberta-large")

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
        ) -> tuple[
            dataset.TaggingDataset, dataset.TaggingDataset,
            DataLoader, DataLoader]:

    tokeniser = transformers.AutoTokenizer.from_pretrained(
        model_path.split("/")[-1], truncation=True, use_fast=True)

    if factorised:
        factorised_max_left_right = get_max_lr(tag_system)
    else:
        factorised_max_left_right = None

    train_dataset = dataset.TaggingDataset(
        "train", tokeniser, tag_system, train_data, device, dataset_name,
        factorised_max_left_right=factorised_max_left_right)
    eval_dataset = dataset.TaggingDataset(
        "eval", tokeniser, tag_system, eval_data, device, dataset_name,
        factorised_max_left_right=factorised_max_left_right)

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
        factorised_max_left_right = get_max_lr(tag_system)
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


SUPPORTED_BACKBONES = {
    "bert",
    "roberta",
    "xlm-roberta",
    "albert",
}


def get_max_lr(tag_system: Mapping[str, int]) -> tuple[int, int]:
    factorised_tags = [extraction.convert_relative_tag_to_factorised(
        extraction.convert_string_to_relative_relation(
            sup)) for sup in tag_system]
    max_l = max([tag[0] for tag in factorised_tags])
    max_r = max([tag[2] for tag in factorised_tags])
    return max_l, max_r


def generate_config(
        model_type: str,
        tag_system: Mapping[str, int],
        model_path: str,
        num_deprel_tags: int,
        num_sup_deprel_tags: int,
        train_deprel: bool = False,
        train_pos: bool = True,
        train_xpos: bool = False,
        num_pos_tags: int = 50,
        num_xpos_tags: int = 50,
        train_arc: bool = False,
        train_sup: bool = True,
        factorised: Literal[
            "structural", "complete", "seen", False] = False,
        num_feats_tags: dict[str, int] = {},
        train_feats: bool = False,
        extra_num_labels: None | dict[str, int] = None,
        train_subtypes: bool = False,
        pos_label_smoothing: float = 0.0,
        xpos_label_smoothing: float = 0.0,
        arc_label_smoothing: float = 0.0,
        deprel_label_smoothing: float = 0.0,
        sup_label_smoothing: float = 0.0,
        feats_label_smoothing: float = 0.0,
        subtypes_label_smoothing: float = 0.0,
        ) -> transformers.PretrainedConfig:

    config = transformers.AutoConfig.from_pretrained(
        model_path,
        num_labels=len(tag_system),
    )

    if config.model_type not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported Hugging Face model type "
            f"{config.model_type!r} for checkpoint {model_path!r}."
        )

    num_encoder_layers = config.num_hidden_layers

    max_l = None
    max_r = None
    if factorised is not None:
        max_l, max_r = get_max_lr(tag_system)

    config.task_specific_params = {
        "model_path": model_path,

        # Architecture-dependent backbone properties.
        "encoder_hidden_size": config.hidden_size,
        "encoder_num_layers": num_encoder_layers,
        "encoder_num_attention_heads": config.num_attention_heads,

        "pos_emb_dim": 256,
        "num_pos_tags": num_pos_tags,
        "num_xpos_tags": num_xpos_tags,
        "extra_num_labels": extra_num_labels,
        "train_subtypes": train_subtypes,
        "dropout": 0.1,
        "use_pos": False,

        "n_heads": config.num_attention_heads,
        "transformer_layers": 0,

        # Relative layer selection:
        # BERT-base: 4, 8, 12
        # XLM-R-large: 8, 16, 24
        "pos_layer": round(num_encoder_layers / 3),
        "supertag_layer": num_encoder_layers,
        # round(2 * num_encoder_layers / 3),
        "parse_layer": num_encoder_layers,

        "train_pos": train_pos,
        "train_xpos": train_xpos,

        "mlp_arc_hidden": 500 if train_arc else None,

        "mlp_lab_hidden": (
            200 if train_deprel or train_subtypes else None
            # previously 100
        ),
        "mlp_dropout": 0.2,
        "mlp_num_labels": (
            num_deprel_tags
            if train_deprel
            else None
        ),
        "deprel_num": num_deprel_tags,
        "sup_deprel_num": num_sup_deprel_tags,
        "train_sup": train_sup,
        "factorised": factorised,
        "max_l": max_l,
        "max_r": max_r,

        "num_feats_tags": num_feats_tags,
        "train_feats": train_feats,
        "pos_label_smoothing": pos_label_smoothing,
        "xpos_label_smoothing": xpos_label_smoothing,
        "arc_label_smoothing": arc_label_smoothing,
        "deprel_label_smoothing": deprel_label_smoothing,
        "sup_label_smoothing": sup_label_smoothing,
        "feats_label_smoothing": feats_label_smoothing,
        "subtypes_label_smoothing": subtypes_label_smoothing,
    }

    return config


def initialize_model(
        model_type: str, tag_system: Mapping[str, int], model_path: str,
        num_deprel_tags: int, num_sup_deprel_tags: int,
        train_deprel: bool = False,
        train_pos: bool = True, train_xpos: bool = False,
        train_feats: bool = False,
        num_pos_tags: int = 50,
        num_xpos_tags: int = 50,
        num_feats_tags: dict[str, int] = {},
        train_arc: bool = False,
        train_sup: bool = True,
        factorised: Literal['structural', 'complete', 'seen', False] = False,
        extra_num_labels: dict[str, int] | None = None,
        train_subtypes: bool = False,
        pos_label_smoothing: float = 0.0,
        xpos_label_smoothing: float = 0.0,
        arc_label_smoothing: float = 0.0,
        deprel_label_smoothing: float = 0.0,
        sup_label_smoothing: float = 0.0,
        feats_label_smoothing: float = 0.0,
        subtypes_label_smoothing: float = 0.0,
        ) -> model.ModelForTagging | None:
    config = generate_config(
        model_type, tag_system, model_path, train_pos=train_pos,
        train_xpos=train_xpos,
        num_pos_tags=num_pos_tags, num_xpos_tags=num_xpos_tags,
        num_deprel_tags=num_deprel_tags,
        num_sup_deprel_tags=num_sup_deprel_tags,
        train_arc=train_arc, train_sup=train_sup,
        factorised=factorised, train_deprel=train_deprel,
        num_feats_tags=num_feats_tags,
        train_feats=train_feats,
        extra_num_labels=extra_num_labels,
        train_subtypes=train_subtypes,
        pos_label_smoothing=pos_label_smoothing,
        xpos_label_smoothing=xpos_label_smoothing,
        arc_label_smoothing=arc_label_smoothing,
        deprel_label_smoothing=deprel_label_smoothing,
        sup_label_smoothing=sup_label_smoothing,
        feats_label_smoothing=feats_label_smoothing,
        subtypes_label_smoothing=subtypes_label_smoothing,
    )
    tagging_model = model.ModelForTagging(config=config)
    tagging_model.compile()
    return tagging_model


def initialize_optimizer_and_scheduler(
        model, num_batches_per_epoch,
        num_epochs=500,
        grad_acc: int = 1,
        warmup_epochs: int = 5,
        encoder_lr: float = 1e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 0.01,
        ):
    num_update_steps_per_epoch = math.ceil(
        num_batches_per_epoch / grad_acc
    )

    num_training_steps = (
        num_update_steps_per_epoch * num_epochs
    )

    num_warmup_steps = (
        warmup_epochs * num_update_steps_per_epoch
    )
    # no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']

    no_decay_param_ids = set()

    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for p in module.parameters(recurse=False):
                no_decay_param_ids.add(id(p))

    for name, p in model.named_parameters():
        if name.endswith(".bias"):
            no_decay_param_ids.add(id(p))

    encoder_param_ids = {
        id(p) for p in model.encoder.parameters()
        }

    grouped_parameters = [
        # Newly initialized heads: with decay
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    id(p) not in encoder_param_ids
                    and not id(p) in no_decay_param_ids
                    and "mix.weights" not in n
                    and p.requires_grad
                )
            ],
            "lr": head_lr,
            "weight_decay": weight_decay,
            "betas": (0.9, 0.999),
        },

        # Newly initialized heads: no decay
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    id(p) not in encoder_param_ids
                    and (
                        id(p) in no_decay_param_ids
                        or "mix.weights" in n
                    )
                    and p.requires_grad
                )
            ],
            "lr": head_lr,
            "weight_decay": 0.0,
            "betas": (0.9, 0.999),
        },

        # Pretrained encoder: with decay
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    id(p) in encoder_param_ids
                    and not id(p) in no_decay_param_ids
                    and p.requires_grad
                )
            ],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
            "betas": (0.9, 0.999),
        },

        # Pretrained encoder: no decay
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    id(p) in encoder_param_ids
                    and id(p) in no_decay_param_ids
                    and p.requires_grad
                )
            ],
            "lr": encoder_lr,
            "weight_decay": 0.0,
            "betas": (0.9, 0.999),
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

    optimizer = AdamW8bit(
        grouped_parameters,
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler, num_training_steps


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


def select_deprel_logits(
        deprel_predictions: np.ndarray,
        heads: np.ndarray,
        ) -> np.ndarray:
    """Select dependency-relation logits for chosen heads.

    deprel_predictions:
        [B, D, H, L]

    heads:
        [B, D], with -1 for ROOT/padding.

    Returns:
        [B, D, L]
    """
    safe_heads = np.maximum(heads, 0).astype(np.intp, copy=False)
    # [B, D]

    head_indices = safe_heads[..., None, None]
    # [B, D, 1, 1]

    # Explicitly broadcast across the label dimension.
    head_indices = np.broadcast_to(
        head_indices,
        (
            *safe_heads.shape,
            1,
            deprel_predictions.shape[-1],
        ),
    )
    # [B, D, 1, L]

    selected = np.take_along_axis(
        deprel_predictions,
        head_indices,
        axis=2,
    )
    # [B, D, 1, L]

    return selected.squeeze(2)
    # [B, D, L]


def get_eval_metric(
        eval_metric_type: Literal[
            "cacc", "a*-las", "a*-uas", "mst-las", "mst-uas"],
        factorised: Literal["complete", "structural", "seen", False],
        deprels_from_supertags: bool,
        combined_acc: float,
        sup_predictions: np.ndarray | None,
        arc_predictions: np.ndarray | None,
        pos_predictions: np.ndarray | None,
        deprel_predictions: np.ndarray | None,
        factorised_predictions: Mapping[str, np.ndarray],
        seen_supertag_logps: np.ndarray | None,
        eval_sup_labels: np.ndarray,
        eval_arc_labels: np.ndarray | None,
        eval_deprel_labels: np.ndarray | None,
        id2pos: Mapping[int, str],
        id2deprel: Mapping[int, str],
        deprel2id: Mapping[str, int],
        id2sup: Mapping[int, str],
        sup2id: Mapping[str, int],
        id2sup_relative: Mapping[int, extraction.RelativeTag],
        valid_id2sup: Mapping[int, str] | None,
        valid_id2sup_relative: Mapping[int, extraction.RelativeTag] | None,
        valid_factors: None | factorisation.SupertagFactors,
        max_l: int,
        max_r: int,
        k_supertag: int,
        k_head_scores: int,
        t_sup: float = 1,
        t_arc: float = 1,
        sup_score_scale: float = 1.0,
        ) -> float:
    eval_metric: float
    match eval_metric_type:
        case "cacc":
            eval_metric = combined_acc

        case "a*-las" | "a*-uas":
            root_supertag = "*+root"

            assert arc_predictions is not None
            assert eval_arc_labels is not None
            # assert pos_predictions is not None

            chart_id2sup: Mapping[int, str]
            chart_id2sup_relative: Mapping[int, extraction.RelativeTag]
            if factorised == "complete":
                argument_logps = {
                    f_name: -utils.neg_log10_softmax(f_pred / t_sup)
                    for f_name, f_pred in
                    factorised_predictions.items() if
                    f_name.startswith("left") or
                    f_name.startswith("right")
                }
                candidates = factorisation.top_k_valid_supertags_batch(
                    argument_logps,
                    -utils.neg_log10_softmax(
                        factorised_predictions["l_arg_nums"] / t_sup),
                    -utils.neg_log10_softmax(
                        factorised_predictions["r_arg_nums"] / t_sup),
                    -utils.neg_log10_softmax(
                        factorised_predictions["aux_positions"] / t_sup),
                    -utils.neg_log10_softmax(
                        factorised_predictions["aux_rel_ids"] / t_sup),
                    id2deprel,
                    max_l, max_r, k=k_supertag,
                    projective_only=True,
                    valid_mask=eval_sup_labels != -1,
                )

                (
                    supertag_scores,
                    chart_id2sup,
                    chart_sup2id,
                ) = factorisation.make_batch_supertag_scores(
                    candidates,
                    root_supertag,
                )
                chart_id2sup_relative = {
                    i: extraction.convert_string_to_relative_relation(
                        tag)
                    for i, tag in chart_id2sup.items()}
                root_sup_id = chart_sup2id[root_supertag]
                chart_deprel_dict = deprel2id
            elif factorised == "structural":
                assert valid_factors is not None
                supertag_scores = (
                    -factorisation.score_structural_supertags_batch(
                        valid_factors,
                        -utils.neg_log10_softmax(
                            factorised_predictions["l_arg_nums"] / t_sup),
                        -utils.neg_log10_softmax(
                            factorised_predictions["r_arg_nums"] / t_sup),
                        -utils.neg_log10_softmax(
                            factorised_predictions["aux_positions"] / t_sup),
                    ))
                assert valid_id2sup is not None
                assert valid_id2sup_relative is not None
                chart_id2sup = valid_id2sup
                chart_id2sup_relative = valid_id2sup_relative
                chart_sup2id = {
                    sup: i for i, sup in chart_id2sup.items()}
                chart_deprel_dict = {"_": 0, "dep": 0, "root": 0}
                root_sup_id = chart_sup2id["*+_"]
            elif factorised == "seen":
                assert seen_supertag_logps is not None
                supertag_scores = -seen_supertag_logps
                # supertag_scores = utils.neg_log10_softmax(
                #     seen_supertag_scores / t_sup)
                chart_id2sup = id2sup
                chart_id2sup_relative = id2sup_relative

                chart_deprel_dict = deprel2id
                root_sup_id = sup2id[root_supertag]
            else:
                assert sup_predictions is not None
                supertag_scores = utils.neg_log10_softmax(
                    sup_predictions / t_sup)
                chart_id2sup = id2sup
                chart_id2sup_relative = id2sup_relative
                chart_deprel_dict = deprel2id
                root_sup_id = sup2id[root_supertag]

            # if epo > -1:
            head_preds_astar, deprel_preds_astar = parsing.chart(
                arc_predictions,
                eval_arc_labels,
                supertag_scores,
                chart_id2sup_relative,
                id2pos,
                chart_deprel_dict,
                pos_predictions.argmax(
                    -1) if pos_predictions is not None else None,
                max_l,
                max_r,
                root_sup_id=root_sup_id,
                k_supertag=k_supertag,
                k_head_scores=k_head_scores,
                t_arc=t_arc,
                sup_score_scale=sup_score_scale,
            )

            assert eval_deprel_labels is not None

            if eval_metric_type == "a*-las":

                if not deprels_from_supertags:
                    assert deprel_predictions is not None

                    # deprel_predictions: [B, D, H, L]
                    # head_preds_astar:   [B, D]
                    deprel_logits_astar = select_deprel_logits(
                        deprel_predictions,
                        head_preds_astar,
                    )
                    # [B, D, L]

                    deprel_preds_astar = (
                        deprel_logits_astar.argmax(-1)
                    )
                    # [B, D]

                eval_metric = parsing.las(
                    head_preds_astar,
                    deprel_preds_astar,
                    eval_arc_labels,
                    eval_deprel_labels,
                    id2deprel=id2deprel
                )

            else:  # a*-uas
                eval_metric = parsing.uas(
                    head_preds_astar,
                    eval_arc_labels,
                )

            # else:
            #     eval_metric = 0
            #     tol = 99999

        case "mst-las" | "mst-uas":
            assert arc_predictions is not None
            assert eval_arc_labels is not None

            mst = parsing.mst(
                arc_predictions,
                eval_arc_labels,
            )
            # mst: [B, D]

            if eval_metric_type == "mst-las":
                assert deprel_predictions is not None
                assert eval_deprel_labels is not None

                # deprel_predictions: [B, D, H, L]
                deprel_logits_mst = select_deprel_logits(
                    deprel_predictions,
                    mst,
                )
                # [B, D, L]

                deprel_predictions_mst = (
                    deprel_logits_mst.argmax(-1)
                )
                # [B, D]

                eval_metric = parsing.las(
                    mst,
                    deprel_predictions_mst,
                    eval_arc_labels,
                    eval_deprel_labels,
                )

            else:  # mst-uas
                eval_metric = parsing.uas(
                    mst,
                    eval_arc_labels,
                )

        case _:
            raise Exception(
                f"args.tagging.eval_metric "
                f"'{eval_metric_type}' unknown"
            )

    return eval_metric


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

    logging.info(f"Loaded {len(train_data)} training sentences.")
    logging.info(f"Loaded {len(dev_data)} evaluation sentences.")
    train_dataset, dev_dataset, train_dataloader, dev_dataloader = (
        prepare_training_data(
            train_data, dev_data, prefix,
            sup2id, args.tagging.model_path, args.tagging.batch_size,
            factorised=args.tagging.factorised is not False))

    id2sup = {i: sup for sup, i in sup2id.items()}
    id2sup_relative = {
        i: extraction.convert_string_to_relative_relation(sup)
        for i, sup in id2sup.items()}

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

    id2pos = {
        i: pos for pos, i in train_dataset.pos_dict.items()}
    id2deprel = {
        i: deprel for deprel, i in train_dataset.deprel_dict.items()}

    logging.info("Initializing the model")
    model = initialize_model(
        args.tagging.model_name, sup2id, args.tagging.model_path,
        train_pos=args.tagging.train_pos,
        train_xpos=args.tagging.train_xpos,
        train_feats=args.tagging.train_feats,
        num_pos_tags=len(train_dataset.pos_dict),
        num_xpos_tags=len(train_dataset.xpos_dict),
        num_deprel_tags=len(train_dataset.deprel_dict),
        num_sup_deprel_tags=len(train_dataset.sup_deprel_dict),
        num_feats_tags={
            feat: len(dic) for feat, dic in train_dataset.feats_dicts.items()},
        train_deprel=args.tagging.train_deprel,
        train_arc=args.tagging.train_arc,
        train_sup=args.tagging.train_sup,
        factorised=args.tagging.factorised,
        extra_num_labels={
            subtype: len(dic) for subtype, dic
            in train_dataset.subtypes_dicts.items()},
        train_subtypes=args.tagging.train_subtypes,
        pos_label_smoothing=args.tagging.pos_label_smoothing,
        xpos_label_smoothing=args.tagging.xpos_label_smoothing,
        arc_label_smoothing=args.tagging.arc_label_smoothing,
        deprel_label_smoothing=args.tagging.deprel_label_smoothing,
        sup_label_smoothing=args.tagging.sup_label_smoothing,
        feats_label_smoothing=args.tagging.feats_label_smoothing,
        subtypes_label_smoothing=args.tagging.subtypes_label_smoothing,
    )
    assert model is not None
    model.to(device)

    run_name = (
        args.file.conllu_file + "-" + args.tagging.model_name + "-"
        # + str(args.tagging.lr) + "-"
        + str(args.tagging.epochs))

    train_set_size = len(train_dataloader)
    optimizer, scheduler, num_training_steps = (
        initialize_optimizer_and_scheduler(
            model,
            num_batches_per_epoch=train_set_size,
            num_epochs=args.tagging.epochs,
            grad_acc=args.tagging.grad_acc,
            # args.tagging.encoder_lr, args.tagging.head_lr
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

    pcgrad_params = [
        p
        for p in model.encoder.parameters()
        if p.requires_grad
    ]

    seen_factors = None
    valid_factors = None
    valid_supertag2id = None
    valid_id2sup = None
    valid_id2sup_relative = None
    if args.tagging.factorised is not False:
        if args.tagging.factorised in ("complete", "seen"):
            seen_factors = factorisation.preprocess_supertags(
                sup2id,
                train_dataset.deprel_dict,
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

    # freeze_factor = 5

    # three load methods: load / continue / init
    # state dict:
    # n_iter, best_acc, tol, args.tagging.epochs, epo

    loss_weights = defaultdict(
        lambda: 1.0,
        args.tagging.loss_weights if args.tagging.loss_weights is not None
        else dict())

    # TODO: change
    k_supertag = args.tagging.k_supertag
    k_head_scores = args.tagging.k_head_scores
    t_sup: float = args.tagging.t_sup
    t_arc: float = args.tagging.t_arc

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

                    outputs = model(**batch)

                    sup_loss = outputs[0]
                    pos_loss = outputs[2]
                    arc_loss = outputs[4]
                    deprel_loss = outputs[6]
                    factorised_losses = outputs[8]
                    xpos_loss = outputs[10]
                    feats_losses = outputs[12]
                    subtypes_losses = outputs[14]
                    losses = {
                        "sup_loss": outputs[0],
                        "pos_loss": outputs[2],
                        "arc_loss": outputs[4],
                        "deprel_loss": outputs[6],
                        "xpos_loss": outputs[10],
                        "feats_losses": outputs[12],
                        "subtypes_losses": outputs[14],
                    }

                    if factorised_losses is not None and len(
                            factorised_losses) > 0:
                        sup_loss = torch.stack(
                            list(factorised_losses.values())).mean()
                        losses["sup_loss"] = sup_loss

                    primary_loss_names = {"arc_loss"}
                    if "a*" in args.tagging.eval_metric:
                        primary_loss_names.add("sup_loss")

                    if (
                            args.tagging.eval_metric == "mst-las"
                            or (
                                not args.tagging.deprels_from_supertags
                                and "uas" not in args.tagging.eval_metric)):
                        primary_loss_names.add("deprel_loss")

                    if (
                            args.deprels.merged is not None
                            and len(args.deprels.merged) > 0
                            and "a*" in args.tagging.eval_metric
                            and args.tagging.deprels_from_supertags):
                        primary_loss_names.add("pos_loss")

                    auxiliary_loss_names = set(
                        losses.keys()) - primary_loss_names

                    num_active_losses = len(primary_loss_names)
                    loss: torch.Tensor = torch.zeros(
                        (1,), device="cpu" if device == torch.device("cpu")
                        else "cuda")

                    primary_loss = torch.stack(
                        [
                            losses[name]*loss_weights[name]
                            for name in primary_loss_names]).sum()
                    loss = primary_loss

                    pcgrad_aux_losses = {}
                    for name in auxiliary_loss_names:
                        aux_loss = losses[name]

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

                    pcgrad_corrections = None
                    pcgrad_stats = {}

                    if pcgrad_aux_losses:
                        (
                            pcgrad_corrections,
                            pcgrad_stats,
                        ) = pcgrad.pcgrad_corrections(
                            primary_loss=primary_loss,
                            aux_losses=pcgrad_aux_losses,
                            shared_params=pcgrad_params,
                            scaler=scaler,
                            grad_acc=args.tagging.grad_acc,
                            global_loss_scale=global_loss_scale,
                        )

                if args.tagging.use_tensorboard:
                    for name, stats in pcgrad_stats.items():
                        if stats["cosine"] is None:
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

                scaler.scale(loss / args.tagging.grad_acc).backward()

                if pcgrad_corrections is not None:
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
                    s_preds, eval_subtypes_labels[s_name], eval_arc_labels
                )
                for s_name, s_preds in subtypes_predictions.items()}

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
                dev_sup_acc, dev_pos_acc, dev_arc_acc,
                dev_deprel_acc, dev_factorised_accs,
                dev_xpos_acc, dev_feats_accs,
                dev_subtypes_accs,) = (
                get_accuracies(
                    writer, n_iter, args.tagging.use_tensorboard,
                    predictions, eval_labels,
                    pos_predictions, eval_pos_labels,
                    arc_predictions, eval_arc_labels,
                    deprel_predictions_, eval_deprel_labels,
                    factorised_predictions, eval_factorised_labels,
                    xpos_predictions, eval_xpos_labels,
                    feats_predictions, eval_feats_labels,
                    subtypes_predictions_, eval_subtypes_labels,
                    f_supertag_logps=seen_supertag_logps, printinfo=False,
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
            eval_metric: float = get_eval_metric(
                args.tagging.eval_metric,
                args.tagging.factorised,
                args.tagging.deprels_from_supertags,
                combined_acc=combined_acc,
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
                sup_score_scale=args.tagging.sup_score_scale,
            )

            writer.add_scalar(
                f'{args.tagging.eval_metric}/dev', eval_metric, n_iter)

            if dev_pos_acc is not None:
                logging.info("current pos acc {}".format(dev_pos_acc))
            if dev_xpos_acc is not None:
                logging.info("current xpos acc {}".format(dev_xpos_acc))
            if dev_arc_acc is not None:
                logging.info("current arc acc {}".format(dev_arc_acc))
            if dev_deprel_acc is not None:
                logging.info("current deprel acc {}".format(dev_deprel_acc))
            if dev_sup_acc is not None:
                logging.info("current supertag acc {}".format(dev_sup_acc))
            for f_name, f_dev_acc in dev_factorised_accs.items():
                logging.info("current {} acc {}".format(f_name, f_dev_acc))
            for f_name, f_dev_acc in dev_feats_accs.items():
                logging.info("current {} acc {}".format(f_name, f_dev_acc))
            for s_name, s_dev_acc in dev_subtypes_accs.items():
                logging.info("current {} acc {}".format(s_name, s_dev_acc))
            if eval_metric is not None:
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

            # print("pos mix:", torch.softmax(
            #     model.pos_mix.weights, dim=0).tolist())
            # print("xpos mix:", torch.softmax(
            #     model.xpos_mix.weights, dim=0).tolist())
            # print("arc mix:", torch.softmax(
            #     model.arc_mix.weights, dim=0).tolist())
            # if model.rel_mix is not None:
            #     print("rel mix:", torch.softmax(
            #         model.rel_mix.weights, dim=0).tolist())
            # if model.sup_mix is not None:
            #     print("sup mix:", torch.softmax(
            #         model.sup_mix.weights, dim=0).tolist())
            # if model.sup_arg_mix is not None:
            #     print("sup arg mix:", torch.softmax(
            #         model.sup_arg_mix.weights, dim=0).tolist())
            # if model.sup_head_mix is not None:
            #     print("sup head mix:", torch.softmax(
            #         model.sup_head_mix.weights, dim=0).tolist())
            # if model.feats_mixes is not None:
            #     for feat, mix in model.feats_mixes.items():
            #         print(f"{feat} mix:", torch.softmax(
            #             mix.weights, dim=0).tolist())

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
                    n_iter, args.tagging.factorised, seen_factors,
                    t_sup=t_sup)
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
        run_name, writer, args.tagging, n_iter,
        args.tagging.factorised, seen_factors,
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

    model = initialize_model(
        args.tagging.model_name, sup2id,
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
        subtypes_label_smoothing=args.tagging.subtypes_label_smoothing,)

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

        deprel_predictions_ = select_deprel_logits(
            deprel_predictions,
            eval_arc_labels,
        )
        # [B, D, L]

    subtypes_predictions_ = {
        s_name: select_deprel_logits(
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
        sup_score_scale= args.tagging.sup_score_scale,
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

    model = initialize_model(
        args.tagging.model_name, sup2id, args.tagging.model_path,
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
        subtypes_label_smoothing=args.tagging.subtypes_label_smoothing,)
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
