from . import model
from .. import settings, extraction

import math
import torch.nn as nn
from bitsandbytes.optim import AdamW8bit
import transformers

from typing import Literal, Mapping


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
        factorised: settings.Factorised = False,
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

    config = model.AutoConfig.from_pretrained(
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
        "proj_drop": 0.1,
        "arc_drop": 0.2,
        "deprel_drop": 0.3,
        "mix_drop": 0.1,
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

        "train_arc": True,
        "train_pos": train_pos,
        "train_xpos": train_xpos,

        "mlp_arc_hidden": 500 if train_arc else None,

        "mlp_lab_hidden": (
            100 if train_deprel or train_subtypes else None
        ),
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
    tagging_model = model.ModelForTagging(config=config)  # type: ignore
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
