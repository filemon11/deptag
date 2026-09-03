from . import standards
from .. import data

import typed_settings as ts
import dataclasses
import pathlib

from typing import Literal

SETTINGS_DIR = pathlib.Path("settings/")
DEFAULT_SETTINGS = "default"
"Directory where settings are located"

Factorised = Literal[
    'structural', 'complete',
    'seen', False]
Mode = Literal["init", "continue", "add"]
EvalMetric = Literal[
    "cacc", "mst-las", "mst-uas", "a*-las", "a*-uas"]
Split = Literal["train", "test", "dev"]

IntOrStr = int | str

converter = ts.converters.get_default_cattrs_converter()


def structure_int_or_str(value, _type):
    if isinstance(value, int):
        return value

    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value

    raise TypeError(f"Expected int or str, got {type(value).__name__}")


converter.register_structure_hook(
    IntOrStr,
    structure_int_or_str,
)


@dataclasses.dataclass(frozen=True)
class FileSettings:
    conllu_file: str
    output_file: str
    split: None | Split = None
    standard: str = "default"
    standards_dir: str = str(standards.STANDARDS_DIR)
    standard_from_xml: bool = False
    save_standard_from_xml_dir: str = str(standards.STANDARDS_DIR)
    allow_partial_underspecification: bool = True
    save_standard_from_xml: bool = True
    ud_folder: str = str(data.UD_DIR)
    data_folder: str = str(data.DATA_DIR)
    train_fraction: float = 1.0
    eval_fraction: float = 1.0


@dataclasses.dataclass(frozen=True)
class DepSettings:
    arguments: tuple[str, ...]
    adjuncts: tuple[str, ...]
    delete: tuple[str, ...]
    labelled: bool
    subtypes: bool
    order_relations: bool = True
    merged: None | dict[str, list[str]] = None
    merged_fallback_subtypes: bool = True
    distinguish_merged_fallback_subtypes: bool = True


@dataclasses.dataclass(frozen=True)
class TaggingSettings:
    batch_size: int
    model_name: str
    model_path: str
    # lr: float
    epochs: int
    grad_acc: int = 1
    eval_steps: int | None = None
    warmup_epochs: int = 5
    encoder_lr: float = 0.00001
    head_lr: float = 0.0001
    weight_decay: float = 0.01
    tol: int = 99999
    tag_vocab_path: str = "vocab"
    output_path: str = "models"
    use_tensorboard: bool = True
    eval_model_name: str = ""
    train_pos: bool = True
    train_xpos: bool = False
    train_arc: bool = False
    train_deprel: bool = False
    train_sup: bool = True
    train_feats: bool = False
    train_subtypes: bool = False
    mode: Mode = "init"
    eval_metric: EvalMetric = "cacc"
    factorised: Factorised = False
    deprels_from_supertags: bool = False
    k_supertag: int = 5
    k_head_scores: int = 5
    t_arc: float = 1
    t_sup: float = 1
    pos_label_smoothing: float = 0.0
    xpos_label_smoothing: float = 0.0
    arc_label_smoothing: float = 0.0
    deprel_label_smoothing: float = 0.0
    sup_label_smoothing: float = 0.0
    feats_label_smoothing: float = 0.0
    subtypes_label_smoothing: float = 0.0
    sup_score_scale: float = 1.0
    loss_weights: None | dict[str, float] = None


@dataclasses.dataclass(frozen=True)
class TaggingRangesSettings:
    warmup_epochs: tuple[int, int] | None = None
    encoder_lr: tuple[float, float] | None = None
    head_lr: tuple[float, float] | None = None
    weight_decay: tuple[float, float] | None = None
    train_pos: tuple[bool, ...] | None = None
    train_xpos: tuple[bool, ...] | None = None
    train_arc: tuple[bool, ...] | None = None
    train_deprel: tuple[bool, ...] | None = None
    train_sup: tuple[bool, ...] | None = None
    train_feats: tuple[bool, ...] | None = None
    train_subtypes: tuple[bool, ...] | None = None
    factorised: tuple[Factorised, ...] | None = None
    deprels_from_supertags: tuple[bool, ...] | None = None
    k_supertag: tuple[int, int] | None = None
    k_head_scores: tuple[int, int] | None = None
    t_arc: tuple[float, float] | None = None
    t_sup: tuple[float, float] | None = None
    pos_label_smoothing: tuple[float, float] | None = None
    xpos_label_smoothing: tuple[float, float] | None = None
    arc_label_smoothing: tuple[float, float] | None = None
    deprel_label_smoothing: tuple[float, float] | None = None
    sup_label_smoothing: tuple[float, float] | None = None
    feats_label_smoothing: tuple[float, float] | None = None
    subtypes_label_smoothing: tuple[float, float] | None = None
    sup_score_scale: tuple[float, float] | None = None
    loss_weights: None | dict[
        str, tuple[float, float]] = None


@dataclasses.dataclass(frozen=True)
class OptSettings:
    ranges: TaggingRangesSettings
    tagging: TaggingSettings
    deprels: DepSettings
    file: FileSettings
    study_name: str
    n_trials: int
    pruner_min_resource: int = 1
    pruner_max_resource: int | str = "auto"
    pruner_reduction_factor: int = 3
    pruner_bootstrap_count: int = 0
    sampler_n_startup_trials: int = 10
    sampler_n_ei_candidates: int = 24
    sampler_multivariate: bool = True


@dataclasses.dataclass(frozen=True)
class ExtractSettings:
    deprels: DepSettings
    file: FileSettings


@dataclasses.dataclass(frozen=True)
class Settings(ExtractSettings):
    tagging: TaggingSettings


# Settings loading
def load_settings(
        name: str = DEFAULT_SETTINGS,
        *, dir: pathlib.Path = SETTINGS_DIR
        ) -> Settings:
    return ts.load(
        Settings, appname="deptag",
        config_files=[dir / f"{name}.toml"])


def load_extract_settings(
        name: str = DEFAULT_SETTINGS,
        *, dir: pathlib.Path = SETTINGS_DIR
        ) -> ExtractSettings:
    return ts.load(
        ExtractSettings, appname="deptag",
        config_files=[dir / f"{name}.toml"])


def load_opt_settings(
        name: str = DEFAULT_SETTINGS,
        *, dir: pathlib.Path = SETTINGS_DIR
        ) -> OptSettings:
    return ts.load_settings(
        OptSettings,
        loaders=ts.default_loaders(
            appname="opt",
            config_files=[dir / f"{name}.toml"],
        ),
        converter=converter,
    )
