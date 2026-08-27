from . import standards
from .. import data

import typed_settings as ts
import dataclasses
import pathlib

from typing import Literal

SETTINGS_DIR = pathlib.Path("settings/")
DEFAULT_SETTINGS = "default"
"Directory where settings are located"


@dataclasses.dataclass(frozen=True)
class FileSettings:
    conllu_file: str
    output_file: str
    split: None | Literal["train", "test", "dev"] = None
    standard: str = "default"
    standards_dir: str = str(standards.STANDARDS_DIR)
    standard_from_xml: bool = False
    save_standard_from_xml_dir: str = str(standards.STANDARDS_DIR)
    allow_partial_underspecification: bool = True
    save_standard_from_xml: bool = True
    ud_folder: str = str(data.UD_DIR)
    data_folder: str = str(data.DATA_DIR)


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
    tol: int = 99999
    tag_vocab_path: str = "vocab"
    output_path: str = "models"
    use_tensorboard: bool = True
    eval_model_name: str = ""
    loss_ratio: float = 0.5
    train_pos: bool = True
    train_xpos: bool = False
    train_arc: bool = False
    train_deprel: bool = False
    train_sup: bool = True
    train_feats: bool = False
    train_subtypes: bool = False
    mode: Literal["init", "continue", "add"] = "init"
    eval_metric: Literal[
        "cacc", "mst-las", "mst-uas", "a*-las", "a*-uas"] = "cacc"
    factorised: Literal[
        "structural", "complete", "seen", False] = False
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
