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
    split: None | Literal["train", "test", "dev"]
    output_file: str
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
    lr: float
    epochs: int
    tag_vocab_path: str = "vocab"
    output_path: str = "models"
    num_warmup_steps: int = 160
    use_tensorboard: bool = True


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
