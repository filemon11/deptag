import typed_settings as ts
import dataclasses
import pathlib

SETTINGS_DIR = pathlib.Path("standards/")
DEFAULT_SETTINGS = "default"
"Directory where settings are located"


@dataclasses.dataclass(frozen=True)
class DepSettings:
    arguments: tuple[str, ...]
    adjuncts: tuple[str, ...]
    delete: tuple[str, ...]


# Settings loading

def load_settings(
        name: str = DEFAULT_SETTINGS,
        *, dir: pathlib.Path = SETTINGS_DIR
        ) -> DepSettings:
    return ts.load(
        DepSettings, appname="deprels",
        config_files=[dir / f"{name}.toml"])
