import typed_settings as ts
import dataclasses


@dataclasses.dataclass(frozen=True)
class DepSettings:
    arguments: tuple[str, ...]
    adjuncts: tuple[str, ...]
    delete: tuple[str, ...]


# settings = ts.load(
#     DepSettings, appname="deprels",
#     config_files=["../../settings/default.toml"])
# print(settings)
