import typed_settings as ts


@ts.settings
class DepSettings:
    arguments: tuple[str, ...]
    adjuncts: tuple[str, ...]


settings = ts.load(
    DepSettings, appname="deprels", config_files=["../settings/default.toml"])
print(settings)
