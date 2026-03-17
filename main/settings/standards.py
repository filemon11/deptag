import typed_settings as ts


@ts.settings
class DeprelStandard:
    labels: dict[str, tuple[str, ...]]


standard = ts.load(
    DeprelStandard, appname="deprels",
    config_files=["../standards/default.toml"])
print(standard)
