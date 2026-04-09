import pathlib

from . import settings, standards, validation

from typing import overload, Literal


# TODO: better integration for UD in data folder

@overload
def load_settings(
        mode: Literal["extract"],
        settings_name: str = settings.DEFAULT_SETTINGS,
        *, settings_dir: pathlib.Path = settings.SETTINGS_DIR
        ) -> settings.ExtractSettings:
    ...


@overload
def load_settings(
        mode: Literal["full"] = "full",
        settings_name: str = settings.DEFAULT_SETTINGS,
        *, settings_dir: pathlib.Path = settings.SETTINGS_DIR
        ) -> settings.Settings:
    ...


def load_settings(
        mode: Literal["extract", "full"] = "full",
        settings_name: str = settings.DEFAULT_SETTINGS,
        *, settings_dir: pathlib.Path = settings.SETTINGS_DIR
        ) -> settings.Settings | settings.ExtractSettings:
    """_summary_

    Args:
        settings_name (str, optional): _description_.
            Defaults to settings.DEFAULT_SETTINGS.
        standard_name (str, optional): _description_.
            Defaults to standards.DEFAULT_STANDARD.
        settings_dir (pathlib.Path, optional): _description_.
            Defaults to settings.STETTINGS_DIR.
        standards_dir (pathlib.Path, optional): _description_.
            Defaults to standards.STANDARDS_DIR.
        standard_from_xml (bool, optional): _description_.
            Defaults to False.
        save_standard_from_xml (bool, optional): _description_.
            Defaults to True.
        save_standard_from_xml_dir (pathlib.Path, optional): _description_.
            Defaults to ( standards.STANDARDS_DIR).

    Returns:
        settings.DepSettings: _description_
    """
    # Load settings
    sett: settings.Settings | settings.ExtractSettings
    if mode == "full":
        sett = settings.load_settings(
            settings_name,
            dir=settings_dir
        )
    else:
        sett = settings.load_extract_settings(
            settings_name,
            dir=settings_dir
        )

    # Load standard
    if sett.file.standard_from_xml:
        # ignores standard_name; if save_standard_from_xml is True,
        # use standard_name as the name of the new toml standard file
        # and the UD corpus name
        stan = standards.load_stats_as_standard(
            pathlib.Path(
                sett.file.data_folder
                ) / sett.file.ud_folder / f"UD_{sett.file.conllu_file}"
        )
        if sett.file.save_standard_from_xml:
            standards.save_standard(
                stan, sett.file.standard,
                dir=pathlib.Path(
                    sett.file.save_standard_from_xml_dir)
            )
    else:
        stan = standards.load_standard(
            sett.file.standard,
            dir=pathlib.Path(
                sett.file.standards_dir)
        )

    validation.assert_dep_settings(
        sett.deprels
    )

    validation.assert_dep_standard(
        sett.deprels, stan,
        allow_partial_underspecification=(
            sett.file.allow_partial_underspecification)
    )

    return sett
