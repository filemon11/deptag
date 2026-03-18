import pathlib

from . import settings, standards, validation

UD_DIR = pathlib.Path(
    "data", "Universal Dependencies 2.17", "ud-treebanks-v2.17")


# TODO: better integration for UD in data folder
def load_settings(
        settings_name: str = settings.DEFAULT_SETTINGS,
        standard_name: str = standards.DEFAULT_STANDARD,
        *, settings_dir: pathlib.Path = settings.SETTINGS_DIR,
        standards_dir: pathlib.Path = standards.STANDARDS_DIR,
        standard_from_xml: bool = False,
        ud_folder: pathlib.Path = UD_DIR,
        save_standard_from_xml: bool = True,
        save_standard_from_xml_dir: pathlib.Path = (
            standards.STANDARDS_DIR)
        ) -> settings.DepSettings:
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
    sett = settings.load_settings(
        settings_name,
        dir=settings_dir
    )

    # Load standard
    if standard_from_xml:
        # ignores standard_name; if save_standard_from_xml is True,
        # use standard_name as the name of the new toml standard file
        # and the UD corpus name
        stan = standards.load_stats_as_standard(
            ud_folder / f"UD_{standard_name}"
        )
        if save_standard_from_xml:
            standards.save_standard(
                stan, standard_name,
                dir=save_standard_from_xml_dir
            )
    else:
        stan = standards.load_standard(
            standard_name,
            dir=standards_dir
        )

    validation.assert_settings(
        sett, stan,
    )

    return sett
