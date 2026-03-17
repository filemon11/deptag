import xml.etree.ElementTree
import pathlib
import dataclasses
import toml

from .. import deprels

STANDARDS_DIR = "standards/"
"Directory where standards are located"


# Standards definitions

# TODO: this assumes that a deprel that has
# subtypes defined for it can also appear alone.
# It is not clear if this is the case.
@dataclasses.dataclass(frozen=True)
class DeprelStandard:
    labels: dict[str, tuple[str, ...]]


# Standards conversion

def load_stats_as_standard(
        stats_file: pathlib.Path,
        ) -> DeprelStandard:
    """_summary_

    Parameters
    ----------
    stats_file : pathlib.Path
        _description_

    Returns
    -------
    DeprelStandard
        _description_
    """

    # Load xml file
    tree = xml.etree.ElementTree.parse(stats_file)

    # Navigate to deps summary
    root = tree.getroot()
    deps = root.find("deps")

    # Iterate over children and save tag names and subtypes
    labels: dict[str, tuple[str, ...]] = {}

    assert deps is not None
    subtype: str | None
    for deprel in deps:
        # If it is a subtype, add subtype.
        # If not, retrieve list to initialise empty list
        # if it is not initialised yet.
        subtype = None
        deprel_name = deprel.attrib["name"]
        if deprels.has_subtype(deprel_name):
            deprel_name, subtype = deprels.split_main_sub(deprel_name)
        if deprel_name not in labels:
            labels[deprel_name] = tuple()
        if subtype is not None:
            labels[deprel_name] = labels[deprel_name] + (subtype,)

    # Create standard
    return DeprelStandard(labels=labels)


def save_standard(
        standard: DeprelStandard,
        name: str,
        ) -> None:
    """_summary_

    Parameters
    ----------
    standard : DeprelStandard
        _description_
    name : str
        _description_
    """

    with open(pathlib.Path(STANDARDS_DIR, f"{name}.toml"), "w") as f:
        toml.dump({"deprels": dataclasses.asdict(standard)}, f)


# Standards loading

# standard = ts.load(
#     DeprelStandard, appname="deprels",
#     config_files=["../../standards/default.toml"])
# print(standard)
