import pathlib
import conllu

from typing import Iterator, Literal


DATA_DIR = pathlib.Path("data")
UD_DIR = pathlib.Path("Universal Dependencies 2.17", "ud-treebanks-v2.17")
STANDARD_SUFFIX = "conllu"


def parse_conllu(
        name: str,
        *,
        dir: pathlib.Path = DATA_DIR,
        suffix: str = STANDARD_SUFFIX,
        encoding: str = "utf-8",
        ) -> Iterator[conllu.TokenList]:
    with open(dir / f"{name}.{suffix}", "r", encoding=encoding) as f:
        for tokentree in conllu.parse_incr(f):
            yield tokentree


def load_conllu(
        name: str, ud_split: None | Literal[
            "test", "dev", "train"] = None,
        *,
        dir: pathlib.Path = DATA_DIR,
        ud_folder: pathlib.Path = UD_DIR,
        suffix: str = STANDARD_SUFFIX,
        encoding: str = "utf-8",
        ) -> Iterator[conllu.TokenList]:

    # If ud_split is specified, loads UD bank

    if ud_split is not None:
        dir = dir / ud_folder / f"UD_{name}"
        conllu_files = list(dir.glob(f"{ud_split}.{suffix}"))

        assert len(conllu_files) > 0, (
            f"Could not {ud_split} split in path {dir}"
        )
        assert len(conllu_files) < 2, (
            f"Found more than one {ud_split} split in path {dir}"
        )
        name = conllu_files[0].stem[:-(len(suffix)+1)]    # remove suffix

    for tokenlist in parse_conllu(
            name, dir=dir, suffix=suffix, encoding=encoding):
        yield tokenlist
