from . import locs

import conllu
import pathlib

from typing import Generator


def write_incr(
        name: str, *, dir: pathlib.Path = locs.DATA_DIR
        ) -> Generator[None, conllu.TokenList, None]:
    with open(dir / f"{name}.conllu", 'w') as f:
        try:
            while True:
                f.writelines((yield).serialize() + "\n")
        except StopIteration:
            pass
