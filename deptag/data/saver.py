from . import locs

import conllu
import pathlib
import tqdm
import itertools

from typing import Generator


def write_incr(
        name: str, *, dir: pathlib.Path = locs.DATA_DIR
        ) -> Generator[None, conllu.TokenList, None]:
    with open(dir / f"{name}.conllu", 'w') as f:
        try:
            for _ in tqdm.tqdm(
                    itertools.count(),
                    desc="Writing conllu"):
                f.writelines((yield).serialize() + "\n")
        except StopIteration:
            pass
