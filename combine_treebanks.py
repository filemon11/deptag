import pathlib
import shutil
import xml.etree.ElementTree
from collections import defaultdict

from typing import DefaultDict


UD_DIR = pathlib.Path(
    "data", "Universal Dependencies 2.15", "ud-treebanks-v2.15")

GOAL_DIR = UD_DIR / "UD_combined"  # pathlib.Path("data", "combined")
pathlib.Path(GOAL_DIR).mkdir(parents=True, exist_ok=True)

SELECTION = (
    "Bulgarian-BTB",
    "Catalan-AnCora",
    "Czech-PDTC",
    "German-GSD",
    "English-Atis",
    "Spanish-AnCora",
    "French-GSD",
    "Italian-ISDT",
    "Dutch-Alpino",
    "Norwegian-Bokmaal",
    "Romanian-RRT",
    "Russian-SynTagRus"
)

SPLITS = ("dev", "test", "train")


def find(dir: pathlib.Path, split: str) -> pathlib.Path:
    return next(dir.glob(f"*{split}.conllu"))


for split in SPLITS:
    with open(GOAL_DIR / f"combined-ud-{split}.conllu", 'wb') as wfd:
        for corpus in SELECTION:
            with open(find(UD_DIR / f"UD_{corpus}", split), 'rb') as fd:
                shutil.copyfileobj(fd, wfd)


deprels: DefaultDict[str, int] = defaultdict(int)

for corpus in SELECTION:
    tree = xml.etree.ElementTree.parse(UD_DIR / f"UD_{corpus}" / "stats.xml")

    # Navigate to deps summary
    root = tree.getroot()
    deps = root.find("deps")

    assert deps is not None
    for deprel in deps:
        deprel_name = deprel.attrib["name"]

        assert deprel.text is not None
        deprel_count = int(deprel.text)
        deprels[deprel_name] += deprel_count


lines = [f'<dep name="{name}">{count}</dep>' for name, count in deprels.items()]
with open(GOAL_DIR / "stats.xml", "w") as f:
    f.write(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<treebank>
  <deps unique="60">
    {"\n    ".join(lines)}
  </deps>
</treebank>
""")
