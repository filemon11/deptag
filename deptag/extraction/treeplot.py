"""
Code origin: https://github.com/explosion/spaCy/discussions/1215
"""

from spacy import displacy
from collections import OrderedDict
import pathlib
import conllu
from . import extractor

from typing import MutableMapping, Collection, Iterable


def set_arrow_direction(word_line):
    """
    Sets the orientation of the arrow that notes the directon of the dependency
    between the two units.

    """
    if int(word_line["id"]) > int(word_line["head"]):
        word_line["dir"] = "right"
    elif int(word_line["id"]) < int(word_line["head"]):
        word_line["dir"] = "left"
    return word_line


def convert2zero_based_numbering(word_line_field: str) -> str:
    "CONLL-U numbering starts at 1, displaCy's at 0..."
    word_line_field = str(int(word_line_field) - 1)
    return word_line_field


def get_start_and_end_(word_line: MutableMapping[str, str | int]) -> None:
    """
    Displacy's 'start' value is the lowest value amongst the ID
    and HEAD values,
    and the 'end' is always the highest. 'Start' and 'End' have nothing to do
    with dependency which is indicated by the arrow direction, not the line
    direction.
    """
    word_line["start"] = min([int(word_line["id"]), int(word_line["head"])])
    word_line["end"] = max([int(word_line["id"]), int(word_line["head"])])


def conll_u_string2displacy_json(
        conll_u_sent_string: str, supertag_for_pos: bool = True
        ) -> dict[str, list[dict[str, str | int]]]:
    """
    Converts a single CONLL-U formatted sentence to the displaCy json format.
    CONLL-U specification: http://universaldependencies.org/format.html
    """
    conll_u_lines = [
        line for line in conll_u_sent_string.split("\n")
        if len(line) > 0 and line[0].isnumeric()]

    displacy_json: dict[
        str, list[dict[str, str | int]]] = {
            "arcs": [], "words": []}
    for tabbed_line in conll_u_lines:

        word_line: OrderedDict[str, str | int] = OrderedDict()
        word_line["id"], word_line["form"], word_line["lemma"], \
            word_line["upostag"], word_line["xpostag"], word_line["feats"], \
            word_line["head"], word_line["deprel"], word_line["deps"], \
            word_line["misc"] = tabbed_line.split("\t")

        assert isinstance(word_line["id"], str)
        if "." in word_line["id"] or "-" in word_line["id"]:
            continue

        if supertag_for_pos:
            assert isinstance(word_line["misc"], str)
            word_line["upostag"] = dict([
                pair.split("=") for pair
                in word_line["misc"].split("|")
                if len(pair.split("=")) > 1])["supertag"]

        word_line["id"] = convert2zero_based_numbering(word_line["id"])
        if word_line["head"] != "_":
            assert isinstance(word_line["head"], str)
            word_line["head"] = convert2zero_based_numbering(word_line["head"])

        if word_line["deprel"] != "root" and word_line["head"] != "_":
            get_start_and_end_(word_line)
            word_line = set_arrow_direction(word_line)
            displacy_json["arcs"].append({
                "dir": word_line["dir"],
                "end": word_line["end"],
                "label": word_line["deprel"],
                "start": word_line["start"]})

        displacy_json["words"].append({
            "tag": word_line["upostag"],
            "text": word_line["form"]})

    return displacy_json


def tokenlist_to_svg(sentence: conllu.TokenList) -> str:
    displacy_json = conll_u_string2displacy_json(sentence.serialize())
    return displacy.render(displacy_json, style="dep", manual=True)


def plot_svg(svg: str, filename: str, dir: pathlib.Path) -> None:
    dir.mkdir(parents=True, exist_ok=True)
    with open(dir / filename, "w", encoding="utf-8", ) as fh:
        fh.write(svg)


def unicorn_plot_pipeline(
        sentences: Iterable[conllu.TokenList],
        unicorns: Collection[str],
        dir: pathlib.Path) -> Iterable[conllu.TokenList]:
    for i, sentence in enumerate(sentences):
        sentence_supertags = extractor.get_string_relations(
            sentence)
        for uni in unicorns:
            if uni in sentence_supertags:
                plot_svg(
                    tokenlist_to_svg(sentence),
                    f"{i}_{uni}.svg",
                    dir)
        yield sentence
