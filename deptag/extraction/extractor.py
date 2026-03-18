import conllu
from .. import data
from ..data import locs
import pathlib
import dataclasses
from collections import defaultdict

from typing import (
    Sequence, Collection, Iterable, Generator, Mapping, DefaultDict)

# extract function
# for each sentence:
# 1. assign labels (own function)
# 2. extract label (own function)
# -> Iterable

# func Iterator with return value (statistics)


RawArc = tuple[int, str]
RawTag = tuple[Sequence[RawArc], RawArc | None, int]


def collect_relations(
        sentence: conllu.TokenList,
        arguments: Collection[str],
        adjuncts: Collection[str],
        delete: Collection[str] = tuple(),
        *,
        without_labels: bool = False,
        distinguish_fallback_subtypes: bool = True
        ) -> list[RawTag]:
    # Returns [([(daughter_position, label), ...],
    # (head_position, label) | None), ...]
    # ignores multiwords

    daughters: list[list[RawArc]] = [[] for _ in sentence]
    heads: list[RawArc | None] = [None]*len(sentence)

    proper_item_num: int = 0
    for token in sentence:
        if isinstance(token["id"], tuple):
            continue
        proper_item_num += 1
        token_id: int = token["id"]-1
        head_id: int = token["head"]-1
        deprel: str = token["deprel"]
        if deprel in arguments:
            daughters[head_id].append(
                (token_id, "" if without_labels else deprel))
        elif deprel in adjuncts:
            heads[token_id] = (
                head_id, "" if without_labels else deprel)
        elif deprel in delete:
            pass
        elif data.has_subtype(deprel):
            deprel_main, subtype = data.split_main_sub(deprel)
            if not distinguish_fallback_subtypes:
                deprel = deprel_main
            # Fallback
            # TODO: give info about this if successful
            if deprel_main in arguments:
                daughters[head_id].append(
                    (token_id, "" if without_labels else deprel))
            elif deprel_main in adjuncts:
                heads[token_id] = (
                    head_id, "" if without_labels else deprel)
            elif deprel_main in delete:
                pass
            else:
                raise Exception(
                    f"dependency relation '{deprel_main}:{subtype}' "
                    "found in data but not known to settings.",
                    f" Fallback to relation '{deprel_main}' was unsuccessful.")
        else:
            raise Exception(
                f"dependency relation '{deprel}' "
                "found in data but not known to settings.")

    return list(zip(daughters, heads, range(proper_item_num)))


RelativeTag = Sequence[tuple[bool | None, str]]


def convert_raw_relation_to_relative(
        tag: RawTag,) -> RelativeTag:
    head: RawArc | None = tag[1]
    anchor_id: int | None = tag[2]
    daughters = tag[0]

    relative_tag: list[tuple[bool | None, str]] = []
    for daughter_id, daughter_label in daughters:
        if anchor_id is not None and anchor_id < daughter_id:
            if head is not None and head[0] < anchor_id:
                relative_tag.append((False, head[1]))
                head = None
            relative_tag.append((None, ""))
            anchor_id = None
        elif head is not None and head[0] < daughter_id:
            if anchor_id is not None and anchor_id < head[0]:
                relative_tag.append((None, ""))
                anchor_id = None
            relative_tag.append((False, head[1]))
            head = None
        relative_tag.append((True, daughter_label))
    if anchor_id is not None:
        if head is not None and head[0] < anchor_id:
            relative_tag.append((False, head[1]))
            head = None
        relative_tag.append((None, ""))
        anchor_id = None
    elif head is not None:
        if anchor_id is not None and anchor_id < head[0]:
            relative_tag.append((None, ""))
            anchor_id = None
        relative_tag.append((False, head[1]))
        head = None

    return tuple(relative_tag)


def convert_relative_relation_to_string(
        tag: RelativeTag,
        ) -> str:
    string = ""
    for entry in tag:
        match entry[0]:
            case None:
                string += "*"
            case True:
                string += f"+{entry[1]}"
            case False:
                string += f"-{entry[1]}"
    return string


@dataclasses.dataclass
class Statistics():
    num_supertags: int
    supertags: set
    supertag_to_nums: Mapping[str, int]
    num_unicorns: int
    unicorns: set
    num_instances: int
    perc_instances_unicorn: float
    perc_unicorn: float


def extract(
        sentences: Iterable[conllu.TokenList],
        arguments: Collection[str],
        adjuncts: Collection[str],
        delete: Collection[str] = tuple(),
        *,
        without_labels: bool = False,
        distinguish_fallback_subtypes: bool = True
        ) -> Generator[
            tuple[
                list[RawTag], list[RelativeTag],
                list[str], conllu.TokenList],
            None, Statistics]:

    supertag_to_nums: DefaultDict[str, int] = defaultdict(int)
    for sentence in sentences:
        raw_relations = collect_relations(
            sentence, arguments, adjuncts, delete,
            without_labels=without_labels,
            distinguish_fallback_subtypes=distinguish_fallback_subtypes
        )
        relative_relations: list[RelativeTag] = [
            convert_raw_relation_to_relative(rel) for rel in raw_relations]
        string_relations: list[str] = [
            convert_relative_relation_to_string(rel)
            for rel in relative_relations]

        sentence_iter = iter(sentence)
        for string in string_relations:
            supertag_to_nums[string] += 1

            token = next(sentence_iter)
            while isinstance(token["id"], tuple):
                token = next(sentence_iter)
            if token["misc"] is not None:
                token["misc"]["supertag"] = string
            else:
                token["misc"] = {"supertag": string}

        yield (raw_relations, relative_relations, string_relations, sentence)

    unicorns = {
        supertag for supertag, num in supertag_to_nums.items() if num == 1}
    num_instances = sum(supertag_to_nums.values())
    return Statistics(
        supertag_to_nums=supertag_to_nums,
        supertags=set(supertag_to_nums.keys()),
        num_supertags=len(supertag_to_nums),
        unicorns=unicorns,
        num_unicorns=len(unicorns),
        num_instances=num_instances,
        perc_instances_unicorn=len(unicorns)/num_instances,
        perc_unicorn=len(unicorns)/len(supertag_to_nums)
    )


def extract_and_write(
        sentences: Iterable[conllu.TokenList],
        file_name: str,
        arguments: Collection[str],
        adjuncts: Collection[str],
        delete: Collection[str] = tuple(),
        *,
        dir: pathlib.Path = locs.DATA_DIR,
        without_labels: bool = False,
        distinguish_fallback_subtypes: bool = True
        ) -> Statistics:
    extractor = iter(extract(
        sentences, arguments, adjuncts, delete,
        without_labels=without_labels,
        distinguish_fallback_subtypes=distinguish_fallback_subtypes)
    )
    writer = data.write_incr(
        file_name, dir=dir
    )
    try:
        writer.send(None)  # type: ignore
        while True:
            extracted_sent = next(extractor)
            writer.send(extracted_sent[3])
    except StopIteration as e:
        return e.value
