from .. import data
from ..data import locs

import conllu
import tqdm

import pathlib
import dataclasses
from collections import defaultdict, Counter

from typing import (
    Sequence, Collection, Iterable,
    Generator, Mapping, DefaultDict)

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
        merged: Mapping[str, Collection[str]] | None = None,
        *,
        without_labels: bool = False,
        distinguish_fallback_subtypes: bool = True,
        merged_fallback_subtypes: bool = True,
        distinguish_merged_fallback_subtypes: bool = True
        ) -> list[RawTag]:
    # Returns [([(daughter_position, label), ...],
    # (head_position, label) | None), ...]
    # ignores multiwords
    deprel: str

    deprel_to_new: dict[str, str] | None = None
    if merged is not None:
        deprel_to_new = {}
        for new, deprel_list in merged.items():
            for deprel in deprel_list:
                deprel_to_new[deprel] = new

    def deprel_merge(deprel: str) -> str:
        if deprel_to_new is not None:
            if deprel in deprel_to_new:
                return deprel_to_new[deprel]
            elif merged_fallback_subtypes and data.has_subtype(
                    deprel) and data.split_main_sub(
                        deprel)[0] in deprel_to_new:
                main_type, subtype = data.split_main_sub(deprel)
                main_type = deprel_to_new[main_type]
                if distinguish_merged_fallback_subtypes:
                    return f"{main_type}:{subtype}"
                return main_type
        return deprel

    daughters: list[list[RawArc]] = [[] for _ in sentence]
    heads: list[RawArc | None] = [None]*len(sentence)

    proper_item_num: int = 0
    for token in sentence:
        if isinstance(token["id"], tuple):
            continue
        proper_item_num += 1
        token_id: int = token["id"]-1
        head_id: int = token["head"]-1
        deprel = token["deprel"]

        if deprel in arguments:
            daughters[head_id].append(
                (token_id, "" if without_labels else deprel_merge(deprel)))
        elif deprel in adjuncts:
            heads[token_id] = (
                head_id, "" if without_labels else deprel_merge(deprel))
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
                    (token_id, "" if without_labels else deprel_merge(deprel)))
            elif deprel_main in adjuncts:
                heads[token_id] = (
                    head_id, "" if without_labels else deprel_merge(deprel))
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
        tag: RawTag,
        *,
        order_relations: bool = True
        ) -> RelativeTag:
    head: RawArc | None = tag[1]
    anchor_id: int | None = tag[2]
    daughters = tag[0]

    if not order_relations:
        assert anchor_id is not None
        left_rels: list[tuple[bool, str]] = []
        right_rels: list[tuple[bool, str]] = []
        for daughter_id, daughter_label in daughters:
            if daughter_id < anchor_id:
                left_rels.append((True, daughter_label))
            else:
                right_rels.append((True, daughter_label))
        left_rels.sort(key=lambda x: x[1])
        right_rels.sort(key=lambda x: x[1])
        if head is not None:
            if head[0] < anchor_id:
                left_rels.append((False, head[1]))
            else:
                right_rels.insert(0, (False, head[1]))
        return tuple(left_rels) + ((None, ""),) + tuple(right_rels)

    relative_tag: list[tuple[bool | None, str]] = []
    for daughter_id, daughter_label in daughters:
        if anchor_id is not None and anchor_id < daughter_id:
            if head is not None and head[0] < anchor_id:
                relative_tag.append((False, head[1]))
                head = None
            relative_tag.append((None, ""))
            anchor_id = None
        if head is not None and head[0] < daughter_id:
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
    if head is not None:
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
    avg_instances_per_supertag: float
    word_to_supertag_to_nums: Mapping[str, Mapping[str, int]]
    word_to_supertag_to_nums_unicorn: Mapping[str, Mapping[str, int]]
    num_adjunct: int
    num_initial: int
    avg_num_edges: float
    avg_supertags_per_type: float
    labels: set[str]
    occurrences_per_label: Mapping[str, int]
    avg_occurrences_per_label: float
    avg_left_args: float
    avg_right_args: float
    num_left_adjuncts: int
    num_right_adjuncts: int
    num_strict_left_adjuncts: int
    num_strict_right_adjuncts: int

    def __add__(self, other: "Statistics") -> "Statistics":
        supertag_to_nums = Counter(self.supertag_to_nums) + Counter(
            other.supertag_to_nums)
        unicorns = {
            supertag for supertag, num in supertag_to_nums.items() if num == 1}
        supertags = self.supertags | other.supertags
        num_supertags = len(supertags)
        num_unicorns = len(unicorns)
        num_instances = self.num_instances+other.num_instances
        word_to_supertag_to_nums: dict[str, Mapping[str, int]] = {}
        self_keys = set(self.word_to_supertag_to_nums.keys())
        other_keys = set(other.word_to_supertag_to_nums.keys())
        common_keys = self_keys.intersection(other_keys)
        for key in common_keys:
            word_to_supertag_to_nums[key] = Counter(
                self.word_to_supertag_to_nums[key]) + Counter(
                    other.word_to_supertag_to_nums[key])
        for key in self_keys - common_keys:
            word_to_supertag_to_nums[key] = self.word_to_supertag_to_nums[key]
        for key in other_keys - common_keys:
            word_to_supertag_to_nums[key] = other.word_to_supertag_to_nums[key]

        word_to_supertag_to_nums_unicorn = {
            supertag: sup2nums for supertag, sup2nums
            in word_to_supertag_to_nums.items()
            if any([sup in unicorns for sup in sup2nums.keys()])}

        num_adjunct = len(
            [tag for tag in supertag_to_nums.keys() if "-" in tag])
        num_initial = len(supertag_to_nums) - num_adjunct

        avg_num_edges = sum([
            tag.count("+")+tag.count("-") for tag in supertag_to_nums.keys()
            ]) / len(supertag_to_nums)

        avg_supertags_per_type = sum([
            len(supertag_to_nums) for supertag_to_nums
            in word_to_supertag_to_nums.values()
            ]) / len(word_to_supertag_to_nums)

        labels = self.labels | other.labels
        occurrences_per_label = Counter(
            self.occurrences_per_label) + Counter(
                other.occurrences_per_label
            )

        avg_occurrences_per_label = sum(
            occurrences_per_label.values()) / len(occurrences_per_label)

        avg_left_args = sum(
            [tag.split("*")[0].count("+") for tag in supertag_to_nums.keys()]
            ) / len(supertag_to_nums)
        avg_right_args = sum(
            [tag.split("*")[1].count("+") for tag in supertag_to_nums.keys()]
            ) / len(supertag_to_nums)

        num_left_adjuncts = len([
            tag for tag in supertag_to_nums.keys()
            if "-" in tag.split("*")[1]
        ])
        num_right_adjuncts = len([
            tag for tag in supertag_to_nums.keys()
            if "-" in tag.split("*")[0]
        ])
        num_strict_left_adjuncts = len([
            tag for tag in supertag_to_nums.keys()
            if "-" in tag.split("*")[1].split("+")[-1]
        ])
        num_strict_right_adjuncts = len([
            tag for tag in supertag_to_nums.keys()
            if "-" in tag.split("*")[0].split("+")[0]
        ])

        return Statistics(
            num_supertags=len(supertags),
            supertags=supertags,
            supertag_to_nums=supertag_to_nums,
            num_unicorns=num_unicorns,
            unicorns=unicorns,
            num_instances=num_instances,
            perc_instances_unicorn=num_unicorns/num_instances,
            perc_unicorn=len(unicorns)/len(supertags),
            avg_instances_per_supertag=num_instances/num_supertags,
            word_to_supertag_to_nums=word_to_supertag_to_nums,
            word_to_supertag_to_nums_unicorn=word_to_supertag_to_nums_unicorn,
            num_adjunct=num_adjunct,
            num_initial=num_initial,
            avg_num_edges=avg_num_edges,
            avg_supertags_per_type=avg_supertags_per_type,
            labels=labels,
            occurrences_per_label=occurrences_per_label,
            avg_occurrences_per_label=avg_occurrences_per_label,
            avg_left_args=avg_left_args,
            avg_right_args=avg_right_args,
            num_left_adjuncts=num_left_adjuncts,
            num_right_adjuncts=num_right_adjuncts,
            num_strict_left_adjuncts=num_strict_left_adjuncts,
            num_strict_right_adjuncts=num_strict_right_adjuncts,
        )


def print_statistics(statistics: Statistics):
    print(f"# instances: {statistics.num_instances}")
    print(f"# supertags: {statistics.num_supertags}")
    print(
        "avg instances per supertag: "
        f"{round(statistics.avg_instances_per_supertag, 2)}")
    print(
        "# supertags occurring once: "
        f"{statistics.num_unicorns}")
    print(
        "-> % of supertags:",
        statistics.perc_unicorn
    )
    print(
        "-> % of instances:",
        statistics.perc_instances_unicorn
    )
    print(
        "# initial trees:",
        statistics.num_initial
    )
    print(
        "# adjunct trees:",
        statistics.num_adjunct
    )
    print(
        "avg # edges per supertag:",
        statistics.avg_num_edges
    )
    print(
        "# supertags per type:",
        statistics.avg_supertags_per_type
    )
    print(
        "avg # of occurrences per dep label:",
        statistics.avg_occurrences_per_label
    )
    print(
        "avg # of args preceding head:",
        statistics.avg_left_args
    )
    print(
        "avg # of args following head:",
        statistics.avg_right_args
    )
    print(
        "# of left adjoining adjunct trees:",
        statistics.num_left_adjuncts
    )
    print(
        "# of right adjoining adjunct trees:",
        statistics.num_right_adjuncts
    )
    print(
        "# of strict left adjoining adjunct trees:",
        statistics.num_strict_left_adjuncts
    )
    print(
        "# of strict right adjoining adjunct trees:",
        statistics.num_strict_right_adjuncts
    )


def extract(
        sentences: Iterable[conllu.TokenList],
        arguments: Collection[str],
        adjuncts: Collection[str],
        delete: Collection[str] = tuple(),
        merged: None | Mapping[str, Collection[str]] = None,
        *,
        without_labels: bool = False,
        distinguish_fallback_subtypes: bool = True,
        merged_fallback_subtypes: bool = True,
        distinguish_merged_fallback_subtypes: bool = True,
        order_relations: bool = True,
        ) -> Generator[
            tuple[
                list[RawTag], list[RelativeTag],
                list[str], conllu.TokenList],
            None, Statistics]:

    supertag_to_nums: DefaultDict[str, int] = defaultdict(int)
    word_to_supertag_to_nums: dict[str, DefaultDict[str, int]]
    word_to_supertag_to_nums = defaultdict(lambda: defaultdict(int))

    relative_tags: set[RelativeTag] = set()
    occurrences_per_label: DefaultDict[str, int] = defaultdict(int)

    for sentence in tqdm.tqdm(
            sentences, desc="Extracting supertags"):
        raw_relations = collect_relations(
            sentence, arguments, adjuncts, delete,
            merged,
            without_labels=without_labels,
            distinguish_fallback_subtypes=distinguish_fallback_subtypes,
            merged_fallback_subtypes=merged_fallback_subtypes,
            distinguish_merged_fallback_subtypes=(
                distinguish_merged_fallback_subtypes),
        )
        relative_relations: list[RelativeTag] = [
            convert_raw_relation_to_relative(
                rel, order_relations=order_relations) for rel in raw_relations]
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

            # Associate supertags with word dict
            word_to_supertag_to_nums[token["form"]][string] += 1

        relative_tags |= set(relative_relations)
        for rel in relative_relations:
            for tag in rel:
                if tag[1] != "":
                    occurrences_per_label[tag[1]] += 1

        yield (raw_relations, relative_relations, string_relations, sentence)

    unicorns = {
        supertag for supertag, num in supertag_to_nums.items() if num == 1}
    num_instances = sum(supertag_to_nums.values())

    word_to_supertag_to_nums_unicorn = {
        supertag: sup2nums for supertag, sup2nums
        in word_to_supertag_to_nums.items()
        if any([sup in unicorns for sup in sup2nums.keys()])}

    num_adjunct = len([tag for tag in supertag_to_nums.keys() if "-" in tag])
    num_initial = len(supertag_to_nums) - num_adjunct

    avg_num_edges = sum([
        tag.count("+")+tag.count("-") for tag in supertag_to_nums.keys()
    ]) / len(supertag_to_nums)

    avg_supertags_per_type = sum(
        [len(supertag_to_nums) for supertag_to_nums
         in word_to_supertag_to_nums.values()]) / len(word_to_supertag_to_nums)

    labels = {
        rel[1] for tag in relative_tags for rel in tag if not rel[1] == ""}
    avg_occurrences_per_label = sum(
        occurrences_per_label.values()) / len(occurrences_per_label)

    avg_left_args = sum(
        [tag.split("*")[0].count("+") for tag in supertag_to_nums.keys()]
        ) / len(supertag_to_nums)
    avg_right_args = sum(
        [tag.split("*")[1].count("+") for tag in supertag_to_nums.keys()]
        ) / len(supertag_to_nums)

    num_left_adjuncts = len([
        tag for tag in supertag_to_nums.keys()
        if "-" in tag.split("*")[1]
    ])
    num_right_adjuncts = len([
        tag for tag in supertag_to_nums.keys()
        if "-" in tag.split("*")[0]
    ])

    num_strict_left_adjuncts = len([
        tag for tag in supertag_to_nums.keys()
        if "-" in tag.split("*")[1].split("+")[-1]
    ])
    num_strict_right_adjuncts = len([
        tag for tag in supertag_to_nums.keys()
        if "-" in tag.split("*")[0].split("+")[0]
    ])

    return Statistics(
        supertag_to_nums=supertag_to_nums,
        supertags=set(supertag_to_nums.keys()),
        num_supertags=len(supertag_to_nums),
        unicorns=unicorns,
        num_unicorns=len(unicorns),
        num_instances=num_instances,
        perc_instances_unicorn=len(unicorns)/num_instances,
        perc_unicorn=len(unicorns)/len(supertag_to_nums),
        avg_instances_per_supertag=num_instances/len(supertag_to_nums),
        word_to_supertag_to_nums=word_to_supertag_to_nums,
        word_to_supertag_to_nums_unicorn=word_to_supertag_to_nums_unicorn,
        num_adjunct=num_adjunct,
        num_initial=num_initial,
        avg_num_edges=avg_num_edges,
        avg_supertags_per_type=avg_supertags_per_type,
        labels=labels,
        occurrences_per_label=occurrences_per_label,
        avg_occurrences_per_label=avg_occurrences_per_label,
        avg_left_args=avg_left_args,
        avg_right_args=avg_right_args,
        num_left_adjuncts=num_left_adjuncts,
        num_right_adjuncts=num_right_adjuncts,
        num_strict_left_adjuncts=num_strict_left_adjuncts,
        num_strict_right_adjuncts=num_strict_right_adjuncts,
    )


def extract_and_write(
        sentences: Iterable[conllu.TokenList],
        file_name: str,
        arguments: Collection[str],
        adjuncts: Collection[str],
        delete: Collection[str] = tuple(),
        merged: Mapping[str, Collection[str]] | None = None,
        *,
        dir: pathlib.Path = locs.DATA_DIR,
        without_labels: bool = False,
        distinguish_fallback_subtypes: bool = True,
        merged_fallback_subtypes: bool = True,
        distinguish_merged_fallback_subtypes: bool = True,
        order_relations: bool = True,
        ) -> Statistics:
    extractor = iter(extract(
        sentences, arguments, adjuncts, delete, merged,
        without_labels=without_labels,
        distinguish_fallback_subtypes=distinguish_fallback_subtypes,
        merged_fallback_subtypes=merged_fallback_subtypes,
        distinguish_merged_fallback_subtypes=(
            distinguish_merged_fallback_subtypes),
        order_relations=order_relations,
        )
    )
    writer = data.write_incr(
        file_name, dir=dir
    )
    try:
        writer.send(None)  # type: ignore
        while True:
            extracted_sent = next(extractor)
            writer.send(extracted_sent[3])
        # never happens; just for type-checker.
        # The StopIteration is thrown by extractor
    except StopIteration as e:
        return e.value


def replace_unicorns_and_write(
        sentences: Iterable[conllu.TokenList],
        unicorns: Collection[str],
        file_name: str,
        *,
        dir: pathlib.Path = locs.DATA_DIR,
        ) -> Statistics:
    extractor = read(
        sentences,
        replace_labels_supertags=unicorns)
    writer = data.write_incr(
        file_name, dir=dir
    )
    try:
        writer.send(None)  # type: ignore
        while True:
            extracted_sent = next(extractor)
            writer.send(extracted_sent[2])
        # never happens; just for type-checker.
        # The StopIteration is thrown by extractor
    except StopIteration as e:
        return e.value


def read_relation(token: conllu.Token) -> str:
    assert token["misc"] is not None
    assert "supertag" in token["misc"]
    return token["misc"]["supertag"]


def get_string_relations(sentence: conllu.TokenList) -> list[str]:
    return [
        read_relation(token) for token in sentence if not isinstance(
            token["id"], tuple
        )]


def get_type(string: str) -> bool | None:
    match string:
        case "+":
            return True
        case "-":
            return False
        case "*":
            return None
        case _:
            raise TypeError(f"Relation type {string} unknown.")


def convert_string_to_relative_relation(relation: str) -> RelativeTag:
    relative_list: list[tuple[bool | None, str]] = list()
    current_item: str = ""
    current_type: None | bool = get_type(relation[0])
    for char in relation[1:]:
        try:
            new_type = get_type(char)
            relative_list.append((current_type, current_item))
            current_type = new_type
            current_item = ""
        except TypeError:
            current_item += char
    relative_list.append((current_type, current_item))
    return relative_list


def read(
        sentences: Iterable[conllu.TokenList],
        *,
        replace_labels_supertags: Collection[str] | None,
        fill_label: str = "dep"
        ) -> Generator[
            tuple[
                list[RelativeTag],
                list[str], conllu.TokenList],
            None, Statistics]:

    supertag_to_nums: DefaultDict[str, int] = defaultdict(int)
    word_to_supertag_to_nums: dict[str, DefaultDict[str, int]]
    word_to_supertag_to_nums = defaultdict(lambda: defaultdict(int))

    relative_tags: set[RelativeTag] = set()
    occurrences_per_label: DefaultDict[str, int] = defaultdict(int)

    for sentence in tqdm.tqdm(
            sentences, desc="Extracting supertags"):
        string_relations: list[str] = get_string_relations(sentence)
        relative_relations: list[RelativeTag] = [
            convert_string_to_relative_relation(
                rel) for rel in string_relations]

        if replace_labels_supertags is not None:
            new_relative_relations: list[RelativeTag] = []
            for str_relation, rel_relation in zip(
                    string_relations, relative_relations):
                if str_relation in replace_labels_supertags:
                    new_relative_relations.append(
                        replace_labels(
                            rel_relation, fill_label)
                    )
                else:
                    new_relative_relations.append(rel_relation)
            relative_relations = new_relative_relations
            string_relations = [
                convert_relative_relation_to_string(tag) for tag
                in relative_relations
            ]
        relative_tags |= set(relative_tags)
        for rel in relative_relations:
            for tag in rel:
                if tag[1] != "":
                    occurrences_per_label[tag[1]] += 1

        sentence_iter = iter(sentence)
        for string in string_relations:
            supertag_to_nums[string] += 1

            token = next(sentence_iter)
            while isinstance(token["id"], tuple):
                token = next(sentence_iter)
            token["misc"]["supertag"] = string

            # Associate supertags with word dict
            word_to_supertag_to_nums[token["form"]][string] += 1

        yield (relative_relations, string_relations, sentence)

    unicorns = {
        supertag for supertag, num in supertag_to_nums.items() if num == 1}
    num_instances = sum(supertag_to_nums.values())
    word_to_supertag_to_nums_unicorn = {
        supertag: sup2nums for supertag, sup2nums
        in word_to_supertag_to_nums.items()
        if any([sup in unicorns for sup in sup2nums.keys()])}
    num_adjunct = len([tag for tag in supertag_to_nums.keys() if "-" in tag])
    num_initial = len(supertag_to_nums) - num_adjunct

    avg_num_edges = sum([
        tag.count("+")+tag.count("-") for tag in supertag_to_nums.keys()
    ]) / len(supertag_to_nums)

    avg_supertags_per_type = sum(
        [len(supertag_to_nums) for supertag_to_nums
         in word_to_supertag_to_nums.values()]) / len(word_to_supertag_to_nums)

    labels = {
        rel[1] for tag in relative_tags for rel in tag if not rel[1] == ""}
    avg_occurrences_per_label = sum(
        occurrences_per_label.values()) / len(occurrences_per_label)

    avg_left_args = sum(
        [tag.split("*")[0].count("+") for tag in supertag_to_nums.keys()]
        ) / len(supertag_to_nums)
    avg_right_args = sum(
        [tag.split("*")[1].count("+") for tag in supertag_to_nums.keys()]
        ) / len(supertag_to_nums)

    num_left_adjuncts = len([
        tag for tag in supertag_to_nums.keys()
        if "-" in tag.split("*")[1]
    ])
    num_right_adjuncts = len([
        tag for tag in supertag_to_nums.keys()
        if "-" in tag.split("*")[0]
    ])

    num_strict_left_adjuncts = len([
        tag for tag in supertag_to_nums.keys()
        if "-" in tag.split("*")[1].split("+")[-1]
    ])
    num_strict_right_adjuncts = len([
        tag for tag in supertag_to_nums.keys()
        if "-" in tag.split("*")[0].split("+")[0]
    ])

    return Statistics(
        supertag_to_nums=supertag_to_nums,
        supertags=set(supertag_to_nums.keys()),
        num_supertags=len(supertag_to_nums),
        unicorns=unicorns,
        num_unicorns=len(unicorns),
        num_instances=num_instances,
        perc_instances_unicorn=len(unicorns)/num_instances,
        perc_unicorn=len(unicorns)/len(supertag_to_nums),
        avg_instances_per_supertag=num_instances/len(supertag_to_nums),
        word_to_supertag_to_nums=word_to_supertag_to_nums,
        word_to_supertag_to_nums_unicorn=word_to_supertag_to_nums_unicorn,
        num_adjunct=num_adjunct,
        num_initial=num_initial,
        avg_num_edges=avg_num_edges,
        avg_supertags_per_type=avg_supertags_per_type,
        labels=labels,
        occurrences_per_label=occurrences_per_label,
        avg_occurrences_per_label=avg_occurrences_per_label,
        avg_left_args=avg_left_args,
        avg_right_args=avg_right_args,
        num_left_adjuncts=num_left_adjuncts,
        num_right_adjuncts=num_right_adjuncts,
        num_strict_left_adjuncts=num_strict_left_adjuncts,
        num_strict_right_adjuncts=num_strict_right_adjuncts,
    )


def replace_labels(
        relative_tag: RelativeTag,
        fill_label: str = "dep") -> RelativeTag:
    new_tag: list[tuple[bool | None, str]] = []
    for relation in relative_tag:
        if relation[0] is None:
            new_tag.append(relation)
        else:
            new_tag.append((relation[0], fill_label))
    return new_tag
