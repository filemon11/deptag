import conllu

from collections import Counter

from typing import Collection, Sequence


def find_constituents(
        sentence: conllu.TokenTree
        ) -> dict[int, frozenset[int]]:
    constituents: dict[int, frozenset[int]] = {}
    const: set[int] = set()
    for child in sentence.children:
        child_consts = find_constituents(child)
        constituents |= child_consts
        const |= child_consts[child.token["id"]]
    const.add(sentence.token["id"])
    constituents[sentence.token["id"]] = frozenset(const)
    return constituents


def gap_number(constituent: Collection[int]) -> int:
    min_idx = min(constituent)
    max_idx = max(constituent)
    gap_idxs = sorted(set(range(min_idx, max_idx+1)) - set(constituent))

    number = 0

    if len(gap_idxs) == 0:
        pass
    else:
        for i, j in zip(gap_idxs, gap_idxs[1:]):
            if i+1 != j:
                number += 1
        number += 1

    return number


def count_gap_numbers(
        constituents: Collection[Collection[int]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for const in constituents:
        counts[gap_number(const)] += 1
    return counts


def find_constituents_without_adj(
        sentence: conllu.TokenTree, supertags: Sequence[str]
        ) -> dict[int, frozenset[int]]:
    constituents: dict[int, frozenset[int]] = {}
    const: set[int] = set()
    for child in sentence.children:
        child_consts = find_constituents_without_adj(child, supertags)
        constituents |= child_consts
        if "-" not in supertags[child.token["id"]-1]:
            const |= child_consts[child.token["id"]]
    const.add(sentence.token["id"])
    constituents[sentence.token["id"]] = frozenset(const)
    return constituents
