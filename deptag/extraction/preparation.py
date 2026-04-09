from . import extractor
import conllu

from typing import Iterable, Collection, Mapping


def prepare_train(
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
        ) -> tuple[list[list[tuple[str, str, str]]], dict[str, int]]:
    # -> word, pos, supertag

    sents: list[list[tuple[str, str, str]]] = []
    sup2id: dict[str, int] = {}
    for sen in extractor.extract(
            sentences,
            arguments,
            adjuncts,
            delete,
            merged,
            without_labels=without_labels,
            distinguish_fallback_subtypes=distinguish_fallback_subtypes,
            merged_fallback_subtypes=merged_fallback_subtypes,
            distinguish_merged_fallback_subtypes=(
                distinguish_merged_fallback_subtypes),
            order_relations=order_relations,
            ):
        sent: list[tuple[str,  str, str]] = []
        for sup, word in zip(sen[2], sen[3]):
            sent.append((word["form"], word["upos"], sup))

            if sup not in sup2id:
                sup2id[sup] = len(sup2id)+1

        sents.append(sent)

    return sents, sup2id


def prepare(
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
        ) -> list[list[tuple[str, str, str]]]:
    # -> word, pos, supertag

    sents: list[list[tuple[str, str, str]]] = []
    for sen in extractor.extract(
            sentences,
            arguments,
            adjuncts,
            delete,
            merged,
            without_labels=without_labels,
            distinguish_fallback_subtypes=distinguish_fallback_subtypes,
            merged_fallback_subtypes=merged_fallback_subtypes,
            distinguish_merged_fallback_subtypes=(
                distinguish_merged_fallback_subtypes),
            order_relations=order_relations,
            ):
        sent: list[tuple[str,  str, str]] = []
        for sup, word in zip(sen[2], sen[3]):
            sent.append((word["form"], word["upos"], sup))

        sents.append(sent)

    return sents
