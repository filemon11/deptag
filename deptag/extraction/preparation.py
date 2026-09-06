from . import extractor
import conllu
import dataclasses

from typing import Iterable, Collection, Mapping


@dataclasses.dataclass(frozen=True)
class Token:
    # Both in surface order, left-to-right.
    word: str
    upos: str
    xpos: str
    sup: str
    head: int
    deprel: str
    sup_deprel: str
    feats: dict[str, str]


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
        subtypes: bool = True,
        ) -> tuple[
            list[list[Token]], dict[str, int]]:
    # -> word, pos, supertag

    deprel_to_new: dict[str, str] = {}
    if merged is not None:
        for new, deprel_list in merged.items():
            for deprel in deprel_list:
                deprel_to_new[deprel] = new

    sents: list[list[Token]] = []
    sup2id: dict[str, int] = {"-root*": 0, "*+root": 1}
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
            subtypes=subtypes,
            ):
        sent: list[Token] = []
        for sup, word in zip(sen[2], sen[3]):
            sent.append(Token(
                word=word["form"],
                upos=word["upos"],
                xpos=word["xpos"],
                sup=sup,
                head=word["head"],
                deprel=word["deprel"],
                sup_deprel=extractor.deprel_merge(
                    word["deprel"],
                    deprel_to_new,
                    merged_fallback_subtypes,
                    distinguish_merged_fallback_subtypes),
                feats=word["feats"] if word["feats"] is not None else {},
            ))

            if sup not in sup2id:
                sup2id[sup] = len(sup2id)  # +1

        sents.append(sent)

    sup2id["-UNK*"] = len(sup2id)
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
        subtypes: bool = True,
        ) -> list[list[Token]]:
    # -> word, pos, supertag

    deprel_to_new: dict[str, str] = {}
    if merged is not None:
        for new, deprel_list in merged.items():
            for deprel in deprel_list:
                deprel_to_new[deprel] = new

    sents: list[list[Token]] = []
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
            subtypes=subtypes,
            ):
        sent: list[Token] = []
        for sup, word in zip(sen[2], sen[3]):
            sent.append(Token(
                word=word["form"],
                upos=word["upos"],
                xpos=word["xpos"],
                sup=sup,
                head=word["head"],
                deprel=word["deprel"],
                sup_deprel=extractor.deprel_merge(
                    word["deprel"],
                    deprel_to_new,
                    merged_fallback_subtypes,
                    distinguish_merged_fallback_subtypes),
                feats=word["feats"] if word["feats"] is not None else {}
            ))

        sents.append(sent)

    return sents
