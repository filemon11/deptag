from .. import extraction

import heapq
import itertools
import numpy as np

import tqdm
import multiprocessing as mp
import os

from dataclasses import dataclass
from itertools import product

from typing import Mapping, Collection, Literal, Sequence


def decode_aux_position(
        cls: int,
        max_l: int,
        ) -> tuple[str, int] | None:
    """Return (side, distance_from_star), or None."""
    no_aux = max_l + 1

    if cls < no_aux:
        # cls=max_l     -> distance 1 left
        # cls=max_l-1   -> distance 2 left
        # cls=0         -> distance max_l+1 left
        return "left", no_aux - cls

    if cls == no_aux:
        return None

    # cls=max_l+2 -> distance 1 right
    return "right", cls - no_aux


def aux_position_is_valid(
        cls: int,
        n_left: int,
        n_right: int,
        max_l: int,
        ) -> bool:
    location = decode_aux_position(cls, max_l)

    if location is None:
        return True

    side, distance = location

    if side == "left":
        return distance <= n_left + 1

    return distance <= n_right + 1


@dataclass(frozen=True)
class FactoredSupertag:
    # Both in surface order, left-to-right.
    left: tuple[str, ...]
    right: tuple[str, ...]

    # Uses classifier class numbering.
    aux_position: int

    # None iff aux_position == max_l + 1.
    aux_deprel: str | None


def render_supertag(
        tag: FactoredSupertag,
        max_l: int,
        ) -> str:

    left = [
        f"+{deprel}"
        for deprel in tag.left
    ]
    right = [
        f"+{deprel}"
        for deprel in tag.right
    ]

    location = decode_aux_position(
        tag.aux_position,
        max_l,
    )

    if location is not None:
        assert tag.aux_deprel is not None

        head = f"-{tag.aux_deprel}"
        side, distance = location

        if side == "left":
            # There are len(left)+1 possible insertion
            # positions.
            #
            # d=1: immediately before *
            # d=len(left)+1: leftmost position
            insertion = len(left) + 1 - distance
            left.insert(insertion, head)

        else:
            # d=1: immediately after *
            insertion = distance - 1
            right.insert(insertion, head)

    return "".join(left) + "*" + "".join(right)


def class_to_aux_position(
        cls: int,
        max_l: int,
        ) -> int | None:
    position = cls - max_l - 1
    return None if position == 0 else position


def aux_position_to_class(
        position: int | None,
        max_l: int,
        ) -> int:
    position = 0 if position is None else position
    return position + max_l + 1


def factor_supertag(
        supertag: str,
        max_l: int,
        ) -> FactoredSupertag:

    (
        n_left,
        left_deprels,
        n_right,
        right_deprels,
        aux_position,
        aux_deprel,
    ) = extraction.convert_relative_tag_to_factorised(
        extraction.convert_string_to_relative_relation(supertag))

    assert n_left == len(left_deprels)
    assert n_right == len(right_deprels)

    return FactoredSupertag(
        left=tuple(left_deprels),
        right=tuple(right_deprels),
        aux_position=aux_position_to_class(aux_position, max_l),
        aux_deprel=aux_deprel,
    )


def score_tag(
        tag: FactoredSupertag,
        *,
        left_count_logp: np.ndarray,
        right_count_logp: np.ndarray,
        argument_logp: dict[str, np.ndarray],
        aux_position_logp: np.ndarray,
        aux_deprel_logp: np.ndarray,
        deprel2id: Mapping[str, int],
        max_l: int,
        ) -> float:

    n_left = len(tag.left)
    n_right = len(tag.right)

    score = (
        left_count_logp[n_left]
        + right_count_logp[n_right]
        + aux_position_logp[tag.aux_position]
    )

    # left_1 is closest to *, hence reverse surface order.
    for i, deprel in enumerate(
            reversed(tag.left), start=1):
        score += argument_logp[
            f"left_{i}"
        ][deprel2id[deprel]]

    # right_1 is assumed to be closest to * and therefore
    # already agrees with surface order.
    for i, deprel in enumerate(
            tag.right, start=1):
        score += argument_logp[
            f"right_{i}"
        ][deprel2id[deprel]]

    no_aux = max_l + 1

    if tag.aux_position != no_aux:
        assert tag.aux_deprel is not None

        score += aux_deprel_logp[
            deprel2id[tag.aux_deprel]
        ]

    return float(score)


@dataclass
class SupertagFactors:
    n_left: np.ndarray
    n_right: np.ndarray
    aux_position: np.ndarray
    aux_deprel: np.ndarray

    left_deprels: np.ndarray
    right_deprels: np.ndarray


def preprocess_supertags(
        supertag2id: Mapping[str, int],
        deprel2id: Mapping[str, int],
        max_l: int,
        max_r: int,
        ) -> SupertagFactors:

    n_tags = len(supertag2id)

    ids = set(supertag2id.values())
    expected_ids = set(range(n_tags))

    if ids != expected_ids:
        raise ValueError(
            "Supertag IDs must be contiguous from 0 to "
            f"{n_tags - 1}, but got IDs "
            f"{sorted(ids)}."
        )

    n_left = np.empty(n_tags, dtype=np.int64)
    n_right = np.empty(n_tags, dtype=np.int64)
    aux_position = np.empty(n_tags, dtype=np.int64)

    aux_deprel = np.full(
        n_tags,
        -1,
        dtype=np.int64,
    )

    left_deprels = np.full(
        (n_tags, max_l),
        -1,
        dtype=np.int64,
    )

    right_deprels = np.full(
        (n_tags, max_r),
        -1,
        dtype=np.int64,
    )

    for supertag, tag_id in supertag2id.items():
        if supertag == "UNK":
            continue

        tag = factor_supertag(
            supertag,
            max_l,
        )

        n_left[tag_id] = len(tag.left)
        n_right[tag_id] = len(tag.right)
        aux_position[tag_id] = tag.aux_position

        # left_i is numbered inside-out, while tag.left
        # is stored in surface order.
        for i, deprel in enumerate(
                reversed(tag.left)):
            left_deprels[tag_id, i] = (
                deprel2id[deprel]
            )

        # right_i agrees with surface order.
        for i, deprel in enumerate(tag.right):
            right_deprels[tag_id, i] = (
                deprel2id[deprel]
            )

        if tag.aux_deprel is not None:
            aux_deprel[tag_id] = (
                deprel2id[tag.aux_deprel]
            )

    return SupertagFactors(
        n_left=n_left,
        n_right=n_right,
        aux_position=aux_position,
        aux_deprel=aux_deprel,
        left_deprels=left_deprels,
        right_deprels=right_deprels,
    )


def score_supertags_batch(
        factors: SupertagFactors,
        argument_logps: Mapping[str, np.ndarray],
        left_count_logps: np.ndarray,
        right_count_logps: np.ndarray,
        aux_position_logps: np.ndarray,
        aux_deprel_logps: np.ndarray,
        ) -> np.ndarray:
    """Score all supertags for every token.

    Parameters
    ----------
    left_count_logps
        [B, S, max_l + 1]
    right_count_logps
        [B, S, max_r + 1]
    argument_logps
        left_i/right_i -> [B, S, num_deprels]
    aux_position_logps
        [B, S, max_l + max_r + 3]
    aux_deprel_logps
        [B, S, num_deprels]

    Returns
    -------
    np.ndarray
        [B, S, num_supertags]
    """

    # Each 1-D index array has shape [T], so NumPy broadcasts
    # this into [B, S, T].
    scores = (
        left_count_logps[..., factors.n_left]
        + right_count_logps[..., factors.n_right]
        + aux_position_logps[..., factors.aux_position]
    )

    max_l = factors.left_deprels.shape[1]
    max_r = factors.right_deprels.shape[1]

    # Left argument labels.
    for i in range(max_l):
        label_ids = factors.left_deprels[:, i]
        active = label_ids >= 0

        # Replace -1 temporarily by a harmless valid index.
        safe_ids = np.maximum(label_ids, 0)

        slot_scores = argument_logps[
            f"left_{i + 1}"
        ][..., safe_ids]

        scores += slot_scores * active

    # Right argument labels.
    for i in range(max_r):
        label_ids = factors.right_deprels[:, i]
        active = label_ids >= 0
        safe_ids = np.maximum(label_ids, 0)

        slot_scores = argument_logps[
            f"right_{i + 1}"
        ][..., safe_ids]

        scores += slot_scores * active

    # Auxiliary dependency label.
    active = factors.aux_deprel >= 0
    safe_ids = np.maximum(
        factors.aux_deprel,
        0,
    )

    aux_scores = aux_deprel_logps[
        ..., safe_ids
    ]

    scores += aux_scores * active

    return scores


def score_supertags(
        supertag2id: Mapping[str, int],
        deprel2id: Mapping[str, int],
        argument_logps: Mapping[str, np.ndarray],
        left_count_logps: np.ndarray,
        right_count_logps: np.ndarray,
        aux_position_logps: np.ndarray,
        aux_deprel_logps: np.ndarray,
        ) -> np.ndarray:
    """Score every supertag in `supertag2id`.

    Returns
    -------
    np.ndarray
        scores[tag_id] is the log-score of that supertag.
        Higher is better.
    """
    scores = np.full(
        max(supertag2id.values()) + 1,
        -np.inf,
        dtype=np.float32,
    )

    max_l = len(left_count_logps) - 1

    for supertag, tag_id in supertag2id.items():
        factored_supertag = factor_supertag(supertag, max_l)

        score = (
            left_count_logps[len(factored_supertag.left)]
            + right_count_logps[len(factored_supertag.right)]
        )

        # `left_deprels` is in surface order:
        #
        #     +A+B*
        #
        # but the neural heads are numbered inside-out:
        #
        #     left_1 = B
        #     left_2 = A
        #
        for i, deprel in enumerate(
                reversed(factored_supertag.left), start=1):
            score += argument_logps[
                f"left_{i}"
            ][deprel2id[deprel]]

        # On the right, surface order and inside-out order coincide:
        #
        #     *+A+B
        #
        #     right_1 = A
        #     right_2 = B
        for i, deprel in enumerate(
                factored_supertag.right, start=1):
            score += argument_logps[
                f"right_{i}"
            ][deprel2id[deprel]]

        # Your signed representation:
        #   -1   = immediately left of *
        #   None = no head slot
        #   +1   = immediately right of *
        #
        # maps directly to the fixed classifier coordinates.

        score += aux_position_logps[
            factored_supertag.aux_position
        ]

        # No head slot => no head dependency relation.
        if factored_supertag.aux_deprel is not None:
            score += aux_deprel_logps[
                deprel2id[factored_supertag.aux_deprel]
            ]

        scores[tag_id] = score

    return scores


def valid_aux_position_mask(
        n_left: int,
        n_right: int,
        max_l: int,
        max_r: int,
        projective_only: bool = False,
        ) -> np.ndarray:

    num_classes = max_l + max_r + 3

    mask = np.zeros(
        num_classes,
        dtype=bool,
    )

    leftmost = max_l - n_left
    no_aux = max_l + 1
    rightmost = max_l + n_right + 2

    if projective_only:
        mask[
            [leftmost, no_aux, rightmost]
        ] = True
    else:
        mask[
            leftmost:rightmost + 1
        ] = True

    return mask


def valid_aux_positions(
        n_left: int,
        n_right: int,
        max_l: int,
        mode: Literal["all", "projective"] = "all",
        ) -> list[int]:

    no_aux = max_l + 1

    # Class corresponding to -(n_left + 1).
    leftmost = max_l - n_left

    # Class corresponding to +(n_right + 1).
    rightmost = max_l + n_right + 2

    if mode == "all":
        return list(range(
            leftmost,
            rightmost + 1,
        ))

    if mode == "projective":
        return [
            leftmost,
            no_aux,
            rightmost,
        ]

    raise ValueError(
        f"Unknown mode: {mode!r}"
    )


def generate_valid_supertag2id(
        deprels: Collection[str],
        max_l: int,
        max_r: int,
        mode: Literal["all", "projective"] = "projective",
        ) -> dict[str, int]:
    """Generate all valid labelled supertags.

    Parameters
    ----------
    deprels
        Dependency relations that may occur on + and - slots.
    max_l
        Maximum number of left argument slots.
    max_r
        Maximum number of right argument slots.
    mode
        "all" allows projective and non-projective head-slot positions.
        "projective" allows only projective head-slot positions.
    """
    supertags: set[str] = set()

    no_aux = max_l + 1

    for n_left in range(max_l + 1):
        for n_right in range(max_r + 1):

            # All ordered dependency-label sequences of the
            # required lengths.
            for left_deprels in product(
                    deprels, repeat=n_left):

                for right_deprels in product(
                        deprels, repeat=n_right):

                    # -----------------------------------------
                    # No head/auxiliary slot.
                    # -----------------------------------------
                    tag = FactoredSupertag(
                        left=left_deprels,
                        right=right_deprels,
                        aux_position=no_aux,
                        aux_deprel=None,
                    )

                    supertags.add(
                        render_supertag(
                            tag,
                            max_l=max_l,
                        )
                    )

                    # -----------------------------------------
                    # Tags containing a head/auxiliary slot.
                    # -----------------------------------------
                    for aux_position in valid_aux_positions(
                            n_left=n_left,
                            n_right=n_right,
                            max_l=max_l,
                            mode=mode,
                            ):

                        # We already generated the no-head case.
                        if aux_position == no_aux:
                            continue

                        for aux_deprel in deprels:
                            tag = FactoredSupertag(
                                left=left_deprels,
                                right=right_deprels,
                                aux_position=aux_position,
                                aux_deprel=aux_deprel,
                            )

                            supertags.add(
                                render_supertag(
                                    tag,
                                    max_l=max_l,
                                )
                            )

    # Sorting makes IDs reproducible across runs.
    return {
        supertag: i
        for i, supertag in enumerate(
            sorted(supertags)
        )
    }


DUMMY_DEPREL = "_"


def structuralize_supertag(
        supertag: str,
        max_l: int,
        ) -> str:
    tag = factor_supertag(
        supertag,
        max_l,
    )

    structural_tag = FactoredSupertag(
        left=(
            DUMMY_DEPREL,
        ) * len(tag.left),

        right=(
            DUMMY_DEPREL,
        ) * len(tag.right),

        aux_position=tag.aux_position,

        aux_deprel=(
            None
            if tag.aux_deprel is None
            else DUMMY_DEPREL
        ),
    )

    return render_supertag(
        structural_tag,
        max_l=max_l,
    )


def generate_valid_structural_supertag2id(
        max_l: int,
        max_r: int,
        mode: Literal[
            "all",
            "projective",
        ] = "projective",
        ) -> dict[str, int]:

    tags: set[str] = set()

    no_aux = max_l + 1

    for n_left in range(max_l + 1):
        for n_right in range(max_r + 1):

            for aux_position in valid_aux_positions(
                    n_left=n_left,
                    n_right=n_right,
                    max_l=max_l,
                    mode=mode,
                    ):

                tag = FactoredSupertag(
                    left=(
                        DUMMY_DEPREL,
                    ) * n_left,

                    right=(
                        DUMMY_DEPREL,
                    ) * n_right,

                    aux_position=aux_position,

                    aux_deprel=(
                        None
                        if aux_position == no_aux
                        else DUMMY_DEPREL
                    ),
                )

                tags.add(
                    render_supertag(
                        tag,
                        max_l=max_l,
                    )
                )

    return {
        tag: i
        for i, tag in enumerate(
            sorted(tags)
        )
    }


def score_structural_supertags_batch(
        factors: SupertagFactors,
        left_count_logps: np.ndarray,
        right_count_logps: np.ndarray,
        aux_position_logps: np.ndarray,
        ) -> np.ndarray:
    """Score structural supertags.

    Returns
    -------
    np.ndarray
        Shape [batch, sequence, num_supertags].
        Higher scores are better.
    """
    return (
        left_count_logps[
            ..., factors.n_left
        ]
        + right_count_logps[
            ..., factors.n_right
        ]
        + aux_position_logps[
            ..., factors.aux_position
        ]
    )


def k_best_product(
        logps: Sequence[np.ndarray],
        k: int,
        ) -> list[tuple[float, tuple[int, ...]]]:
    """Return the k best assignments to independent categorical factors.

    Parameters
    ----------
    logps
        One 1-D log-probability vector per factor.
    k
        Number of assignments to return.

    Returns
    -------
    list
        (summed_logp, assignment), ordered best first.
    """
    if not logps:
        return [(0.0, ())]

    # Alternatives for each factor, best first.
    orders = [
        np.argsort(-factor)
        for factor in logps
    ]

    sorted_logps = [
        factor[order]
        for factor, order in zip(logps, orders)
    ]

    # `ranks[d]` says which ranked alternative is selected
    # for factor d.
    start = tuple(0 for _ in logps)

    start_score = sum(
        float(factor[0])
        for factor in sorted_logps
    )

    heap = [(-start_score, start)]
    seen = {start}

    result: list[tuple[float | int, tuple[int, ...]]] = []

    while heap and len(result) < k:
        neg_score, ranks = heapq.heappop(heap)

        assignment = tuple(
            int(orders[d][rank])
            for d, rank in enumerate(ranks)
        )

        result.append(
            (-neg_score, assignment)
        )

        # Generate neighboring assignments by moving one
        # factor to its next-best value.
        for d in range(len(logps)):
            next_rank = ranks[d] + 1

            if next_rank >= len(sorted_logps[d]):
                continue

            new_ranks = list(ranks)
            new_ranks[d] = next_rank
            new_ranks_ = tuple(new_ranks)

            if new_ranks_ in seen:
                continue

            seen.add(new_ranks_)

            new_score = sum(
                float(sorted_logps[j][rank])
                for j, rank in enumerate(new_ranks_)
            )

            heapq.heappush(
                heap,
                (-new_score, new_ranks_),
            )

    return result


# def top_k_valid_supertags(
#         argument_logps: Mapping[str, np.ndarray],
#         left_count_logps: np.ndarray,
#         right_count_logps: np.ndarray,
#         aux_position_logps: np.ndarray,
#         aux_deprel_logps: np.ndarray,
#         id2deprel: Mapping[int, str],
#         max_l: int,
#         max_r: int,
#         k: int,
#         projective_only: bool = True,
#         ) -> list[tuple[float, str]]:
#     """Generate exact k-best valid factorized supertags for one token.

#     All inputs are for a single token, i.e. the batch and sequence
#     dimensions have already been selected.

#     Returns
#     -------
#     list
#         (log_probability, supertag_string), best first.
#     """
#     # Global min-heap containing only the current k best tags.
#     best: list[tuple[float, int, str]] = []
#     tie_breaker = itertools.count()

#     def offer(score: float, supertag: str) -> None:
#         entry = (
#             score,
#             next(tie_breaker),
#             supertag,
#         )

#         if len(best) < k:
#             heapq.heappush(best, entry)

#         elif score > best[0][0]:
#             heapq.heapreplace(best, entry)

#     mode: Literal["projective", "all"] = (
#         "projective"
#         if projective_only
#         else "all"
#     )

#     no_aux = max_l + 1

#     for n_left in range(max_l + 1):
#         for n_right in range(max_r + 1):

#             count_score = (
#                 float(left_count_logps[n_left])
#                 + float(right_count_logps[n_right])
#             )

#             for aux_position in valid_aux_positions(
#                     n_left=n_left,
#                     n_right=n_right,
#                     max_l=max_l,
#                     mode=mode,
#                     ):

#                 structure_score = (
#                     count_score
#                     + float(
#                         aux_position_logps[
#                             aux_position
#                         ]
#                     )
#                 )

#                 factors: list[np.ndarray] = []

#                 # left_1, left_2, ... are inside-out.
#                 for i in range(1, n_left + 1):
#                     factors.append(
#                         argument_logps[f"left_{i}"]
#                     )

#                 # right_1, right_2, ... are already in
#                 # surface order.
#                 for i in range(1, n_right + 1):
#                     factors.append(
#                         argument_logps[f"right_{i}"]
#                     )

#                 has_aux = aux_position != no_aux

#                 if has_aux:
#                     factors.append(aux_deprel_logps)

#                 # A single structural configuration cannot
#                 # contribute more than k items to the global
#                 # top-k, so obtaining its own k best is sufficient.
#                 for factor_score, assignment in k_best_product(
#                         factors,
#                         k,
#                         ):

#                     cursor = 0

#                     # Classifier order is inside-out. FactoredSupertag
#                     # expects surface order, so reverse the left side.
#                     left_classifier = assignment[
#                         cursor:cursor + n_left
#                     ]
#                     cursor += n_left

#                     left = tuple(
#                         id2deprel[label]
#                         for label in reversed(
#                             left_classifier
#                         )
#                     )

#                     right_classifier = assignment[
#                         cursor:cursor + n_right
#                     ]
#                     cursor += n_right

#                     right = tuple(
#                         id2deprel[label]
#                         for label in right_classifier
#                     )

#                     if has_aux:
#                         aux_deprel = id2deprel[
#                             assignment[cursor]
#                         ]
#                     else:
#                         aux_deprel = None

#                     tag = FactoredSupertag(
#                         left=left,
#                         right=right,
#                         aux_position=aux_position,
#                         aux_deprel=aux_deprel,
#                     )

#                     supertag = render_supertag(
#                         tag,
#                         max_l=max_l,
#                     )

#                     offer(
#                         structure_score + factor_score,
#                         supertag,
#                     )

#     return [
#         (score, supertag)
#         for score, _, supertag in sorted(
#             best,
#             reverse=True,
#         )
#     ]


def top_k_valid_supertags(
        argument_logps: Mapping[str, np.ndarray],
        left_count_logps: np.ndarray,
        right_count_logps: np.ndarray,
        aux_position_logps: np.ndarray,
        aux_deprel_logps: np.ndarray,
        id2deprel: Mapping[int, str],
        max_l: int,
        max_r: int,
        k: int,
        projective_only: bool = True,
        ) -> list[tuple[float, str]]:

    if k <= 0:
        return []

    mode: Literal["projective", "all"] = (
        "projective"
        if projective_only
        else "all"
    )

    no_aux = max_l + 1

    # ---------------------------------------------------------
    # Rank every categorical factor ONCE.
    #
    # prepared[name] =
    #     (class IDs in descending score order,
    #      corresponding descending scores)
    # ---------------------------------------------------------

    def prepare(
            values: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray]:

        order = np.argsort(
            -values,
            kind="stable",
        )

        return order, values[order]

    prepared_arguments = {
        name: prepare(values)
        for name, values in argument_logps.items()
    }

    prepared_aux = prepare(aux_deprel_logps)

    # A configuration is:
    #
    # (
    #     n_left,
    #     n_right,
    #     aux_position,
    #     structure_score,
    #     factors,
    # )
    #
    # factors is a tuple of pre-ranked categorical factors.
    configs = []

    # Global heap entries:
    #
    # (-total_score, tie_breaker, config_id, ranks)
    #
    # ranks[d] is the rank chosen for factor d.
    heap = []
    tie_breaker = itertools.count()

    # Prevent reaching the same product state by multiple paths.
    seen: set[tuple[int, tuple[int, ...]]] = set()

    for n_left in range(max_l + 1):
        for n_right in range(max_r + 1):

            count_score = (
                float(left_count_logps[n_left])
                + float(right_count_logps[n_right])
            )

            for aux_position in valid_aux_positions(
                    n_left=n_left,
                    n_right=n_right,
                    max_l=max_l,
                    mode=mode,
                    ):

                structure_score = (
                    count_score
                    + float(
                        aux_position_logps[
                            aux_position
                        ]
                    )
                )

                factors = []

                for i in range(1, n_left + 1):
                    factors.append(
                        prepared_arguments[
                            f"left_{i}"
                        ]
                    )

                for i in range(1, n_right + 1):
                    factors.append(
                        prepared_arguments[
                            f"right_{i}"
                        ]
                    )

                has_aux = aux_position != no_aux

                if has_aux:
                    factors.append(prepared_aux)

                factors_ = tuple(factors)

                config_id = len(configs)

                configs.append((
                    n_left,
                    n_right,
                    aux_position,
                    structure_score,
                    factors_,
                ))

                ranks = (0,) * len(factors_)

                # Best possible assignment for this structure.
                score = structure_score

                for _, sorted_scores in factors_:
                    score += float(sorted_scores[0])

                state = (config_id, ranks)
                seen.add(state)

                heapq.heappush(
                    heap,
                    (
                        -score,
                        next(tie_breaker),
                        config_id,
                        ranks,
                    ),
                )

    # ---------------------------------------------------------
    # Global best-first enumeration.
    # ---------------------------------------------------------

    result: list[tuple[float, str]] = []

    while heap and len(result) < k:

        (
            neg_score,
            _,
            config_id,
            ranks,
        ) = heapq.heappop(heap)

        score = -neg_score

        (
            n_left,
            n_right,
            aux_position,
            structure_score,
            factors,
        ) = configs[config_id]

        # Convert ranked positions back into classifier IDs.
        assignment = tuple(
            int(order[rank])
            for (order, _), rank
            in zip(factors, ranks)
        )

        cursor = 0

        left_classifier = assignment[
            cursor:cursor + n_left
        ]
        cursor += n_left

        left = tuple(
            id2deprel[label]
            for label in reversed(
                left_classifier
            )
        )

        right_classifier = assignment[
            cursor:cursor + n_right
        ]
        cursor += n_right

        right = tuple(
            id2deprel[label]
            for label in right_classifier
        )

        if aux_position != no_aux:
            aux_deprel = id2deprel[
                assignment[cursor]
            ]
        else:
            aux_deprel = None

        tag = FactoredSupertag(
            left=left,
            right=right,
            aux_position=aux_position,
            aux_deprel=aux_deprel,
        )

        result.append((
            score,
            render_supertag(
                tag,
                max_l=max_l,
            ),
        ))

        # Expand neighboring product states.
        for d, (_, sorted_scores) in enumerate(
                factors):

            old_rank = ranks[d]
            new_rank = old_rank + 1

            if new_rank >= len(sorted_scores):
                continue

            new_ranks = list(ranks)
            new_ranks[d] = new_rank
            new_ranks_ = tuple(new_ranks)

            state = (
                config_id,
                new_ranks_,
            )

            if state in seen:
                continue

            seen.add(state)

            # O(1) update instead of recomputing the
            # sum over every factor.
            new_score = (
                score
                - float(
                    sorted_scores[old_rank]
                )
                + float(
                    sorted_scores[new_rank]
                )
            )

            heapq.heappush(
                heap,
                (
                    -new_score,
                    next(tie_breaker),
                    config_id,
                    new_ranks_,
                ),
            )

    return result


_TOP_K_STATE = None


def _init_top_k_worker(
        argument_logps,
        left_count_logps,
        right_count_logps,
        aux_position_logps,
        aux_deprel_logps,
        id2deprel,
        max_l,
        max_r,
        k,
        projective_only,
        ):
    """Initialize read-only state for a top-k worker."""
    global _TOP_K_STATE

    _TOP_K_STATE = {
        "argument_logps": argument_logps,
        "left_count_logps": left_count_logps,
        "right_count_logps": right_count_logps,
        "aux_position_logps": aux_position_logps,
        "aux_deprel_logps": aux_deprel_logps,
        "id2deprel": id2deprel,
        "max_l": max_l,
        "max_r": max_r,
        "k": k,
        "projective_only": projective_only,
    }


def _top_k_valid_supertags_worker(
        index: tuple[int, int],
        ):
    """Process one token."""
    b, s = index
    state = _TOP_K_STATE

    token_argument_logps = {
        name: values[b, s]
        for name, values
        in state["argument_logps"].items()
    }

    return top_k_valid_supertags(
        argument_logps=token_argument_logps,
        left_count_logps=state["left_count_logps"][b, s],
        right_count_logps=state["right_count_logps"][b, s],
        aux_position_logps=state["aux_position_logps"][b, s],
        aux_deprel_logps=state["aux_deprel_logps"][b, s],
        id2deprel=state["id2deprel"],
        max_l=state["max_l"],
        max_r=state["max_r"],
        k=state["k"],
        projective_only=state["projective_only"],
    )


def get_eval_workers() -> int:
    value = os.environ.get("DEPTAG_EVAL_WORKERS")

    if value is None:
        # Safe fallback: never silently launch one process per node CPU.
        return 1

    workers = int(value)
    if workers < 1:
        raise ValueError(
            f"DEPTAG_EVAL_WORKERS must be >= 1, got {workers}"
        )

    return workers


def top_k_valid_supertags_batch(
        argument_logps: Mapping[str, np.ndarray],
        left_count_logps: np.ndarray,
        right_count_logps: np.ndarray,
        aux_position_logps: np.ndarray,
        aux_deprel_logps: np.ndarray,
        id2deprel: Mapping[int, str],
        max_l: int,
        max_r: int,
        k: int,
        projective_only: bool,
        valid_mask: np.ndarray,
        chunksize: int = 8,
        ):
    batch_size, sequence_length = (
        left_count_logps.shape[:2]
    )

    # Only process actual words, not padding positions.
    indices = [
        (b, s)
        for b in range(batch_size)
        for s in range(sequence_length)
        if valid_mask[b, s]
    ]

    with mp.Pool(
            processes=get_eval_workers(),
            initializer=_init_top_k_worker,
            initargs=(
                argument_logps,
                left_count_logps,
                right_count_logps,
                aux_position_logps,
                aux_deprel_logps,
                id2deprel,
                max_l,
                max_r,
                k,
                projective_only,
            ),
            ) as pool:

        flat_result = list(
            tqdm.tqdm(
                pool.imap(
                    _top_k_valid_supertags_worker,
                    indices,
                    chunksize=chunksize,
                ),
                total=len(indices),
                desc="Reconstructing k best supertags",
            )
        )

    # Restore [batch][sequence] layout.
    # Padding positions stay empty.
    result = [
        [
            []
            for _ in range(sequence_length)
        ]
        for _ in range(batch_size)
    ]

    for (b, s), token_result in zip(
            indices,
            flat_result):
        result[b][s] = token_result

    return result


def make_sentence_supertag_scores(
        token_candidates: list[
            list[tuple[float, str]]
        ],
        ) -> tuple[
            np.ndarray,
            dict[int, str],
            dict[str, int],
        ]:

    tags = sorted({
        tag
        for candidates in token_candidates
        for _, tag in candidates
    })

    supertag2id = {
        tag: i
        for i, tag in enumerate(tags)
    }

    id2sup = {
        i: tag
        for tag, i in supertag2id.items()
    }

    scores = np.full(
        (
            len(token_candidates),
            len(tags),
        ),
        -np.inf,
        dtype=np.float32,
    )

    for token, candidates in enumerate(
            token_candidates):

        for score, tag in candidates:
            scores[
                token,
                supertag2id[tag],
            ] = -score

    return scores, id2sup, supertag2id


def make_batch_supertag_scores(
        batch_candidates: list[
            list[list[tuple[float, str]]]
        ],
        root_supertag: str = "*+root",
        ) -> tuple[
            np.ndarray,
            dict[int, str],
            dict[str, int],
        ]:
    """Convert [B][S][K] candidates to a dense [B,S,T] matrix.

    Candidate scores are assumed to be log-probabilities, i.e.
    higher is better.

    The returned matrix contains -log10 probabilities, i.e.
    lower is better, as expected by A*.
    """
    batch_size = len(batch_candidates)
    sequence_length = len(batch_candidates[0])

    # Union of all top-k tags occurring anywhere in this batch.
    tags = {
        tag
        for sentence_candidates in batch_candidates
        for token_candidates in sentence_candidates
        for _, tag in token_candidates
    }

    # A* needs to be able to refer to the root supertag even if
    # it wasn't among the generated top-k candidates.
    tags.add(root_supertag)

    # Deterministic IDs.
    ordered_tags = sorted(tags)

    supertag2id = {
        tag: i
        for i, tag in enumerate(ordered_tags)
    }

    id2sup = {
        i: tag
        for tag, i in supertag2id.items()
    }

    # Missing token/tag combinations are impossible candidates.
    supertag_scores = np.full(
        (
            batch_size,
            sequence_length,
            len(ordered_tags),
        ),
        np.inf,
        dtype=np.float32,
    )

    for b, sentence_candidates in enumerate(
            batch_candidates):

        for s, token_candidates in enumerate(
                sentence_candidates):

            for logp, tag in token_candidates:
                supertag_scores[
                    b,
                    s,
                    supertag2id[tag],
                ] = -logp   #  / np.log(10.0)

    return (
        supertag_scores,
        id2sup,
        supertag2id,
    )
