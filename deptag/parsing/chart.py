import numpy as np
from numba import njit, prange
from math import inf

from ufal.chu_liu_edmonds import chu_liu_edmonds

import frozendict
import heapq
import os

import tqdm

import multiprocessing

from .. import extraction, utils
from . import deprels

from timeit import default_timer as timer
from datetime import timedelta

from collections import namedtuple, deque
import itertools

from typing import (
    Self, Literal, Mapping, Callable,
    Any, Collection, Generic, Hashable,
    TypeVar)


Aux = Literal["-", "<", ">"]
Auxnum = Literal[0, 1, 2]
AUXMAP: Mapping[Aux, int] = frozendict.frozendict(
    {"-": 0, "<": 1, ">": 2}
)

N: Literal[0] = 0
L: Literal[1] = 1
R: Literal[2] = 2

REVERSE_AUXMAP: Mapping[int, Aux] = frozendict.frozendict(
    {value: key for key, value in AUXMAP.items()}
)

DTYPE = np.float32


# @functools.total_ordering
class Weight():
    __slots__ = ("inside", "out_estimate", "sum")

    def __init__(self, inside: float | int, out_estimate: float | int):
        self.inside: float = float(inside)
        self.out_estimate: float = float(out_estimate)
        self.sum: float = inside + out_estimate

    def __float__(self) -> float:
        return self.sum

    # def lt(self, other: "Weight") -> bool:
    #     return self.sum < other.sum

    def __lt__(self, other: "Weight") -> bool:
        return self.sum < other.sum

    # def __lt__(self, other: object) -> bool:
    #     if not isinstance(other, Weight):
    #         return NotImplemented

    #     if self.sum != other.sum:
    #         return self.sum < other.sum

    #     return self.inside > other.inside

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Weight):
            return NotImplemented
        return self.sum == other.sum   # and self.inside == other.inside

    def to_array(self) -> np.ndarray:
        return np.array(
            (self.inside, self.out_estimate),
            dtype=DTYPE)

    def __str__(self) -> str:
        return (
            f"({round(self.inside, 2)},{round(self.out_estimate, 2)})")

    def __repr__(self) -> str:
        return str(self)


class WeightPointer(Weight):
    __slots__ = ("back1", "back2", "supertag_ind")

    def __init__(
            self: Self,
            inside: float,
            out_estimate: float,
            back1: "Item",
            back2: "Item",
            supertag_ind: int):
        super().__init__(inside, out_estimate)
        self.back1: Item = back1
        self.back2: Item = back2
        self.supertag_ind: int = supertag_ind


_ItemTuple = namedtuple(
    "_ItemTuple",
    ("start", "end", "anchor", "l_args", "r_args", "aux"),
)


class Item(_ItemTuple):
    __slots__ = ()

    def __new__(
            cls,
            start: int | None,
            end: int | None,
            anchor: int | None,
            l_args: int | None,
            r_args: int | None,
            aux: int | None | Aux,
            ) -> Self:
        aux = AUXMAP[aux] if isinstance(aux, str) else aux

        assert (
            (start is None)
            == (end is None)
            == (anchor is None)
            == (aux is None)
        )

        if start is None:
            assert l_args is None and r_args is None
        else:
            assert end is not None
            assert anchor is not None
            assert start != 0
            assert start <= anchor <= end

        if r_args is None:
            assert l_args is None

        if l_args is not None:
            assert start is not None
            assert aux is not None
            assert l_args + (aux == L) < start

        return _ItemTuple.__new__(
            cls,
            start,
            end,
            anchor,
            l_args,
            r_args,
            aux,
        )

    @classmethod
    def unchecked(
            cls,
            start: int | None,
            end: int | None,
            anchor: int | None,
            l_args: int | None,
            r_args: int | None,
            aux: int | None,
            ) -> Self:
        return _ItemTuple.__new__(
            cls,
            start,
            end,
            anchor,
            l_args,
            r_args,
            aux,
        )

    @property
    def tup(self) -> tuple[
            int | None,
            int | None,
            int | None,
            int | None,
            int | None,
            int | None,
            ]:
        return self

    @property
    def is_axiom(self) -> bool:
        return self.start is None

    @property
    def is_initial(self) -> bool:
        return self.aux == N

    @property
    def is_adjunct(self) -> bool:
        return self.aux == R or self.aux == L

    @property
    def is_left_adjunct(self) -> bool:
        return self.aux == R

    @property
    def is_right_adjunct(self) -> bool:
        return self.aux == L

    @property
    def is_complete(self) -> bool:
        return self.r_args is None and self.start is not None

    @property
    def is_uncomplete(self) -> bool:
        return self.r_args is not None and self.start is not None

    @property
    def is_switched(self) -> bool:
        return (
            self.r_args is not None
            and self.start is not None
            and self.l_args is None)

    @property
    def is_unswitched(self) -> bool:
        return (
            self.r_args is not None
            and self.start is not None
            and self.l_args is not None
        )

    def is_goal(self, length: int) -> bool:
        return (
            self.is_complete
            and self.is_initial
            and self.start == 1
            and self.end == length)

    def __str__(self) -> str:
        if self.is_axiom:
            return "AXIOM"
        assert self.aux is not None
        if self.is_complete:
            return (
                f"[{self.start},{self.end},"
                f"{self.anchor},-,-,{REVERSE_AUXMAP[self.aux]}]")
        if self.is_switched:
            return (
                f"[{self.start},{self.end},{self.anchor},"
                f"-,{self.r_args},{REVERSE_AUXMAP[self.aux]}]")
        return (
            f"[{self.start},{self.end},{self.anchor},"
            f"{self.l_args},{self.r_args},{REVERSE_AUXMAP[self.aux]}]")

    def __repr__(self) -> str:
        return str(self)

    def __lt__(self, other: Self) -> bool:  # type: ignore
        assert not self.is_axiom
        assert not other.is_axiom
        assert self.start is not None
        assert other.start is not None
        return self.start < other.start

    # def __eq__(self, other: Any) -> bool:
    #     if isinstance(other, Item):
    #         return (
    #             self.start == other.start
    #             and self.end == other.end
    #             and self.anchor == other.anchor
    #             and self.l_args == other.l_args
    #             and self.r_args == other.r_args
    #             and self.aux == other.aux
    #         )
    #     elif isinstance(other, Sequence):
    #         if len(other) == 6 and isinstance(other[5], str):
    #             other = list(other)
    #             other[5] = AUXMAP[other[5]]
    #         return self.tup == other
    #     return self.tup == other

    # def __hash__(self):
    #     return hash(self.tup)

    def __len__(self):
        if self.end is None or self.start is None:
            return 0
        return self.end-self.start+1


AXIOM = Item(None, None, None, None, None, None)

EntryType = tuple[Item, WeightPointer]


class Chart():
    # index 1: span start, length: |w| + 1 (None)
    # index 2: span end, length: |w| + 1 (None)
    # (empty span not possible)
    # index 3: span anchor, length |w| + 1 (None)
    # index 4: left args, length max_l + 2 (0 and None)
    # (0 for zero left args and -1 for switched item)
    # index 5: right args, length max_r + 2
    # (0 for zero right args and -1 for completed item)
    # index 6: auxiliary marker, length 3 (<-, ->, -) + 1 (None)
    # index 7: inside cost/outside estimate,
    # backpointers, supertag index -> length 2+2*6+1

    def __init__(self, length: int, max_l: int, max_r: int) -> None:

        self._chart: np.ndarray = np.zeros(
            (length+1, length+1, length+1, max_l+2, max_r+2, 4, 2+2*6+1),
            dtype=np.float32)
        self._chart[:, :, :, :, :, :, :2] = np.inf

        assert max_l < length, "max_l must be smaller than length"

        # self._complete_by_start: list[list[list[
        #     EntryType]]] = [
        #     [[] for _ in range(4)]
        #     for _ in range(length + 2)
        # ]
        self._complete_by_start_: list[list[list[list[
                    EntryType]]]] = [
                    [[[] for _ in range(length + 2)] for _ in range(4)]
                    for _ in range(length + 2)
                ]
        # self._complete_by_end: list[list[list[EntryType]]] = [
        #     [[] for _ in range(4)]
        #     for _ in range(length + 2)
        # ]
        self._complete_by_end_: list[list[list[list[EntryType]]]] = [
                [[[] for _ in range(length + 2)] for _ in range(4)]
                for _ in range(length + 2)
            ]

        # Unswitched items:
        # l_args is an integer and r_args is an integer.
        # self._unswitched_by_start: list[list[EntryType]] = [
        #     [] for _ in range(length + 2)
        # ]
        self._unswitched_by_start_: list[list[list[EntryType]]] = [
            [[] for _ in range(max_l + 2)] for _ in range(length + 2)
        ]

        # Switched items:
        # l_args is None and r_args is an integer.
        # self._switched_by_end: list[list[
        #     EntryType]] = [
        #     [] for _ in range(length + 2)
        # ]
        self._switched_by_end_: list[list[list[
            EntryType]]] = [
            [[] for _ in range(max_r + 2)] for _ in range(length + 2)
        ]

    @staticmethod
    def get_index(
            p: Item
            ) -> tuple[int, int, int, int, int, int]:
        return Chart.item2chartidxs(p)
        # if isinstance(p, Item):
        #     p = Chart.item2chartidxs(p)
        # else:
        #     assert len(p) == 6, f"Indices missing from '{p}'"
        # p = cast(tuple[int, int, int, int, int, int], p)
        # return p

    def __getitem__(
            self, p: Item
            ) -> WeightPointer:
        t = self.get_index(p)
        weight_pointer: np.ndarray = self._chart[t]
        # assert len(
        #     weight_pointer.shape) == 1 and weight_pointer.shape[
        # -1] == 2+2*6+1
        # print("weight_pointer", weight_pointer)
        return WeightPointer(
            inside=weight_pointer[0].item(),
            out_estimate=weight_pointer[1].item(),
            back1=self.chartidxs2item(
                tuple(weight_pointer[2:8].astype(int).tolist())),
            back2=self.chartidxs2item(
                tuple(weight_pointer[8:14].astype(int).tolist())),
            supertag_ind=int(weight_pointer[14].astype(int).tolist())
            )

    def __setitem__(
            self, p: Item,
            weight_pointer: WeightPointer) -> None:
        t = self.get_index(p)

        self._chart[t] = (
            *weight_pointer.to_array(),
            *self.item2chartidxs(weight_pointer.back1),
            *self.item2chartidxs(weight_pointer.back2),
            weight_pointer.supertag_ind)

        # if not isinstance(p, Item):
        #     p = self.chartidxs2item(p)
        # start, end, anchor, l_args, r_args, aux
        #   0     1     2       3        4     5
        entry = p, weight_pointer

        if t[4] == 0:

            self._complete_by_start_[t[0]][t[5]][t[1]].append(
                entry)
            self._complete_by_end_[t[1]][t[5]][t[0]].append(
                entry
            )

        elif t[3] == 0:
            self._switched_by_end_[t[1]][t[4]].append(
                entry
            )

        else:
            self._unswitched_by_start_[t[0]][t[3]].append(
                entry
            )

    def peek(
            self: Self, p: Item
            ) -> None | WeightPointer:
        weight = self.__getitem__(p)
        if weight.sum < inf:
            return weight
        return None

    @staticmethod
    def item2chartidxs(
            item: Item) -> tuple[
                int, int, int, int, int, int]:
        return (
            0 if item.start is None else item.start,
            0 if item.end is None else item.end,
            0 if item.anchor is None else item.anchor,
            0 if item.l_args is None else item.l_args+1,
            0 if item.r_args is None else item.r_args+1,
            0 if item.aux is None else item.aux+1)

    @staticmethod
    def chartidxs2item(
            idxs: tuple[
                int, int, int, int, int, int]) -> Item:
        return Item.unchecked(
            start=None if idxs[0] == 0 else idxs[0],
            end=None if idxs[1] == 0 else idxs[1],
            anchor=None if idxs[2] == 0 else idxs[2],
            l_args=None if idxs[3] == 0 else idxs[3]-1,
            r_args=None if idxs[4] == 0 else idxs[4]-1,
            aux=None if idxs[5] == 0 else idxs[5]-1,
        )


T = TypeVar("T", bound=Hashable)


# class Agenda_(Generic[T]):
#     def __init__(self) -> None:
#         self._heap = heapdict.heapdict()

#     def add_update(self, item: T, weight: WeightPointer) -> None:
#         try:
#             current_weight = self._heap[item]
#             if weight.lt(current_weight):
#                 self._heap[item] = weight
#         except KeyError:
#             self._heap[item] = weight

#     def pop(self) -> tuple[T, WeightPointer]:
#         item, weight = self._heap.popitem()
#         return item, weight

#     def __str__(self) -> str:
#         items = list(self._heap)
#         items = list(sorted(items, key=lambda x: self._heap[x].sum))
#         return ", ".join(
#             f'{str(self._heap[item])}:{str(item)}' for item in items)

#     def __repr__(self) -> str:
#         return str(self)

#     @property
#     def is_empty(self) -> bool:
#         return len(self._heap) == 0


class Agenda(Generic[T]):
    """A mutable A* agenda using heapq and lazy deletion.

    Items may receive improved weights while they are open. Old heap
    entries remain in the heap and are discarded when encountered.
    """

    __slots__ = (
        "_heap",
        "_current",
        "_counter",
    )

    def __init__(self) -> None:
        # Entries:
        #
        # (
        #     f = inside + outside,
        #     -inside,
        #     insertion sequence,
        #     item,
        #     weight,
        # )
        #
        # The unique sequence number ensures that Item objects are never
        # compared by heapq.
        self._heap: list[
            tuple[float, float, int, T, WeightPointer]
        ] = []

        # item -> (sequence number, current best WeightPointer)
        self._current: dict[
            T,
            tuple[int, WeightPointer],
        ] = {}

        self._counter = itertools.count()

    @staticmethod
    def _priority(
            weight: WeightPointer,
            ) -> tuple[float, float]:
        # Primary A* priority: smaller f = g + h.
        #
        # For equal f, prefer larger g. This moves the search
        # toward more complete items.
        return weight.sum, -weight.inside

    def add_update(
            self,
            item: T,
            weight: WeightPointer,
            ) -> bool:
        """Insert an item or replace its open entry with a better one.

        Returns
        -------
        bool
            True when the agenda changed.
        """
        current = self._current.get(item)

        if current is not None:
            _, current_weight = current

            # The outside estimate is a property of the item, so two
            # derivations of the same item differ only in inside cost.
            # if __debug__:
            #     assert (
            #         weight.out_estimate
            #         == current_weight.out_estimate
            #     ), (
            #         "The same item received different outside "
            #         "estimates."
            #     )

            if weight.inside >= current_weight.inside:
                return False

        sequence = next(self._counter)
        self._current[item] = sequence, weight

        f_score, inside_tiebreak = self._priority(weight)

        heapq.heappush(
            self._heap,
            (
                f_score,
                inside_tiebreak,
                sequence,
                item,
                weight,
            ),
        )

        # Optional protection against an excessive number of stale
        # entries.
        if (
            len(self._heap)
            > 4 * len(self._current) + 1024
        ):
            self._rebuild()

        return True

    def pop(self) -> tuple[T, WeightPointer]:
        """Remove and return the open item with lowest A* priority."""
        heap = self._heap
        current_entries = self._current

        while heap:
            (
                _,
                _,
                sequence,
                item,
                weight,
            ) = heapq.heappop(heap)

            current = current_entries.get(item)

            if current is None:
                # The item has already been popped.
                continue

            if current[0] != sequence:
                # A better entry was added after this one.
                continue

            del current_entries[item]
            return item, weight

        raise KeyError("Cannot pop from an empty agenda.")

    def peek(self) -> tuple[T, WeightPointer]:
        """Return the minimum item without removing it."""
        heap = self._heap
        current_entries = self._current

        while heap:
            (
                _,
                _,
                sequence,
                item,
                weight,
            ) = heap[0]

            current = current_entries.get(item)

            if current is not None and current[0] == sequence:
                return item, weight

            # Remove a stale minimum entry.
            heapq.heappop(heap)

        raise KeyError("Cannot peek into an empty agenda.")

    def _rebuild(self) -> None:
        """Discard all stale entries in linear time."""
        self._heap = [
            (
                weight.sum,
                -weight.inside,
                sequence,
                item,
                weight,
            )
            for item, (sequence, weight)
            in self._current.items()
        ]

        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        return bool(self._current)

    def __len__(self) -> int:
        return len(self._current)

    @property
    def is_empty(self) -> bool:
        return not self._current

    def __str__(self) -> str:
        entries = sorted(
            self._current.items(),
            key=lambda entry: self._priority(
                entry[1][1]
            ),
        )

        return ", ".join(
            f"{weight}:{item}"
            for item, (_, weight) in entries
        )

    def __repr__(self) -> str:
        return str(self)


def item_axiom(idx: int, l_args: int, r_args, aux: int) -> Item:
    return Item(
        idx, idx, idx, l_args, r_args, aux
    )


def supertag_to_item(
        idx: int, supertag: extraction.RelativeTag) -> Item | None:
    projective = extraction.process_relative_tag_to_projective(supertag)
    if projective is None:
        return None

    l_args, r_args, aux, _ = projective
    return item_axiom(
        idx, len(l_args), len(r_args), AUXMAP[aux])


def item_complete(item: Item) -> Item:
    assert item.is_uncomplete, "Can only complete uncomplete item."
    assert item.r_args == 0, "Can only complete item with 0 open right args."
    return Item.unchecked(
        item.start,  # type: ignore
        item.end,  # type: ignore
        item.anchor,  # type: ignore
        item.l_args,
        None,
        item.aux  # type: ignore
    )


def item_complete_unchecked(item: Item) -> Item:
    return Item.unchecked(
        item.start,  # type: ignore
        item.end,  # type: ignore
        item.anchor,  # type: ignore
        item.l_args,
        None,
        item.aux  # type: ignore
    )


def item_switch(item: Item) -> Item:
    assert item.is_unswitched, "Can only switch unswitched item."
    assert item.l_args == 0, "Can only switch item with 0 open left args."
    return Item.unchecked(
        item.start,  # type: ignore
        item.end,  # type: ignore
        item.anchor,  # type: ignore
        None,
        item.r_args,
        item.aux  # type: ignore
    )


def item_switch_unchecked(item: Item) -> Item:
    return Item.unchecked(
        item.start,  # type: ignore
        item.end,  # type: ignore
        item.anchor,  # type: ignore
        None,
        item.r_args,
        item.aux  # type: ignore
    )


def item_left_subst(main: Item, arg: Item) -> Item:
    assert main.is_unswitched, (
        "Can only perform left substitution at unswitched item.")
    assert arg.is_initial, "Substitute cannot be auxiliary."
    assert main.l_args is not None
    assert main.l_args > 0, (
        "There must be at least one open left "
        "argument available for left substitution")
    assert arg.end is not None
    assert main.start is not None
    assert arg.end+1 == main.start, (
        f"{arg} should end at {main.start-1} "
        f"to be substituted into {main}")
    assert arg.is_complete
    return Item.unchecked(
        arg.start,  # type: ignore
        main.end,  # type: ignore
        main.anchor,  # type: ignore
        main.l_args-1,
        main.r_args,
        main.aux  # type: ignore
    )


def item_left_subst_unchecked(main: Item, arg: Item) -> Item:
    return Item.unchecked(
        arg.start,  # type: ignore
        main.end,  # type: ignore
        main.anchor,  # type: ignore
        main.l_args-1,
        main.r_args,
        main.aux  # type: ignore
    )


def item_right_subst(main: Item, arg: Item) -> Item:
    assert main.is_switched, (
        "Can only perform right substitution at switched item.")
    assert arg.is_initial, "Substitute cannot be auxiliary."
    assert main.r_args is not None
    assert main.r_args > 0, (
        "There must be at least one open right "
        "argument available for left substitution")
    assert arg.start is not None
    assert main.end is not None
    assert main.end+1 == arg.start, (
        f"{arg} should start at {main.end+1} "
        f"to be substituted into {main}")
    assert arg.is_complete
    return Item.unchecked(
        main.start,  # type: ignore
        arg.end,  # type: ignore
        main.anchor,  # type: ignore
        main.l_args,
        main.r_args-1,
        main.aux  # type: ignore
    )


def item_right_subst_unchecked(main: Item, arg: Item) -> Item:
    return Item.unchecked(
        main.start,  # type: ignore
        arg.end,  # type: ignore
        main.anchor,  # type: ignore
        main.l_args,
        main.r_args-1,
        main.aux  # type: ignore
    )


def item_left_adjoin(main: Item, adj: Item) -> Item:
    assert main.is_unswitched, (
        "Can only perform left adjunction at unswitched item.")
    assert adj.is_left_adjunct, "Adjunction item must be left adjunct."
    assert adj.end is not None
    assert main.start is not None
    assert adj.end+1 == main.start, (
        f"{adj} should end at {main.start-1} "
        f"to be adjoined into {main}")
    assert adj.is_complete
    return Item.unchecked(
        adj.start,  # type: ignore
        main.end,  # type: ignore
        main.anchor,  # type: ignore
        main.l_args,
        main.r_args,
        main.aux  # type: ignore
    )


def item_left_adjoin_unchecked(main: Item, adj: Item) -> Item:
    return Item.unchecked(
        adj.start,  # type: ignore
        main.end,  # type: ignore
        main.anchor,  # type: ignore
        main.l_args,
        main.r_args,
        main.aux  # type: ignore
    )


def item_right_adjoin(main: Item, adj: Item) -> Item:
    assert main.is_switched, (
        "Can only perform right adjunction at switched item.")
    assert adj.is_right_adjunct, "Adjunction item must be right adjunct."
    assert adj.start is not None
    assert main.end is not None
    assert main.end+1 == adj.start, (
        f"{adj} should start at {main.end+1} "
        f"to be adjoined into {main}")
    assert adj.is_complete
    return Item.unchecked(
        main.start,  # type: ignore
        adj.end,  # type: ignore
        main.anchor,  # type: ignore
        main.l_args,
        main.r_args,
        main.aux  # type: ignore
    )


def item_right_adjoin_unchecked(main: Item, adj: Item) -> Item:
    return Item.unchecked(
        main.start,  # type: ignore
        adj.end,  # type: ignore
        main.anchor,  # type: ignore
        main.l_args,
        main.r_args,
        main.aux  # type: ignore
    )


AUXALL = (N, R, L)
AUXOPTLEFT = (N, L)
AUXOPTRIGHT = (N, R)
AUXNONLY = (N,)


def aux_check(left_space: int, right_space: int) -> tuple[Auxnum, ...]:
    if left_space > 0:
        if right_space > 0:
            return AUXALL
        return AUXOPTLEFT
    if right_space > 0:
        return AUXOPTRIGHT
    return AUXNONLY


class System():

    def __init__(
            self,
            head_scores: np.ndarray, supertag_scores: np.ndarray,
            id2sup: Mapping[int, extraction.RelativeTag],
            max_r: int, max_l: int, k_supertag: int = 10,
            k_head_scores: int = 10,
            estimate_type: Literal["simple", "advanced"] = "advanced",
            contains_root: bool = False,
            ) -> None:

        self._head_scores = head_scores
        self._supertag_scores = supertag_scores
        self._id2sup = id2sup
        self._k_supertag = min(k_supertag, self._supertag_scores.shape[-1])
        self._k_head_scores = min(k_head_scores, self._head_scores.shape[-1])

        kth_smallest: np.ndarray = -np.partition(
            -self._head_scores,
            self._k_head_scores-1, axis=-1)

        kth_smallest = kth_smallest[
                :, -self._k_head_scores]  # type: ignore

        self._unmasked_head_scores = self._head_scores.copy()

        self._head_scores = self._head_scores.copy()
        self._head_scores[
            self._head_scores > kth_smallest[:, np.newaxis]] = np.inf

        self._contains_root = contains_root

        self._head_scores[
            np.arange(
                (int(self._contains_root)), self._head_scores.shape[0]),
            np.arange(
                (int(self._contains_root)), self._head_scores.shape[1])
            ] = np.inf

        self._k_sup_inds: np.ndarray = np.argpartition(
            -self._supertag_scores, -self._k_supertag,
            axis=-1)[:, -self._k_supertag:]

        self._step: int = 0
        self._length: int = self._head_scores.shape[0]
        self._max_r: int = min(max_r, self._length-1)
        self._max_l: int = min(max_l, self._length-1)
        self._chart: Chart = Chart(self._length, self._max_l, self._max_r)
        self._agenda: Agenda = Agenda()

        self._estimate_type = estimate_type
        self.outside_estimates: np.ndarray
        if self._estimate_type == "simple":
            self.outside_estimates = self.compute_outside_estimates(
                self._supertag_scores, self._head_scores
            )
        else:
            self.outside_estimates = self.compute_advanced_outside_estimates(
                self._supertag_scores, self._head_scores, self._contains_root
            )

        self._fallback_complete_by_end: list[
            dict[int, tuple[Item, WeightPointer, float]]
        ] = [
            {} for _ in range(self._length + 1)
        ]

        # Best lexical axiom for each position.
        self._fallback_axiom: list[
            tuple[Item, WeightPointer, float] | None
        ] = [
            None for _ in range(self._length + 1)
        ]

        self._rules_complete_initial = (
            self.left_subst_dep_unchecked_bucket_,
            self.right_subst_dep_unchecked_bucket_,
        )

        self._rules_complete_left_aux = (
            self.right_adjoin_dep_unchecked_bucket_,
        )

        self._rules_complete_right_aux = (
            self.left_adjoin_dep_unchecked_bucket_,
        )

        self._rules_finish = (
            self.complete_unchecked_,
            self.right_adjoin_head_unchecked_bucket_,
        )

        self._rules_right_arguments = (
            self.right_subst_head_unchecked_bucket_,
            self.right_adjoin_head_unchecked_bucket_,
        )

        self._rules_switch = (
            self.switch_unchecked_,
            self.left_adjoin_head_unchecked_bucket_,
        )

        self._rules_left_arguments = (
            self.left_subst_head_unchecked_bucket_,
            self.left_adjoin_head_unchecked_bucket_,
        )

    @staticmethod
    def compute_outside_estimates(
            supertag_scores: np.ndarray,
            head_scores: np.ndarray
            ) -> np.ndarray:
        mins_sup = supertag_scores.min(axis=-1)
        mins_head = head_scores.min(axis=-1)

        cumsum = np.cumsum(mins_sup + mins_head)

        span_sums = (cumsum[:, np.newaxis] - np.concatenate(
            (np.array([0]), cumsum[:-1]))[np.newaxis, :]).T
        return cumsum[-1] - span_sums

    @staticmethod
    @njit(cache=True, parallel=True)
    def _advanced_outside_kernel(
            min_supertag: np.ndarray,
            head_scores_t: np.ndarray,
            prefix_head_min_t: np.ndarray,
            suffix_head_min_t: np.ndarray,
            contains_root: bool
            ) -> np.ndarray:
        """Compute h(i, j, c).

        Transposed inputs have the following indexing:

            head_scores_t[c, k]       = w(k, c)
            prefix_head_min_t[i, k]   = min_{m < i} w(k, m)
            suffix_head_min_t[j+1, k] = min_{m > j} w(k, m)

        The second axis k is contiguous in memory.
        """
        n = head_scores_t.shape[0]
        dtype = head_scores_t.dtype

        estimates = np.full((n, n, n), np.inf, dtype=dtype)

        supertag_prefix = np.empty(n + 1, dtype=dtype)
        supertag_prefix[0] = 0.0

        for k in range(n):
            supertag_prefix[k + 1] = (
                supertag_prefix[k] + min_supertag[k]
            )

        total_supertag = supertag_prefix[n]

        # Different i values write to disjoint parts of estimates, so this
        # loop can safely be parallelized.
        for i in prange(n):  # type: ignore
            prefix_for_i = prefix_head_min_t[i]

            for j in range(i, n):
                suffix_for_j = suffix_head_min_t[j + 1]

                outside_supertag = (
                    supertag_prefix[i]
                    + total_supertag
                    - supertag_prefix[j + 1]
                )

                for c in range(i, j + 1):
                    heads_for_c = head_scores_t[c]
                    outside_head = 0.0

                    # Dependents left of the span.
                    for k in range(int(contains_root), i):
                        best = prefix_for_i[k]

                        candidate = suffix_for_j[k]
                        if candidate < best:
                            best = candidate

                        candidate = heads_for_c[k]
                        if candidate < best:
                            best = candidate

                        outside_head += best

                    # Dependents right of the span.
                    for k in range(j + 1, n):
                        best = prefix_for_i[k]

                        candidate = suffix_for_j[k]
                        if candidate < best:
                            best = candidate

                        candidate = heads_for_c[k]
                        if candidate < best:
                            best = candidate

                        outside_head += best

                    # Include anchor head estimate
                    if contains_root and c == 0:
                        anchor_head = 0.0

                    else:
                        anchor_head = prefix_head_min_t[i, c]

                        candidate = suffix_head_min_t[j + 1, c]
                        if candidate < anchor_head:
                            anchor_head = candidate

                    estimates[i, j, c] = (
                        outside_supertag
                        + outside_head
                        + anchor_head
                    )

        return estimates

    @staticmethod
    def compute_advanced_outside_estimates(
            supertag_scores: np.ndarray,
            head_scores: np.ndarray,
            contains_root: bool,
            ) -> np.ndarray:
        """Compute anchor-sensitive outside estimates.

        Parameters
        ----------
        supertag_scores:
            Shape [n, num_supertags].

        head_scores:
            Shape [n, n], with head_scores[k, m] = w(k, m).

        Returns
        -------
        np.ndarray
            Shape [n, n, n]. Valid entries satisfy i <= c <= j.
            All other entries are infinity.
        """
        if supertag_scores.ndim != 2:
            raise ValueError(
                "supertag_scores must have shape [n, num_supertags]."
            )

        n = supertag_scores.shape[0]

        if head_scores.shape != (n, n):
            raise ValueError(
                f"head_scores must have shape {(n, n)}, "
                f"not {head_scores.shape}."
            )

        # Do not automatically force float64. Keeping float32 can provide a
        # substantial speed improvement when its precision is sufficient.
        dtype = np.result_type(
            supertag_scores.dtype,
            head_scores.dtype,
            np.float32,
        )

        supertag_scores = np.ascontiguousarray(
            supertag_scores,
            dtype=dtype,
        )
        head_scores = np.ascontiguousarray(
            head_scores,
            dtype=dtype,
        )

        min_supertag = supertag_scores.min(axis=-1)

        # prefix_head_min[k, i] = min_{m < i} omega(k, m)
        prefix_head_min = np.full(
            (n, n + 1),
            np.inf,
            dtype=dtype,
        )
        prefix_head_min[:, 1:] = np.minimum.accumulate(
            head_scores,
            axis=1,
        )

        # suffix_head_min[k, j+1] = min_{m > j} omega(k, m)
        suffix_head_min = np.full(
            (n, n + 1),
            np.inf,
            dtype=dtype,
        )
        suffix_head_min[:, :n] = np.minimum.accumulate(
            head_scores[:, ::-1],
            axis=1,
        )[:, ::-1]

        # The innermost kernel loop runs over dependent k. Transposing and
        # copying makes k the contiguous dimension in every input.
        head_scores_t = np.ascontiguousarray(head_scores.T)
        prefix_head_min_t = np.ascontiguousarray(prefix_head_min.T)
        suffix_head_min_t = np.ascontiguousarray(suffix_head_min.T)

        return System._advanced_outside_kernel(
            min_supertag,
            head_scores_t,
            prefix_head_min_t,
            suffix_head_min_t,
            contains_root,
        )

    def get_outside_estimate(self, item: Item) -> float:
        assert not item.is_axiom
        assert item.start is not None and item.end is not None
        if self._estimate_type == "simple":
            return self.outside_estimates[
                item.start-1, item.end-1].item()
        else:
            return self.outside_estimates[
                item.start-1, item.end-1, item.anchor-1].item()  # type: ignore

    def get_attachment_weight(self, head: Item, dependent: Item) -> float:
        assert not head.is_axiom and not dependent.is_axiom
        assert dependent.anchor is not None
        assert head.anchor is not None
        return self._head_scores[dependent.anchor-1, head.anchor-1].item()

    def get_item_weight_pointer_pair(
            self,
            new_item: Item,
            back1: Item, weight1: Weight,
            back2: Item = AXIOM, weight2: Weight | None = None,
            supertag_ind: int = 0
            ) -> tuple[Item, WeightPointer]:
        weight: float
        if back2.is_axiom:
            weight = weight1.inside
        else:
            assert weight2 is not None
            weight = weight1.inside+weight2.inside+self.get_attachment_weight(
                back1, back2)

        return new_item, WeightPointer(
            weight,
            self.get_outside_estimate(new_item),
            back1, back2, supertag_ind)

    def add_if_finite_(
            self, item: Item, head: Item, head_weight: Weight,
            dep: Item = AXIOM, dep_weight: Weight | None = None,
            supertag_ind: int = 0) -> WeightPointer:

        result_item, result_weight = (
            self.get_item_weight_pointer_pair(
                item, head, head_weight, dep, dep_weight, supertag_ind))

        if result_weight.sum < inf:
            closed_weight = self._chart.peek(result_item)
            if closed_weight is None:
                self._agenda.add_update(
                    result_item, result_weight)
        return result_weight

    def complete_(
            self,
            item: Item, weight: Weight
            ) -> None:
        if not item.is_switched:
            return
        if not item.r_args == 0:
            return
        self.complete_unchecked_(item, weight)

    def complete_unchecked_(
            self,
            item: Item, weight: Weight
            ) -> None:
        self.add_if_finite_(
            item_complete_unchecked(item),
            item, weight)

    def switch_(
            self,
            item: Item, weight: Weight
            ) -> None:
        if not item.l_args == 0:
            return
        self.switch_unchecked_(item, weight)

    def switch_unchecked_(
            self,
            item: Item, weight: Weight
            ) -> None:
        self.add_if_finite_(
            item_switch_unchecked(item),
            item, weight)

    def left_subst_head_(
            self,
            item: Item, weight: Weight
            ) -> None:

        if not item.l_args > 0:  # type: ignore
            return

        self.left_subst_head_unchecked_(item, weight)

    def left_subst_head_unchecked_(
            self,
            item: Item, weight: Weight
            ) -> None:

        for i in range(
                item.l_args+(1 if item.is_right_adjunct else 0),
                item.start):  # type: ignore
            for c in range(i, item.start):  # type: ignore
                dep = Item.unchecked(
                    i, item.start-1, c, None, None, N)  # type: ignore
                other_weight = self._chart.peek(dep)
                if other_weight is not None:
                    self.add_if_finite_(
                        item_left_subst_unchecked(item, dep),
                        item, weight, dep, other_weight)

    def left_subst_head_unchecked_bucket_(
            self,
            item: Item, weight: Weight
            ) -> None:
        start = item.start
        l_args = item.l_args
        minimum_start = (
            l_args
            + (1 if item.aux == L else 0)
        )
        candidates = self._chart._complete_by_end_[
            start-1][N+1][minimum_start:]
        add = self.add_if_finite_

        for li in candidates:
            for dep, other_weight in li:
                # if dep.start < minimum_start:
                #     continue

                add(
                    item_left_subst_unchecked(item, dep),
                    item,
                    weight,
                    dep,
                    other_weight,
                )

    def left_subst_dep_(
            self,
            item: Item, weight: Weight
            ) -> None:
        if not item.is_complete:
            return

        if not item.is_initial:
            return

        self.left_subst_dep_unchecked_(item, weight)

    def left_subst_dep_unchecked_(
            self,
            item: Item, weight: Weight
            ) -> None:

        possible_l_args = min(self._max_l, item.start)  # type: ignore

        for i in range(item.end+1, self._length+1):  # type: ignore
            possible_r_args = min(self._max_r, self._length-i)

            for l in range(1, possible_l_args+1):  # noqa: E741

                for r in range(0, possible_r_args+1):
                    possible_aux = aux_check(
                        possible_l_args - l, possible_r_args - r)
                    for c in range(item.end+1, i+1):  # type: ignore
                        for aux in possible_aux:
                            head = Item.unchecked(
                                item.end+1, i, c, l, r, aux)
                            other_weight = self._chart.peek(head)
                            if other_weight is not None:
                                self.add_if_finite_(
                                    item_left_subst_unchecked(head, item),
                                    head, other_weight, item, weight)

    def left_subst_dep_unchecked_bucket_(
            self,
            item: Item,
            weight: Weight,
            ) -> None:
        """Use the completed initial `item` as a left dependent."""
        item_start = item.start
        item_end = item.end

        maximum_l_args = min(self._max_l, item_start)
        candidates = self._chart._unswitched_by_start_[
            item_end + 1][2:maximum_l_args+2]

        length = self._length
        max_r = self._max_r
        add = self.add_if_finite_

        for li in candidates:
            for head, other_weight in li:
                l_args = head.l_args
                r_args = head.r_args
                head_end = head.end
                aux = head.aux

                # if l_args < 1 or l_args > maximum_l_args:
                #     continue

                maximum_r_args = min(max_r, length - head_end)

                if r_args > maximum_r_args:
                    continue

                if aux == L:
                    if l_args >= maximum_l_args:
                        continue
                elif aux == R:
                    if r_args >= maximum_r_args:
                        continue

                add(
                    item_left_subst_unchecked(head, item),
                    head,
                    other_weight,
                    item,
                    weight,
                )

    def right_subst_head_(
                self,
                item: Item, weight: Weight
                ) -> None:
        if not item.is_switched:
            return

        if not item.r_args > 0:
            return

        self.right_subst_head_unchecked_(item, weight)

    def right_subst_head_unchecked_(
                self,
                item: Item, weight: Weight
                ) -> None:

        for i in range(
                item.end+1, self._length-item.r_args-(  # type: ignore
                    1 if item.is_left_adjunct else 0)+2):
            for c in range(item.end+1, i+1):  # type: ignore
                dep = Item.unchecked(
                    item.end+1, i, c, None, None, N)  # type: ignore
                other_weight = self._chart.peek(dep)
                if other_weight is not None:
                    self.add_if_finite_(
                        item_right_subst_unchecked(item, dep),
                        item, weight, dep, other_weight)

    def right_subst_head_unchecked_bucket_(
            self,
            item: Item,
            weight: Weight,
            ) -> None:
        """Substitute a completed initial item to the right of `item`."""
        end = item.end
        r_args = item.r_args

        dependent_start = end + 1

        maximum_dep_end = (
            self._length
            - r_args
            - (item.aux == R)
            + 1
        )

        candidates = self._chart._complete_by_start_[
            dependent_start][N+1][:maximum_dep_end+1]
        add = self.add_if_finite_

        for li in candidates:
            for dep, other_weight in li:
                # if dep.end > maximum_dep_end:  # type: ignore[operator]
                #     continue

                add(
                    item_right_subst_unchecked(item, dep),
                    item,
                    weight,
                    dep,
                    other_weight,
                )

    def right_subst_dep_(
            self,
            item: Item, weight: Weight
            ) -> None:
        if not item.is_complete:
            return
        if not item.is_initial:
            return

        self.right_subst_dep_unchecked_(item, weight)

    def right_subst_dep_unchecked_(
            self,
            item: Item, weight: Weight
            ) -> None:

        for i in range(1, item.start):

            possible_r_args = min(
                self._max_r, self._length-item.end+1)  # type: ignore
            for r in range(1, possible_r_args+1):

                possible_aux = aux_check(
                    i-1, possible_r_args - r)
                for c in range(i, item.start):  # type: ignore
                    for aux in possible_aux:
                        head = Item.unchecked(
                            i, item.start-1,
                            c, None, r, aux)  # type: ignore
                        other_weight = self._chart.peek(head)
                        if other_weight is not None:
                            self.add_if_finite_(
                                item_right_subst_unchecked(head, item),
                                head, other_weight, item, weight)

    def right_subst_dep_unchecked_bucket_(
            self,
            item: Item,
            weight: Weight,
            ) -> None:
        """Use the completed initial `item` as a right dependent."""
        item_start = item.start
        item_end = item.end

        maximum_r_args = min(
            self._max_r,
            self._length - item_end + 1,
        )

        candidates = self._chart._switched_by_end_[item_start - 1][
            2:maximum_r_args+2]
        add = self.add_if_finite_

        for li in candidates:
            for head, other_weight in li:
                head_start = head.start
                r_args = head.r_args
                aux = head.aux

                # if r_args < 1 or r_args > maximum_r_args:
                #     continue

                if aux == L:
                    if head_start <= 1:
                        continue
                elif aux == R:
                    if r_args >= maximum_r_args:
                        continue

                add(
                    item_right_subst_unchecked(head, item),
                    head,
                    other_weight,
                    item,
                    weight,
                )

    def left_adjoin_head_(
            self,
            item: Item, weight: Weight
            ) -> None:
        if not item.is_unswitched:
            return

        self.left_adjoin_head_unchecked_(item, weight)

    def left_adjoin_head_unchecked_(
            self,
            item: Item, weight: Weight
            ) -> None:

        for i in range(
                item.l_args+1+(1 if item.is_right_adjunct else 0),
                item.start):  # type: ignore
            for c in range(i, item.start):
                dep = Item.unchecked(
                    i, item.start-1, c, None, None, R)  # type: ignore
                other_weight = self._chart.peek(dep)
                if other_weight is not None:
                    self.add_if_finite_(
                        item_left_adjoin_unchecked(item, dep),
                        item, weight, dep, other_weight)

    def left_adjoin_head_unchecked_bucket_(
            self,
            item: Item,
            weight: Weight,
            ) -> None:
        """Adjoin a completed left-adjunct item to the left of `item`."""
        start = item.start
        l_args = item.l_args

        minimum_dep_start = (
            l_args
            + 1
            + (item.aux == L)
        )

        candidates = self._chart._complete_by_end_[
            start - 1][R+1][minimum_dep_start:]
        add = self.add_if_finite_

        for li in candidates:
            for dep, other_weight in li:
                # if dep.start < minimum_dep_start:  # type: ignore[operator]
                #     continue

                add(
                    item_left_adjoin_unchecked(item, dep),
                    item,
                    weight,
                    dep,
                    other_weight,
                )

    def left_adjoin_dep_(
                self,
                item: Item, weight: Weight
                ) -> None:
        if not item.is_complete:
            return
        if not item.is_left_adjunct:
            return

        self.left_adjoin_dep_unchecked_(item, weight)

    def left_adjoin_dep_unchecked_(
                self,
                item: Item, weight: Weight
                ) -> None:

        for i in range(item.end+1, self._length+1):  # type: ignore
            possible_l_args = min(self._max_l, item.start-1)  # type: ignore
            for l in range(0, possible_l_args+1):  # noqa: E741
                possible_r_args = min(self._max_r, self._length-i)
                for r in range(0, possible_r_args+1):
                    possible_aux = aux_check(
                        possible_l_args - l, possible_r_args - r)

                    for c in range(item.end+1, i+1):
                        for aux in possible_aux:
                            head = Item.unchecked(
                                item.end+1, i, c, l, r, aux)  # type: ignore
                            other_weight = self._chart.peek(head)
                            if other_weight is not None:
                                self.add_if_finite_(
                                    item_left_adjoin_unchecked(head, item),
                                    head, other_weight, item, weight)

    def left_adjoin_dep_unchecked_bucket_(
            self,
            item: Item,
            weight: Weight,
            ) -> None:
        """Use completed left adjunct `item` as a left dependent."""
        item_start = item.start
        item_end = item.end

        maximum_l_args = min(
            self._max_l,
            item_start - 1,
        )

        candidates = self._chart._unswitched_by_start_[item_end + 1][
            :maximum_l_args+2]

        length = self._length
        max_r = self._max_r
        add = self.add_if_finite_

        for li in candidates:
            for head, other_weight in li:
                l_args = head.l_args
                r_args = head.r_args
                head_end = head.end
                aux = head.aux

                # Original:
                #
                # for l in range(0, maximum_l_args + 1)
                # if l_args > maximum_l_args:
                #     continue

                maximum_r_args = min(max_r, length - head_end)

                # Original:
                #
                # for r in range(0, maximum_r_args + 1)
                if r_args > maximum_r_args:
                    continue

                if aux == L:
                    if l_args >= maximum_l_args:
                        continue
                elif aux == R:
                    if r_args >= maximum_r_args:
                        continue

                add(
                    item_left_adjoin_unchecked(head, item),
                    head,
                    other_weight,
                    item,
                    weight,
                )

    def right_adjoin_head_(
                self,
                item: Item, weight: Weight
                ) -> None:
        if not item.is_switched:
            return

        self.right_adjoin_head_unchecked_(item, weight)

    def right_adjoin_head_unchecked_(
                self,
                item: Item, weight: Weight
                ) -> None:

        for i in range(
                item.end+1, self._length-item.r_args-(  # type: ignore
                    1 if item.is_left_adjunct else 0)+1):
            for c in range(item.end+1, i+1):  # type: ignore
                dep = Item.unchecked(
                    item.end+1, i, c, None, None, L)
                other_weight = self._chart.peek(dep)
                if other_weight is not None:
                    self.add_if_finite_(
                        item_right_adjoin_unchecked(item, dep),
                        item, weight, dep, other_weight)

    def right_adjoin_head_unchecked_bucket_(
            self,
            item: Item,
            weight: Weight,
            ) -> None:
        """Adjoin a completed right-adjunct item to the right of `item`."""
        end = item.end
        r_args = item.r_args

        dependent_start = end + 1

        maximum_dep_end = (
            self._length
            - r_args
            - (item.aux == R)
        )

        candidates = self._chart._complete_by_start_[
            dependent_start][L+1][:maximum_dep_end+1]
        add = self.add_if_finite_

        for li in candidates:
            for dep, other_weight in li:
                # if dep.end > maximum_dep_end:  # type: ignore[operator]
                #     continue

                add(
                    item_right_adjoin_unchecked(item, dep),
                    item,
                    weight,
                    dep,
                    other_weight,
                )

    def right_adjoin_dep_(
                self,
                item: Item, weight: Weight
                ) -> None:
        if not item.is_complete:
            return
        if not item.is_right_adjunct:
            return

        self.right_adjoin_dep_unchecked_(item, weight)

    def right_adjoin_dep_unchecked_(
                self,
                item: Item, weight: Weight
                ) -> None:

        for i in range(1, item.start):  # type: ignore
            possible_r_args = min(
                self._max_r, self._length-item.end)  # type: ignore
            for r in range(0, possible_r_args+1):
                possible_aux = aux_check(
                    i - 1, possible_r_args - r)

                for c in range(i, item.start):  # type: ignore
                    for aux in possible_aux:
                        head = Item.unchecked(
                            i, item.start-1, c, None, r, aux)  # type: ignore
                        other_weight = self._chart.peek(head)
                        if other_weight is not None:
                            self.add_if_finite_(
                                item_right_adjoin_unchecked(head, item),
                                head, other_weight, item, weight)

    def right_adjoin_dep_unchecked_bucket_(
            self,
            item: Item,
            weight: Weight,
            ) -> None:
        """Use completed right adjunct `item` as a right dependent."""
        item_start = item.start
        item_end = item.end

        maximum_r_args = min(
            self._max_r,
            self._length - item_end,
        )

        candidates = self._chart._switched_by_end_[item_start - 1][
            :maximum_r_args+2]
        add = self.add_if_finite_

        for i in range(0, maximum_r_args+1):
            for head, other_weight in candidates[i]:
                head_start = head.start
                r_args = i-1  # head.r_args
                aux = head.aux

                # if r_args > maximum_r_args:
                #     continue

                if aux == L:
                    if head_start <= 1:
                        continue
                elif aux == R:
                    if r_args == maximum_r_args:
                        continue

                add(
                    item_right_adjoin_unchecked(head, item),
                    head,
                    other_weight,
                    item,
                    weight,
                )

    def axiom(self) -> None:
        for i in range(1, self._length+1):
            for sup_id in self._k_sup_inds[i-1]:
                if sup_id.item() not in self._id2sup:
                    continue
                sup = self._id2sup[sup_id.item()]
                l_args = 0
                r_args = 0
                check_left = True
                for typ, _ in sup:
                    if typ is None:
                        check_left = False
                    else:
                        if check_left:
                            l_args += 1
                        else:
                            r_args += 1

                # left, right = sup.split("*")
                # l_args = left.count("+") + left.count("-")
                # r_args = right.count("+") + right.count("-")
                # print(l_args, r_args)

                if l_args < i and r_args <= self._length - i:
                    projective = supertag_to_item(i, self._id2sup[sup_id])
                    if projective is None:
                        continue

                    weight = self.add_if_finite_(
                        projective,
                        AXIOM, Weight(
                            self._supertag_scores[i-1, sup_id], 0),
                        supertag_ind=sup_id
                        )
                    self._record_fallback_axiom(projective, weight)

    def run(
            self,
            printinfo: bool = False
            ) -> tuple[Item, WeightPointer] | None:

        rules_complete_initial = self._rules_complete_initial
        rules_complete_left_aux = self._rules_complete_left_aux
        rules_complete_right_aux = self._rules_complete_right_aux
        rules_finish = self._rules_finish
        rules_right_arguments = self._rules_right_arguments
        rules_switch = self._rules_switch
        rules_left_arguments = self._rules_left_arguments

        agenda = self._agenda
        chart = self._chart

        # add_update = agenda.add_update
        pop = agenda.pop

        goal: tuple[Item, WeightPointer] | None = None

        # longest_item: Item = AXIOM
        # longest_weight_pointer: WeightPointer = WeightPointer(
        #     0, 0, AXIOM, AXIOM, 0)

        # covered = set()
        # for new, weight in
        self.axiom()
        # if weight.sum < inf:
        #     self._record_fallback_axiom(new, weight)
        #     add_update(new, weight)
        #     covered.add(new.start)
        # else:
        #     print(new, weight)  # TODO: [1, 1, 1, 0, 1, >] is generated.
        #                           Why?
        # print(
        #     f"Does not provide axioms for {self._length - len(covered)} "
        #     "tokens.")

        if printinfo:
            self.print()

        rules: Collection[Callable[[Item, Weight], None]]

        while not agenda.is_empty:
            self._step += 1
            item, weight_pointer = pop()
            chart[item] = weight_pointer
            self._record_fallback_complete(item, weight_pointer)

            # if item.is_complete:
            #     if len(item) > len(longest_item):
            #         longest_item = item
            #         longest_weight_pointer = weight_pointer

            if item.is_goal(self._length):
                goal = item, weight_pointer
                if printinfo:
                    self.print(item, weight_pointer, goal=True)
                break

            l_args = item.l_args
            r_args = item.r_args

            if r_args is None:
                aux = item.aux

                if aux == N:
                    rules = rules_complete_initial
                elif aux == R:
                    rules = rules_complete_right_aux
                else:
                    rules = rules_complete_left_aux
                # Must be L since item can never be AXIOM

            elif l_args is None:
                if r_args == 0:
                    rules = rules_finish
                else:
                    rules = rules_right_arguments

            elif l_args == 0:
                rules = rules_switch

            else:
                rules = rules_left_arguments

            for rule in rules:
                rule(item, weight_pointer)

            if printinfo:
                self.print(item)

            # Upper time limit
            # if self._step > self._length*10:
            #     break
            # if self._step >= 200:
            #     break

        # print(
        #     f"{self._step} steps total, goal found? {goal is not None}."
        #     f" Longest item: {len(longest_item)}/{self._length}")
        # if allow_incomplete and goal is None:
        #     goal = longest_item, longest_weight_pointer
        return goal

    # def _backtrack(
    #         self, weight_pointer: WeightPointer) -> tuple[
    #             list[int], list[int], list[int], list[int], list[int],
    #             list[str], list[str], list[str], list[str], str | None]:

    #     if weight_pointer.back1.is_axiom:
    #         projective_tag = extraction.process_relative_tag_to_projective(
    #             self._id2sup[weight_pointer.supertag_ind]
    #         )
    #         assert projective_tag is not None

    #         l_args, r_args, _, auxdep = projective_tag

    #         return (
    #             [], [], [], [], [weight_pointer.supertag_ind],
    #             [], [], l_args, r_args, auxdep)

    #     if weight_pointer.back2.is_axiom:
    #         return self._backtrack(
    #             self._chart[weight_pointer.back1])

    #     (
    #         h_l_heads, h_r_heads, h_l_lab, h_r_lab, h_supertag_inds,
    #         h_l_dep, h_r_dep, h_l_mdep,
    # h_r_mdep, h_auxdep) = self._backtrack(
    #         self._chart[weight_pointer.back1]
    #     )
    #     (
    #         d_l_heads, d_r_heads, d_l_lab, d_r_lab, d_supertag_inds,
    #         d_l_dep, d_r_dep, _, _, d_auxdep) = self._backtrack(
    #         self._chart[weight_pointer.back2]
    #     )

    #     assert weight_pointer.back1.anchor is not None
    #     if weight_pointer.back2.is_adjunct:
    #         assert weight_pointer.back1.start is not None
    #         assert weight_pointer.back2.start is not None
    #         assert d_auxdep is not None

    #         if weight_pointer.back1.start < weight_pointer.back2.start:
    #             return (
    #                 h_l_heads,
    #                 h_r_heads + d_l_heads + [
    #                     weight_pointer.back1.anchor] + d_r_heads,
    #                 h_l_lab, h_r_lab + d_l_lab + [1] + d_r_lab,
    #                 h_supertag_inds + d_supertag_inds,
    #                 h_l_dep, h_r_dep + d_l_dep + [d_auxdep] + d_r_dep,
    #                 h_l_mdep, h_r_mdep, h_auxdep)
    #         else:
    #             return (
    #                 d_l_heads + [
    #                     weight_pointer.back1.anchor] + d_r_heads + h_l_heads,
    #                 h_r_heads,
    #                 d_l_lab + [1] + d_r_lab + h_l_lab, h_r_lab,
    #                 d_supertag_inds + h_supertag_inds,
    #                 d_l_dep + [d_auxdep] + d_r_dep + h_l_dep, h_r_dep,
    #                 h_l_mdep, h_r_mdep, h_auxdep)

    #     if weight_pointer.back1.is_switched:

    #         return (
    #             h_l_heads,
    #             h_r_heads + d_l_heads + [
    #                 weight_pointer.back1.anchor] + d_r_heads,
    #             h_l_lab, h_r_lab + d_l_lab + [0] + d_r_lab,
    #             h_supertag_inds + d_supertag_inds,
    #             h_l_dep, h_r_dep + d_l_dep + [h_r_mdep[0]] + d_r_dep,
    #             h_l_mdep, (h_r_mdep[1:] if len(h_r_mdep) > 1 else []),
    #             h_auxdep)
    #     else:
    #         return (
    #             d_l_heads + [
    #                 weight_pointer.back1.anchor] + d_r_heads + h_l_heads,
    #             h_r_heads,
    #             d_l_lab + [0] + d_r_lab + h_l_lab, h_r_lab,
    #             d_supertag_inds + h_supertag_inds,
    #             d_l_dep + [h_l_mdep[-1]] + d_r_dep + h_l_dep, h_r_dep,
    #             h_l_mdep[:-1], h_r_mdep, h_auxdep)

    # def backtrack(
    #         self, weight_pointer: WeightPointer, pad: bool = False
    #         ) -> tuple[
    #             list[int], list[int], list[int], list[str]]:
    #     """returns a list of head indices (0 is root, 1 the first token),
    #     a list of binary scores
    #     (0 if the position was substituted into its head
    #     and 1 if it was adjoined), a list of supertag indices and
    #     a list of (simplified) dependency relations."""

    #     # TODO: we can get a lot better if also taking the top length
    #     # items for the remaining spans

    #     if not weight_pointer.back1.is_axiom:
    #         (
    #             l_heads, r_heads, l_lab, r_lab, supertag_inds,
    #             l_dep, r_dep, _, _, _) = self._backtrack(weight_pointer)

    #         heads = l_heads + [0] + r_heads
    #         lab = l_lab + [0] + r_lab
    #         supertags = supertag_inds
    #         dep = l_dep + ["root"] + r_dep

    #     else:
    #         heads = []
    #         lab = []
    #         supertags = []
    #         dep = []

    #     if pad:
    #         if not weight_pointer.back1.is_axiom:
    #             assert weight_pointer.back1.start is not None
    #             assert weight_pointer.back1.end is not None
    #             start = int(min(
    #                 weight_pointer.back1.start,
    #                 (
    #                     inf if weight_pointer.back2.start is None
    #                     else weight_pointer.back2.start)))
    #             end = int(max(
    #                 weight_pointer.back1.end,
    #                 (
    #                     -inf if weight_pointer.back2.end is None
    #                     else weight_pointer.back2.end)))
    #         else:
    #             start = 1
    #             end = 0

    #         l_pad = start-1
    #         r_pad = self._length-end

    #         heads = [0]*l_pad + heads + [0]*r_pad
    #         lab = [0]*l_pad + lab + [0]*r_pad
    #         supertags = [0]*l_pad + supertags + [0]*r_pad
    #         dep = ["root"]*l_pad + dep + ["root"]*r_pad

    #     return heads, lab, supertags, dep

    def _backtrack_into(
            self,
            item: Item,
            weight_pointer: WeightPointer,
            heads: list[int],
            labels: list[int],
            supertag_inds: list[int],
            dependencies: list[str],
            ) -> tuple[
                deque[str],
                deque[str],
                str | None,
            ]:
        """Fill the output arrays while traversing the derivation.

        Returns the unconsumed left and right dependency labels of the
        subtree anchor, plus its auxiliary dependency label.
        """
        back1 = weight_pointer.back1
        back2 = weight_pointer.back2

        # Lexical/supertag axiom.
        if back1.is_axiom:
            anchor = item.anchor
            assert anchor is not None

            supertag_index = weight_pointer.supertag_ind

            projective_tag = (
                extraction.process_relative_tag_to_projective(
                    self._id2sup[supertag_index]
                )
            )
            assert projective_tag is not None

            left_arguments, right_arguments, _, auxiliary_dependency = (
                projective_tag
            )

            supertag_inds[anchor - 1] = supertag_index

            # deque permits O(1) consumption from either end.
            return (
                deque(left_arguments),
                deque(right_arguments),
                auxiliary_dependency,
            )

        # Unary deduction.
        if back2.is_axiom:
            return self._backtrack_into(
                back1,
                self._chart[back1],
                heads,
                labels,
                supertag_inds,
                dependencies,
            )

        # Binary deduction: recursively process the head and dependent
        # subderivations.
        (
            head_left_arguments,
            head_right_arguments,
            head_auxiliary_dependency,
        ) = self._backtrack_into(
            back1,
            self._chart[back1],
            heads,
            labels,
            supertag_inds,
            dependencies,
        )

        (
            _dependent_left_arguments,
            _dependent_right_arguments,
            dependent_auxiliary_dependency,
        ) = self._backtrack_into(
            back2,
            self._chart[back2],
            heads,
            labels,
            supertag_inds,
            dependencies,
        )

        head_anchor = back1.anchor
        dependent_anchor = back2.anchor

        assert head_anchor is not None
        assert dependent_anchor is not None

        dependent_index = dependent_anchor - 1
        heads[dependent_index] = head_anchor

        if back2.is_adjunct:
            assert dependent_auxiliary_dependency is not None

            labels[dependent_index] = 1
            dependencies[dependent_index] = (
                dependent_auxiliary_dependency
            )

        elif back1.is_switched:
            assert head_right_arguments

            labels[dependent_index] = 0
            dependencies[dependent_index] = (
                head_right_arguments.popleft()
            )

        else:
            assert head_left_arguments

            labels[dependent_index] = 0
            dependencies[dependent_index] = (
                head_left_arguments.pop()
            )

        return (
            head_left_arguments,
            head_right_arguments,
            head_auxiliary_dependency,
        )

    def backtrack(
            self,
            goal_item: Item,
            weight_pointer: WeightPointer,
            ) -> tuple[
                list[int],
                list[int],
                list[int],
                list[str],
            ]:
        """Recover heads, arc labels, supertags and dependency relations."""
        length = self._length

        # Sentinel values make incomplete reconstruction detectable.
        heads = [-1] * length
        labels = [-1] * length
        supertag_inds = [-1] * length
        dependencies = [""] * length

        self._backtrack_into(
            goal_item,
            weight_pointer,
            heads,
            labels,
            supertag_inds,
            dependencies,
        )

        root_anchor = goal_item.anchor
        assert root_anchor is not None

        root_index = root_anchor - 1
        heads[root_index] = 0
        labels[root_index] = 0
        dependencies[root_index] = "root"

        assert all(head >= 0 for head in heads)
        assert all(label >= 0 for label in labels)
        assert all(index >= 0 for index in supertag_inds)
        assert all(dependency for dependency in dependencies)

        # `pad` is unused in the original implementation as well. Remove it
        # unless padding should be implemented here.
        return heads, labels, supertag_inds, dependencies

    # def _backtrack_disconnected(self, start: int, end: int) -> tuple[
    #         list[int], list[int], list[int], list[str]]:
    #     # TODO: This recursive definition is not efficient.
    #     # Replace this with a chart-based approach:
    #     # a cell either contains a reference to a System chart item
    #     # or a reference to two other chart items
    #     # it also keeps track of the maximum length System chart subsequence
    #     # in its span. This is how the split points are compared
    #     # Is there a better approach?

    #     min_item: Item = AXIOM
    #     min_weight_pointer: WeightPointer = WeightPointer(
    #         inf, inf, AXIOM, AXIOM, 0
    #     )

    #     for l in list(range(1, end-start+2))[::-1]:  # noqa: E741
    #         if not min_item.is_axiom:
    #             break
    #         for i in range(start, end-l+2):

    #             for c in range(i, i+l):
    #                 item = Item(i, i+l-1, c, None, None, N)
    #                 weight = self._chart.peek(item)
    #                 if weight is not None:
    #                     if weight < min_weight_pointer:
    #                         min_item = item
    #                         min_weight_pointer = weight

    #     if min_item.is_axiom:
    #         length = end-start+1
    #         return [0]*length, [0]*length, [0]*length, ["root"]*length

    #     else:
    #         assert min_item.start is not None
    #         assert min_item.end is not None
    #         l_head, l_lab, l_tag, l_dep = self._backtrack_disconnected(
    #             start, min_item.start-1
    #         )
    #         r_head, r_lab, r_tag, r_dep = self._backtrack_disconnected(
    #             min_item.end+1, end
    #         )
    #         head, lab, tag, dep = self.backtrack(
    #             min_item, min_weight_pointer)

    #         return (
    #             l_head + head + r_head,
    #             l_lab + lab + r_lab,
    #             l_tag + tag + r_tag,
    #             l_dep + dep + r_dep
    #         )

    # def backtrack_disconnected(self) -> tuple[
    #         list[int], list[int], list[int], list[str]]:
    #     output = self._backtrack_disconnected(1, self._length)
    #     return output

    def print(
            self, popped_item: Item | None = None,
            weight: Weight | None = None, goal: bool = False):
        popped = ""
        if popped_item is not None:
            popped = str(popped_item)
        print(
            f"{self._step: <3} | {str(popped): <20} "
            f"| {str(self._agenda)}")
        if goal:
            assert weight is not None
            print(
                f"GOAL: weight {round(float(weight.sum), 2)}, "
                f"probability {round(10**(-weight.sum), 20)}")

    def _record_fallback_axiom(
            self,
            item: Item,
            weight_pointer: WeightPointer,
            ) -> None:
        anchor = item.anchor
        assert anchor is not None

        cost = weight_pointer.inside
        previous = self._fallback_axiom[anchor]

        if previous is None or cost < previous[2]:
            self._fallback_axiom[anchor] = (
                item,
                weight_pointer,
                cost,
            )

    def _record_fallback_complete(
            self,
            item: Item,
            weight_pointer: WeightPointer,
            ) -> None:
        if not item.is_complete:
            return

        start = item.start
        end = item.end
        anchor = item.anchor

        assert start is not None
        assert end is not None
        assert anchor is not None

        # Any component containing the artificial root must itself be
        # rooted at the artificial root. Otherwise token 1 would receive
        # a head.
        if start == 1 and anchor != 1:
            return

        cost = weight_pointer.inside
        candidates = self._fallback_complete_by_end[end]
        previous = candidates.get(start)

        # For a given span, only the lowest-cost complete item can improve
        # the requested coverage/cost objective.
        if previous is None or cost < previous[2]:
            candidates[start] = (
                item,
                weight_pointer,
                cost,
            )

    # def _write_fallback_component(
    #         self,
    #         item: Item,
    #         weight_pointer: WeightPointer,
    #         heads: list[int],
    #         labels: list[int],
    #         supertag_inds: list[int],
    #         dependencies: list[str],
    #         root_position: int,
    #         ) -> None:
    #     """Backtrack one complete fallback component into global arrays."""
    #     assert item.is_complete
    #     assert item.start is not None
    #     assert item.end is not None
    #     assert item.anchor is not None

    #     (
    #         _remaining_left_dependencies,
    #         _remaining_right_dependencies,
    #         auxiliary_dependency,
    #     ) = self._backtrack_into(
    #         item,
    #         weight_pointer,
    #         heads,
    #         labels,
    #         supertag_inds,
    #         dependencies,
    #     )

    #     anchor_index = item.anchor - 1

    #     # Internal dependents were assigned by _backtrack_into(), but the
    #     # component anchor has no parent within its own derivation.
    #     assert heads[anchor_index] == -1, (
    #         f"Fallback component anchor {item.anchor} already received "
    #         f"head {heads[anchor_index]} while backtracking {item}."
    #     )

    #     if item.anchor == root_position:
    #         heads[anchor_index] = 0
    #         labels[anchor_index] = 0
    #         dependencies[anchor_index] = "root"
    #     else:
    #         # Artificially connect the disconnected component to the root.
    #         heads[anchor_index] = root_position

    #         if auxiliary_dependency is not None:
    #             labels[anchor_index] = 1
    #             dependencies[anchor_index] = auxiliary_dependency
    #         else:
    #             labels[anchor_index] = 0
    #             dependencies[anchor_index] = "dep"

    def _write_fallback_component(
            self,
            item: Item,
            weight_pointer: WeightPointer,
            heads: list[int],
            labels: list[int],
            supertag_inds: list[int],
            dependencies: list[str],
            ) -> str | None:
        """Backtrack one complete fallback component into global arrays."""
        assert item.is_complete
        assert item.anchor is not None

        (
            _remaining_left_dependencies,
            _remaining_right_dependencies,
            auxiliary_dependency,
        ) = self._backtrack_into(
            item,
            weight_pointer,
            heads,
            labels,
            supertag_inds,
            dependencies,
        )

        anchor_index = item.anchor - 1

        # All internal arcs are reconstructed, but the component root
        # deliberately remains unattached until the component-level MST.
        assert heads[anchor_index] == -1

        return auxiliary_dependency

    def fallback_backtrack(
            self,
            root_position: int = 1,
            ) -> tuple[
                list[int],
                list[int],
                list[int],
                list[str],
            ]:
        """Return a fallback parse when no full goal item was found.

        The method:

        1. maximizes coverage by non-overlapping complete chart items;
        2. minimizes total inside cost among equally covering selections;
        3. fills remaining positions from their best lexical axioms;
        4. treats every reconstructed subtree/singleton as one component;
        5. connects the components with a minimum-cost dependency tree
        derived from the neural head scores.
        """
        length = self._length

        if not 1 <= root_position <= length:
            raise ValueError(
                f"Root position {root_position} is invalid for "
                f"sentence length {length}."
            )

        # ------------------------------------------------------------
        # 1. Select the best set of non-overlapping complete components.
        # ------------------------------------------------------------

        best_coverage = [0] * (length + 1)
        best_cost = [0.0] * (length + 1)

        previous_position = [0] * (length + 1)

        selected_entry: list[
            tuple[Item, WeightPointer, float] | None
        ] = [None] * (length + 1)

        entry: tuple[Item, WeightPointer, float] | None

        for end in range(1, length + 1):
            # Option 1: leave position `end` uncovered.
            best_coverage[end] = best_coverage[end - 1]
            best_cost[end] = best_cost[end - 1]
            previous_position[end] = end - 1

            # Option 2: select a complete item ending at `end`.
            for entry in self._fallback_complete_by_end[end].values():
                item, _, item_cost = entry

                start = item.start
                assert start is not None

                prefix = start - 1

                candidate_coverage = (
                    best_coverage[prefix]
                    + end - start + 1
                )

                candidate_cost = (
                    best_cost[prefix]
                    + item_cost
                )

                if (
                    candidate_coverage > best_coverage[end]
                    or (
                        candidate_coverage == best_coverage[end]
                        and candidate_cost < best_cost[end]
                    )
                ):
                    best_coverage[end] = candidate_coverage
                    best_cost[end] = candidate_cost
                    previous_position[end] = prefix
                    selected_entry[end] = entry

        # Recover selected components.
        selected_components: list[
            tuple[Item, WeightPointer]
        ] = []

        position = length

        while position > 0:
            entry = selected_entry[position]

            if entry is not None:
                item, weight_pointer, _ = entry

                selected_components.append(
                    (item, weight_pointer)
                )

            position = previous_position[position]

        selected_components.reverse()

        # ------------------------------------------------------------
        # 2. Reconstruct the selected components.
        # ------------------------------------------------------------

        heads = [-1] * length
        labels = [-1] * length
        supertag_inds = [-1] * length
        dependencies = [""] * length

        covered = [False] * length

        # Information required for the component-level MST.
        #
        # All positions are one-based, like Item.anchor and UD heads.
        component_tokens: list[list[int]] = []
        component_anchors: list[int] = []
        component_aux_dependencies: list[str | None] = []

        for item, weight_pointer in selected_components:
            assert item.start is not None
            assert item.end is not None
            assert item.anchor is not None

            auxiliary_dependency = self._write_fallback_component(
                item,
                weight_pointer,
                heads,
                labels,
                supertag_inds,
                dependencies,
            )

            # The complete chart item covers a contiguous span.
            component_tokens.append(
                list(range(item.start, item.end + 1))
            )
            component_anchors.append(item.anchor)
            component_aux_dependencies.append(
                auxiliary_dependency
            )

            covered[item.start - 1:item.end] = (
                [True] * (item.end - item.start + 1)
            )

        # ------------------------------------------------------------
        # 3. Turn every still-uncovered token into a singleton component.
        # ------------------------------------------------------------

        for position in range(1, length + 1):
            index = position - 1

            if covered[index]:
                continue

            entry = self._fallback_axiom[position]

            if entry is None:
                # No lexical axiom survived at all. We still need a lexical
                # representation so that this token can participate in the
                # final dependency tree.
                supertag_inds[index] = 0

                auxiliary_dependency = None

            else:
                axiom_item, axiom_weight, _ = entry

                (
                    _left_dependencies,
                    _right_dependencies,
                    auxiliary_dependency,
                ) = self._backtrack_into(
                    axiom_item,
                    axiom_weight,
                    heads,
                    labels,
                    supertag_inds,
                    dependencies,
                )

                assert axiom_item.anchor is not None
                assert axiom_item.anchor == position, (
                    f"Fallback lexical axiom for position {position} "
                    f"has anchor {axiom_item.anchor}."
                )

                # A lexical axiom has no internal parent.
                assert heads[index] == -1

            # Do NOT assign this token to the artificial root here.
            # It is a singleton component and the MST will determine
            # its head together with all other component roots.
            component_tokens.append([position])
            component_anchors.append(position)
            component_aux_dependencies.append(
                auxiliary_dependency
            )

        # ------------------------------------------------------------
        # 4. Connect all components.
        #
        # The connector:
        #
        #   * keeps every internal A* dependency untouched;
        #   * attaches only component anchors;
        #   * permits any token of another component to serve as head;
        #   * constructs costs
        #
        #       C[D, H] =
        #           min_{h in H}
        #               head_scores[anchor(D), h]
        #
        #   * uses Chu-Liu/Edmonds to obtain one rooted tree.
        # ------------------------------------------------------------

        self._connect_fallback_components(
            component_tokens,
            component_anchors,
            component_aux_dependencies,
            heads,
            labels,
            dependencies,
            root_position,
        )

        # ------------------------------------------------------------
        # 5. Verify that the fallback really produced a complete parse.
        # ------------------------------------------------------------

        assert all(head >= 0 for head in heads), heads
        assert all(label >= 0 for label in labels), labels
        assert all(index >= 0 for index in supertag_inds), supertag_inds
        assert all(dependency for dependency in dependencies), dependencies

        return heads, labels, supertag_inds, dependencies

    def _connect_fallback_components(
            self,
            component_tokens: list[list[int]],
            component_anchors: list[int],
            component_aux_dependencies: list[str | None],
            heads: list[int],
            labels: list[int],
            dependencies: list[str],
            root_position: int,
            ) -> None:
        """Connect fallback components according to neural head scores.

        `component_tokens` and `component_anchors` contain one-based parser
        positions.

        Internal component arcs have already been reconstructed. Only each
        component anchor lacks a head.
        """
        num_components = len(component_tokens)

        assert num_components == len(component_anchors)
        assert num_components == len(component_aux_dependencies)

        if num_components == 0:
            raise RuntimeError("No fallback components.")

        # Find the component containing the artificial root.
        root_components = [
            component
            for component, tokens in enumerate(component_tokens)
            if root_position in tokens
        ]

        if len(root_components) != 1:
            raise RuntimeError(
                f"Expected one component containing root {root_position}, "
                f"got {root_components}."
            )

        root_component = root_components[0]

        if component_anchors[root_component] != root_position:
            raise RuntimeError(
                "Component containing the artificial root must itself "
                "be rooted at the artificial root."
            )

        # UFAL expects node 0 to be the root, so reorder components.
        component_order = [
            root_component,
            *(
                component
                for component in range(num_components)
                if component != root_component
            ),
        ]

        # contracted_costs[dependent_component, head_component]
        contracted_costs = np.full(
            (num_components, num_components),
            np.inf,
            dtype=np.float64,
        )

        # For every contracted edge, remember which actual token realizes
        # the optimal head.
        actual_heads = np.full(
            (num_components, num_components),
            -1,
            dtype=np.int64,
        )

        for dependent_mst, dependent_component in enumerate(component_order):
            if dependent_mst == 0:
                # Artificial-root component receives no parent.
                continue

            dependent_anchor = component_anchors[dependent_component]
            dependent_index = dependent_anchor - 1

            for head_mst, head_component in enumerate(component_order):
                if head_mst == dependent_mst:
                    continue

                candidate_positions = np.asarray(
                    component_tokens[head_component],
                    dtype=np.int64,
                )

                # Convert one-based parser positions to score-matrix indices.
                candidate_indices = candidate_positions - 1

                # head_scores[dependent, head]
                candidate_costs = self._unmasked_head_scores[
                    dependent_index,
                    candidate_indices,
                ]

                best_local = int(np.argmin(candidate_costs))
                best_cost = float(candidate_costs[best_local])

                if not np.isfinite(best_cost):
                    continue

                contracted_costs[
                    dependent_mst,
                    head_mst,
                ] = best_cost

                actual_heads[
                    dependent_mst,
                    head_mst,
                ] = candidate_positions[best_local]

        # Self-components cannot be heads of themselves.
        np.fill_diagonal(contracted_costs, np.inf)

        # Root component must have no incoming edge.
        contracted_costs[0, :] = np.inf

        # UFAL maximizes scores; our matrix contains costs.
        scores = -contracted_costs

        # UFAL denotes impossible arcs with NaN.
        scores[~np.isfinite(scores)] = np.nan

        mst_heads, _mst_score = chu_liu_edmonds(scores)

        # Artificial root.
        root_index = root_position - 1
        heads[root_index] = 0
        labels[root_index] = 0
        dependencies[root_index] = "root"

        # Expand component-level edges back into token-level arcs.
        for dependent_mst in range(1, num_components):
            head_mst = int(mst_heads[dependent_mst])

            dependent_component = component_order[dependent_mst]

            dependent_anchor = component_anchors[
                dependent_component
            ]
            dependent_index = dependent_anchor - 1

            actual_head = int(
                actual_heads[
                    dependent_mst,
                    head_mst,
                ]
            )

            if actual_head < 1:
                raise RuntimeError(
                    f"No actual head for component edge "
                    f"{head_mst} -> {dependent_mst}."
                )

            heads[dependent_index] = actual_head

            auxiliary_dependency = (
                component_aux_dependencies[
                    dependent_component
                ]
            )

            if auxiliary_dependency is not None:
                labels[dependent_index] = 1
                dependencies[dependent_index] = auxiliary_dependency
            else:
                labels[dependent_index] = 0
                dependencies[dependent_index] = "dep"


def process(
        inp
        ) -> tuple[np.ndarray, np.ndarray]:
    # prefix: np.ndarray, ma: np.ndarray, ig: np.ndarray,
    #             supertag_scores: np.ndarray, pad_len: int,
    #             predicted_pos: np.ndarray
    (
        prefix, ma, ig, supertag_scores, pad_len, predicted_pos,
        deprel2id, id2sup_relative, id2pos, root_sup_id, max_l, max_r,
        k_head_scores, k_supertag) = inp
    ignore = ig == -1
    # temp = ma.copy()
    # temp[ignore][:, ignore] = float("-inf")
    # temp = temp.argmax(-1)
    # temp[ignore] = -1

    ma = ma[~ignore][:, np.logical_or(
        ~ignore, prefix[0][:ma.shape[-1]])]

    inpt = utils.softmax(ma)

    inpt = np.concatenate((prefix[:, :inpt.shape[-1]], inpt))  #
    with np.errstate(divide='ignore'):
        inpt = -np.log10(inpt)       # inpt[:, 1:])

    supertag_scores = utils.softmax(supertag_scores[~ignore])
    root = np.zeros((1, supertag_scores.shape[-1],))
    root[0, root_sup_id] = 1
    supertag_scores = np.concatenate(
        (root, supertag_scores),
    )
    with np.errstate(divide='ignore'):
        supertag_scores = -np.log10(supertag_scores)

    system = System(
        inpt, supertag_scores, id2sup_relative,
        max_r=max_r, max_l=max_l,
        k_supertag=k_supertag, k_head_scores=k_head_scores,
        contains_root=True)
    result = system.run(printinfo=False)

    if result is None:
        # head_result = np.zeros((inpt.shape[0],), dtype=int)
        # deprel_result = np.zeros((inpt.shape[0],), dtype=int)
        backtracked = system.fallback_backtrack()
    else:
        backtracked = system.backtrack(result[0], result[1])

    predicted_pos = np.concatenate(
        (predicted_pos[0, np.newaxis], predicted_pos[~ignore]))
    predicted_pos += predicted_pos < 1  # WHY?
    # TODO: what is the root POS tag?

    head_result = np.array(backtracked[0], dtype=int)-1
    head_result += head_result < 0
    head_result[0] = 0
    # print(head_result.shape, head_result)
    deprel_str_result: list[str] = backtracked[3]
    predicted_head_pos = predicted_pos[head_result[1:]]
    # print(predicted_pos.shape, predicted_head_pos.shape)
    deprel_result = np.array([
        deprel2id[deprels.reconstruct(
            deprel, id2pos[dep_pos.item()], id2pos[head_pos.item()])]
        for deprel, dep_pos, head_pos in zip(
            deprel_str_result[1:], predicted_pos[1:],
            predicted_head_pos)
    ])

    # shape: [unpadded_len]
    # The indices refer to the head positions without ignored tokens.
    # Now: add cumulative sum of ignore occurrences to the respective
    # references

    head_result = np.concatenate(
        (prefix[0, 0, np.newaxis], head_result[1:]), axis=0)

    # print("heads_long1", heads_long)
    cumsum = np.cumsum(ignore) - 1  # [0, (0, 0), 1, (1,)]
    # print("cumsum", cumsum)
    head_cumsum = cumsum[~ignore]
    # [1, (1, 1), 2, (2,)][False, True, True, False, True] -1 = [0, 0, 1]
    head_cumsum = head_cumsum[head_result[1:]-1]
    head_cumsum[head_result[1:] == 0] = 0
    # [h_c[x], h_c[y], h_c[z]]

    # print("result", result)
    # print("h_cums", head_cumsum)
    head_result[1:] += head_cumsum

    # heads = np.full((pad_len,), -1)
    # heads[ig != -1] = result[1:]
    # ignore_sum = np.cumsum(ignore)-1
    # return heads + ignore_sum
    heads = np.full((pad_len,), -1)
    heads[~ignore] = head_result[1:]
    # print("mst   ", heads)
    # print("argmax", temp)
    # print("gold  ", ig)

    deprels_ = np.full((pad_len,), 0)
    deprels_[~ignore] = deprel_result
    return heads, deprels_


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


def chart(
        score_matrix: np.ndarray,
        ignore_deprels: np.ndarray,
        supertag_scores: np.ndarray,
        id2sup: Mapping[int, str],
        id2pos: Mapping[int, str],
        deprel2id: Mapping[str, int],
        predicted_pos_ids: np.ndarray,
        max_l: int,
        max_r: int,
        root_sup_id: int,
        k_supertag: int = 10,
        k_head_scores: int = 10,
        ) -> np.ndarray:

    prefix = np.zeros((1, score_matrix.shape[-1]), dtype=int)
    prefix[0, 0] = 1

    id2sup_relative = {
        i: extraction.convert_string_to_relative_relation(tag)
        for i, tag in id2sup.items()}

    # def process(
    #         prefix: np.ndarray, ma: np.ndarray, ig: np.ndarray,
    #         supertag_scores: np.ndarray, pad_len: int,
    #         predicted_pos: np.ndarray
    #         ) -> tuple[np.ndarray, np.ndarray]:
    #     # prefix: np.ndarray, ma: np.ndarray, ig: np.ndarray,
    #     #             supertag_scores: np.ndarray, pad_len: int,
    #     #             predicted_pos: np.ndarray
    #     # prefix, ma, ig, supertag_scores, pad_len, predicted_pos = inp
    #     ignore = ig == -1
    #     # temp = ma.copy()
    #     # temp[ignore][:, ignore] = float("-inf")
    #     # temp = temp.argmax(-1)
    #     # temp[ignore] = -1

    #     ma = ma[~ignore][:, np.logical_or(
    #         ~ignore, prefix[0][:ma.shape[-1]])]

    #     inpt = utils.softmax(ma)

    #     inpt = np.concatenate((prefix[:, :inpt.shape[-1]], inpt))  #
    #     inpt = -np.log10(inpt)       # inpt[:, 1:])prefix: np.ndarray,
    # ma: np.ndarray, ig: np.ndarray,
    #     #             supertag_scores: np.ndarray, pad_len: int,
    #     #             predicted_pos: np.ndarray

    #     supertag_scores = utils.softmax(supertag_scores[~ignore])
    #     root = np.zeros((1, supertag_scores.shape[-1],))
    #     root[0, root_sup_id] = 1
    #     supertag_scores = np.concatenate(
    #         (root, supertag_scores),
    #     )
    #     supertag_scores = -np.log10(supertag_scores)

    #     system = System(
    #         inpt, supertag_scores, id2sup_relative,
    #         max_r=max_r, max_l=max_l,
    #         k_supertag=k_supertag, k_head_scores=k_head_scores,
    #         contains_root=True)
    #     result = system.run(allow_incomplete=False, printinfo=False)

    #     if result is None:
    #         # head_result = np.zeros((inpt.shape[0],), dtype=int)
    #         # deprel_result = np.zeros((inpt.shape[0],), dtype=int)
    #         backtracked = system.backtrack_disconnected()
    #     else:
    #         backtracked = system.backtrack(result[1])

    #     predicted_pos = np.concatenate(
    #         (predicted_pos[0, np.newaxis], predicted_pos[~ignore]))
    #     predicted_pos += predicted_pos < 1  # WHY?
    #     # TODO: what is the root POS tag?

    #     head_result = np.array(backtracked[0], dtype=int)-1
    #     head_result += head_result < 0
    #     head_result[0] = 0
    #     # print(head_result.shape, head_result)
    #     deprel_str_result: list[str] = backtracked[3]
    #     predicted_head_pos = predicted_pos[head_result[1:]]
    #     # print(predicted_pos.shape, predicted_head_pos.shape)
    #     deprel_result = np.array([
    #         deprel2id[deprels.reconstruct(
    #             deprel, id2pos[dep_pos.item()], id2pos[head_pos.item()])]
    #         for deprel, dep_pos, head_pos in zip(
    #             deprel_str_result[1:], predicted_pos[1:],
    #             predicted_head_pos)
    #     ])

    #     # shape: [unpadded_len]
    #     # The indices refer to the head positions without ignored tokens.
    #     # Now: add cumulative sum of ignore occurrences to the respective
    #     # references

    #     head_result = np.concatenate(
    #         (prefix[0, 0, np.newaxis], head_result[1:]), axis=0)

    #     # print("heads_long1", heads_long)
    #     cumsum = np.cumsum(ignore) - 1  # [0, (0, 0), 1, (1,)]
    #     # print("cumsum", cumsum)
    #     head_cumsum = cumsum[~ignore]
    #     # [1, (1, 1), 2, (2,)][False, True, True, False, True] -1 = [0, 0, 1]
    #     head_cumsum = head_cumsum[head_result[1:]-1]
    #     head_cumsum[head_result[1:] == 0] = 0
    #     # [h_c[x], h_c[y], h_c[z]]

    #     # print("result", result)
    #     # print("h_cums", head_cumsum)
    #     head_result[1:] += head_cumsum

    #     # heads = np.full((pad_len,), -1)
    #     # heads[ig != -1] = result[1:]
    #     # ignore_sum = np.cumsum(ignore)-1
    #     # return heads + ignore_sum
    #     heads = np.full((pad_len,), -1)
    #     heads[~ignore] = head_result[1:]
    #     # print("mst   ", heads)
    #     # print("argmax", temp)
    #     # print("gold  ", ig)

    #     deprels_ = np.full((pad_len,), 0)
    #     deprels_[~ignore] = deprel_result
    #     return heads, deprels_

    start = timer()
    # stack = [
    #     process(
    #         prefix, ma, ig, sup, score_matrix.shape[1], pp)
    #     for ma, ig, sup, pp in zip(
    #         score_matrix, ignore_deprels, supertag_scores, predicted_pos_ids)]

    # prefix, ma, ig, supertag_scores, pad_len, predicted_pos,
    # deprel2id, id2sup_relative, id2pos, root_sup_id, max_l, max_r,
    # k_head_scores, k_supertag) = inp

    le = score_matrix.shape[0]
    # num_workers = multiprocessing.cpu_count()
    # chunksize = max(1, (le + num_workers * 16 - 1) // (num_workers * 16))
    chunksize = 1
    stack = multiprocessing.Pool(processes=get_eval_workers()).map(
        process,
        tqdm.tqdm(
            zip(
                [prefix]*le,
                score_matrix, ignore_deprels, supertag_scores,
                [score_matrix.shape[1]]*le,
                predicted_pos_ids,
                [deprel2id]*le, [id2sup_relative]*le, [id2pos]*le,
                [root_sup_id]*le,
                [max_l]*le, [max_r]*le, [k_head_scores]*le, [k_supertag]*le),
            desc="Chart parsing",
            total=le),
        chunksize=chunksize)

    end = timer()
    print("Chart took", timedelta(seconds=end-start), "seconds")
    return np.stack([s[0] for s in stack]), np.stack([s[1] for s in stack])

# TODO: keep track of widest complete item, return it if not finding goal,
# allow backtracking
# ==> retrieve at least part of the tree; all other tokens are connected with
# the root


# ### Schema ###
# A few functions:
# Item x Weight x Item x Weight -> Item x WeightPointer | None
# Rules with two antecedencts have to be added twice (symmetric)


# head_probs: np.ndarray = np.array(
#     [
#         [1, 0, 0, 0, 0, 0, 0],
#         [0.1, 0, 0.1, 0.8, 0.3, 0.1, 0.1],
#         [0.5, 0.3, 0.1, 0.1, 0, 0.1, 0.8,],
#         [0.05, 0.3, 0.6, 0.05, 0.3, 0.6, 0.05],
#         [0.05, 0.3, 0.6, 0.05, 0.3, 0.6, 0.05],
#         [0.5, 0.3, 0.1, 0.1, 0, 0.1, 0.8, ],
#         [0.1, 0.01, 0.09, 0.8, 0.02, 0.05, 0.03,],
#     ])
# head_probs /= head_probs.sum(-1, keepdims=True)
# 
# supertag_probs: np.ndarray = np.array(
#     [
#         [0.75, 0.15, 0.05, 0.05],
#         [0.3, 0.6, 0.05, 0.05],
#         [0.05, 0.05, 0.5, 0.4],
#         [0.3, 0.6, 0.05, 0.05],
#         [0.05, 0.05, 0.5, 0.4],
#         [0.75, 0.15, 0.05, 0.05],
#     ]
# )
# 
# id2sup = {
#     0: "*",
#     1: "*+dep1",
#     2: "+dep2*",
#     3: "-aux+dep3*",
# }
# 
# system = System(
#     head_scores=-np.log10(head_probs[1:, 1:]),
#     supertag_scores=-np.log10(supertag_probs),
#     id2sup={
#         i: extraction.convert_string_to_relative_relation(sup)
#         for i, sup in id2sup.items()},
#     max_r=10,
#     max_l=10,
#     k_head_scores=200,
#     k_supertag=100
# )
# 
# result = system.run(printinfo=True)
# if result is not None:
#     system.backtrack(result[1])

# TODO: head conversion
# TODO: deterministic deptag conversion