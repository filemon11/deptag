import pytest
import numpy as np

from ...parsing.chart import (
    Weight, Item, AUXMAP, AXIOM, Chart, WeightPointer, Agenda,
    supertag_to_item, item_axiom, item_complete, item_switch,
    item_left_subst, item_right_subst, item_left_adjoin, item_right_adjoin,
    System, R, L, N)
from ...extraction import RelativeTag


class TestWeight:

    def test_sum(self):
        weight = Weight(10, 15)

        assert weight.sum == 25, "Should be 25"
        assert float(weight) == 25, "Should be 25"

    def test_comparisons(self):
        weight1 = Weight(10, 15)
        weight2 = Weight(2, 16)
        weight3 = Weight(2, 16)

        assert weight2 < weight1
        assert weight1 > weight2
        assert not weight1 == weight2
        assert weight2 == weight3


class TestItem:

    def test_tuple(self):
        tup = (1, 2, 2, 0, 4, 0)
        assert Item(*tup).tup == tup

    def test_constructor(self):

        # Indexing starts at 1
        with pytest.raises(AssertionError):
            Item(-1, 3, 1, 0, 0, "-")
        with pytest.raises(AssertionError):
            Item(0, 3, 1, 0, 0, "-")

        # Start and end order
        with pytest.raises(AssertionError):
            Item(3, 1, 3, 0, 0, "-")

        # Anchor positioning
        with pytest.raises(AssertionError):
            Item(1, 3, 4, 0, 0, "-")
        with pytest.raises(AssertionError):
            Item(2, 3, 1, 0, 0, "-")

        # Too many left arguments
        with pytest.raises(AssertionError):
            Item(10, 11, 10, 10, 0, "-")
        with pytest.raises(AssertionError):
            Item(2, 4, 2, 1, 0, "<")

        # None must be complete
        with pytest.raises(AssertionError):
            Item(1, None, None, None, None, None)
        with pytest.raises(AssertionError):
            Item(None, 7, None, None, None, None)
        with pytest.raises(AssertionError):
            Item(None, None, 2, None, None, None)
        with pytest.raises(AssertionError):
            Item(None, None, None, 3, None, None)
        with pytest.raises(AssertionError):
            Item(None, None, None, None, 4, None)
        with pytest.raises(AssertionError):
            Item(None, None, None, None, None, "-")

        # Unspecified right args
        with pytest.raises(AssertionError):
            Item(1, 3, 2, 0, None, ">")

        # aux translation
        assert (
            Item(1, 3, 2, 0, 0, ">").aux == Item(
                1, 3, 2, 0, 0, AUXMAP[">"]).aux)

    def test_is_axiom(self):
        assert Item(None, None, None, None, None, None).is_axiom
        assert not Item(1, 3, 2, 0, 1, "-").is_axiom
        assert not Item(1, 3, 2, None, 1, "-").is_axiom
        assert not Item(1, 3, 2, None, None, "-").is_axiom

    def test_is_initial(self):
        assert Item(1, 3, 2, 0, 1, "-").is_initial
        assert Item(1, 3, 2, None, 1, "-").is_initial
        assert Item(1, 3, 2, None, None, "-").is_initial
        assert not Item(2, 3, 2, 0, 1, "<").is_initial
        assert not Item(2, 3, 2, 0, 1, ">").is_initial
        assert not Item(None, None, None, None, None, None).is_initial

    def test_is_adjunct(self):
        assert Item(1, 3, 2, 0, 1, ">").is_adjunct
        assert Item(2, 3, 2, None, 1, "<").is_adjunct
        assert Item(1, 3, 2, None, None, ">").is_adjunct
        assert not Item(2, 3, 2, 0, 1, "-").is_adjunct
        assert not Item(2, 3, 2, 0, 1, "-").is_adjunct
        assert not Item(1, 3, 2, None, None, "-").is_adjunct
        assert not Item(None, None, None, None, None, None).is_adjunct

    def test_is_left_adjunct(self):
        assert Item(1, 3, 2, 0, 1, ">").is_left_adjunct
        assert Item(2, 3, 2, None, 1, ">").is_left_adjunct
        assert Item(1, 3, 2, None, None, ">").is_left_adjunct
        assert not Item(2, 3, 2, 0, 1, "-").is_left_adjunct
        assert not Item(2, 3, 2, 0, 1, "-").is_left_adjunct
        assert not Item(1, 3, 2, None, None, "-").is_left_adjunct
        assert not Item(2, 3, 2, 0, 1, "<").is_left_adjunct
        assert not Item(2, 3, 2, 0, 1, "<").is_left_adjunct
        assert not Item(3, 3, 3, None, None, "<").is_left_adjunct
        assert not Item(None, None, None, None, None, None).is_left_adjunct

    def test_is_right_adjunct(self):
        assert Item(2, 3, 2, 0, 1, "<").is_right_adjunct
        assert Item(3, 5, 4, None, 1, "<").is_right_adjunct
        assert Item(2, 3, 2, None, None, "<").is_right_adjunct
        assert not Item(2, 3, 2, 0, 1, "-").is_right_adjunct
        assert not Item(2, 3, 2, 0, 1, "-").is_right_adjunct
        assert not Item(1, 3, 2, None, None, "-").is_right_adjunct
        assert not Item(2, 3, 2, 0, 1, ">").is_right_adjunct
        assert not Item(2, 3, 2, 0, 1, ">").is_right_adjunct
        assert not Item(3, 3, 3, None, None, ">").is_right_adjunct
        assert not Item(None, None, None, None, None, None).is_right_adjunct

    def test_is_complete(self):
        assert Item(3, 5, 4, None, None, "<").is_complete
        assert Item(3, 5, 4, None, None, "-").is_complete
        assert not Item(3, 5, 4, None, 1, "-").is_complete
        assert not Item(3, 5, 4, None, 0, "-").is_complete
        assert not Item(3, 5, 4, 1, 0, "-").is_complete
        assert not Item(3, 5, 4, 0, 0, "-").is_complete
        assert not Item(None, None, None, None, None, None).is_complete

    def test_is_uncomplete(self):
        assert Item(3, 5, 4, None, 1, "-").is_uncomplete
        assert Item(3, 5, 4, None, 0, "-").is_uncomplete
        assert Item(3, 5, 4, 1, 0, "-").is_uncomplete
        assert Item(3, 5, 4, 0, 0, "-").is_uncomplete

        assert not Item(3, 5, 4, None, None, "<").is_uncomplete
        assert not Item(3, 5, 4, None, None, "-").is_uncomplete
        assert not Item(None, None, None, None, None, None).is_uncomplete

    def test_is_switched(self):
        assert Item(3, 5, 4, None, 0, "-").is_switched
        assert Item(2, 6, 2, None, 1, ">").is_switched
        assert not Item(2, 6, 2, None, None, ">").is_switched
        assert not Item(2, 6, 2, 0, 0, ">").is_switched
        assert not Item(2, 6, 2, 1, 0, ">").is_switched
        assert not Item(2, 6, 2, 0, 1, ">").is_switched
        assert not Item(None, None, None, None, None, None).is_switched

    def test_is_unswitched(self):
        assert Item(3, 5, 4, 0, 0, "-").is_unswitched
        assert Item(2, 6, 2, 1, 0, ">").is_unswitched
        assert not Item(2, 6, 2, None, 0, ">").is_unswitched
        assert not Item(2, 6, 2, None, None, ">").is_unswitched
        assert not Item(None, None, None, None, None, None).is_unswitched

    def test_is_goal(self):
        assert Item(1, 5, 5, None, None, "-").is_goal(5)
        assert not Item(1, 5, 5, None, 0, "-").is_goal(5)
        assert not Item(1, 5, 5, 0, 0, "-").is_goal(5)
        assert not Item(1, 5, 5, None, None, "-").is_goal(4)
        assert not Item(1, 5, 5, None, None, "-").is_goal(6)
        assert not Item(1, 5, 5, None, None, ">").is_goal(5)
        assert not Item(None, None, None, None, None, None).is_goal(6)

    def test_lt(self):
        assert Item(1, 5, 5, None, None, "-") < Item(6, 6, 6, None, None, "-")
        assert Item(1, 5, 5, None, None, "-") < Item(6, 6, 6, 0, 0, "-")
        assert Item(7, 7, 7, None, None, "-") > Item(6, 6, 6, 0, 0, "-")
        assert not Item(7, 7, 7, None, None, "-") < Item(6, 6, 6, 0, 0, "-")

        with pytest.raises(AssertionError):
            Item(None, None, None, None, None, None) < Item(6, 6, 6, 0, 0, "-")

    def test_eq(self):
        assert Item(1, 5, 5, None, None, "-") == Item(1, 5, 5, None, None, "-")
        assert Item(1, 5, 5, None, None, "-") == (
            1, 5, 5, None, None, AUXMAP["-"])
        assert Item(1, 5, 5, None, None, "-") != Item(6, 6, 6, 0, 0, "-")
        assert Item(7, 7, 7, None, None, "-") != Item(
            6, 6, 6, 0, 0, AUXMAP["-"])

    def test_len(self):
        assert len(Item(1, 1, 1, None, None, N)) == 1
        assert len(Item(2, 4, 3, None, None, R)) == 3
        assert len(Item(None, None, None, None, None, None)) == 0


class TestAxiom:
    def test_axiom(self):
        assert AXIOM.is_axiom


class TestChart:
    def test_get_index(self):
        with pytest.raises(AssertionError):
            Chart.get_index((0, 0, 0, 0, 0))

        with pytest.raises(AssertionError):
            Chart.get_index((0, 0, 0, 0, 0, 0, 0))

        item = Item(1, 3, 2, 0, 4, "-")
        assert Chart.get_index(item) == Chart.item2chartidxs(item)

    def test_item2chartidxs(self):
        assert Chart.item2chartidxs(
            Item(1, 3, 2, 0, 0, "-")) == (1, 3, 2, 1, 1, AUXMAP["-"]+1)
        assert Chart.item2chartidxs(
            Item(1, 3, 2, None, 0, ">")) == (1, 3, 2, 0, 1, AUXMAP[">"]+1)
        assert Chart.item2chartidxs(
            Item(1, 3, 2, None, None, ">")) == (1, 3, 2, 0, 0, AUXMAP[">"]+1)
        assert Chart.item2chartidxs(
            Item(None, None, None, None, None, None)) == (0, 0, 0, 0, 0, 0)

    def test_chartidxs2item(self):
        assert Item(
            1, 3, 2, 0, 0, "-") == Chart.chartidxs2item(
                (1, 3, 2, 1, 1, AUXMAP["-"]+1)), (
                    Item(1, 3, 2, 0, 0, "-"), Chart.chartidxs2item(
                        (1, 3, 2, 1, 1, AUXMAP["-"]+1))
                )
        assert Item(
            1, 3, 2, None, 0, ">") == Chart.chartidxs2item(
                (1, 3, 2, 0, 1, AUXMAP[">"]+1))
        assert Item(
            1, 3, 2, None, None, ">") == Chart.chartidxs2item(
                (1, 3, 2, 0, 0, AUXMAP[">"]+1))
        assert Item(
            None, None, None, None, None, None) == Chart.chartidxs2item(
                (0, 0, 0, 0, 0, 0))

    def test_setitem(self):
        chart = Chart(6, 4, 4)
        weight1 = WeightPointer(3.0, 2.2, AXIOM, AXIOM, 0)
        item1 = Item(1, 3, 2, 0, 0, "-")
        # item2 = Item(5, 5, 5, None, None, "<")
        weight2 = WeightPointer(4.1, 0, AXIOM, item1, 1)

        chart[item1] = weight1
        assert np.array_equal(
            chart._chart[1, 3, 2, 1, 1, AUXMAP["-"]+1],
            np.array(
                (3.0, 2.2, *[0]*12, 0),
                dtype=np.float32))

        chart[5, 5, 5, 0, 0, AUXMAP["<"]+1] = weight2
        assert np.array_equal(
            chart._chart[5, 5, 5, 0, 0, AUXMAP["<"]+1],
            np.array(
                (4.1, 0, *[0]*6, 1, 3, 2, 1, 1, AUXMAP["<"], 1),
                dtype=np.float32),
            )

    def test_getitem(self):
        chart = Chart(6, 4, 4)
        # weight1 = WeightPointer(3.0, 2.2, AXIOM, AXIOM, 0)
        item1 = Item(1, 3, 2, 0, 0, "-")
        # item2 = Item(5, 5, 5, None, None, "<")
        # weight2 = WeightPointer(4.1, 0, AXIOM, item1, 1)

        chart._chart[1, 3, 2, 1, 1, AUXMAP["-"]+1] = (
            3.0, 2.2, *[0]*12, 0)
        weight1 = chart[item1]
        assert np.array_equal(weight1.to_array(), np.array((
            3.0, 2.2), dtype=np.float32))
        assert weight1.back1 == AXIOM
        assert weight1.back2 == AXIOM
        assert weight1.supertag_ind == 0

        chart._chart[5, 5, 5, 0, 0, AUXMAP["<"]+1] = (
            4.1, 0, *[0]*6, 1, 3, 2, 1, 1, AUXMAP["-"]+1, 1)
        weight2 = chart[5, 5, 5, 0, 0, AUXMAP["<"]+1]
        assert np.array_equal(weight2.to_array(), np.array((
            4.1, 0.0), dtype=np.float32))
        assert weight2.back1 == AXIOM
        assert weight2.back2 == item1
        assert weight2.supertag_ind == 1

        item3 = Item(1, 2, 1, None, None, "-")
        weight3 = chart[item3]
        assert np.array_equal(weight3.to_array(), np.array((
            np.inf, np.inf), dtype=np.float32))
        assert weight3.back1 == AXIOM
        assert weight3.back2 == AXIOM
        assert weight3.supertag_ind == 0

    def test_peek(self):
        chart = Chart(6, 4, 4)
        # weight1 = WeightPointer(3.0, 2.2, AXIOM, AXIOM, 0)
        item1 = Item(1, 3, 2, 0, 0, "-")
        # item2 = Item(5, 5, 5, None, None, "<")
        # weight2 = WeightPointer(4.1, 0, AXIOM, item1, 1)

        chart._chart[1, 3, 2, 1, 1, AUXMAP["-"]+1] = (
            3.0, 2.2, *[0]*12, 0)
        weight1 = chart.peek(item1)
        assert weight1 is not None
        assert np.array_equal(weight1.to_array(), np.array((
            3.0, 2.2), dtype=np.float32))
        assert weight1.back1 == AXIOM
        assert weight1.back2 == AXIOM
        assert weight1.supertag_ind == 0

        chart._chart[5, 5, 5, 0, 0, AUXMAP["<"]+1] = (
            4.1, 0, *[0]*6, 1, 3, 2, 1, 1, AUXMAP["-"]+1, 1)
        weight2 = chart.peek((5, 5, 5, 0, 0, AUXMAP["<"]+1))
        assert weight2 is not None
        assert np.array_equal(weight2.to_array(), np.array((
            4.1, 0.0), dtype=np.float32))
        assert weight2.back1 == AXIOM
        assert weight2.back2 == item1
        assert weight2.supertag_ind == 1

        item3 = Item(1, 2, 1, None, None, "-")
        weight3 = chart.peek(item3)
        assert weight3 is None


class TestAgenda:
    def test_add_update(self):
        agenda = Agenda()

        item1 = Item(3, 5, 4, 1, 1, ">")
        item2 = Item(3, 5, 4, 1, 1, "-")
        item3 = Item(3, 5, 4, 1, 1, "<")

        weight1 = WeightPointer(
            0.0, 0.0, AXIOM, AXIOM, 0)
        # weight2 = WeightPointer(
        #     0.1, 0.2, item1, AXIOM, 0)
        weight3 = WeightPointer(
            0.5, 0.0, item1, item2, 0)

        agenda.add_update(item1, weight1)
        assert set(agenda._heap) == {item1}
        assert agenda._heap[item1.tup] == weight1

        agenda.add_update(item3, weight3)
        assert set(agenda._heap) == {item1, item3}
        assert agenda._heap[item3.tup] == weight3

        agenda.add_update(item3, weight1)
        assert set(agenda._heap) == {item1, item3}
        assert agenda._heap[item3.tup] == weight1

        agenda.add_update(item3, weight3)
        assert set(agenda._heap) == {item1, item3}
        assert agenda._heap[item3.tup] == weight1

    def test_pop(self):
        agenda = Agenda()

        item1 = Item(3, 5, 4, 1, 1, ">")
        item2 = Item(3, 5, 4, 1, 1, "-")
        item3 = Item(3, 5, 4, 1, 1, "<")

        weight1 = WeightPointer(
            0.0, 0.0, AXIOM, AXIOM, 0)
        weight2 = WeightPointer(
            0.1, 0.2, item1, AXIOM, 0)
        weight3 = WeightPointer(
            0.5, 0.0, item1, item2, 0)

        agenda.add_update(item1, weight1)
        item, weight = agenda.pop()
        assert item == item1
        assert weight.sum == weight1.sum

        agenda.add_update(item3, weight3)
        item, weight = agenda.pop()
        assert item == item3
        assert weight.sum == weight3.sum

        agenda.add_update(item3, weight3)
        agenda.add_update(item1, weight1)
        agenda.add_update(item2, weight2)
        item, weight = agenda.pop()
        assert item == item1
        assert weight.sum == weight1.sum

        agenda.add_update(item2, weight1)
        item, weight = agenda.pop()
        assert item == item2
        assert weight.sum == weight1.sum


class TestItemFunctions:
    def test_item_axiom(self):
        assert item_axiom(
            1, 0, 4, AUXMAP[">"]) == Item(
                1, 1, 1, 0, 4, AUXMAP[">"])

        assert item_axiom(
            3, 1, 4, AUXMAP["<"]) == Item(
                3, 3, 3, 1, 4, AUXMAP["<"])

    def test_supertag_to_item(self):
        tag1: RelativeTag = ((True, "dep1"), (False, "dep2"), (None, "*"))
        assert supertag_to_item(
            1, tag1) is None

        tag2: RelativeTag = ((None, "*"), (False, "dep2"), (True, "dep1"))
        assert supertag_to_item(
            1, tag2) is None

        tag3: RelativeTag = ((None, "*"), (True, "dep2"), (False, "dep1"))
        assert (
            supertag_to_item(
                1, tag3)
            == Item(1, 1, 1, 0, 1, ">")
        )

        tag4: RelativeTag = (
            (False, "dep1"), (True, "dep3"),
            (None, "*"), (True, "dep2"))
        assert (
            supertag_to_item(
                5, tag4)
            == Item(5, 5, 5, 1, 1, "<")
        )

        tag5: RelativeTag = (
            (False, "dep1"), (True, "dep3"),
            (None, "*"), (False, "dep2"))
        with pytest.raises(AssertionError):
            supertag_to_item(
                5, tag5)

    def test_item_complete(self):
        with pytest.raises(AssertionError):
            item_complete(Item(1, 1, 1, 0, 2, "-"))

        with pytest.raises(AssertionError):
            item_complete(Item(1, 1, 1, None, None, "-"))

        assert item_complete(
            Item(2, 4, 3, None, 0, ">")).is_complete

    def test_item_switch(self):
        with pytest.raises(AssertionError):
            item_switch(Item(2, 2, 2, 1, 2, "-"))

        with pytest.raises(AssertionError):
            item_switch(Item(2, 2, 2, None, None, "-"))

        assert item_switch(
            Item(2, 2, 2, 0, 2, "<")).is_switched

    def test_item_left_subst(self):
        with pytest.raises(AssertionError):
            item_left_subst(
                Item(2, 2, 2, None, 2, "-"),
                Item(1, 1, 1, 0, 0, "-"))

        with pytest.raises(AssertionError):
            item_left_subst(
                Item(2, 2, 2, 1, 2, "-"),
                Item(1, 1, 1, 0, 0, "-"))

        with pytest.raises(AssertionError):
            item_left_subst(
                Item(2, 2, 2, 1, 2, "-"),
                Item(1, 1, 1, None, 1, "-"))

        with pytest.raises(AssertionError):
            item_left_subst(
                Item(2, 2, 2, 1, 2, "-"),
                Item(1, 1, 1, None, None, ">"))

        with pytest.raises(AssertionError):
            item_left_subst(
                Item(2, 2, 2, 1, 2, "-"),
                Item(3, 3, 3, None, None, "-"))

        assert item_left_subst(
            Item(2, 2, 2, 1, 2, "-"),
            Item(1, 1, 1, None, None, "-")
        ) == Item(1, 2, 2, 0, 2, "-")

    def test_item_right_subst(self):
        with pytest.raises(AssertionError):
            item_right_subst(
                Item(2, 2, 2, 1, 2, "-"),
                Item(3, 3, 3, None, None, "-"))

        with pytest.raises(AssertionError):
            item_right_subst(
                Item(2, 2, 2, None, 2, "-"),
                Item(3, 3, 3, 0, 0, "-"))

        with pytest.raises(AssertionError):
            item_right_subst(
                Item(2, 2, 2, None, 2, "-"),
                Item(3, 3, 3, None, 1, "-"))

        with pytest.raises(AssertionError):
            item_right_subst(
                Item(2, 2, 2, None, 2, "-"),
                Item(3, 3, 3, None, None, ">"))

        with pytest.raises(AssertionError):
            item_right_subst(
                Item(2, 2, 2, None, 2, "-"),
                Item(1, 1, 1, None, None, "-"))

        assert item_right_subst(
            Item(2, 2, 2, None, 2, "-"),
            Item(3, 3, 3, None, None, "-")
        ) == Item(2, 3, 2, None, 1, "-")

    def test_item_left_adjoin(self):
        with pytest.raises(AssertionError):
            item_left_adjoin(
                Item(2, 2, 2, None, 2, "-"),
                Item(1, 1, 1, 0, 0, ">"))

        with pytest.raises(AssertionError):
            item_left_adjoin(
                Item(2, 2, 2, 0, 2, "-"),
                Item(1, 1, 1, 0, 0, ">"))

        with pytest.raises(AssertionError):
            item_left_adjoin(
                Item(2, 2, 2, 0, 2, "-"),
                Item(1, 1, 1, None, 1, ">"))

        with pytest.raises(AssertionError):
            item_left_adjoin(
                Item(2, 2, 2, 0, 2, "-"),
                Item(1, 1, 1, None, None, "-"))

        with pytest.raises(AssertionError):
            item_left_adjoin(
                Item(3, 3, 3, 0, 2, "-"),
                Item(2, 2, 2, None, None, "<"))

        with pytest.raises(AssertionError):
            item_left_adjoin(
                Item(2, 2, 2, 0, 2, "-"),
                Item(3, 3, 3, None, None, ">"))

        assert item_left_adjoin(
            Item(2, 2, 2, 0, 2, "-"),
            Item(1, 1, 1, None, None, ">")
        ) == Item(1, 2, 2, 0, 2, "-")

        assert item_left_adjoin(
            Item(3, 3, 3, 1, 2, "-"),
            Item(2, 2, 2, None, None, ">")
        ) == Item(2, 3, 3, 1, 2, "-")

    def test_item_right_adjoin(self):
        with pytest.raises(AssertionError):
            item_right_adjoin(
                Item(2, 2, 2, None, None, "-"),
                Item(3, 3, 3, None, None, "<"))

        with pytest.raises(AssertionError):
            item_right_adjoin(
                Item(2, 2, 2, 1, 2, "-"),
                Item(3, 3, 3, None, None, "<"))

        with pytest.raises(AssertionError):
            item_right_adjoin(
                Item(2, 2, 2, None, 2, "-"),
                Item(3, 3, 3, 0, 0, "<"))

        with pytest.raises(AssertionError):
            item_right_adjoin(
                Item(2, 2, 2, None, 2, "-"),
                Item(3, 3, 3, None, 1, "<"))

        with pytest.raises(AssertionError):
            item_right_adjoin(
                Item(2, 2, 2, None, 2, "-"),
                Item(1, 1, 1, None, None, "<"))

        assert item_right_adjoin(
            Item(2, 2, 2, None, 2, "<"),
            Item(3, 3, 3, None, None, "<")
        ) == Item(2, 3, 2, None, 2, "<")


class TestSystem:
    def test_compute_outside_estimates(self):
        supertag_scores = np.array([
            [0.1, 0.5, 0.3, 0.1],
            [0.4, 0.1, 0.5, 0.0],
            [0.25, 0.05, 0.2, 0.5],
            [0.25, 0.05, 0.05, 0.65],
            [0.25, 0.25, 0.25, 0.25],
        ])

        head_scores = np.array([
            [0.25, 0.05, 0.2, 0.5, 1.0],
            [0.25, 0.05, 0.05, 0.65, 1.0],
            [0.25, 0.25, 0.25, 0.25, 1.0],
            [0.1, 0.5, 0.3, 0.1, 1.0],
            [0.4, 0.1, 0.5, 0.0, 1.0],
        ])

        estimates = np.array([
            [0.75, 0.70, 0.40, 0.25, 0.00],
            [0.90, 0.85, 0.55, 0.40, 0.15],
            [0.90, 0.90, 0.60, 0.45, 0.20],
            [0.90, 0.90, 0.90, 0.75, 0.50],
            [0.90, 0.90, 0.90, 0.90, 0.65],
        ])

        assert np.allclose(
            np.triu(System.compute_outside_estimates(
                supertag_scores, head_scores)),
            np.triu(estimates)
            )

    def test_compute_advanced_outside_estimates(self):
        # TODO: add anchor head estimate check
        supertag_scores = np.array([
            [0.1, 0.5],  # 0.1
            [0.4, 0.1],  # 0.1
            [0.25, 0.05],  # 0.05
        ])

        head_scores = np.array([
            [0.25, 0.05, 0.2],
            [0.25, 0.05, 0.05],
            [0.25, 0.25, 0.25],
        ])

        # <1,1,1>: (0.05,) 0.05, 0.25 -> 0.3 + 0.15 = 0.45
        # <2,2,2>: 0.05, (0.05,) 0.25 -> 0.3 + 0.15 = 0.45
        # <3,3,3>: 0.05, 0.05, (0.25) -> 0.1 + 0.2 = 0.3
        # <1,2,1>: (0.2, 0.05,) 0.25 -> 0.25 + 0.05 = 0.3
        # <1,2,2>: (0.05, 0.05,) 0.25 -> 0.25 + 0.05 = 0.3
        # <2,3,2>: 0.05, (0.05, 0.25) -> 0.05 + 0.1 = 0.15
        # <2,3,3>: 0.2, (0.05, 0.25) -> 0.2 + 0.1 = 0.3

        estimates = System.compute_advanced_outside_estimates(
                supertag_scores, head_scores)

        assert np.isclose(estimates[0, 0, 0], 0.45)
        assert np.isclose(estimates[2, 2, 2], 0.30)
        assert np.isclose(estimates[0, 1, 0], 0.30)
        assert np.isclose(estimates[1, 2, 1], 0.15)
        assert np.isclose(estimates[1, 2, 2], 0.30)

    def test_get_attachment_weight(self):
        supertag_scores = np.array([
            [0.1, 0.5],  # 0.1
            [0.4, 0.1],  # 0.1
            [0.25, 0.05],  # 0.05
        ])

        head_scores = np.array([
            [0.25, 0.05, 0.2],
            [0.25, 0.05, 0.05],
            [0.25, 0.25, 0.25],
        ])

        system = System(
            head_scores, supertag_scores,
            {}, 0, 0, 2, 2, "advanced"
        )

        assert system.get_attachment_weight(
            Item(1, 1, 1, 0, 0, "-"),
            Item(2, 2, 2, None, None, "<")
        ) == np.inf

        assert system.get_attachment_weight(
            Item(2, 2, 2, 1, 0, "-"),
            Item(1, 1, 1, None, None, "-")
        ) == 0.05

        with pytest.raises(AssertionError):
            system.get_attachment_weight(
                Item(2, 2, 2, 1, 0, "-"),
                Item(None, None, None, None, None, None)
            )

        with pytest.raises(AssertionError):
            system.get_attachment_weight(
                Item(None, None, None, None, None, None),
                Item(1, 1, 1, None, None, "-")
            )

    def test_get_item_weight_pointer_pair(self):
        supertag_scores = np.array([
            [0.1, 0.5],  # 0.1
            [0.4, 0.1],  # 0.1
            [0.25, 0.05],  # 0.05
        ])

        head_scores = np.array([
            [0.25, 0.05, 0.2],
            [0.25, 0.05, 0.05],
            [0.25, 0.25, 0.25],
        ])

        system = System(
            head_scores, supertag_scores,
            {}, 0, 0, 2, 2, "advanced"
        )

        item, weight = system.get_item_weight_pointer_pair(
            Item(1, 1, 1, None, 0, "-"),
            back1=Item(1, 1, 1, 0, 0, "-"),
            weight1=Weight(0.1, 0.5))
        assert item == Item(1, 1, 1, None, 0, "-")
        assert weight.back1 == Item(1, 1, 1, 0, 0, "-")
        assert weight.back2 == AXIOM
        assert np.isclose(weight.inside, 0.1)

        item, weight = system.get_item_weight_pointer_pair(
            Item(1, 2, 1, None, 0, "-"),
            back1=Item(1, 1, 1, None, 1, "-"),
            weight1=Weight(0.1, 0.5),
            back2=Item(2, 2, 2, None, None, "-"),
            weight2=Weight(0.5, 0.15),
            supertag_ind=2)
        assert item == Item(1, 2, 1, None, 0, "-")
        assert weight.back1 == Item(1, 1, 1, None, 1, "-")
        assert weight.back2 == Item(2, 2, 2, None, None, "-")
        assert np.isclose(weight.inside, 0.1+0.5+np.inf)

        item, weight = system.get_item_weight_pointer_pair(
            Item(1, 2, 2, 0, 1, "-"),
            back1=Item(2, 2, 2, 1, 1, "-"),
            weight1=Weight(0.1, 0.5),
            back2=Item(1, 1, 1, None, None, "-"),
            weight2=Weight(0.5, 0.15),
            supertag_ind=2)
        assert np.isclose(weight.inside, 0.1+0.5+0.05)

    def test_run(self):
        # head, arg, adj

        supertag_scores = np.array([
            [0.1, 0.5, 0.6],  # *+dep
            [0.5, 0.2, 0.4],  # *
            [0.25, 0.7, 0.6],  # -aux*
        ])

        head_scores = np.array([
            [0.05, 0.25, 0.6],
            [0.45, 0.4, 0.3],
            [0.20, 0.15, 0.0],
        ])

        mapping = {
            0: ((None, "*"), (True, "dep"),),
            1: ((None, "*"),),
            2: ((False, "aux"), (None, "*"),),
        }

        system = System(
            head_scores, supertag_scores,
            mapping, 2, 2, 3, 3, "advanced"
        )

        result = system.run(printinfo=False)

        assert result[0] == Item(1, 3, 1, None, None, "-")
        assert np.isclose(result[1].inside, 0.1+0.2+0.6+0.45+0.15)

        system = System(
            head_scores, supertag_scores,
            mapping, 2, 2, 2, 2, "advanced"
        )

        result = system.run(printinfo=False)

        assert result is None

    def test_axiom(self):
        # head, arg, adj

        supertag_scores = np.array([
            [0.6, 0.5, 0.1],  # *+dep
            [0.5, 0.2, 0.4],  # *
            [0.25, 0.1, 0.6],  # -aux*
        ])

        head_scores = np.array([
            [0.05, 0.25, 0.6],
            [0.45, 0.4, 0.3],
            [0.20, 0.15, 0.0],
        ])

        mapping = {
            0: ((None, "*"), (True, "dep"), (False, "aux"),),
            1: ((None, "*"),),
            2: ((False, "aux"), (None, "*"), (True, "dep"),),
        }

        system = System(
            head_scores, supertag_scores,
            mapping, 2, 2, 2, 2, "advanced"
        )

        axioms = system.axiom()
        items = {ax[0] for ax in axioms}

        assert items == {
            Item(1, 1, 1, 0, 0, N),
            Item(2, 2, 2, 0, 0, N),
            Item(2, 2, 2, 0, 1, L),
            Item(3, 3, 3, 0, 0, N)
        }

        supertag_scores = np.array([
            [0.1, 0.2, 0.2],  # *+dep
            [0.5, 0.1, 0.4],  # *
            [0.25, 0.7, 0.1],  # -aux*
        ])

        mapping = {
            0: ((None, "*"), (True, "dep"), (False, "aux"),),
            1: ((None, "*"),),
            2: ((True, "dep"), (True, "dep"), (None, "*"),),
        }

        system = System(
            head_scores, supertag_scores,
            mapping, 2, 2, 1, 1, "advanced"
        )
        axioms = system.axiom()
        items = {ax[0] for ax in axioms}

        assert items == {
            Item(1, 1, 1, 0, 1, R),
            Item(2, 2, 2, 0, 0, N),
            Item(3, 3, 3, 2, 0, N),
        }

    def test_backtrack(self):
        # head, arg, adj

        supertag_scores = np.array([
            [0.1, 0.5, 0.6],  # *+dep
            [0.5, 0.2, 0.4],  # *
            [0.25, 0.7, 0.6],  # -aux*
        ])

        head_scores = np.array([
            [0.05, 0.25, 0.6],
            [0.45, 0.4, 0.3],
            [0.20, 0.15, 0.0],
        ])

        mapping = {
            0: ((None, "*"), (True, "dep"),),
            1: ((None, "*"),),
            2: ((False, "aux"), (None, "*"),),
        }

        system = System(
            head_scores, supertag_scores,
            mapping, 2, 2, 3, 3, "advanced"
        )

        result = system.run(printinfo=False)

        assert result[0] == Item(1, 3, 1, None, None, "-")

        heads, action, supertags, relations = system.backtrack(result[1])

        assert heads == [0, 1, 2]
        assert action == [0, 0, 1]
        assert supertags == [0, 1, 2]
        assert relations == ["root", "dep", "aux"]

    # TODO: test backtrack_disconnected
    # TODO: remove backtrack pad

    def test_switch(self):
        # head, arg, adj

        supertag_scores = np.array([
            [0.6, 0.5, 0.1],  # *+dep
            [0.5, 0.2, 0.4],  # *
            [0.25, 0.1, 0.6],  # -aux*
        ])

        head_scores = np.array([
            [0.05, 0.25, 0.6],
            [0.45, 0.4, 0.3],
            [0.20, 0.15, 0.0],
        ])

        mapping = {
            0: ((None, "*"), (True, "dep"), (False, "aux"),),
            1: ((None, "*"),),
            2: ((False, "aux"), (None, "*"), (True, "dep"),),
        }

        system = System(
            head_scores, supertag_scores,
            mapping, 2, 2, 2, 2, "advanced"
        )

        system.run()

        item = Item(2, 2, 2, 0, 0, N)
        result = system.switch(item, system._chart[item])
        assert len(result) == 1
        assert result[0][0] == Item(2, 2, 2, None, 0, N)
        assert np.isclose(result[0][1].inside, system._chart[item].inside)
        assert np.isclose(
            result[0][1].out_estimate, system._chart[item].out_estimate)

    def test_complete(self):
        # head, arg, adj

        supertag_scores = np.array([
            [0.1, 0.5, 0.1],  # *+dep
            [0.5, 0.2, 0.4],  # *
            [0.25, 0.1, 0.05],  # -aux*
        ])

        head_scores = np.array([
            [0.05, 0.25, 0.6],
            [0.2, 0.4, 0.3],
            [0.20, 0.15, 0.0],
        ])

        mapping = {
            0: ((None, "*"), (True, "dep"),),
            1: ((None, "*"),),
            2: ((False, "aux"), (None, "*"),),
        }

        system = System(
            head_scores, supertag_scores,
            mapping, 2, 2, 2, 2, "advanced"
        )

        system.run()

        item = Item(1, 3, 1, None, 0, N)
        result = system.complete(item, system._chart[item])
        assert len(result) == 1
        assert result[0][0] == Item(1, 3, 1, None, None, N)
        assert np.isclose(result[0][1].inside, system._chart[item].inside)
        assert np.isclose(
            result[0][1].out_estimate, system._chart[item].out_estimate)

    def test_right_subst_head(self):

        supertag_scores = np.array([
            [0.1, 0.5, 0.1],  # *+dep
            [0.5, 0.2, 0.4],  # *
            [0.25, 0.1, 0.05],  # -aux*
        ])

        head_scores = np.array([
            [0.05, 0.25, 0.6],
            [0.2, 0.4, 0.3],
            [0.20, 0.15, 0.0],
        ])

        mapping = {
            0: ((None, "*"), (False, "aux"),),
            1: ((None, "*"),),
            2: ((False, "aux"), (None, "*"),),
        }

        system = System(
            head_scores, supertag_scores,
            mapping, 2, 2, 2, 2, "advanced"
        )

        system.run(printinfo=True)

        assert system._chart.peek(Item(2, 2, 2, None, None, N)) is not None
        assert system._chart.peek(Item(2, 3, 2, None, None, N)) is not None

        item = Item(1, 1, 1, None, 1, N)
        result = system.right_subst_head(
            item, WeightPointer(0.1, 0.0, AXIOM, AXIOM, 0))
        assert len(result) == 2
        result_items = [tup[0] for tup in result]
        result_inside_weights = {tup[1].inside for tup in result}
        result_outside_weights = {tup[1].out_estimate for tup in result}
        assert Item(1, 2, 1, None, 0, N) in result_items
        assert Item(1, 3, 1, None, 0, N) in result_items

        assert len(result_inside_weights) == 2
        assert any([np.isclose(0.4+0.1, x) for x in result_inside_weights])
        assert any([np.isclose(0.6+0.1, x) for x in result_inside_weights])

        assert len(result_outside_weights) == 2
        assert any([np.isclose(
            system.get_outside_estimate(Item(1, 2, 1, None, 0, N)), x)
            for x in result_outside_weights])
        assert any([np.isclose(
            system.get_outside_estimate(Item(1, 3, 1, None, 0, N)), x)
            for x in result_outside_weights])

        item = Item(1, 1, 1, None, 1, R)
        result = system.right_subst_head(
            item, WeightPointer(0.0, 0.0, AXIOM, AXIOM, 0))
        assert len(result) == 1
        assert Item(1, 2, 1, None, 0, R) == result[0][0]

    def test_left_subst_head(self):
        # TODO

        supertag_scores = np.array([
            [0.5, 0.05, 0.1],  # *+dep
            [0.5, 0.2, 0.4],  # *
            [0.25, 0.1, 0.05],  # -aux*
        ])

        head_scores = np.array([
            [0.05, 0.25, 0.2],
            [0.2, 0.4, 0.3],
            [0.20, 0.15, 0.85],
        ])

        mapping = {
            0: ((None, "*"), (False, "aux"),),
            1: ((None, "*"),),
            2: ((False, "aux"), (None, "*"),),
        }

        system = System(
            head_scores, supertag_scores,
            mapping, 2, 2, 2, 2, "advanced"
        )

        system.run(printinfo=True)

        assert system._chart.peek(Item(2, 2, 2, None, None, N)) is not None
        assert system._chart.peek(Item(1, 2, 1, None, None, N)) is not None

        item = Item(3, 3, 3, 1, 0, N)
        result = system.left_subst_head(
            item, WeightPointer(0.1, 0.0, AXIOM, AXIOM, 0))
        assert len(result) == 2
        result_items = [tup[0] for tup in result]
        result_inside_weights = {tup[1].inside for tup in result}
        result_outside_weights = {tup[1].out_estimate for tup in result}
        assert Item(2, 3, 3, 0, 0, N) in result_items
        assert Item(1, 3, 3, 0, 0, N) in result_items

        assert len(result_inside_weights) == 2
        assert any([np.isclose(0.2+0.3+0.1, x) for x in result_inside_weights])
        assert any(
            [np.isclose(
                0.05+0.4+0.2+0.2+0.1, x) for x in result_inside_weights])

        assert len(result_outside_weights) == 2
        assert any([np.isclose(
            system.get_outside_estimate(Item(2, 3, 3, 0, 0, N)), x)
            for x in result_outside_weights])
        assert any([np.isclose(
            system.get_outside_estimate(Item(1, 3, 3, 0, 0, N)), x)
            for x in result_outside_weights])

        item = Item(3, 3, 3, 1, 0, L)
        result = system.left_subst_head(
            item, WeightPointer(0.0, 0.0, AXIOM, AXIOM, 0))
        assert len(result) == 1
        assert Item(2, 3, 3, 0, 0, L) == result[0][0]

# TODO: the four double item functions
# TODO: test head setting
