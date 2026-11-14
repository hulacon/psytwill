"""Packing short-file corpora into grid-aligned extraction units."""

import pytest

from psytwill.staging import (
    Segment,
    item_units,
    layout_unit,
    normalize_gigaspeech_text,
    normalize_librispeech_text,
    pack_groups,
)


def seg(i, group, dur, text=""):
    return Segment(segment=f"s{i:03d}", group=group, duration=dur, text=text)


class TestLayoutUnit:
    def test_every_onset_on_grid_and_gap_respected(self):
        segs = [seg(0, "g", 1.3), seg(1, "g", 0.51), seg(2, "g", 7.0)]
        u = layout_unit("u000", segs, hop=0.5, gap=0.5)
        assert u.onsets == [0.0, 2.0, 3.5]
        for k in range(1, len(segs)):
            assert u.onsets[k] - u.offsets[k - 1] >= 0.5 - 1e-9
            assert abs(u.onsets[k] / 0.5 - round(u.onsets[k] / 0.5)) < 1e-9
        assert u.duration == 11.0  # 3.5 + 7.0 + 0.5 gap, already on grid

    def test_unit_length_is_whole_frames(self):
        u = layout_unit("u", [seg(0, "g", 0.48)], hop=0.5, gap=0.5)
        assert u.duration == 1.0

    def test_empty(self):
        u = layout_unit("u", [], hop=0.5, gap=0.5)
        assert u.duration == 0.0 and u.onsets == []

    def test_bad_params(self):
        with pytest.raises(ValueError):
            layout_unit("u", [], hop=0.0)


class TestPackGroups:
    def test_groups_never_split_and_target_respected(self):
        groups = [(f"g{k}", [seg(k * 10 + j, f"g{k}", 4.0) for j in range(4)])
                  for k in range(5)]  # each group ~ 4 x (4 + 0.5) = 18 s
        units = pack_groups(groups, target=40.0)
        assert [len(u.segments) for u in units] == [8, 8, 4]
        for u in units:
            assert u.duration <= 40.0
            groups_in = {s.group for s in u.segments}
            for g in groups_in:  # whole group present
                assert sum(s.group == g for s in u.segments) == 4

    def test_oversize_group_gets_its_own_unit(self):
        groups = [("a", [seg(0, "a", 2.0)]), ("big", [seg(1, "big", 100.0)]),
                  ("b", [seg(2, "b", 2.0)])]
        units = pack_groups(groups, target=10.0)
        assert [[s.group for s in u.segments] for u in units] == [["a"], ["big"], ["b"]]

    def test_deterministic_ids_with_prefix(self):
        groups = [("a", [seg(0, "a", 1.0)]), ("b", [seg(1, "b", 1.0)])]
        units = pack_groups(groups, target=1.0, unit_prefix="spk12-")
        assert [u.unit for u in units] == ["spk12-000", "spk12-001"]
        assert pack_groups(groups, target=1.0, unit_prefix="spk12-")[0].onsets == units[0].onsets

    def test_group_mismatch_refused(self):
        with pytest.raises(ValueError, match="carries group"):
            pack_groups([("a", [seg(0, "b", 1.0)])], target=5.0)

    def test_empty_groups_skipped(self):
        assert pack_groups([("a", [])], target=5.0) == []


class TestItemUnits:
    """E5's "re-extract per item" (DECIDED 2026-11-13): one unit per item, no neighbour."""

    def test_one_unit_per_group(self):
        groups = [("w1", [seg(0, "w1", 0.6)]), ("w2", [seg(1, "w2", 0.7)]), ("w3", [seg(2, "w3", 1.2)])]
        units = item_units(groups, unit_prefix="echo-")
        assert [u.unit for u in units] == ["echo-w1", "echo-w2", "echo-w3"]
        assert [[s.group for s in u.segments] for u in units] == [["w1"], ["w2"], ["w3"]]
        assert all(u.onsets[0] == 0.0 for u in units)

    def test_no_trailing_gap_only_grid_rounding(self):
        # a packed unit ends one gap after the item (1.0 s here); an item unit
        # ends on the first grid point at or after it, so < one hop of padding
        u = item_units([("w", [seg(0, "w", 0.48)])], hop=0.5, gap=0.5)[0]
        assert u.duration == 0.5
        assert item_units([("w", [seg(0, "w", 1.0)])], hop=0.5)[0].duration == 1.0
        assert layout_unit("u", [seg(0, "w", 0.48)], hop=0.5, gap=0.5).duration == 1.0  # unchanged

    def test_a_multi_file_group_keeps_its_internal_gap(self):
        u = item_units([("ch", [seg(0, "ch", 1.3), seg(1, "ch", 0.51)])], hop=0.5, gap=0.5)[0]
        assert u.onsets == [0.0, 2.0] and u.duration == 3.0  # 2.0 + 0.51, up to the grid

    def test_ids_are_native_ids_and_unique(self):
        with pytest.raises(ValueError, match="native id"):
            item_units([("Bad Key", [seg(0, "Bad Key", 1.0)])])
        with pytest.raises(ValueError, match="twice"):
            item_units([("a", [seg(0, "a", 1.0)]), ("a", [seg(1, "a", 1.0)])])

    def test_group_mismatch_refused_and_empty_skipped(self):
        with pytest.raises(ValueError, match="carries group"):
            item_units([("a", [seg(0, "b", 1.0)])])
        assert item_units([("a", [])]) == []


class TestTextNormalisation:
    def test_gigaspeech_tokens(self):
        t = "HELLO <COMMA> WORLD <PERIOD> <SIL> REALLY <QUESTIONMARK> YES <EXCLAMATIONPOINT>"
        assert normalize_gigaspeech_text(t) == "hello, world. really? yes!"

    def test_librispeech_lowercases_only(self):
        assert normalize_librispeech_text("THE  QUICK BROWN ") == "the quick brown"
