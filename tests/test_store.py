"""Reading the long-form table back into per-space matrices.

Offline and synthetic, like the rest of the suite: the fixtures below are
miniatures of the three shapes the real store actually contains — a plain
image-level group, a group carrying a declared-string family, and a group with
replicate rows per stimulus.
"""

import numpy as np
import pandas as pd
import pytest

from psytwill.exceptions import InputError, SpaceError
from psytwill.store import (
    LoadReport,
    align_spaces,
    dedupe_spaces,
    load_spaces,
    model_inventory,
)


def write_table(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def long_rows(stimuli, model, dim, *, value=None, replicates=1, string=False):
    """Long-form rows for one model, optionally with replicate rows per stimulus."""
    out = []
    for s_i, s in enumerate(stimuli):
        for r in range(replicates):
            for d in range(dim):
                v = value(s_i, r, d) if callable(value) else float(s_i + d)
                out.append(
                    {
                        "stimulus_id": s,
                        "chunk_idx": s_i * replicates + r,
                        "modality": "visual",
                        "extractor": "viz2psy",
                        "model": model,
                        "feature": f"{model}_{d:03d}",
                        "value": None if string else v,
                        "value_str": "a caption" if string else None,
                    }
                )
    return out


@pytest.fixture
def image_table(tmp_path):
    rng = np.random.RandomState(0)
    stimuli = [f"img{i:03d}" for i in range(12)]
    rows = long_rows(stimuli, "clip", 4, value=lambda s, r, d: rng.randn())
    rows += long_rows(stimuli, "resmem", 1, value=lambda s, r, d: float(s))
    rows += long_rows(stimuli, "caption", 1, string=True)
    return write_table(tmp_path / "image.parquet", rows)


def test_inventory_flags_string_families(image_table):
    inv = model_inventory(image_table).set_index("model")
    assert bool(inv.loc["caption", "is_string"]) is True
    assert bool(inv.loc["clip", "is_string"]) is False


def test_string_families_are_not_loaded_as_spaces(image_table):
    rep = LoadReport()
    spaces = load_spaces(image_table, report=rep)
    assert "caption" not in spaces
    assert rep.skipped_string == ["caption"]
    assert set(spaces) == {"clip", "resmem"}


def test_features_are_ordered_and_rows_are_stimuli(image_table):
    s = load_spaces(image_table)["clip"]
    assert s.features == ["clip_000", "clip_001", "clip_002", "clip_003"]
    assert s.X.shape == (12, 4)
    assert s.labels[:2] == ["img000", "img001"]
    assert s.n_replicates == 1


def test_prefix_namespaces_spaces(image_table):
    assert set(load_spaces(image_table, prefix="image")) == {
        "image:clip",
        "image:resmem",
    }


def test_missing_key_column_names_the_available_ones(image_table):
    with pytest.raises(SpaceError, match="stimulus_id"):
        load_spaces(image_table, key=("time",))


def test_absent_file_is_an_input_error(tmp_path):
    with pytest.raises(InputError, match="No such feature table"):
        load_spaces(tmp_path / "nope.parquet")


# --- replicates ------------------------------------------------------------


@pytest.fixture
def replicate_table(tmp_path):
    """Five 'captions' per image: same stimulus_id, distinct chunk_idx."""
    stimuli = [f"img{i:03d}" for i in range(6)]
    rows = long_rows(
        stimuli, "minilm", 3, replicates=5, value=lambda s, r, d: float(s + r)
    )
    return write_table(tmp_path / "reps.parquet", rows)


def test_replicates_pool_to_one_row_per_stimulus(replicate_table):
    rep = LoadReport()
    s = load_spaces(replicate_table, report=rep)["minilm"]
    assert s.X.shape == (6, 3)          # rows stay stimuli, not captions
    assert s.n_replicates == 5
    assert rep.pooled == {"minilm": 5}
    # mean over r in 0..4 of (s + r) = s + 2
    assert s.X[0, 0] == pytest.approx(2.0)
    assert s.X[3, 0] == pytest.approx(5.0)


def test_refusing_to_pool_states_the_fix(replicate_table):
    with pytest.raises(SpaceError, match="chunk_idx"):
        load_spaces(replicate_table, pool=None)


def test_keying_finer_keeps_replicates_apart(replicate_table):
    s = load_spaces(replicate_table, key=("stimulus_id", "chunk_idx"))["minilm"]
    assert s.X.shape == (30, 3)         # 6 stimuli x 5 captions
    assert "|" in s.labels[0]


# --- temporal windows ------------------------------------------------------


def grid_rows(stimuli, model, dim, times, *, value=None):
    """Long-form rows for one model on a temporal grid."""
    out = []
    for s_i, s in enumerate(stimuli):
        for t_i, t in enumerate(times):
            for d in range(dim):
                v = value(s_i, t_i, d) if callable(value) else float(t_i)
                out.append(
                    {
                        "stimulus_id": s,
                        "time": float(t),
                        "modality": "visual",
                        "extractor": "viz2psy",
                        "model": model,
                        "feature": f"{model}_{d:03d}",
                        "value": v,
                        "value_str": None,
                    }
                )
    return out


def test_window_pools_rows_and_records_the_count(tmp_path):
    # 0.5 s grid, 4 bins; window=1.0 pools bins pairwise
    t = write_table(
        tmp_path / "grid.parquet",
        grid_rows(["clipA"], "m1", 2, [0.0, 0.5, 1.0, 1.5]),
    )
    rep = LoadReport()
    s = load_spaces(t, key=("stimulus_id", "time"), window=1.0, report=rep)["m1"]
    assert s.X.shape == (2, 2)
    assert s.labels == ["clipA|0.0", "clipA|1.0"]
    assert s.n_replicates == 2
    assert rep.pooled == {"m1": 2}
    # bins 0,1 mean to 0.5 and bins 2,3 to 2.5 (value = bin index)
    assert s.X[:, 0] == pytest.approx([0.5, 2.5])


def test_window_reconciles_bin_start_and_bin_center_stamps(tmp_path):
    # The real store: visual groups stamp 0.0, 0.5, ... and audio groups
    # stamp 0.25, 0.75, ... — unbinned, the two grids share no labels.
    vis = write_table(
        tmp_path / "vis.parquet", grid_rows(["c"], "m1", 2, [0.0, 0.5, 1.0])
    )
    aud = write_table(
        tmp_path / "aud.parquet", grid_rows(["c"], "m2", 2, [0.25, 0.75, 1.25])
    )
    key = ("stimulus_id", "time")
    with pytest.raises(SpaceError, match="share no labels"):
        align_spaces(
            load_spaces(vis, key=key) | load_spaces(aud, key=key)
        )
    spaces = load_spaces(vis, key=key, window=0.5) | load_spaces(
        aud, key=key, window=0.5
    )
    _, labels = align_spaces(spaces)
    assert labels == ["c|0.0", "c|0.5", "c|1.0"]


def test_window_without_time_in_key_states_the_fix(image_table):
    with pytest.raises(SpaceError, match="'time'"):
        load_spaces(image_table, window=0.5)


# --- alignment and dedupe --------------------------------------------------


def test_align_restricts_to_shared_labels(tmp_path):
    a = write_table(tmp_path / "a.parquet", long_rows(["x", "y", "z"], "m1", 2))
    b = write_table(tmp_path / "b.parquet", long_rows(["y", "z", "w"], "m2", 2))
    spaces = load_spaces(a) | load_spaces(b)
    aligned, labels = align_spaces(spaces)
    assert labels == ["y", "z"]
    assert all(s.X.shape[0] == 2 for s in aligned.values())


def test_align_preserves_temporal_order_not_string_order(tmp_path):
    # "c|100.5" string-sorts before "c|12.5"; a block-permutation null on
    # string-sorted rows would shuffle blocks that are not temporal blocks.
    times = [2.0, 12.5, 100.5]
    a = write_table(tmp_path / "a.parquet", grid_rows(["c"], "m1", 2, times))
    b = write_table(tmp_path / "b.parquet", grid_rows(["c"], "m2", 2, times))
    key = ("stimulus_id", "time")
    _, labels = align_spaces(load_spaces(a, key=key) | load_spaces(b, key=key))
    assert labels == ["c|2.0", "c|12.5", "c|100.5"]


def test_align_refuses_disjoint_sets(tmp_path):
    a = write_table(tmp_path / "a.parquet", long_rows(["x"], "m1", 2))
    b = write_table(tmp_path / "b.parquet", long_rows(["q"], "m2", 2))
    with pytest.raises(SpaceError, match="share no labels"):
        align_spaces(load_spaces(a) | load_spaces(b))


def test_dedupe_drops_the_same_space_under_a_second_name(tmp_path):
    rows = long_rows(["x", "y", "z"], "ebind", 4)
    a = write_table(tmp_path / "grp_a.parquet", rows)
    b = write_table(tmp_path / "grp_b.parquet", rows)  # byte-identical
    spaces = load_spaces(a, prefix="image") | load_spaces(b, prefix="ebindgrp")
    rep = LoadReport()
    kept = dedupe_spaces(spaces, report=rep)
    assert set(kept) == {"image:ebind"}
    assert rep.deduped == {"ebindgrp:ebind": "image:ebind"}


def test_dedupe_survives_differing_dictionary_order(tmp_path):
    # Two files, same rows, opposite row order on disk. pyarrow gives each
    # file its own category order, so without lexical re-ordering the two
    # copies load in different row orders and dedupe misses them — exactly
    # how the movie ebind duplicate slipped through the Phase-0 probe.
    rows = grid_rows(["clipA", "clipB"], "ebind", 3, [0.0, 0.5])
    a = write_table(tmp_path / "grp_a.parquet", rows)
    b = write_table(tmp_path / "grp_b.parquet", list(reversed(rows)))
    key = ("stimulus_id", "time")
    spaces = load_spaces(a, key=key, prefix="frames") | load_spaces(
        b, key=key, prefix="ebindgrp"
    )
    rep = LoadReport()
    kept = dedupe_spaces(spaces, report=rep)
    assert set(kept) == {"frames:ebind"}
    assert rep.deduped == {"ebindgrp:ebind": "frames:ebind"}


def test_dedupe_keeps_genuinely_different_spaces(tmp_path):
    a = write_table(tmp_path / "a.parquet", long_rows(["x", "y"], "m1", 3))
    b = write_table(
        tmp_path / "b.parquet",
        long_rows(["x", "y"], "m2", 3, value=lambda s, r, d: float(s * 7 + d)),
    )
    spaces = load_spaces(a) | load_spaces(b)
    assert len(dedupe_spaces(spaces)) == 2


def provenance_rows(stimuli, model, suffix="_n_pooled", scale=100.0):
    """One prefixed bookkeeping column per stimulus, on a much larger scale."""
    return [
        {
            "stimulus_id": s,
            "chunk_idx": i,
            "modality": "text",
            "extractor": "word2psy",
            "model": model,
            "feature": f"{model}{suffix}",
            "value": scale * (i + 1),
            "value_str": None,
        }
        for i, s in enumerate(stimuli)
    ]


class TestProvenanceColumns:
    """``{model}_n_pooled`` is provenance, not a dimension (Contract B §4.1 gap)."""

    def _table(self, tmp_path, model="fasttext"):
        stimuli = [f"img{i:03d}" for i in range(8)]
        rng = np.random.RandomState(1)
        rows = long_rows(stimuli, model, 4, value=lambda s, r, d: rng.randn())
        rows += provenance_rows(stimuli, model)
        return write_table(tmp_path / "prov.parquet", rows), stimuli

    def test_dropped_from_the_matrix_and_reported(self, tmp_path):
        path, stimuli = self._table(tmp_path)
        rep = LoadReport()
        spaces = load_spaces(path, report=rep)
        space = spaces["fasttext"]
        assert space.dim == 4
        assert space.features == [f"fasttext_{d:03d}" for d in range(4)]
        assert rep.dropped_provenance == {"fasttext": ["fasttext_n_pooled"]}
        assert space.n == len(stimuli)

    def test_the_count_would_otherwise_be_pc1(self, tmp_path):
        """The regression this guards: a large-scale count dominates the spectrum."""
        path, _ = self._table(tmp_path)
        X = load_spaces(path)["fasttext"].X
        counts = np.arange(1, X.shape[0] + 1, dtype=float) * 100.0

        def top_share(M):
            L = np.linalg.svd(M - M.mean(0), compute_uv=False) ** 2
            return L[0] / L.sum()

        assert top_share(X) < 0.6
        assert top_share(np.column_stack([X, counts])) > 0.95

    def test_a_provenance_only_model_is_skipped_not_emptied(self, tmp_path):
        stimuli = [f"img{i:03d}" for i in range(5)]
        path = write_table(
            tmp_path / "only.parquet", provenance_rows(stimuli, "lonely")
        )
        rep = LoadReport()
        spaces = load_spaces(path, report=rep)
        assert "lonely" not in spaces
        assert rep.skipped_empty == ["lonely"]
        assert rep.dropped_provenance == {}

    def test_a_real_feature_ending_in_a_digit_run_is_untouched(self, tmp_path):
        """Only the declared suffixes go; ordinary features are never guessed at."""
        stimuli = [f"img{i:03d}" for i in range(6)]
        rows = long_rows(stimuli, "llstat", 3)
        rows += [
            {
                "stimulus_id": s,
                "chunk_idx": i,
                "modality": "visual",
                "extractor": "viz2psy",
                "model": "llstat",
                "feature": "llstat_n_edges",
                "value": float(i),
                "value_str": None,
            }
            for i, s in enumerate(stimuli)
        ]
        rep = LoadReport()
        spaces = load_spaces(write_table(tmp_path / "kept.parquet", rows), report=rep)
        assert "llstat_n_edges" in spaces["llstat"].features
        assert rep.dropped_provenance == {}
