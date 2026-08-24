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


# --- alignment and dedupe --------------------------------------------------


def test_align_restricts_to_shared_labels(tmp_path):
    a = write_table(tmp_path / "a.parquet", long_rows(["x", "y", "z"], "m1", 2))
    b = write_table(tmp_path / "b.parquet", long_rows(["y", "z", "w"], "m2", 2))
    spaces = load_spaces(a) | load_spaces(b)
    aligned, labels = align_spaces(spaces)
    assert labels == ["y", "z"]
    assert all(s.X.shape[0] == 2 for s in aligned.values())


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


def test_dedupe_keeps_genuinely_different_spaces(tmp_path):
    a = write_table(tmp_path / "a.parquet", long_rows(["x", "y"], "m1", 3))
    b = write_table(
        tmp_path / "b.parquet",
        long_rows(["x", "y"], "m2", 3, value=lambda s, r, d: float(s * 7 + d)),
    )
    spaces = load_spaces(a) | load_spaces(b)
    assert len(dedupe_spaces(spaces)) == 2
