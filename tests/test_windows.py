"""Read-time window handling (Contract B 1.4): item masks and the causal shift.

Synthetic packed units: each unit holds two items (groups) of 9.5 s with the
staging gap of 0.5 s, on a 0.5 s grid. Member ``a`` reads a centred 4 s window
on every column; member ``b`` reads only its grid step.
"""

import json

import numpy as np
import pandas as pd
import pytest

from psytwill.cli import main
from psytwill.exceptions import InputError
from psytwill.store import SpaceMatrix
from psytwill.windows import (causal_shifts, item_masks, numeric_windows, read_absent, record_table_segments,
                              shift_columns, table_segments)

HOP, N_UNITS, N_BINS = 0.5, 8, 40  # 20 s units
SEG = pd.DataFrame({"onset": [0.0, 10.0], "offset": [9.5, 19.5], "group": ["g1", "g2"]})
LAB = {"interpretability": "opaque", "derivation": "embedding"}


def _space(n_units=1, w=4.0):
    labels = [f"u{u:02d}|{b * HOP}" for u in range(n_units) for b in range(N_BINS)]
    X = np.tile(np.arange(N_BINS, dtype=float), n_units)[:, None]
    return SpaceMatrix(name="a", labels=labels, X=X, features=["a_00"])


def test_item_mask_marks_rows_whose_window_reads_two_groups():
    sp = _space()
    masks = item_masks(sp, {"a_00": (4.0, "centred")}, {"u00": SEG}, HOP)
    hit = [lab.split("|")[1] for lab, h in zip(sp.labels, masks["a_00"]) if h]
    # window [c-2, c+2] with c = t + 0.25 reaches both items when 8 < c < 11.5
    assert hit == [str(t) for t in np.arange(8.0, 11.5, 0.5)]
    assert item_masks(sp, {"a_00": (4.0, "centred")}, {}, HOP) == {}  # no segments: one item per clip
    assert item_masks(sp, {}, {"u00": SEG}, HOP) == {}  # no windowed column


def test_a_trailing_window_masks_only_after_the_boundary():
    masks = item_masks(_space(), {"a_00": (4.0, "trailing")}, {"u00": SEG}, HOP)
    hit = [float(lab.split("|")[1]) for lab, h in zip(_space().labels, masks["a_00"]) if h]
    assert min(hit) >= 9.5 and max(hit) < 14.0  # [c-4, c] reaches g1 until c - 4 >= 9.5


def test_causal_shift_relabels_centred_values_half_a_window_later():
    sh = causal_shifts({"a_00": (4.0, "centred"), "t": (3.0, "trailing")}, HOP)
    assert sh == {"a_00": {"window_sec": 4.0, "shift_sec": 2.0, "rounding_sec": 0.0}}
    assert causal_shifts({"r": (8.92, "centred")}, HOP)["r"]["shift_sec"] == 4.5  # rounded, recorded
    sp = _space()
    out, absent = shift_columns(sp, sh, HOP)
    assert np.isnan(out.X[:4, 0]).all() and absent[:4].all() and not absent[4:].any()
    assert out.X[4, 0] == 0.0 and out.X[10, 0] == 6.0  # the value at t moved to t + 2 s
    assert sp.X[4, 0] == 4.0  # the input is untouched


def test_masked_values_stay_masked_where_the_shift_lands():
    sp = _space()
    cols = {"a_00": {**LAB, "window_sec": 4.0, "window_align": "centred"}}
    _, absent, rec = read_absent(sp, cols, {"u00": SEG}, HOP, causal=True)
    t = np.array([float(lab.split("|")[1]) for lab in sp.labels])
    # first 2 s (no source) + the 7 masked source rows, now 2 s later
    assert set(t[absent]) == set(np.arange(0.0, 2.0, 0.5)) | set(np.arange(10.0, 13.5, 0.5))
    assert rec["item_masked_rows"] == 7 and rec["absent_rows"] == 11
    assert rec["causal_shifts"]["a_00"]["shift_sec"] == 2.0


def test_grid_and_event_columns_are_never_masked_or_shifted():
    cols = {"g": {**LAB, "window_sec": "grid"}, "e": {**LAB, "window_sec": "event"}}
    assert numeric_windows(cols) == {}


# --- the segments record and the CLI ----------------------------------------------------


def _packed_table(tmp_path, windowed=True):
    rng = np.random.default_rng(0)
    rows = []
    for u in range(N_UNITS):
        sid = f"ext-tc-unit-{u:03d}"
        for b in range(N_BINS):
            z = rng.normal(size=2)
            for model, d in (("a", 4), ("b", 3)):
                x = z @ rng.normal(size=(2, d)) * 0 + np.r_[z, z][:d] + 0.05 * rng.normal(size=d)
                for j, v in enumerate(x):
                    rows.append((sid, 0.25 + HOP * b, model, f"{model}_{j:02d}", float(v)))
    df = pd.DataFrame(rows, columns=["stimulus_id", "time", "model", "feature", "value"])
    df["value_str"] = pd.Series([None] * len(df), dtype="string")
    df["modality"], df["extractor"], df["extractor_version"] = "audio", "testx", "0.0"
    path = tmp_path / "tc" / "groups" / "tc_audio_frames_features.parquet"
    path.parent.mkdir(parents=True)
    df.to_parquet(path, index=False)
    a_win = {"window_sec": 4.0, "window_align": "centred"} if windowed else {}
    meta = {"schema_version": "1.3", "models": ["a", "b"],
            "model_labels": {"a": {"default": {**LAB, **a_win}, "columns": {}},
                             "b": {"default": {**LAB, **({"window_sec": "grid"} if windowed else {})},
                                   "columns": {}}}}
    path.with_suffix(".meta.json").write_text(json.dumps(meta))
    inputs = tmp_path / "tc" / "inputs"
    inputs.mkdir()
    for u in range(N_UNITS):
        SEG.assign(stimulus_id=f"ext-tc-unit-{u:03d}").to_csv(inputs / f"unit-{u:03d}_segments.csv", index=False)
    return path, inputs


FIT = ["--key", "stimulus_id,time", "--window", "0.5", "--members", "a,b", "--block", "T",
       "--fixed-k", "2", "--no-criterion", "--n-splits", "3"]


def test_segments_are_recorded_and_checked(tmp_path):
    table, inputs = _packed_table(tmp_path)
    assert table_segments(table) == {}
    rec = record_table_segments(table, inputs)
    assert len(rec["files"]) == N_UNITS and rec["files"]["ext-tc-unit-000"]["file"] == "unit-000_segments.csv"
    assert list(table_segments(table)["ext-tc-unit-003"]["group"]) == ["g1", "g2"]
    (inputs / "unit-003_segments.csv").write_text("stimulus_id,onset,offset,group\next-tc-unit-003,0,19.5,g1\n")
    with pytest.raises(InputError, match="changed since it was recorded"):
        table_segments(table)
    (inputs / "unit-003_segments.csv").unlink()
    with pytest.raises(InputError, match="is missing"):
        table_segments(table)
    with pytest.raises(InputError, match="holds no"):
        record_table_segments(table, tmp_path)


def test_fit_masks_items_and_projection_reads_causally(tmp_path, capsys):
    table, inputs = _packed_table(tmp_path)
    assert main(["features", str(table), "--refresh-segments", str(inputs)]) == 0
    out = tmp_path / "space"
    assert main(["space", "fit", "--features", str(table), *FIT, "-o", str(out), "--stem", "T"]) == 0
    m = json.loads((out / "T.json").read_text())
    assert m["per_member"]["a"]["item_masked_rows"] == 7 * N_UNITS  # 7 rows per unit, as above
    assert m["per_member"]["b"]["item_masked_rows"] == 0
    assert m["n_rows"] == N_UNITS * N_BINS  # masking never drops a row
    assert m["item_masks"]["segments_units_per_table"] == {str(table): N_UNITS}
    assert "window reads two items" in capsys.readouterr().out
    scores = tmp_path / "s.parquet"
    assert main(["space", "project", "--features", str(table), "--key", "stimulus_id,time", "--window", "0.5",
                 "--space", str(out / "T.json"), "--causal", "-o", str(scores)]) == 0
    meta = json.loads((tmp_path / "s.meta.json").read_text())
    assert meta["causal"] is True
    assert meta["window_reads"]["a"]["absent_rows"] == 11 * N_UNITS
    assert meta["window_reads"]["a"]["causal_shifts"]["a_00"]["shift_sec"] == 2.0
    assert "b" not in meta["window_reads"]
    # rows where `a` is absent are still placed by `b` (k 2 from b's 3 columns), not dropped
    assert len(pd.read_parquet(scores)) == N_UNITS * N_BINS


def test_causal_refuses_a_fit_without_window_labels(tmp_path, capsys):
    table, _ = _packed_table(tmp_path, windowed=False)
    out = tmp_path / "space"
    assert main(["space", "fit", "--features", str(table), *FIT, "-o", str(out), "--stem", "T"]) == 0
    assert main(["space", "project", "--features", str(table), "--key", "stimulus_id,time", "--window", "0.5",
                 "--space", str(out / "T.json"), "--causal", "-o", str(tmp_path / "s.parquet")]) == 1
    assert "--causal needs the fit's per-column window labels" in capsys.readouterr().err
