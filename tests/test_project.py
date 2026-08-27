"""grid->interval projection: pooling, coverage, and the alignment contract.

The projection exists so machine spaces acquire rows at the grain where
irregular annotations natively live. The failure modes it must not have are
all silent: a bin-center stamp landing one interval late, a sparse model
shrinking every other space's n at alignment, a provenance count entering
the geometry as a dimension. Each has a test that fails if the guard goes.
"""

import json

import numpy as np
import pandas as pd
import pytest

from psytwill.exceptions import InputError, SpaceError
from psytwill.project import project_onto_intervals
from psytwill.store import load_spaces


def long_form(rows):
    """(stimulus_id, time, model, feature, value) rows -> store-schema frame."""
    df = pd.DataFrame(rows, columns=["stimulus_id", "time", "model", "feature", "value"])
    df["modality"] = "test"
    df["extractor"] = "testx"
    df["extractor_version"] = "0.0"
    return df


@pytest.fixture
def gridded(tmp_path):
    """Two clips on a 0.5 s grid: one bin-start model, one bin-center model.

    clipA times 0.0..3.5 (8 bins), value = its time, so pooled means are
    arithmetic and checkable by hand. The audio-style model stamps centers
    (0.25, 0.75, ...) with value = the bin-start it belongs to.
    """
    rows = []
    for t in np.arange(0, 4, 0.5):
        rows.append(("clipA", t, "vis", "f0", float(t)))
        rows.append(("clipA", t + 0.25, "aud", "f0", float(t)))
    for t in np.arange(0, 2, 0.5):
        rows.append(("clipB", t, "vis", "f0", float(t)))
    p = tmp_path / "gridded.parquet"
    long_form(rows).to_parquet(p, index=False)
    return p


@pytest.fixture
def intervals(tmp_path):
    """clipA: [0,1) and [1,3); a tail gap [3,4). clipB: one interval [0,2)."""
    df = pd.DataFrame(
        {
            "stimulus_id": ["clipA", "clipA", "clipB"],
            "chunk_idx": [0, 1, 0],
            "onset": [0.0, 1.0, 0.0],
            "offset": [1.0, 3.0, 2.0],
            "model": "desc",
            "feature": "d0",
            "value": [0.1, 0.2, 0.3],
        }
    )
    p = tmp_path / "intervals.parquet"
    df.to_parquet(p, index=False)
    return p


def cell(frame, sid, chunk, model, feature):
    m = frame[
        (frame.stimulus_id == sid)
        & (frame.chunk_idx == chunk)
        & (frame.model == model)
        & (frame.feature == feature)
    ]
    assert len(m) == 1
    return m["value"].iloc[0]


# --- pooling ---------------------------------------------------------------


def test_means_pool_by_containment(gridded, intervals):
    out, meta = project_onto_intervals(gridded, intervals, window=0.5)
    # clipA [0,1): bins 0.0, 0.5 -> mean 0.25; [1,3): bins 1.0..2.5 -> 1.75
    assert cell(out, "clipA", 0, "vis", "f0") == pytest.approx(0.25)
    assert cell(out, "clipA", 1, "vis", "f0") == pytest.approx(1.75)
    assert cell(out, "clipB", 0, "vis", "f0") == pytest.approx(0.75)


def test_bin_center_stamps_land_in_the_same_bins(gridded, intervals):
    """Audio stamps 0.25/0.75/... must pool identically to visual 0.0/0.5/..."""
    out, _ = project_onto_intervals(gridded, intervals, window=0.5)
    assert cell(out, "clipA", 0, "aud", "f0") == pytest.approx(0.25)
    assert cell(out, "clipA", 1, "aud", "f0") == pytest.approx(1.75)


def test_tail_gap_bins_are_dropped(gridded, intervals):
    """clipA bins 3.0/3.5 fall in no interval and must reach no row."""
    out, _ = project_onto_intervals(gridded, intervals, window=0.5)
    n = cell(out, "clipA", 1, "vis", "vis_n_pooled")
    assert n == 4  # bins 1.0..2.5, not 1.0..3.5


# --- the alignment contract ------------------------------------------------


def test_every_interval_is_a_row_even_when_empty(tmp_path, intervals):
    """A model absent from clipB must still emit clipB rows (as NaN)."""
    rows = [("clipA", t, "vis", "f0", float(t)) for t in np.arange(0, 3, 0.5)]
    p = tmp_path / "sparse.parquet"
    long_form(rows).to_parquet(p, index=False)
    out, _ = project_onto_intervals(p, intervals, window=0.5)
    v = cell(out, "clipB", 0, "vis", "f0")
    assert np.isnan(v)
    assert cell(out, "clipB", 0, "vis", "vis_n_pooled") == 0


def test_loads_and_aligns_as_spaces(tmp_path, gridded, intervals):
    """The output round-trips through load_spaces at the interval key, with
    the provenance count dropped into the report, never the geometry."""
    out, _ = project_onto_intervals(gridded, intervals, window=0.5)
    p = tmp_path / "projected.parquet"
    out.to_parquet(p, index=False)
    from psytwill.store import LoadReport

    report = LoadReport()
    spaces = load_spaces(
        p, key=("stimulus_id", "chunk_idx"), prefix="proj", report=report
    )
    assert set(spaces) == {"proj:vis", "proj:aud"}
    assert spaces["proj:vis"].X.shape == (3, 1)  # 3 intervals, f0 only
    assert any(
        c.endswith("_n_pooled")
        for cols in report.dropped_provenance.values()
        for c in cols
    )


# --- the ceiling halves ----------------------------------------------------


def test_odd_even_bins_partition(gridded, intervals):
    full, _ = project_onto_intervals(gridded, intervals, window=0.5)
    odd, _ = project_onto_intervals(gridded, intervals, window=0.5, bins="odd")
    even, _ = project_onto_intervals(gridded, intervals, window=0.5, bins="even")
    n_f = cell(full, "clipA", 1, "vis", "vis_n_pooled")
    n_o = cell(odd, "clipA", 1, "vis", "vis_n_pooled")
    n_e = cell(even, "clipA", 1, "vis", "vis_n_pooled")
    assert n_o + n_e == n_f
    assert abs(n_o - n_e) <= 1
    # the two halves pool disjoint bins, so their means differ by one bin step
    m_o = cell(odd, "clipA", 1, "vis", "f0")
    m_e = cell(even, "clipA", 1, "vis", "f0")
    assert m_o != m_e


# --- input errors say the fix ---------------------------------------------


def test_interval_table_without_spans_is_an_error(tmp_path, gridded):
    df = pd.DataFrame(
        {"stimulus_id": ["c"], "chunk_idx": [0], "model": "m", "feature": "f",
         "value": [1.0]}
    )
    p = tmp_path / "no_spans.parquet"
    df.to_parquet(p, index=False)
    with pytest.raises(SpaceError, match="interval index"):
        project_onto_intervals(gridded, p, window=0.5)


def test_inverted_span_is_an_error(tmp_path, gridded):
    df = pd.DataFrame(
        {"stimulus_id": ["clipA"], "chunk_idx": [0], "onset": [2.0],
         "offset": [1.0], "model": "m", "feature": "f", "value": [1.0]}
    )
    p = tmp_path / "bad.parquet"
    df.to_parquet(p, index=False)
    with pytest.raises(InputError, match="span"):
        project_onto_intervals(gridded, p, window=0.5)


def test_interval_table_as_gridded_input_is_an_error(intervals):
    with pytest.raises(SpaceError, match="'time'"):
        project_onto_intervals(intervals, intervals, window=0.5)


def test_disjoint_stimuli_is_an_error(tmp_path, gridded):
    df = pd.DataFrame(
        {"stimulus_id": ["elsewhere"], "chunk_idx": [0], "onset": [0.0],
         "offset": [5.0], "model": "m", "feature": "f", "value": [1.0]}
    )
    p = tmp_path / "disjoint.parquet"
    df.to_parquet(p, index=False)
    with pytest.raises(SpaceError, match="disjoint|units"):
        project_onto_intervals(gridded, p, window=0.5)


# --- overlap rule ----------------------------------------------------------


def test_overlap_owned_by_latest_start_and_counted(tmp_path, gridded):
    df = pd.DataFrame(
        {
            "stimulus_id": ["clipA", "clipA"],
            "chunk_idx": [0, 1],
            "onset": [0.0, 1.0],
            "offset": [2.0, 3.0],  # [1,2) is covered by both
            "model": "m",
            "feature": "f",
            "value": [1.0, 2.0],
        }
    )
    p = tmp_path / "overlap.parquet"
    df.to_parquet(p, index=False)
    out, meta = project_onto_intervals(gridded, p, window=0.5)
    # chunk 0 keeps only [0,1): bins 0.0, 0.5
    assert cell(out, "clipA", 0, "vis", "f0") == pytest.approx(0.25)
    # chunk 1 owns [1,3): bins 1.0..2.5
    assert cell(out, "clipA", 1, "vis", "f0") == pytest.approx(1.75)
    assert meta["n_overlapped_bins"] > 0
