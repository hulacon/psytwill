"""Project gridded feature rows onto an irregular interval index.

The time-aware cross mode's irregular half. :mod:`psytwill.compare` defines
its measures only between spaces sharing a row index, and the store's
temporal groups sit at two kinds of grain: regular windows stamped ``time``
(frames, audio, caption chunks) and intervals stamped ``onset``/``offset``
(transcript chunks, scene-annotation segments). ``--window`` binning is the
regular half of the join; this module is the other half — pool the grid
into each interval, so machine spaces acquire rows at the grain where the
irregular annotations natively live.

Direction matters. An interval row is a unit somebody produced (a scene a
human segmented, a dialogue chunk); a grid row is a clock tick. Pooling
grid→interval keeps the row a real unit, which is why it is the measured
direction; the reverse (stamping each bin with its covering interval) is a
resampling of annotation content onto arbitrary ticks and is deliberately
not built until something needs it.

Three rules, matching :func:`psytwill.store.load_spaces` so the two halves
of the join cannot disagree:

**Bin first, then contain.** Grid groups disagree on what ``time`` stamps —
visual/caption groups stamp bin starts, audio groups bin centers — so rows
are binned to ``floor(time / window) * window`` before the containment test,
exactly as ``--window`` does. An interval owns the bins whose binned stamp
lies in ``[onset, offset)``.

**Every interval is a row, present or not.** Downstream alignment intersects
labels across spaces, so a model must emit a row for every interval in the
index — NaN where the interval covers no bins or only NaN values — or one
sparse model silently shrinks every other space's n.

**The bin count is provenance, never a dimension.** Each model emits a
``{model}_n_pooled`` feature (its non-NaN bin count per interval), named
with the reserved suffix so :func:`psytwill.store.load_spaces` drops it into
the load report instead of the covariance matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from psytwill.exceptions import InputError, SpaceError
from psytwill.store import META_COLUMNS, _read, _schema_names

PROJECT_SCHEMA_VERSION = "1.0"

INTERVAL_KEY = ("stimulus_id", "chunk_idx")


def _interval_index(path: Path) -> pd.DataFrame:
    """The distinct (stimulus_id, chunk_idx, onset, offset) rows of a table.

    Any interval-stamped group file works as an index source; its feature
    content is ignored here. Overlapping intervals are allowed (transcript
    crosstalk) — the containment rule below resolves them deterministically —
    but an interval with a missing or inverted span is an input error.
    """
    names = _schema_names(path)
    missing = [c for c in (*INTERVAL_KEY, "onset", "offset") if c not in names]
    if missing:
        raise SpaceError(
            f"'{path.name}' has no column(s) {missing}, so it cannot serve as "
            "an interval index. It needs stimulus_id, chunk_idx, onset, offset."
        )
    iv = _read(path, [*INTERVAL_KEY, "onset", "offset"])
    for c in INTERVAL_KEY:
        iv[c] = iv[c].astype(str) if c == "stimulus_id" else iv[c]
    iv = iv.drop_duplicates(subset=list(INTERVAL_KEY)).reset_index(drop=True)
    if iv["chunk_idx"].isna().any():
        raise InputError(
            f"'{path.name}' has rows with a null chunk_idx; an interval index "
            "needs every interval identified."
        )
    iv["chunk_idx"] = iv["chunk_idx"].astype("int64")
    iv["onset"] = iv["onset"].astype(float)
    iv["offset"] = iv["offset"].astype(float)
    bad = iv[iv["onset"].isna() | iv["offset"].isna() | (iv["offset"] <= iv["onset"])]
    if len(bad):
        raise InputError(
            f"{len(bad)} interval(s) in '{path.name}' have a missing or "
            "non-positive span (offset <= onset). Fix the table; a silent "
            "drop here would change what n means downstream."
        )
    return iv.sort_values(["stimulus_id", "onset", "chunk_idx"]).reset_index(drop=True)


def _assign_bins(
    bins: pd.DataFrame, intervals: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Map each (stimulus_id, t) bin to the interval containing its start.

    Where intervals overlap, the latest-starting containing interval owns the
    bin — a deterministic rule, and the count of bins with more than one
    candidate is returned so the caller can record it rather than hide it.
    """
    out = []
    n_overlap = 0
    for sid, grp in bins.groupby("stimulus_id", observed=True):
        iv = intervals[intervals["stimulus_id"] == sid]
        if iv.empty:
            continue
        onsets = iv["onset"].to_numpy()
        offsets = iv["offset"].to_numpy()
        t = grp["t"].to_numpy()
        pos = np.searchsorted(onsets, t, side="right") - 1
        ok = (pos >= 0) & (t < offsets[np.clip(pos, 0, None)])
        # overlap accounting: a bin whose owner's predecessor also contains it
        prev = pos - 1
        overlapped = ok & (prev >= 0) & (t < offsets[np.clip(prev, 0, None)])
        n_overlap += int(overlapped.sum())
        sel = grp.loc[ok].copy()
        sel["chunk_idx"] = iv["chunk_idx"].to_numpy()[pos[ok]]
        out.append(sel)
    if not out:
        raise SpaceError(
            "No grid bin fell inside any interval. Either the two tables "
            "cover disjoint stimuli or the units disagree (both must be "
            "seconds)."
        )
    return pd.concat(out, ignore_index=True), n_overlap


def project_onto_intervals(
    gridded: str | Path,
    intervals: str | Path,
    *,
    window: float,
    models: Iterable[str] | None = None,
    bins: Literal["all", "odd", "even"] = "all",
) -> tuple[pd.DataFrame, dict]:
    """Pool a gridded group table onto an interval index.

    Returns ``(long_form_frame, meta)``. The frame is store-schema long form
    keyed ``(stimulus_id, chunk_idx)`` with ``onset``/``offset`` carried
    through, one row per (interval, model, feature) for **every** interval in
    the index, plus a ``{model}_n_pooled`` provenance feature per model.

    ``bins`` selects all bins (the measurement), or the odd/even-ranked bins
    within each interval (the two halves of a split-half stability ceiling).
    """
    gpath, ipath = Path(gridded), Path(intervals)
    names = _schema_names(gpath)
    if "time" not in names:
        raise SpaceError(
            f"'{gpath.name}' has no 'time' column; only a gridded group can "
            "be projected. Interval-keyed tables are index sources, not inputs."
        )
    cols = ["stimulus_id", "time", "model", "feature", "value"] + [
        c for c in META_COLUMNS if c in names
    ]
    frame = _read(gpath, cols, models=list(models) if models else None)
    if not len(frame):
        raise InputError(f"'{gpath.name}' has no rows for the requested models.")
    frame = frame.copy()
    frame["stimulus_id"] = frame["stimulus_id"].astype(str)
    # +1e-9 as in load_spaces: float-noise 0.9999... lands in its true bin.
    frame["t"] = np.floor(frame["time"].to_numpy() / window + 1e-9) * window

    iv = _interval_index(ipath)

    bin_index = frame[["stimulus_id", "t"]].drop_duplicates()
    assigned, n_overlap = _assign_bins(bin_index, iv)
    frame = frame.merge(assigned, on=["stimulus_id", "t"], how="inner")

    if bins != "all":
        rank = (
            frame[["stimulus_id", "chunk_idx", "t"]]
            .drop_duplicates()
            .sort_values(["stimulus_id", "chunk_idx", "t"])
        )
        rank["rk"] = rank.groupby(["stimulus_id", "chunk_idx"], observed=True).cumcount()
        keep = rank[rank["rk"] % 2 == (1 if bins == "odd" else 0)]
        frame = frame.merge(
            keep[["stimulus_id", "chunk_idx", "t"]],
            on=["stimulus_id", "chunk_idx", "t"],
            how="inner",
        )

    key = ["stimulus_id", "chunk_idx", "model", "feature"]
    pooled = frame.groupby(key, observed=True)["value"].mean().reset_index()

    # per-model provenance: non-NaN bin count per interval
    nn = frame.dropna(subset=["value"])
    counts = (
        nn.groupby(["stimulus_id", "chunk_idx", "model"], observed=True)["t"]
        .nunique()
        .reset_index(name="value")
    )

    # every interval x every (model, feature) — absence must be a NaN row,
    # not a missing label, or alignment shrinks every other space
    feats = frame[["model", "feature"]].drop_duplicates()
    full = iv.assign(_k=1).merge(feats.assign(_k=1), on="_k").drop(columns="_k")
    out = full.merge(pooled, on=key, how="left")

    counts["feature"] = counts["model"].astype(str) + "_n_pooled"
    prov = iv.assign(_k=1).merge(
        counts[["model"]].drop_duplicates().assign(_k=1), on="_k"
    ).drop(columns="_k")
    prov["feature"] = prov["model"].astype(str) + "_n_pooled"
    prov = prov.merge(
        counts, on=["stimulus_id", "chunk_idx", "model", "feature"], how="left"
    )
    prov["value"] = prov["value"].fillna(0.0)
    out = pd.concat([out, prov], ignore_index=True)

    meta_first = (
        frame[["model"] + [c for c in META_COLUMNS if c in frame.columns]]
        .drop_duplicates(subset=["model"])
    )
    out = out.merge(meta_first, on="model", how="left")
    # full store schema: readers select value_str unconditionally
    out["value_str"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out = out.sort_values(["stimulus_id", "chunk_idx", "model", "feature"]).reset_index(
        drop=True
    )

    meta = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "kind": "psytwill_project",
        "rule": (
            f"grid->interval: bins of width {window}s pooled (mean) into the "
            "interval containing the binned stamp in [onset, offset); "
            "latest-starting interval owns an overlapped bin"
        ),
        "source": str(gpath),
        "intervals": str(ipath),
        "window": window,
        "bins": bins,
        "models": sorted(map(str, out["model"].dropna().unique())),
        "n_intervals": int(len(iv)),
        "n_stimuli": int(iv["stimulus_id"].nunique()),
        "n_overlapped_bins": n_overlap,
        "rows": int(len(out)),
    }
    return out, meta
