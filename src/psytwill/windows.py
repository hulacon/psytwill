"""Read-time window handling for time-keyed tables (Contract B 1.4 window labels).

Two steps, both driven by each column's declared ``window_sec`` /
``window_align`` and neither ever writing a value an extractor did not:

**Item masks.** Short-file corpora are packed into extraction units
(:mod:`psytwill.staging`): items concatenated on the grid with a silence gap,
a *group* (a track, a chapter, a word, an episode) never split, and a
segments table recording where each item sits. A windowed column's value on a
row near an item boundary was computed over audio from two unrelated items.
:func:`item_masks` finds those rows: the row's window (centred on the row,
clamped to the unit as the producers clamp it; or trailing) overlaps a
segment of a different group. The member is then **absent** from that row
(the "absent unless complete" path), never filled and never dropped for the
other members. A unit with no segments table is one item.

**Causal shift** (`space project --causal`). A centred window of w seconds
on the row at t reads [t - w/2, t + w/2]. Relabelling that value to the row
at t + w/2 makes it read [T - w, T]: a trailing, causal window, exact on the
grid when w/2 is a multiple of the grid step (rounded to the nearest step
otherwise, and recorded). The first w/2 of every clip then has no value;
the member is absent there.

The segments tables are generated data on GPFS, not repo content. A features
table records where they came from (`segments` in its sidecar, with each
file's sha256), and a read that finds one missing or changed refuses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from psytwill.exceptions import InputError, SpaceError
from psytwill.store import SpaceMatrix

SEGMENTS_SUFFIX = "_segments.csv"
SEGMENT_COLUMNS = ("onset", "offset", "group")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# the segments record in a features table's sidecar
# --------------------------------------------------------------------------


def _unit_files(segments_dir: Path) -> dict[str, Path]:
    """``{stimulus_id: file}`` over a directory's ``*_segments.csv`` (each names its unit)."""
    out = {}
    for p in sorted(segments_dir.glob(f"*{SEGMENTS_SUFFIX}")):
        ids = pd.read_csv(p, usecols=["stimulus_id"], dtype=str)["stimulus_id"].unique()
        if len(ids) != 1:
            raise InputError(f"{p}: a segments table describes one unit, found stimulus_id {list(ids)[:3]}")
        if ids[0] in out:
            raise InputError(f"{segments_dir}: unit {ids[0]} has two segments tables ({out[ids[0]].name}, {p.name})")
        out[ids[0]] = p
    return out


def segments_record(segments_dir: str | Path, stimulus_ids: Iterable[str]) -> dict:
    """``{"dir", "files": {stimulus_id: {"file", "sha256"}}}`` for the units of a table.

    Units with no segments table are continuous media (one item per clip). A
    directory with a table for none of the ids is refused: it is not this
    corpus's staging directory.
    """
    d = Path(segments_dir).resolve()
    if not d.is_dir():
        raise InputError(f"segments directory {d} does not exist")
    have = _unit_files(d)
    files = {sid: {"file": have[sid].name, "sha256": _sha256(have[sid])}
             for sid in sorted(set(stimulus_ids)) if sid in have}
    if not files:
        raise InputError(f"{d} holds no `*{SEGMENTS_SUFFIX}` for any of this table's stimuli; "
                         "point --segments-dir at the corpus's staging `inputs/` directory")
    return {"dir": str(d), "files": files}


def table_segments(table: str | Path) -> dict[str, pd.DataFrame]:
    """``{stimulus_id: segments}`` recorded by a features table (empty when none).

    Refuses when a recorded file is missing or its sha256 changed: the rows
    were extracted against that layout, and a different one would mask the
    wrong rows.
    """
    meta_path = Path(table).with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    rec = json.loads(meta_path.read_text()).get("segments")
    if not rec:
        return {}
    out = {}
    for sid, info in rec["files"].items():
        p = Path(rec["dir"]) / info["file"]
        if not p.exists():
            raise InputError(f"{table}: its segments table {p} is missing; the item masks cannot be "
                             "computed. Restore it, or re-record with `psytwill features --refresh-segments`.")
        if _sha256(p) != info["sha256"]:
            raise InputError(f"{table}: {p} changed since it was recorded (sha256); re-stage and re-extract "
                             "the unit, or re-record with `psytwill features --refresh-segments` if the "
                             "layout is known to be unchanged.")
        df = pd.read_csv(p, dtype={"group": str})
        missing = [c for c in SEGMENT_COLUMNS if c not in df.columns]
        if missing:
            raise InputError(f"{p} lacks column(s) {missing}")
        out[sid] = df[list(SEGMENT_COLUMNS)].sort_values("onset", kind="stable").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# per-column windows from resolved labels
# --------------------------------------------------------------------------


def numeric_windows(columns: Mapping[str, Mapping] | None) -> dict[str, tuple[float, str]]:
    """``{column: (seconds, align)}`` for the columns with a numeric window."""
    out = {}
    for col, e in (columns or {}).items():
        w = e.get("window_sec")
        if isinstance(w, (int, float)) and not isinstance(w, bool):
            out[col] = (float(w), e.get("window_align", "centred"))
    return out


def _row_times(labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sid = np.array([lab.split("|")[0] for lab in labels], dtype=object)
    t = np.array([float(lab.split("|")[1]) for lab in labels])
    return sid, t


def _window_bounds(c: np.ndarray, w: float, align: str, duration: float) -> tuple[np.ndarray, np.ndarray]:
    if align == "trailing":
        return np.clip(c - w, 0.0, None), np.minimum(c, duration)
    if duration <= w:
        lo = np.zeros_like(c)
    else:
        lo = np.clip(c - w / 2, 0.0, duration - w)
    return lo, np.minimum(lo + w, duration)


def item_masks(space: SpaceMatrix, windows: Mapping[str, tuple[float, str]],
               segments: Mapping[str, pd.DataFrame], hop: float) -> dict[str, np.ndarray]:
    """``{column: bool per row}``: the column's window overlaps another group's segment.

    Rows are ``stimulus_id|time`` with ``time`` the start of a ``hop`` bin; the
    window is placed on the bin's centre. A unit's duration is taken as its
    last row's end (the producers clamp windows to the audio). A row in a
    silence gap is masked when its window reaches two groups.
    """
    if not windows or not segments:
        return {}
    sid, t = _row_times(space.labels)
    c = t + hop / 2
    out = {col: np.zeros(len(t), dtype=bool) for col in windows}
    for unit in np.unique(sid):
        seg = segments.get(unit)
        if seg is None:
            continue
        rows = np.flatnonzero(sid == unit)
        on, off = seg["onset"].to_numpy(float), seg["offset"].to_numpy(float)
        grp = seg["group"].astype(str).to_numpy()
        duration = float(t[rows].max() + hop)
        cr = c[rows]
        for col, (w, align) in windows.items():
            lo, hi = _window_bounds(cr, w, align, duration)
            first = np.searchsorted(off, lo, side="right")  # first segment ending after lo
            last = np.searchsorted(on, hi, side="left") - 1  # last segment starting before hi
            hit = np.zeros(len(rows), dtype=bool)
            span = last > first
            for i in np.flatnonzero(span):
                hit[i] = len(set(grp[first[i]:last[i] + 1])) > 1
            out[col][rows] = hit
    return {col: m for col, m in out.items() if m.any()}


# --------------------------------------------------------------------------
# causal shift
# --------------------------------------------------------------------------


def causal_shifts(windows: Mapping[str, tuple[float, str]], hop: float) -> dict[str, dict]:
    """``{column: {"window_sec", "shift_sec", "rounding_sec"}}`` for centred numeric windows."""
    out = {}
    for col, (w, align) in windows.items():
        if align != "centred":
            continue
        steps = int(round((w / 2) / hop))
        out[col] = {"window_sec": w, "shift_sec": steps * hop, "rounding_sec": round(steps * hop - w / 2, 6)}
    return out


def shift_columns(space: SpaceMatrix, shifts: Mapping[str, dict], hop: float,
                  masks: Mapping[str, np.ndarray] | None = None) -> tuple[SpaceMatrix, np.ndarray]:
    """Move each shifted column's values ``shift_sec`` later within its stimulus.

    Returns the shifted space and a row mask of where the member has no
    value to show: a shifted column with no source row (the first w/2 of a
    clip) or whose source row was item-masked. The cells there are NaN, and
    the caller makes the member absent on those rows; nothing is filled.
    """
    masks = masks or {}
    if not shifts:
        none = np.zeros(space.n, dtype=bool)
        for m in masks.values():
            none |= m
        return space, none
    sid, t = _row_times(space.labels)
    step = np.rint(t / hop).astype(np.int64)
    index = {(s, k): i for i, (s, k) in enumerate(zip(sid, step))}
    X = np.array(space.X, dtype=float, copy=True)
    absent = np.zeros(space.n, dtype=bool)
    col_of = {f: j for j, f in enumerate(space.features)}
    for col, sh in shifts.items():
        if col not in col_of:
            continue
        j = col_of[col]
        k = int(round(sh["shift_sec"] / hop))
        src = np.array([index.get((s, kk - k), -1) for s, kk in zip(sid, step)])
        have = src >= 0
        new = np.full(space.n, np.nan)
        new[have] = space.X[src[have], j]
        X[:, j] = new
        bad = ~have
        if col in masks:
            bad[have] |= masks[col][src[have]]
        absent |= bad
    for col, m in masks.items():
        if col not in shifts:
            absent |= m
    return replace(space, X=X), absent


def read_absent(space: SpaceMatrix, columns: Mapping[str, Mapping] | None,
                segments: Mapping[str, pd.DataFrame], hop: float, *, causal: bool = False
                ) -> tuple[SpaceMatrix, np.ndarray, dict]:
    """The member as it is read: item-masked and, with ``causal``, shifted.

    Returns ``(space, absent rows, record)``. ``record`` notes the shifts and
    how many rows each step made the member absent from.
    """
    windows = numeric_windows(columns)
    masks = item_masks(space, windows, segments, hop)
    shifts = causal_shifts(windows, hop) if causal else {}
    out, absent = shift_columns(space, shifts, hop, masks)
    rec = {"item_masked_rows": int(np.any(np.stack(list(masks.values())), axis=0).sum()) if masks else 0,
           "absent_rows": int(absent.sum())}
    if shifts:
        rec["causal_shifts"] = shifts
    return out, absent, rec


def merged_segments(tables: Iterable[str | Path]) -> dict[str, pd.DataFrame]:
    """The segments of every table, keyed by stimulus_id (ids are distinct across tables)."""
    out: dict[str, pd.DataFrame] = {}
    for table in tables:
        got = table_segments(table)
        clash = set(got) & set(out)
        if clash:
            raise SpaceError(f"stimulus id(s) {sorted(clash)[:3]} carry segments in two tables")
        out.update(got)
    return out


def record_table_segments(table: str | Path, segments_dir: str | Path) -> dict:
    """Write ``segments`` into a features table's sidecar (JSON only); returns the record."""
    from datetime import datetime, timezone

    from psytwill import __version__
    from psytwill.store import distinct_values

    meta_path = Path(table).with_suffix(".meta.json")
    if not meta_path.exists():
        raise InputError(f"{table} has no sidecar ({meta_path}); build it with `psytwill features` first")
    meta = json.loads(meta_path.read_text())
    rec = segments_record(segments_dir, distinct_values(table, "stimulus_id"))
    if meta.get("segments") != rec:
        meta["segments"] = rec
        meta.setdefault("refreshed", []).append({
            "by": f"psytwill {__version__}",
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fields": ["segments"]})
        tmp = meta_path.with_name(meta_path.name + ".tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(meta_path)
    return rec
