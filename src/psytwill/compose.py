"""Compose sparse presentation timelines onto a dense movie-style grid.

The interval→grid direction :mod:`psytwill.project` names and defers, and
the dense half of the timelines item (mmmdata-agents contracts §4.3 item 5):
a movie is a dense timeline, a trial-based run a sparse one. This module
produces, per events file, the table a movie extractor *would have written*
had the run been a film — one long-form features table per stream, keyed and
sidecar'd like a movie group table, loading through
:func:`psytwill.store.load_spaces` beside real movie tables with no special
case.

The composition rule is driven by the **item store row's temporal grain**,
not by any dataset's trial vocabulary, so the verb composes any events file
that resolves (see :func:`psytwill.timelines.read_events`) with any features
table built by ``psytwill features``:

* **untimed** rows (``time``/``onset``/``offset``/``chunk_idx`` all null — a
  static image, any static item) are repeated at each grid bin of the
  presentation window ``[onset, onset + duration)``;
* **gridded** rows (an internal ``time`` axis — word-audio frames, a movie's
  frame grid) keep their internal grid, shifted so it starts at the bin
  containing the presentation onset. The store's own stamping convention
  (bin starts vs bin centers) survives the shift untouched;
* **chunk-grain** rows (``chunk_idx``, optionally ``word_idx`` and internal
  ``onset``/``offset``) become transcript rows: spans shifted by the
  presentation onset (or set to the presentation window when the item has no
  internal timing), ``chunk_idx`` renumbered by presentation order and
  ``word_idx`` renumbered globally — the movies transcript convention.

**A bin belongs to an item only if the item covers most of it.** Untimed and
gridded rows both land only on bins the presentation window overlaps by at
least ``min_coverage`` of the bin (default 0.5); a presentation shorter than
that keeps its single best-covered bin, so nothing presented disappears. The
rule exists because an item rarely lasts a whole number of bins: a 0.54 s
word on a 0.5 s grid spills 40 ms into a second bin, and the store's frame
for that bin describes the file's tail, while a render of the run has 460 ms
of silence there. Majority coverage leaves such a bin empty, which is what
the run actually held.

Streams and their target stems mirror the movie group tables, so every
existing consumer (``store.load_spaces``, ``psytwill viz movies``, mmmview's
composed-run dispatch) works unchanged: visual and audio models land on the
grid (``movies_frames`` / ``movies_audio_frames``, starts vs centers), text
models land at word grain (``movies_transcript_words``).

**A family composes beside the battery, not into it.** ``family=NAME``
writes ``movies_<NAME>_frames`` / ``movies_<NAME>_audio_frames`` /
``movies_<NAME>_transcript_chunks`` — the stems a projection family's group
tables carry (e.g. psytwill-space's ``movies_psytwill_space_*``) — so a second
compose into a run root that already holds the battery tables adds tables
instead of overwriting them. A model whose rows say ``modality='shared'`` (a
cross-block relation such as ``pspace_vl``) takes the modality of the other
models in its store, so the same relation composes once per side, each into
its side's stream; model names collide only within one stream.

**Grid stamps follow the store.** A gridded store stamps either bin starts
or bin centers; compose reads which from the rows and stamps fill rows and
untimed expansions in that stream the same way. A stream whose gridded
stores disagree is refused. A stream with no gridded rows takes the movie
group tables' convention (visual starts, audio centers).

**Empty bins are explicit.** Fixation, rest and lead-in/out bins get rows
with a NaN ``value`` for every (model, feature) the stream carries (unless
``sparse=True``): a gap the fit-time structural-undefined fill can act on is
a row, not an absence — "nothing on screen" is a direction, not a zero, and
that direction is applied at fit time (space schema 1.2), never here.

**Time is experimental time.** Onsets as the events file records them
(``onset_column`` selects which recording); no HRF, no TR — Contract C and
braintwill's, exactly as :mod:`psytwill.timelines` says.

**Comparability is a measured property, not an assumption.** The sidecar
carries ``comparable: null`` per model until a render falsifier (synthesize
the run as a real movie, run the real extractor battery, compare) writes a
verdict. Composing is exact by construction only for untimed models with an
identical checkpoint; anything with a context window over a continuous track
must be measured before a composed table is treated as what the extractor
would have emitted. The verdict comes in through ``comparability=`` (a TSV
``model, comparable[, note]``) using the vocabulary in
:data:`COMPARABILITY_LABELS`, and is written per model into the sidecar, so a
consumer reads it from the file.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from psytwill import __version__
from psytwill.exceptions import InputError
from psytwill.timelines import Registry, read_events

#: 1.1: bins need majority coverage (``grid.min_coverage``); per-model
#: ``comparable`` labels from a comparability TSV.
COMPOSE_SCHEMA_VERSION = "1.1"

#: What a model's composed value is relative to the extractor's readout of a
#: render of the same run. A consumer pooling composed and real movie tables
#: decides per label; none of them is "discard".
COMPARABILITY_LABELS = {
    "item": "reproduces the render's readout of the item (display context "
            "such as a canvas around the item aside)",
    "item+offset": "identifies the item in the render's readout but shifted "
                   "or rescaled; a joint fit absorbs the shift as an axis",
    "window": "the model's context window spans more than the presentation; "
              "the composed value is the item's own readout, a render reads "
              "the item inside whatever surrounds it",
    "none": "the extractor emits nothing on a render of the run; composed "
            "rows have no render counterpart",
}

#: Full features-table schema (§4.2.4), plus one provenance column: which
#: item produced a composed row. Extra columns are harmless to every reader
#: (they all select columns), and losing the item identity would not be.
FEATURE_COLUMNS = [
    "stimulus_id", "voice", "time", "onset", "offset", "chunk_idx", "word_idx",
    "modality", "extractor", "extractor_version", "model", "feature",
    "value", "value_str",
]
SOURCE_COLUMN = "source_stimulus_id"

#: modality (normalized) -> target stem for grid-bound grains, matching the
#: movie group tables byte-for-byte so no consumer needs a special case.
GRID_STREAMS = {"visual": "movies_frames", "audio": "movies_audio_frames"}
TRANSCRIPT_STREAM = "movies_transcript_words"

#: The audio grid stamps bin centers, visual stamps bin starts — the store's
#: measured convention (see store.load_spaces docstring). It is the default
#: for a stream with no gridded rows; otherwise the gridded stores' own
#: stamping wins (:func:`grid_stamp`), so a start-stamped projection family
#: composes without mixing conventions.
STREAM_STAMP = {"movies_frames": "start", "movies_audio_frames": "center"}

#: Family stems: ``movies_<family>_<suffix>``, the release-family group-table
#: naming (``psytwill space release project``). Text lands at chunk grain
#: there, so its stem says chunks.
FAMILY_SUFFIXES = {"visual": "frames", "audio": "audio_frames",
                   "text": "transcript_chunks"}
_FAMILY_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: Rows of a cross-block relation carry this modality; the model composes
#: on the side its store holds (:func:`classify_store`).
SHARED_MODALITY = "shared"

MODALITY_ALIASES = {
    "visual": "visual", "image": "visual", "video": "visual",
    "audio": "audio", "auditory": "audio",
    "text": "text", "language": "text",
}

_GRAINS = ("untimed", "gridded", "chunked")


# --------------------------------------------------------------------------
# events -> runs
# --------------------------------------------------------------------------

def slug_for(events_path: str | Path) -> str:
    """The composed run's ``stimulus_id``: the events file stem minus ``_events``."""
    stem = Path(events_path).stem
    if stem.endswith("_events"):
        stem = stem[: -len("_events")]
    return stem


def read_runs(
    events: Sequence[str | Path],
    registry: Optional[Registry],
    *,
    onset_column: str = "onset",
    lead_out: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """One presentations frame across runs, plus per-run info.

    Each events file is one run (one composed "movie"); files are read one
    at a time so nothing about cross-file ordering matters here. Returns
    ``(presentations, runs)`` where presentations has one row per resolved
    stimulus row with columns ``_slug, _sid, _voice, _onset, _dur, _ord``
    and runs maps slug -> {events, entities, n_presentations, run_end}.
    """
    frames = []
    runs: dict[str, dict[str, Any]] = {}
    for p in events:
        p = Path(p)
        tl = read_events([p], registry)
        slug = slug_for(p)
        if slug in runs:
            raise InputError(
                f"two events files reduce to the composed id {slug!r}; each "
                "run must compose under a distinct name."
            )
        if onset_column not in tl.columns:
            raise InputError(
                f"{p.name} has no column {onset_column!r} to take onsets from "
                "(--onset-column); available: "
                f"{[c for c in tl.columns if 'onset' in str(c)]}"
            )
        onset = pd.to_numeric(tl[onset_column], errors="coerce")
        dur = pd.to_numeric(tl["duration"], errors="coerce")
        pres = tl[tl["is_stimulus"]]
        if onset[pres.index].isna().any():
            n = int(onset[pres.index].isna().sum())
            raise InputError(
                f"{p.name}: {n} stimulus row(s) have no usable {onset_column!r}; "
                "a presentation without an onset cannot be placed on the grid."
            )
        span = (onset + dur).dropna()
        run_end = float(span.max()) + lead_out if len(span) else lead_out
        frames.append(pd.DataFrame({
            "_slug": slug,
            "_sid": pres["stimulus_id"].astype(str).to_numpy(),
            "_voice": pres["voice"].to_numpy(),
            "_onset": onset[pres.index].to_numpy(dtype=float),
            "_dur": dur[pres.index].to_numpy(dtype=float),
        }))
        ents = (pres.iloc[0][["subject", "session", "task", "run"]].tolist()
                if len(pres) else [None] * 4)
        runs[slug] = {
            "events": str(p.resolve()),
            "entities": {k: (None if pd.isna(v) else str(v)) for k, v in
                         zip(("subject", "session", "task", "run"), ents)},
            "n_presentations": int(len(pres)),
            "run_end": run_end,
        }
    if not frames:
        raise InputError("no events files given")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["_slug", "_onset"], kind="stable").reset_index(drop=True)
    out["_ord"] = out.groupby("_slug", sort=False).cumcount()
    return out, runs


# --------------------------------------------------------------------------
# stores: classification and reading
# --------------------------------------------------------------------------

def _pq(path: Path, columns: Sequence[str], ids: Optional[Sequence[str]] = None) -> pd.DataFrame:
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(path).schema_arrow.names)
    need = [c for c in columns if c in names]
    filters = [("stimulus_id", "in", list(ids))] if ids is not None else None
    df = pq.read_table(path, columns=need, filters=filters).to_pandas()
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan if c not in ("value_str",) else pd.NA
    return df


def classify_store(path: str | Path, modality_map: Optional[dict[str, str]] = None) -> pd.DataFrame:
    """Per-model grain and modality of one features table.

    Cheap relative to composing: reads the identity columns only, never a
    value. Returns a frame with columns ``model, modality, grain, n_rows,
    n_stimuli``. A model with both an internal grid and chunk identity is
    refused — no composition rule exists for it, and guessing one would put
    rows on the wrong axis silently.
    """
    p = Path(path)
    if not p.exists():
        raise InputError(f"No such feature table: '{p}'.")
    if p.suffix != ".parquet":
        raise InputError(
            f"'{p.name}': compose reads parquet features tables; aggregate "
            "CSVs with `psytwill features -o table.parquet` first."
        )
    df = _pq(p, ["stimulus_id", "model", "modality", "time", "onset", "chunk_idx"])
    rows = []
    for model, sub in df.groupby("model", observed=True, dropna=False):
        if pd.isna(model):
            continue
        timed = sub["time"].notna().any()
        chunked = sub["chunk_idx"].notna().any() or sub["onset"].notna().any()
        if timed and chunked:
            raise InputError(
                f"'{p.name}' model {model!r} carries both a time grid and "
                "chunk identity; no composition rule exists for that mix."
            )
        grain = "gridded" if timed else ("chunked" if chunked else "untimed")
        modality = None
        if "modality" in sub.columns and sub["modality"].notna().any():
            modality = str(sub["modality"].dropna().iloc[0])
        if modality_map and str(model) in modality_map:
            modality = modality_map[str(model)]
        if modality is not None and modality.lower() == SHARED_MODALITY:
            rows.append({"model": str(model), "modality": SHARED_MODALITY,
                         "grain": grain, "n_rows": int(len(sub)),
                         "n_stimuli": int(sub["stimulus_id"].nunique())})
            continue
        if modality is None or modality.lower() not in MODALITY_ALIASES:
            raise InputError(
                f"'{p.name}' model {model!r} has no usable modality "
                f"({modality!r}); pass --modality-map {model}=visual|audio|text."
            )
        rows.append({
            "model": str(model),
            "modality": MODALITY_ALIASES[modality.lower()],
            "grain": grain,
            "n_rows": int(len(sub)),
            "n_stimuli": int(sub["stimulus_id"].nunique()),
        })
    if not rows:
        raise InputError(f"'{p.name}' has no models.")
    out = pd.DataFrame(rows)
    shared = out["modality"] == SHARED_MODALITY
    if shared.any():
        sides = sorted(set(out.loc[~shared, "modality"]))
        if len(sides) != 1:
            names = sorted(out.loc[shared, "model"])
            raise InputError(
                f"'{p.name}': {names} carry modality 'shared', which takes "
                "the modality of the other models in its store, but the "
                f"store holds {sides or 'no other models'}; pass "
                f"--modality-map {names[0]}=visual|audio|text."
            )
        out.loc[shared, "modality"] = sides[0]
    return out


def stream_for(modality: str, grain: str, family: Optional[str] = None) -> str:
    """Target stem for one (modality, grain), or a named refusal.

    With ``family``, the stem is ``movies_<family>_<suffix>``
    (:data:`FAMILY_SUFFIXES`); the grain rules are the same.
    """
    if family is not None:
        stream_for(modality, grain)  # same refusals
        return f"movies_{family}_{FAMILY_SUFFIXES[modality]}"
    if modality == "text":
        if grain == "gridded":
            raise InputError(
                "a gridded text model has no composition rule (transcripts "
                "are word/chunk grain); re-aggregate it, or exclude it with "
                "--models."
            )
        return TRANSCRIPT_STREAM
    if grain in ("untimed", "gridded"):
        return GRID_STREAMS[modality]
    raise InputError(
        f"no composition rule for a {grain} {modality} model (beats-style "
        "grains are not composed yet); exclude it with --models."
    )


def grid_stamp(times: np.ndarray, window: float) -> Optional[str]:
    """``'start'`` or ``'center'`` from where a gridded store's ``time``
    values sit within their bins; None when there are none. Anything else
    (an off-grid store) is refused rather than snapped."""
    t = np.asarray(times, dtype=float)
    t = t[np.isfinite(t)]
    if not len(t):
        return None
    frac = np.round(np.mod(t / window, 1.0), 6) % 1.0
    if np.all(np.isclose(frac, 0.0, atol=1e-6) | np.isclose(frac, 1.0, atol=1e-6)):
        return "start"
    if np.all(np.isclose(frac, 0.5, atol=1e-6)):
        return "center"
    raise InputError(
        f"gridded rows sit neither on bin starts nor bin centers of a "
        f"{window}s grid (e.g. time {t[~np.isclose(frac, 0.0) & ~np.isclose(frac, 0.5)][0]}); "
        "re-aggregate on the grid, or pass the matching --window."
    )


def store_checkpoints(path: str | Path) -> dict[str, str]:
    """model -> checkpoint from a features table's sidecar, if one exists."""
    meta_path = Path(str(Path(path)).removesuffix(".parquet") + ".meta.json")
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return {}
    out: dict[str, str] = {}
    for entry in meta.get("inputs", []) or []:
        for name, m in (entry.get("models") or {}).items():
            ckpt = m.get("checkpoint") if isinstance(m, dict) else None
            if isinstance(ckpt, str):
                out[name] = ckpt
    return out


# --------------------------------------------------------------------------
# composition rules
# --------------------------------------------------------------------------

def _voice_join(pres: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Presentations x store rows, following the attach_features convention:
    voiced store rows join on (id, voice); voice-less rows on id alone."""
    voiced = store[store["voice"].notna()]
    unvoiced = store[store["voice"].isna()]
    parts = []
    if len(unvoiced):
        parts.append(pres.merge(unvoiced, left_on="_sid", right_on="stimulus_id"))
    if len(voiced):
        parts.append(pres.merge(
            voiced, left_on=["_sid", "_voice"], right_on=["stimulus_id", "voice"]
        ))
    if not parts:
        return store.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def _bin_coverage(bin_start: np.ndarray, onset: np.ndarray, end: np.ndarray,
                  window: float) -> np.ndarray:
    """Fraction of each bin ``[bin_start, bin_start + window)`` the
    presentation ``[onset, end)`` covers."""
    overlap = np.minimum(bin_start + window, end) - np.maximum(bin_start, onset)
    return np.clip(overlap, 0.0, None) / window


def _keep_covered(cov: pd.Series, key: pd.Series, min_coverage: float) -> np.ndarray:
    """Bins at or above ``min_coverage``; a presentation with none keeps its
    best-covered bin(s), so a short presentation is never dropped outright."""
    best = cov.groupby(key).transform("max")
    return ((cov >= min_coverage - 1e-9)
            | ((best < min_coverage - 1e-9) & (cov >= best - 1e-9) & (best > 0))
            ).to_numpy()


def compose_untimed(pres: pd.DataFrame, store: pd.DataFrame,
                    window: float, stamp: str,
                    min_coverage: float = 0.5) -> pd.DataFrame:
    """Repeat each item's untimed rows at every grid bin of its window.

    Bins are the grid bins ``[onset, onset + duration)`` covers by at least
    ``min_coverage`` (module docstring), stamped per the target stream.
    """
    onset = pres["_onset"].to_numpy(dtype=float)
    end = onset + pres["_dur"].to_numpy(dtype=float)
    first = np.floor(onset / window + 1e-9)
    counts = np.maximum(0, np.ceil(end / window - 1e-9) - first).astype(int)
    if not counts.sum():
        return store.iloc[0:0].assign(_slug=None, _sid=None, _t=np.nan)
    rep = np.repeat(np.arange(len(pres)), counts)
    within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    grid = pres.iloc[rep][["_slug", "_sid", "_voice"]].reset_index(drop=True)
    grid["_t"] = (first[rep] + within) * window
    cov = pd.Series(_bin_coverage(grid["_t"].to_numpy(), onset[rep], end[rep], window))
    grid = grid[_keep_covered(cov, pd.Series(rep), min_coverage)].reset_index(drop=True)
    out = _voice_join(grid, store)
    offset = window / 2 if stamp == "center" else 0.0
    out["time"] = out["_t"] + offset
    return out


def compose_gridded(pres: pd.DataFrame, store: pd.DataFrame,
                    window: float, min_coverage: float = 0.5) -> pd.DataFrame:
    """Shift each item's internal grid to start at the bin containing its
    onset, keeping only bins the presentation covers by ``min_coverage``.

    A bin is identified by its start whichever way the store stamps it
    (``floor(time / window)``), so the rule is the same for center- and
    start-stamped grids.
    """
    p = pres.copy()
    p["_shift"] = np.floor(p["_onset"].to_numpy() / window + 1e-9) * window
    out = _voice_join(p, store)
    out["time"] = out["time"].astype(float) + out["_shift"]
    if not len(out):
        return out
    bin_start = np.floor(out["time"].to_numpy() / window + 1e-9) * window
    onset = out["_onset"].to_numpy(dtype=float)
    cov = pd.Series(_bin_coverage(bin_start, onset,
                                  onset + out["_dur"].to_numpy(dtype=float), window))
    key = out["_slug"].astype(str) + "\x00" + out["_ord"].astype(str)
    return out[_keep_covered(cov, key, min_coverage)].reset_index(drop=True)


def compose_chunked(pres: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Transcript rows: spans shifted, chunk/word identity renumbered.

    ``chunk_idx`` counts chunks in presentation order across the run,
    ``word_idx`` counts words globally — the movies transcript convention,
    so the viewer's rebasing logic applies unchanged. An item with internal
    ``onset``/``offset`` keeps them, shifted; an item without gets the
    presentation window.
    """
    out = _voice_join(pres, store)
    if not len(out):
        return out
    src_chunk = out["chunk_idx"].fillna(0).astype(float) if "chunk_idx" in out else 0.0
    src_word = out["word_idx"].fillna(0).astype(float) if "word_idx" in out else 0.0
    out = out.assign(_src_chunk=src_chunk, _src_word=src_word)
    out["onset"] = np.where(out["onset"].notna(),
                            out["_onset"] + out["onset"].astype(float),
                            out["_onset"])
    out["offset"] = np.where(out["offset"].notna(),
                             out["_onset"] + out["offset"].astype(float),
                             out["_onset"] + out["_dur"])
    out = out.sort_values(["_slug", "_ord", "_src_chunk", "_src_word"],
                          kind="stable").reset_index(drop=True)
    chunk_key = out[["_slug", "_ord", "_src_chunk"]].apply(tuple, axis=1)
    word_key = out[["_slug", "_ord", "_src_chunk", "_src_word"]].apply(tuple, axis=1)
    for name, key in (("chunk_idx", chunk_key), ("word_idx", word_key)):
        codes = pd.Series(pd.factorize(key)[0], index=out.index)
        out[name] = codes - codes.groupby(out["_slug"]).transform("min")
    return out.drop(columns=["_src_chunk", "_src_word"])


def fill_grid(frame: pd.DataFrame, runs: dict[str, dict[str, Any]],
              window: float, stamp: str) -> pd.DataFrame:
    """Explicit NaN rows for every unoccupied bin of every run.

    Every (model, feature) the stream carries gets a row at every bin the
    run spans — mirroring project.py's rule that absence must be a NaN row,
    not a missing label. Meta columns are carried per model so a fill row is
    attributable like any other.
    """
    if not len(frame):
        return frame
    offset = window / 2 if stamp == "center" else 0.0
    feats = frame[["model", "feature", "modality", "extractor",
                   "extractor_version"]].drop_duplicates()
    fills = []
    for slug, info in runs.items():
        n_bins = int(np.ceil(info["run_end"] / window - 1e-9))
        if n_bins <= 0:
            continue
        all_bins = np.arange(n_bins)
        have = frame.loc[frame["stimulus_id"] == slug, "time"]
        occupied = set(np.floor(have.to_numpy(dtype=float) / window + 1e-9).astype(int))
        missing = np.array(sorted(set(all_bins) - occupied))
        if not len(missing):
            continue
        times = missing * window + offset
        fills.append(pd.DataFrame({"stimulus_id": slug, "time": np.repeat(
            times, len(feats))}).assign(**{
                c: np.tile(feats[c].to_numpy(), len(times))
                for c in feats.columns}))
    if not fills:
        return frame
    fill = pd.concat(fills, ignore_index=True)
    fill["value"] = np.nan
    fill["value_str"] = pd.NA
    for c in FEATURE_COLUMNS + [SOURCE_COLUMN]:
        if c not in fill.columns:
            fill[c] = pd.NA if c in ("voice", "value_str", SOURCE_COLUMN) else np.nan
    return pd.concat([frame, fill], ignore_index=True)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Composed rows -> features schema + provenance, sorted deterministically."""
    out = frame.copy()
    out["stimulus_id"] = out["_slug"]
    out[SOURCE_COLUMN] = out["_sid"] if "_sid" in out else pd.NA
    for c in FEATURE_COLUMNS + [SOURCE_COLUMN]:
        if c not in out.columns:
            out[c] = pd.NA if c in ("voice", "value_str", SOURCE_COLUMN) else np.nan
    out = out[FEATURE_COLUMNS + [SOURCE_COLUMN]]
    sort_cols = [c for c in ("stimulus_id", "time", "onset", "chunk_idx",
                             "word_idx", "model", "feature") if c in out.columns]
    return out.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def read_comparability(path: str | Path) -> dict[Any, dict[str, Optional[str]]]:
    """``model -> {comparable, note}`` from a TSV with columns
    ``model, comparable[, note]``; labels must be in :data:`COMPARABILITY_LABELS`.

    An optional ``stream`` column labels a model per target stream — a
    relation composed on two sides can differ by side. Such rows key as
    ``(stream, model)``; a blank stream keys as the model alone, which
    :func:`_label_for` uses for any stream without its own row. An empty
    ``comparable`` cell is an explicit "no verdict" (null in the sidecar)
    with a note saying why.
    """
    p = Path(path)
    if not p.exists():
        raise InputError(f"comparability table not found: {p}")
    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    missing = {"model", "comparable"} - set(df.columns)
    if missing:
        raise InputError(
            f"{p.name} lacks column(s) {sorted(missing)}; expected "
            "model, comparable[, note]")
    bad = sorted(set(df["comparable"]) - set(COMPARABILITY_LABELS) - {""})
    if bad:
        raise InputError(
            f"{p.name}: unknown comparable label(s) {bad}; use one of "
            f"{sorted(COMPARABILITY_LABELS)}")
    stream = df["stream"] if "stream" in df.columns else pd.Series("", index=df.index)
    keys = [m if not s else (s, m) for m, s in zip(df["model"], stream)]
    dup = sorted({str(k) for k in pd.Series(keys)[pd.Series(keys).duplicated()]})
    if dup:
        raise InputError(f"{p.name}: model(s) listed twice: {dup}")
    note = df["note"] if "note" in df.columns else pd.Series("", index=df.index)
    return {k: {"comparable": c or None, "note": n or None}
            for k, c, n in zip(keys, df["comparable"], note)}


def _label_for(comp: dict, stream: str, model: str) -> Optional[dict[str, Optional[str]]]:
    """The comparability row for ``model`` in ``stream``: a stream-specific
    row first, then the model-wide one; None when the table lists neither."""
    return comp.get((stream, model)) or comp.get(model)


def _signature(events: Sequence[Path], stores: Sequence[Path],
               registry_dir: Optional[Path], params: dict[str, Any],
               comparability: Optional[Path] = None) -> dict[str, Any]:
    def stat(p: Path) -> dict[str, Any]:
        return {"path": str(p.resolve()), "size": p.stat().st_size}

    sig = {
        "compose_schema_version": COMPOSE_SCHEMA_VERSION,
        "events": [stat(Path(p)) for p in events],
        "stores": [stat(Path(p)) for p in stores],
        "registry": None if registry_dir is None else str(Path(registry_dir).resolve()),
        "params": params,
    }
    if comparability is not None:
        # content, not size: a relabel can keep the byte count
        sig["comparability"] = {
            "path": str(comparability.resolve()),
            "sha256": hashlib.sha256(comparability.read_bytes()).hexdigest(),
        }
    return sig


def _summary_key(plan: dict[str, Any], plans: Sequence[dict[str, Any]]) -> str:
    """The model name, or ``model@stream`` when the model composes into more
    than one stream (a relation's two sides)."""
    n = sum(1 for q in plans if q["model"] == plan["model"])
    return plan["model"] if n == 1 else f"{plan['model']}@{plan['stream']}"


def _stream_stamps(plans: Sequence[dict[str, Any]], window: float,
                   ids: Sequence[str]) -> dict[str, str]:
    """Grid stream -> stamp convention: the gridded stores' own (all must
    agree), else the movie group tables' default. Transcript streams have no
    grid and no entry. Reads ``time`` for the presented ``ids`` only."""
    out: dict[str, str] = {}
    for stem in sorted({p["stream"] for p in plans}):
        stem_plans = [p for p in plans if p["stream"] == stem]
        if all(p["modality"] == "text" for p in stem_plans):
            continue
        found: dict[str, str] = {}
        for p in stem_plans:
            if p["grain"] != "gridded":
                continue
            t = _pq(p["store"], ["time", "model"], ids=ids)
            st = grid_stamp(t.loc[t["model"] == p["model"], "time"].to_numpy(), window)
            if st is not None:
                found[f"{p['store'].name}:{p['model']}"] = st
        kinds = set(found.values())
        if len(kinds) > 1:
            raise InputError(
                f"{stem}: gridded stores disagree on stamping ({found}); "
                "compose them into separate runs or re-aggregate one.")
        if kinds:
            out[stem] = kinds.pop()
        else:
            modality = stem_plans[0]["modality"]
            out[stem] = STREAM_STAMP[GRID_STREAMS[modality]]
    return out


def build_composed(
    events: Sequence[str | Path],
    stores: Sequence[str | Path],
    output_dir: str | Path,
    *,
    registry_dir: Optional[str | Path] = None,
    window: float = 0.5,
    onset_column: str = "onset",
    lead_out: float = 0.0,
    models: Optional[Sequence[str]] = None,
    modality_map: Optional[dict[str, str]] = None,
    sparse: bool = False,
    min_coverage: float = 0.5,
    comparability: Optional[str | Path] = None,
    family: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """The ``compose`` verb: events + item stores -> movie-schema tables.

    Writes ``<output_dir>/features/<stream>_features.parquet`` (+
    ``.meta.json``) per stream present — the composed-run layout mmmview and
    the movies viewer already dispatch. Idempotent: when every output exists
    and records this exact input signature, nothing is rewritten (``force``
    overrides). ``dry_run`` reports the plan without reading feature values
    or writing anything. ``comparability`` stamps per-model labels into the
    sidecars (see :func:`read_comparability`); models it does not list stay
    ``null`` and are named in the summary. ``family`` names the stems
    (module docstring) so a projection family composes beside the battery.
    """
    if window <= 0:
        raise InputError("--window must be positive seconds")
    if not 0 < min_coverage <= 1:
        raise InputError("--min-coverage must be in (0, 1]")
    if family is not None and not _FAMILY_RE.match(family):
        raise InputError(
            f"--family {family!r}: use lowercase letters, digits and '_' "
            "(it becomes part of the table stems)")
    comp_path = None if comparability is None else Path(comparability)
    comp = {} if comp_path is None else read_comparability(comp_path)
    out_root = Path(output_dir)
    feat_dir = out_root / "features"
    registry = None if registry_dir is None else Registry.from_dir(registry_dir)
    pres, runs = read_runs(events, registry, onset_column=onset_column,
                           lead_out=lead_out)

    wanted = set(models) if models else None
    plans: list[dict[str, Any]] = []
    seen_models: dict[tuple[str, str], str] = {}
    for spath in stores:
        spath = Path(spath)
        cls = classify_store(spath, modality_map)
        if wanted is not None:
            cls = cls[cls["model"].isin(wanted)]
        checkpoints = store_checkpoints(spath)
        for r in cls.itertuples(index=False):
            stream = stream_for(r.modality, r.grain, family)
            if (stream, r.model) in seen_models:
                raise InputError(
                    f"model {r.model!r} appears in both "
                    f"'{seen_models[(stream, r.model)]}' and '{spath.name}' "
                    f"for {stream}; composed keys would collide — drop one "
                    "store or use --models."
                )
            seen_models[(stream, r.model)] = spath.name
            plans.append({
                "store": spath, "model": r.model, "modality": r.modality,
                "grain": r.grain, "stream": stream,
                "n_rows": r.n_rows, "n_stimuli": r.n_stimuli,
                "checkpoint": checkpoints.get(r.model),
            })
    if wanted is not None:
        found = {m for _, m in seen_models}
        missing = sorted(wanted - found)
        if missing:
            raise InputError(
                f"--models {missing} not found in any given store; available: "
                f"{sorted(found)}"
            )
    if not plans:
        raise InputError("the given stores contribute no models")

    streams = sorted({p["stream"] for p in plans})
    params = {
        "window": window, "onset_column": onset_column, "lead_out": lead_out,
        "sparse": sparse, "models": sorted(wanted) if wanted else None,
        "modality_map": modality_map or None, "min_coverage": min_coverage,
    }
    if family is not None:
        # only when set, so pre-family composes keep their signatures
        params["family"] = family
    signature = _signature([Path(p) for p in events], [Path(s) for s in stores],
                           None if registry_dir is None else Path(registry_dir),
                           params, comp_path)

    summary: dict[str, Any] = {
        "output_dir": str(out_root),
        "runs": {s: {"n_presentations": r["n_presentations"],
                     "run_end": r["run_end"]} for s, r in runs.items()},
        "streams": {},
        "models": {_summary_key(p, plans): {
            "stream": p["stream"], "grain": p["grain"], "store": p["store"].name}
            for p in plans},
    }
    if comp_path is not None:
        summary["comparability"] = {
            "table": str(comp_path),
            "unlabelled": sorted(_summary_key(p, plans) for p in plans
                                 if _label_for(comp, p["stream"], p["model"]) is None),
        }

    if dry_run:
        n_pres = len(pres)
        for stem in streams:
            stem_plans = [p for p in plans if p["stream"] == stem]
            est = sum(
                int(p["n_rows"] / max(p["n_stimuli"], 1)) * n_pres
                for p in stem_plans
            )
            summary["streams"][stem] = {
                "models": sorted(p["model"] for p in stem_plans),
                "estimated_rows_lower_bound": est,
                "output": str(feat_dir / f"{stem}_features.parquet"),
            }
        summary["dry_run"] = True
        return summary

    expected = {stem: feat_dir / f"{stem}_features.parquet" for stem in streams}
    if not force and all(p.exists() for p in expected.values()):
        up_to_date = True
        for stem, p in expected.items():
            meta_p = Path(str(p).removesuffix(".parquet") + ".meta.json")
            try:
                prior = json.loads(meta_p.read_text()).get("inputs_signature")
            except Exception:
                prior = None
            if prior != signature:
                up_to_date = False
                break
        if up_to_date:
            summary["up_to_date"] = True
            summary["streams"] = {
                stem: {"output": str(p), "skipped": True}
                for stem, p in expected.items()
            }
            return summary

    ids = sorted(pres["_sid"].unique())
    stamps = _stream_stamps(plans, window, ids)
    matched_sids: set[str] = set()
    feat_dir.mkdir(parents=True, exist_ok=True)

    for stem in streams:
        stem_plans = [p for p in plans if p["stream"] == stem]
        parts = []
        for spath in sorted({p["store"] for p in stem_plans}, key=str):
            spath_models = [p for p in stem_plans if p["store"] == spath]
            store = _pq(spath, FEATURE_COLUMNS, ids=ids)
            store = store[store["model"].isin([p["model"] for p in spath_models])]
            if not len(store):
                continue
            store["stimulus_id"] = store["stimulus_id"].astype(str)
            for grain in _GRAINS:
                sub = store[store["model"].isin(
                    [p["model"] for p in spath_models if p["grain"] == grain])]
                if not len(sub):
                    continue
                if grain == "untimed" and stem in stamps:
                    part = compose_untimed(pres, sub, window, stamps[stem],
                                           min_coverage)
                elif grain == "gridded":
                    part = compose_gridded(pres, sub, window, min_coverage)
                else:
                    part = compose_chunked(pres, sub)
                if len(part):
                    matched_sids.update(part["_sid"].unique())
                    parts.append(part)
        if not parts:
            continue
        frame = _finalize(pd.concat(parts, ignore_index=True))
        if stem in stamps and not sparse:
            frame = fill_grid(frame, runs, window, stamps[stem])
            sort_cols = ["stimulus_id", "time", "model", "feature"]
            frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        out_path = expected[stem]
        frame.to_parquet(out_path, index=False)
        key_cols = (["stimulus_id", "time", "model", "feature"]
                    if stem in stamps
                    else ["stimulus_id", "chunk_idx", "word_idx", "model", "feature"])
        meta = {
            "schema_version": COMPOSE_SCHEMA_VERSION,
            "table": "composed",
            "stream": stem,
            "extractor": "psytwill",
            "extractor_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "time_semantics": (
                "experimental time: presentation onsets as recorded in the "
                f"events column {onset_column!r}, expanded onto a {window}s "
                f"grid ({stamps.get(stem, 'word/chunk')} stamping); no "
                "HRF, no TR (Contract C)."
            ),
            "grid": None if stem not in stamps else {
                "window": window,
                "stamp": stamps[stem],
                "fill": "sparse" if sparse else "full",
                "min_coverage": min_coverage,
            },
            "output": {
                "path": str(out_path.resolve()),
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "key_columns": key_cols,
            },
            "runs": runs,
            "inputs": {
                "stores": [{
                    "path": str(p["store"].resolve()),
                    "model": p["model"], "grain": p["grain"],
                    "checkpoint": p["checkpoint"],
                } for p in stem_plans],
                "registry": None if registry_dir is None else str(Path(registry_dir).resolve()),
                **params,
            },
            "models": {
                p["model"]: {
                    "checkpoint": p["checkpoint"],
                    "grain": p["grain"],
                    "source_store": str(p["store"].resolve()),
                    "comparable": (_label_for(comp, stem, p["model"]) or {}).get("comparable"),
                    "comparability_note": (_label_for(comp, stem, p["model"]) or {}).get("note"),
                } for p in stem_plans
            },
            "comparability": {
                "table": None if comp_path is None else str(comp_path.resolve()),
                "labels": COMPARABILITY_LABELS,
                "null_means": "no render falsifier verdict for this model; do "
                              "not treat its composed values as an extractor "
                              "readout of the run",
            },
            "inputs_signature": signature,
        }
        meta_path = Path(str(out_path).removesuffix(".parquet") + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))
        summary["streams"][stem] = {
            "output": str(out_path),
            "meta_path": str(meta_path),
            "rows": int(len(frame)),
            "models": sorted(p["model"] for p in stem_plans),
        }

    unmatched = sorted(set(ids) - matched_sids)
    if unmatched:
        shown = ", ".join(unmatched[:10])
        more = "" if len(unmatched) <= 10 else f" (+{len(unmatched) - 10} more)"
        raise InputError(
            f"{len(unmatched)} presented stimulus id(s) have no rows in any "
            f"given store: {shown}{more}. Every presentation must compose — "
            "add the store that covers them, or fix the events/registry."
        )
    return summary
