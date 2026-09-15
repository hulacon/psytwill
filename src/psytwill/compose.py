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

Streams and their target stems mirror the movie group tables, so every
existing consumer (``store.load_spaces``, ``psytwill viz movies``, mmmview's
composed-run dispatch) works unchanged: visual and audio models land on the
grid (``movies_frames`` / ``movies_audio_frames``, starts vs centers), text
models land at word grain (``movies_transcript_words``).

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
would have emitted.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from psytwill import __version__
from psytwill.exceptions import InputError
from psytwill.timelines import Registry, read_events

COMPOSE_SCHEMA_VERSION = "1.0"

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
#: measured convention (see store.load_spaces docstring). Composed fill rows
#: and untimed expansions follow the stream they land in.
STREAM_STAMP = {"movies_frames": "start", "movies_audio_frames": "center"}

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
    return pd.DataFrame(rows)


def stream_for(modality: str, grain: str) -> str:
    """Target stem for one (modality, grain), or a named refusal."""
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


def compose_untimed(pres: pd.DataFrame, store: pd.DataFrame,
                    window: float, stamp: str) -> pd.DataFrame:
    """Repeat each item's untimed rows at every grid bin of its window.

    Bins are the grid times covering ``[onset, onset + duration)``, starting
    at the first bin boundary at or after the onset (an item is on a bin
    only once it is actually on), stamped per the target stream.
    """
    onset = pres["_onset"].to_numpy(dtype=float)
    end = onset + pres["_dur"].to_numpy(dtype=float)
    t0 = np.ceil(onset / window - 1e-9) * window
    counts = np.maximum(0, np.ceil((end - t0) / window - 1e-9)).astype(int)
    if not counts.sum():
        return store.iloc[0:0].assign(_slug=None, _sid=None, _t=np.nan)
    rep = np.repeat(np.arange(len(pres)), counts)
    within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    grid = pres.iloc[rep][["_slug", "_sid", "_voice"]].reset_index(drop=True)
    grid["_t"] = t0[rep] + within * window
    out = _voice_join(grid, store)
    offset = window / 2 if stamp == "center" else 0.0
    out["time"] = out["_t"] + offset
    return out


def compose_gridded(pres: pd.DataFrame, store: pd.DataFrame,
                    window: float) -> pd.DataFrame:
    """Shift each item's internal grid to start at the bin containing its onset."""
    p = pres.copy()
    p["_shift"] = np.floor(p["_onset"].to_numpy() / window + 1e-9) * window
    out = _voice_join(p, store)
    out["time"] = out["time"].astype(float) + out["_shift"]
    return out


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


def _signature(events: Sequence[Path], stores: Sequence[Path],
               registry_dir: Optional[Path], params: dict[str, Any]) -> dict[str, Any]:
    def stat(p: Path) -> dict[str, Any]:
        return {"path": str(p.resolve()), "size": p.stat().st_size}

    return {
        "compose_schema_version": COMPOSE_SCHEMA_VERSION,
        "events": [stat(Path(p)) for p in events],
        "stores": [stat(Path(p)) for p in stores],
        "registry": None if registry_dir is None else str(Path(registry_dir).resolve()),
        "params": params,
    }


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
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """The ``compose`` verb: events + item stores -> movie-schema tables.

    Writes ``<output_dir>/features/<stream>_features.parquet`` (+
    ``.meta.json``) per stream present — the composed-run layout mmmview and
    the movies viewer already dispatch. Idempotent: when every output exists
    and records this exact input signature, nothing is rewritten (``force``
    overrides). ``dry_run`` reports the plan without reading feature values
    or writing anything.
    """
    if window <= 0:
        raise InputError("--window must be positive seconds")
    out_root = Path(output_dir)
    feat_dir = out_root / "features"
    registry = None if registry_dir is None else Registry.from_dir(registry_dir)
    pres, runs = read_runs(events, registry, onset_column=onset_column,
                           lead_out=lead_out)

    wanted = set(models) if models else None
    plans: list[dict[str, Any]] = []
    seen_models: dict[str, str] = {}
    for spath in stores:
        spath = Path(spath)
        cls = classify_store(spath, modality_map)
        if wanted is not None:
            cls = cls[cls["model"].isin(wanted)]
        checkpoints = store_checkpoints(spath)
        for r in cls.itertuples(index=False):
            if r.model in seen_models:
                raise InputError(
                    f"model {r.model!r} appears in both "
                    f"'{seen_models[r.model]}' and '{spath.name}'; composed "
                    "keys would collide — drop one store or use --models."
                )
            seen_models[r.model] = spath.name
            plans.append({
                "store": spath, "model": r.model, "modality": r.modality,
                "grain": r.grain, "stream": stream_for(r.modality, r.grain),
                "n_rows": r.n_rows, "n_stimuli": r.n_stimuli,
                "checkpoint": checkpoints.get(r.model),
            })
    if wanted is not None:
        missing = sorted(wanted - set(seen_models))
        if missing:
            raise InputError(
                f"--models {missing} not found in any given store; available: "
                f"{sorted(seen_models)}"
            )
    if not plans:
        raise InputError("the given stores contribute no models")

    streams = sorted({p["stream"] for p in plans})
    params = {
        "window": window, "onset_column": onset_column, "lead_out": lead_out,
        "sparse": sparse, "models": sorted(wanted) if wanted else None,
        "modality_map": modality_map or None,
    }
    signature = _signature([Path(p) for p in events], [Path(s) for s in stores],
                           None if registry_dir is None else Path(registry_dir),
                           params)

    summary: dict[str, Any] = {
        "output_dir": str(out_root),
        "runs": {s: {"n_presentations": r["n_presentations"],
                     "run_end": r["run_end"]} for s, r in runs.items()},
        "streams": {},
        "models": {p["model"]: {"stream": p["stream"], "grain": p["grain"],
                                "store": p["store"].name} for p in plans},
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
                if grain == "untimed" and stem in STREAM_STAMP:
                    part = compose_untimed(pres, sub, window, STREAM_STAMP[stem])
                elif grain == "gridded":
                    part = compose_gridded(pres, sub, window)
                else:
                    part = compose_chunked(pres, sub)
                if len(part):
                    matched_sids.update(part["_sid"].unique())
                    parts.append(part)
        if not parts:
            continue
        frame = _finalize(pd.concat(parts, ignore_index=True))
        if stem in STREAM_STAMP and not sparse:
            frame = fill_grid(frame, runs, window, STREAM_STAMP[stem])
            sort_cols = ["stimulus_id", "time", "model", "feature"]
            frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        out_path = expected[stem]
        frame.to_parquet(out_path, index=False)
        key_cols = (["stimulus_id", "time", "model", "feature"]
                    if stem in STREAM_STAMP
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
                f"grid ({STREAM_STAMP.get(stem, 'word/chunk')} stamping); no "
                "HRF, no TR (Contract C)."
            ),
            "grid": None if stem not in STREAM_STAMP else {
                "window": window,
                "stamp": STREAM_STAMP[stem],
                "fill": "sparse" if sparse else "full",
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
                    "comparable": None,
                } for p in stem_plans
            },
            "comparability_note": (
                "'comparable' is null until a render falsifier measures "
                "composed-vs-rendered agreement for this model; do not treat "
                "a composed table as an extractor readout before then."
            ),
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
