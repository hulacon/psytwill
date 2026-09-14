"""Per-film viewer payloads from the movies features tables.

Assembles, for each film, everything the timeline viewer renders:

- **scalar feature series** from the long-form features tables
  (``movies_frames``, ``movies_audio_frames``, ``movies_audio_beats``,
  ``movies_transcript_words``, ``movies_transcript_chunks``). Embedding
  dimensions never ship: any feature named ``<stem>_NNN`` (a 3-4 digit
  dimension index) is dropped at load time, per the design rule that raw
  embeddings become derived artifacts, not payload.
- **captions** — the per-frame ``caption`` model's ``value_str``,
  collapsed to change-points.
- **transcript words and chunks** with onsets, and **speaker turns**,
  from the film directory's ``transcribe_transcript_words.csv`` /
  ``transcribe_transcript.csv`` / ``diarize_speakers.csv``. The word
  *text* is an identity column of the extractor CSVs and is not carried
  by the features table, so this is the one place the viewer reads
  beside it.
- **media references** — the ``frames/`` screengrab times and the
  ``audio.m4a`` track, both relative paths resolved by the page.

Text series are scoped to the **transcript corpus**. The features table
keys rows by stimulus and coordinates but not by source corpus (the same
``sentiment`` model scores transcripts, captions, and annotations into
identically-shaped rows in different tables), so the table *stem* is the
corpus key here; caption- and annotation-corpus features are a later
phase.

Series values are aligned three ways, recorded in each series dict:
``grid`` (regular ``t0``/``dt``), ``times`` (explicit ``t`` array), or
``word``/``chunk`` (index-aligned to the payload's ``words``/``chunks``
lists).
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from psytwill.exceptions import InputError

#: feature named like an embedding dimension (``clip_000``, ``ebind_1023``)
EMBEDDING_DIM_RE = re.compile(r"_\d{3,4}$")

#: feature named like a spatial-grid cell (``saliency_00_23``) — a map to
#: derive from at build time someday, not a timeline series
GRID_CELL_RE = re.compile(r"_\d{2}_\d{2}$")

#: features-table stem -> (payload modality, alignment grain)
TABLES = {
    "movies_frames": ("visual", "time"),
    "movies_audio_frames": ("audio", "time"),
    "movies_audio_beats": ("audio", "time"),
    "movies_transcript_words": ("text", "word"),
    "movies_transcript_chunks": ("text", "chunk"),
}

CAPTION_MODEL = "caption"
GRID_TOLERANCE = 1e-6


def _round5(x: float) -> float:
    """5 significant digits — plenty for a pixel plot, much smaller JSON."""
    return float(f"{x:.5g}")


def scalar_features(path: Path) -> list[str]:
    """Distinct non-embedding feature names in one table (cheap column scan)."""
    import pyarrow.parquet as pq

    names = pq.read_table(path, columns=["feature"])["feature"].unique().to_pylist()
    return [n for n in names
            if not EMBEDDING_DIM_RE.search(n) and not GRID_CELL_RE.search(n)]


def load_scalar_table(path: Path) -> pd.DataFrame:
    """One features table restricted to scalar features and payload columns."""
    keep = scalar_features(path)
    if not keep:
        return pd.DataFrame()
    columns = ["stimulus_id", "time", "onset", "offset", "chunk_idx",
               "word_idx", "model", "feature", "value", "value_str"]
    df = pd.read_parquet(path, columns=columns,
                         filters=[("feature", "in", keep)])
    return df


def load_tables(features_dir: Path) -> dict[str, pd.DataFrame]:
    """All viewer tables under ``features_dir``, keyed by stem.

    A missing table is a warning, not an error — a film set scored by only
    two extractors still gets a viewer over what exists.
    """
    features_dir = Path(features_dir)
    tables: dict[str, pd.DataFrame] = {}
    for stem in TABLES:
        path = features_dir / f"{stem}_features.parquet"
        if not path.exists():
            warnings.warn(f"no {path.name} under {features_dir} — "
                          f"{TABLES[stem][0]} ({TABLES[stem][1]}) series skipped")
            continue
        tables[stem] = load_scalar_table(path)
    if not tables:
        raise InputError(
            f"no movies_*_features.parquet under {features_dir} — pass the "
            "directory holding the `psytwill features` movie tables")
    return tables


def film_slugs(tables: dict[str, pd.DataFrame]) -> list[str]:
    slugs: set[str] = set()
    for df in tables.values():
        if not df.empty:
            slugs.update(df["stimulus_id"].unique())
    return sorted(slugs)


def _series_from_times(sub: pd.DataFrame, modality: str) -> list[dict]:
    """Time-aligned series dicts for one film from one table."""
    out = []
    for (model, feature), grp in sub.groupby(["model", "feature"], sort=True):
        grp = grp.dropna(subset=["value"]).sort_values("time")
        if grp.empty:
            continue
        t = grp["time"].to_numpy(dtype=float)
        v = [_round5(x) for x in grp["value"].to_numpy(dtype=float)]
        entry = {"m": modality, "model": model, "f": feature}
        dt = np.diff(t)
        if len(t) > 2 and dt.size and np.ptp(dt) < GRID_TOLERANCE:
            entry.update(align="grid", t0=_round5(t[0]), dt=_round5(dt[0]), v=v)
        else:
            entry.update(align="times", t=[_round5(x) for x in t], v=v)
        out.append(entry)
    return out


def _series_from_index(sub: pd.DataFrame, modality: str, grain: str,
                       index: dict[tuple[int, int], int] | dict[int, int],
                       n: int,
                       chunk_base: dict[int, int] | None = None) -> list[dict]:
    """Index-aligned series (values in words/chunks order, null where absent).

    Word-grain rows carry word2psy's *global* running ``word_idx`` over its
    own tokenization, while the transcript CSV's ``word_idx`` restarts per
    chunk over whisper's — and the token counts can differ (contractions
    split differently). Alignment is therefore by within-chunk *position*
    (``word_idx - chunk_base[chunk]``), exact wherever the two tokenizers
    agree on a chunk's word count and off by at most the count difference
    where they don't. Exact word alignment needs the word string in the
    features table — a Contract B ingest gap, not something to paper over
    here.
    """
    out = []
    for (model, feature), grp in sub.groupby(["model", "feature"], sort=True):
        grp = grp.dropna(subset=["value"])
        if grp.empty:
            continue
        v: list[float | None] = [None] * n
        if grain == "word":
            base = chunk_base or {}
            keys = ((c, w - base.get(c, 0)) for c, w in
                    zip(grp["chunk_idx"].astype(int), grp["word_idx"].astype(int)))
        else:
            keys = grp["chunk_idx"].astype(int)
        for key, value in zip(keys, grp["value"].to_numpy(dtype=float)):
            pos = index.get(key)
            if pos is not None:
                v[pos] = _round5(value)
        if any(x is not None for x in v):
            out.append({"m": modality, "model": model, "f": feature,
                        "align": grain, "v": v})
    return out


def _captions(sub: pd.DataFrame) -> list[list]:
    """[[time, text], ...] change-points from the per-frame caption model."""
    cap = sub[(sub["model"] == CAPTION_MODEL) & sub["value_str"].notna()]
    cap = cap.sort_values("time")
    out: list[list] = []
    for t, text in zip(cap["time"], cap["value_str"]):
        if not out or out[-1][1] != text:
            out.append([_round5(float(t)), str(text)])
    return out


def _read_csv(path: Path, required: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise InputError(f"{path} lacks column(s) {missing} — not a "
                         "transcript CSV this viewer understands")
    return df


def film_media(film_dir: Path) -> dict:
    """Frames, audio, transcript words/chunks and speaker turns for one film.

    Everything is optional-graceful: a film without audio (or speech) still
    renders its visual lanes. Media paths are returned relative to the
    *films directory* — the page knows where it sits relative to that.
    """
    slug = film_dir.name
    media: dict = {"frames": [], "frames_dir": None, "audio": None}
    frames_dir = film_dir / "frames"
    if frames_dir.is_dir():
        times = []
        for f in frames_dir.glob("frame_*.jpg"):
            try:
                times.append(float(f.stem.split("_", 1)[1]))
            except ValueError:
                continue
        if times:
            media["frames"] = sorted(_round5(t) for t in times)
            media["frames_dir"] = f"{slug}/frames"
    if (film_dir / "audio.m4a").exists():
        media["audio"] = f"{slug}/audio.m4a"

    words_df = _read_csv(film_dir / "transcribe_transcript_words.csv",
                         ["word", "onset", "offset", "chunk_idx", "word_idx"])
    words: list[list] = []
    if words_df is not None:
        for r in words_df.itertuples():
            words.append([_round5(float(r.onset)), _round5(float(r.offset)),
                          int(r.chunk_idx), int(r.word_idx), str(r.word)])
    media["words"] = words

    chunks_df = _read_csv(film_dir / "transcribe_transcript.csv",
                          ["onset", "offset", "chunk_idx", "transcribe_text"])
    chunks: list[list] = []
    if chunks_df is not None:
        for r in chunks_df.itertuples():
            chunks.append([_round5(float(r.onset)), _round5(float(r.offset)),
                           int(r.chunk_idx), str(r.transcribe_text)])
    media["chunks"] = chunks

    spk_df = _read_csv(film_dir / "diarize_speakers.csv",
                       ["speaker", "onset", "offset"])
    speakers: list[list] = []
    if spk_df is not None:
        for r in spk_df.itertuples():
            speakers.append([_round5(float(r.onset)), _round5(float(r.offset)),
                             str(r.speaker)])
    media["speakers"] = speakers
    return media


def film_payload(slug: str, tables: dict[str, pd.DataFrame],
                 films_dir: Path, registry_row: dict | None = None) -> dict:
    """The complete viewer payload for one film."""
    film_dir = Path(films_dir) / slug
    if not film_dir.is_dir():
        raise InputError(f"{film_dir} does not exist — features tables name "
                         f"stimulus_id {slug!r} but the films directory has "
                         "no matching folder")
    media = film_media(film_dir)

    word_index = {(w[2], w[3]): i for i, w in enumerate(media["words"])}
    chunk_index = {c[2]: i for i, c in enumerate(media["chunks"])}

    series: list[dict] = []
    captions: list[list] = []
    for stem, (modality, grain) in TABLES.items():
        df = tables.get(stem)
        if df is None or df.empty:
            continue
        sub = df[df["stimulus_id"] == slug]
        if sub.empty:
            continue
        if stem == "movies_frames":
            captions = _captions(sub)
            sub = sub[sub["model"] != CAPTION_MODEL]
        if grain == "time":
            series.extend(_series_from_times(sub, modality))
            continue
        # word2psy numbers chunks GLOBALLY over the whole concatenated
        # transcript corpus (film N's first chunk is not 0), while the
        # per-film transcript CSV restarts at 0 — rebase to the film
        sub = sub.assign(chunk_idx=sub["chunk_idx"].astype(int)
                         - int(sub["chunk_idx"].min()))
        if grain == "word":
            # likewise word_idx is a global running index over word2psy's
            # own tokenization -> within-chunk position; the base is
            # computed over ALL models so OOV-dropped rows can't shift it
            chunk_base = (sub.groupby("chunk_idx")["word_idx"]
                          .min().astype(int).to_dict())
            series.extend(_series_from_index(sub, modality, "word",
                                             word_index, len(media["words"]),
                                             chunk_base))
        else:
            series.extend(_series_from_index(sub, modality, "chunk",
                                             chunk_index, len(media["chunks"])))

    duration = 0.0
    for entry in series:
        if entry["align"] == "grid":
            duration = max(duration, entry["t0"] + entry["dt"] * (len(entry["v"]) - 1))
        elif entry["align"] == "times":
            duration = max(duration, entry["t"][-1])
    if media["frames"]:
        duration = max(duration, media["frames"][-1])
    if media["words"]:
        duration = max(duration, media["words"][-1][1])

    payload = {
        "slug": slug,
        "title": slug,
        "style": None,
        "duration": _round5(duration),
        "media": media,
        "captions": captions,
        "series": series,
    }
    if registry_row:
        payload["title"] = registry_row.get("movie_name") or slug
        payload["style"] = registry_row.get("style")
        reg_dur = registry_row.get("duration_s")
        if reg_dur and float(reg_dur) > duration:
            payload["duration"] = _round5(float(reg_dur))
    return payload


def load_registry(registry: Path) -> dict[str, dict]:
    """movies.tsv rows keyed by stimulus_id. Accepts the file or its dir."""
    registry = Path(registry)
    path = registry / "movies.tsv" if registry.is_dir() else registry
    if not path.exists():
        raise InputError(f"{path} does not exist — pass the "
                         "stimuli/stimulus_registry/ directory or movies.tsv")
    df = pd.read_csv(path, sep="\t")
    if "stimulus_id" not in df.columns:
        raise InputError(f"{path} has no stimulus_id column — not a "
                         "stimulus-registry table")
    return {r["stimulus_id"]: r.to_dict() for _, r in df.iterrows()}


def payload_js(payload: dict) -> str:
    """The lazy-loaded per-film script the page injects on selection."""
    body = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return ("window.PSYTWILL_VIZ = window.PSYTWILL_VIZ || {films:{}};\n"
            f"window.PSYTWILL_VIZ.films[{json.dumps(payload['slug'])}] = {body};\n")
