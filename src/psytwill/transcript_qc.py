"""Flags for ASR transcript chunks: repetition loops, compressed timing, style.

A Whisper-style decoder conditioned on its own previous text can fall into a
repetition loop: one short line emitted again and again for many seconds,
at high confidence, though it was said once. It can also switch transcription
style partway through a stimulus (lowercase, unpunctuated text), and it can
put real words into implausibly short segments. This module flags all three
on a transcript table and turns only the first into an exclusion:

- **Loop** (``loop_exclude``): a run of ``N_LOOP`` or more consecutive chunks
  of one stimulus with identical normalised text. Chunks 2..n of the run are
  excluded and the first is kept, because the line that seeded the loop was
  usually said. Shorter runs are flagged (``run_len``) and kept: a scripted
  repeat ("Hey. Hey. Hey.") is data.
- **Timing** (``timing_compressed``): duration below ``TIMING_MIN_DURATION``
  seconds or more than ``TIMING_MAX_WPS`` words per second. The words are
  real; only the segment bounds are tight. Flag only.
- **Style** (``lower_style``): text that starts lowercase and carries no
  sentence punctuation. Flag only; ``style_share`` gives the per-stimulus share.

Rows are keyed ``(stimulus_id, chunk_idx)``, where ``chunk_idx`` counts rows
within ``stimulus_id`` in input order. That is how word2psy numbers the chunks
of the transcript CSV it reads, so an exclusion list built here keys a block
fitted on those chunks. A decoder's own segment index is not used: it can skip
numbers where empty segments were dropped.

The constants below are the rules. ``RULES_VERSION`` changes whenever one of
them does, and every table written from them records it.
"""

from __future__ import annotations

import re

import pandas as pd

RULES_VERSION = "1"
N_LOOP = 5
TIMING_MIN_DURATION = 0.1  # seconds
TIMING_MAX_WPS = 15.0
INPUT_COLUMNS = ("stimulus_id", "text", "onset", "offset")
KEY = ("stimulus_id", "chunk_idx")

_NORM_DROP = re.compile(r"[^a-z0-9' ]+")
_SENTENCE_PUNCT = re.compile(r"[.?!,]")


def rules() -> dict:
    """The rule constants, for a sidecar."""
    return {"rules_version": RULES_VERSION, "n_loop": N_LOOP,
            "timing_min_duration": TIMING_MIN_DURATION, "timing_max_wps": TIMING_MAX_WPS}


def normalise(text: object) -> str:
    """Lower-case, keep letters, digits, apostrophes and spaces, strip."""
    return _NORM_DROP.sub("", str(text).lower()).strip()


def is_lower_style(text: object) -> bool:
    """Starts with a lowercase letter and carries no sentence punctuation."""
    s = str(text)
    return bool(re.match(r"[a-z]", s)) and not _SENTENCE_PUNCT.search(s)


def flag_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """One row per input chunk, in input order, with the flag columns added.

    ``df`` needs ``stimulus_id, text, onset, offset`` (a transcript input CSV).
    Added: ``chunk_idx, n_words, duration, words_per_s, run_id, run_len,
    run_pos, loop_exclude, timing_compressed, lower_style``. ``run_id`` counts
    runs within ``stimulus_id``; ``run_pos`` is 0 for a run's first chunk.
    A chunk whose text normalises to nothing never extends a run.
    """
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"transcript table lacks column(s) {missing}; it needs {list(INPUT_COLUMNS)}")
    out = df.reset_index(drop=True).copy()
    out["text"] = out["text"].fillna("").astype(str)
    out["chunk_idx"] = out.groupby("stimulus_id", sort=False).cumcount()
    out["n_words"] = out["text"].str.split().str.len().astype(int)
    out["duration"] = (out["offset"].astype(float) - out["onset"].astype(float))
    out["words_per_s"] = out["n_words"] / out["duration"].clip(lower=1e-3)

    norm = out["text"].map(normalise)
    prev = norm.groupby(out["stimulus_id"], sort=False).shift()
    same_prev = norm.eq(prev) & norm.ne("")
    out["run_id"] = (~same_prev).astype(int).groupby(out["stimulus_id"], sort=False).cumsum() - 1
    grp = out.groupby(["stimulus_id", "run_id"], sort=False)
    out["run_len"] = grp["text"].transform("size").astype(int)
    out["run_pos"] = grp.cumcount().astype(int)
    out["loop_exclude"] = (out["run_len"] >= N_LOOP) & (out["run_pos"] > 0)
    out["timing_compressed"] = (out["duration"] < TIMING_MIN_DURATION) | (out["words_per_s"] > TIMING_MAX_WPS)
    out["lower_style"] = out["text"].map(is_lower_style)
    return out


def attach_asr(flags: pd.DataFrame, asr: pd.DataFrame, *, text_column: str = "text",
               columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """Join decoder columns (e.g. confidences) back onto flagged chunks.

    The join is on ``(stimulus_id, onset, offset, text)``, with the decoder's
    text stripped, because the transcript input drops empty segments and so
    shares no index with the decoder's table. Row count and order are kept;
    a chunk with no match gets NaN. A decoder table with two rows on one join
    key is refused rather than guessed between.
    """
    cols = list(columns)
    right = asr[["stimulus_id", "onset", "offset", text_column, *cols]].copy()
    right["text"] = right.pop(text_column).fillna("").astype(str).str.strip()
    on = ["stimulus_id", "onset", "offset", "text"]
    dup = right.duplicated(on, keep=False)
    if dup.any():
        raise ValueError(f"decoder table has {int(dup.sum())} rows sharing a (stimulus_id, onset, offset, text) key")
    merged = flags.merge(right, on=on, how="left", validate="m:1")
    assert len(merged) == len(flags)
    return merged


def style_share(flags: pd.DataFrame) -> pd.Series:
    """Per-stimulus share of ``lower_style`` chunks."""
    return flags.groupby("stimulus_id", sort=False)["lower_style"].mean().rename("lower_style_share")


def exclusion_rows(flags: pd.DataFrame) -> pd.DataFrame:
    """``(stimulus_id, chunk_idx)`` of the loop-excluded chunks."""
    return flags.loc[flags["loop_exclude"], list(KEY)].reset_index(drop=True)
