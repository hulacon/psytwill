"""Carry a packed unit's feature families onto its per-recording units
(psytwill-space 0.3.0, rule C; DECIDED 2026-11-27, "carry by exact remap").

Re-staging a packed root one unit per recording (``stage.py --restage-from``,
:func:`psytwill.staging.split_recordings`) cuts each recording out of its old
unit at a grid point ``shift``: old time ``t`` is new time ``t - shift``, and
the new unit ends on a grid point, so every frame window of the new unit is
the same audio as one frame window of the old unit. A family whose rows
depend only on their own window, or on a table that is itself carried, is
therefore *carried*: its rows are sliced out of the packed table and
re-timed, and nothing is re-extracted.

Which families carry is read from each sidecar, not from a list:

- **word2psy** transcript families carry: every row is a word or a chunk,
  and a chunk is one transcript row (one segment) of one recording.
- the **producers** ``diarize`` (turns) and ``transcribe`` (chunks + words)
  carry: they are the 0.2.x tables the carried consumers were computed from,
  and the windowed members re-extracted on the new units are gated by the
  sliced turns.
- an **aud2psy** member carries when every column's ``window_sec`` label is
  ``grid``, ``event`` or a number no larger than the grid hop.
- every other aud2psy member is **not carried**: a window longer than the
  hop reads across the old unit's recordings (re-extract it per recording),
  and a member with no window label (an event tracker) says nothing about
  what it read.

A unit that already held one recording (``unchanged`` in the recording map)
is the same wav, so all of its families carry as they are.

Every carried sidecar gains a ``carried_from`` block (the old unit, its
stimulus id, the shift, the old sidecar's sha256, and what the slice
changed); paths and per-unit counts are rewritten for the new unit. Counts
that cannot be recomputed from the carried tables are listed under
``carried_from.packed_summaries`` and keep the packed unit's values.

Tables are handled as CSV rows of strings, so every value the carry does
not re-time is written back byte for byte.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

#: Recorded in every carried sidecar.
CARRY_RULE = "psytwill-space 0.3.0 rule C: carry by exact remap (DECIDED 2026-11-27)"
#: Window labels that read only the row's own grid window (or its events).
WINDOW_FREE = ("grid", "event")
#: aud2psy producers whose tables are carried (consumers read them).
PRODUCERS = ("diarize", "transcribe")
#: Onset tolerance when assigning a row to a recording (matches stage.py).
EPS = 1e-6

Row = dict  # one CSV row, values as strings


@dataclass(frozen=True)
class Piece:
    """Where one recording sits in its old unit (a ``recording_map.csv`` row)."""

    old_stimulus_id: str
    new_stimulus_id: str
    shift: float
    duration: float
    hop: float = 0.5

    @property
    def hi(self) -> float:
        return self.shift + self.duration

    def inside(self, onset: float) -> bool:
        """Does a row starting at ``onset`` (old seconds) belong to this recording?"""
        return self.shift - EPS <= onset < self.hi


def fmt_time(x: float) -> str:
    """A re-timed value: shortest repr after rounding away the subtraction's noise."""
    v = round(x, 6)
    return repr(v + 0.0)  # -0.0 -> 0.0


def _retime(value: str, shift: float) -> str:
    return value if value == "" else fmt_time(float(value) - shift)


def _ids(row: Row, piece: Piece) -> Row:
    out = dict(row)
    for col in ("stimulus_id", "chunk_label"):
        if out.get(col) == piece.old_stimulus_id:
            out[col] = piece.new_stimulus_id
    return out


# ---------------------------------------------------------------- verdicts --

def window_labels(model_meta: Mapping[str, Any]) -> list:
    """Every ``window_sec`` a model's labels declare (default and per column)."""
    labels = model_meta.get("labels") or {}
    out = []
    if "window_sec" in (labels.get("default") or {}):
        out.append(labels["default"]["window_sec"])
    for col in (labels.get("columns") or {}).values():
        if isinstance(col, Mapping) and "window_sec" in col:
            out.append(col["window_sec"])
    if not out and "window_sec" in model_meta:
        out.append(model_meta["window_sec"])
    return out


def verdict(sidecar: Mapping[str, Any], hop: float = 0.5) -> tuple[bool, str]:
    """(carried?, why) for one family's sidecar."""
    extractor = sidecar.get("extractor")
    models = sidecar.get("models") or {}
    if extractor == "word2psy":
        return True, "text rows (words, chunks): chunk-local"
    if extractor != "aud2psy":
        return False, f"{extractor} family: not an audio-unit family"
    if len(models) != 1:
        return False, f"{len(models)} models in one sidecar"
    (model, meta), = models.items()
    if model in PRODUCERS:
        return True, "producer table"
    windows = window_labels(meta)
    if not windows:
        return False, "no window label: re-extract per recording"
    long = [w for w in windows if w not in WINDOW_FREE
            and not (isinstance(w, (int, float)) and float(w) <= hop + EPS)]
    if long:
        return False, f"windowed ({', '.join(sorted({str(w) for w in long}))} s): re-extract per recording"
    return True, "grid-local"


# ----------------------------------------------------------------- slicing --

def slice_frames(rows: Sequence[Row], piece: Piece) -> tuple[list[Row], int]:
    """The recording's grid windows, re-timed; and how many end past it.

    Window ``k`` covers ``[k*hop, (k+1)*hop)`` with ``time`` its centre. The
    new unit has ``ceil(duration / hop)`` windows starting at old window
    ``shift / hop``; each must occur exactly once in the packed table.
    """
    hop = piece.hop
    k0 = round(piece.shift / hop)
    if abs(k0 * hop - piece.shift) > EPS:
        raise ValueError(f"shift {piece.shift} is not on the {hop} s grid")
    n = max(1, math.ceil(round(piece.duration / hop, 9)))
    got: dict[int, Row] = {}
    for r in rows:
        k = round(float(r["time"]) / hop - 0.5)
        if k0 <= k < k0 + n:
            if k in got:
                raise ValueError(f"window {k} occurs twice")
            got[k] = r
    missing = [k for k in range(k0, k0 + n) if k not in got]
    if missing:
        raise ValueError(f"{len(missing)} of {n} windows missing from the packed table "
                         f"(first at {(missing[0] + 0.5) * hop} s)")
    out = []
    for k in range(k0, k0 + n):
        r = _ids(got[k], piece)
        r["time"] = fmt_time((k - k0 + 0.5) * hop)
        out.append(r)
    tail = int(n * hop - piece.duration > EPS)
    return out, tail


def slice_turns(rows: Sequence[Row], piece: Piece) -> tuple[list[Row], int]:
    """Diarize turns overlapping the recording, clipped to it and re-timed;
    and how many were clipped."""
    out, clipped = [], 0
    for r in rows:
        on, off = float(r["onset"]), float(r["offset"])
        if off <= piece.shift + EPS or on >= piece.hi - EPS:
            continue
        r = _ids(r, piece)
        if on < piece.shift - EPS or off > piece.hi + EPS:
            clipped += 1
            r["onset"] = fmt_time(max(on, piece.shift) - piece.shift)
            r["offset"] = fmt_time(min(off, piece.hi) - piece.shift)
        else:
            r["onset"], r["offset"] = _retime(r["onset"], piece.shift), _retime(r["offset"], piece.shift)
        r["turn_idx"] = str(len(out))
        out.append(r)
    return out, clipped


def _chunk_map(chunks: Sequence[Row], piece: Piece) -> dict[str, str]:
    """Old chunk_idx -> new chunk_idx for the chunks starting in the recording."""
    if any(c.get("onset", "") == "" for c in chunks):
        raise ValueError("a chunk has no onset, so it cannot be placed in a recording")
    mine = [c["chunk_idx"] for c in chunks if piece.inside(float(c["onset"]))]
    return {old: str(i) for i, old in enumerate(mine)}


def _retimed(r: Row, piece: Piece) -> Row:
    r = _ids(r, piece)
    for col in ("onset", "offset"):
        if col in r:
            r[col] = _retime(r[col], piece.shift)
    return r


def slice_transcribe(chunks: Sequence[Row], words: Sequence[Row], piece: Piece
                     ) -> tuple[list[Row], list[Row], int]:
    """Whisper chunks starting in the recording, with their words (word_idx is
    per chunk, kept); and how many of those chunks end past the recording."""
    cmap = _chunk_map(chunks, piece)
    out_c, overrun = [], 0
    for c in chunks:
        if c["chunk_idx"] in cmap:
            if c["offset"] != "" and float(c["offset"]) > piece.hi + EPS:
                overrun += 1
            r = _retimed(c, piece)
            r["chunk_idx"] = cmap[c["chunk_idx"]]
            out_c.append(r)
    out_w = []
    for w in words:
        if w["chunk_idx"] in cmap:
            r = _retimed(w, piece)
            r["chunk_idx"] = cmap[w["chunk_idx"]]
            out_w.append(r)
    return out_c, out_w, overrun


def slice_text(chunks: Sequence[Row], words: Sequence[Row], piece: Piece
               ) -> tuple[list[Row], list[Row]]:
    """A word2psy family's chunks starting in the recording and their words.
    ``chunk_idx``, ``word_idx`` and ``sentence_idx`` restart at 0, as they
    would in a file holding only this recording's transcript."""
    cmap = _chunk_map(chunks, piece)
    out_c = []
    for c in chunks:
        if c["chunk_idx"] in cmap:
            r = _retimed(c, piece)
            r["chunk_idx"] = cmap[c["chunk_idx"]]
            out_c.append(r)
    out_w, smap = [], {}
    sentence_chunk: dict[str, str] = {}
    for w in words:
        s = w.get("sentence_idx")
        if s is not None:
            if sentence_chunk.setdefault(s, w["chunk_idx"]) != w["chunk_idx"]:
                raise ValueError(f"sentence {s} spans chunks {sentence_chunk[s]} and {w['chunk_idx']}")
        if w["chunk_idx"] not in cmap:
            continue
        r = _retimed(w, piece)
        r["chunk_idx"] = cmap[w["chunk_idx"]]
        r["word_idx"] = str(len(out_w))
        if s is not None:
            r["sentence_idx"] = smap.setdefault(s, str(len(smap)))
        out_w.append(r)
    return out_c, out_w


def slice_family(sidecar: Mapping[str, Any], rows: Mapping[str, Sequence[Row]], piece: Piece
                 ) -> tuple[dict[str, list[Row]], dict[str, int]]:
    """Slice one family's tables (``rows`` keyed as the sidecar's ``output``).

    Returns the new tables and notes on what the slice changed
    (``partial_tail_windows``, ``clipped_turns``, ``chunks_past_end``).
    """
    kind = (sidecar.get("extractor"), tuple(sorted(rows)))
    notes: dict[str, int] = {}
    if kind == ("aud2psy", ("frames",)):
        frames, tail = slice_frames(rows["frames"], piece)
        notes["partial_tail_windows"] = tail
        out = {"frames": frames}
    elif kind == ("aud2psy", ("speakers",)):
        turns, clipped = slice_turns(rows["speakers"], piece)
        notes["clipped_turns"] = clipped
        out = {"speakers": turns}
    elif kind == ("aud2psy", ("transcript", "transcript_words")):
        c, w, over = slice_transcribe(rows["transcript"], rows["transcript_words"], piece)
        notes["chunks_past_end"] = over
        out = {"transcript": c, "transcript_words": w}
    elif kind == ("word2psy", ("chunks", "words")):
        c, w = slice_text(rows["chunks"], rows["words"], piece)
        out = {"chunks": c, "words": w}
    else:
        raise ValueError(f"no carry rule for {kind[0]} tables {list(kind[1])}")
    return out, {k: v for k, v in notes.items() if v}


def producer_rows(tables: Mapping[str, Sequence[Row]]) -> dict[str, int]:
    """The carried producer table sizes a consumer's sidecar records."""
    out = {}
    if "speakers" in tables:
        out["diarize"] = len(tables["speakers"])
    if "transcript_words" in tables:
        out["transcribe"] = len(tables["transcript_words"])
    return out


# ----------------------------------------------------------------- sidecar --

def _rewrite_strings(obj: Any, mapping: Mapping[str, str]) -> Any:
    if isinstance(obj, str):
        for old, new in mapping.items():
            if obj == old or obj.startswith(old.rstrip("/") + "/"):
                return new + obj[len(old):]
        return obj
    if isinstance(obj, list):
        return [_rewrite_strings(v, mapping) for v in obj]
    if isinstance(obj, dict):
        return {k: _rewrite_strings(v, mapping) for k, v in obj.items()}
    return obj


def carried_sidecar(sidecar: Mapping[str, Any], *, piece: Piece, paths: Mapping[str, str],
                    tables: Mapping[str, Sequence[Row]], old_unit: str, old_root: str,
                    old_sha256: str, unchanged: bool, producer_rows: Mapping[str, int] | None = None,
                    notes: Mapping[str, Any] | None = None) -> dict:
    """The new unit's sidecar for one carried family.

    ``paths`` maps old path prefixes to new ones (the old unit's feature dir,
    its wav, its transcript input); ``tables`` maps each ``output`` key to the
    rows written for it; ``producer_rows`` gives the new unit's carried
    producer table sizes (``diarize`` turns, ``transcribe`` words) for the
    consumers that record how many they read.
    """
    out = _rewrite_strings(copy.deepcopy(dict(sidecar)), paths)
    packed: list[str] = []
    if not unchanged:
        inp = out.get("input") or {}
        if "duration_sec" in inp:
            inp["duration_sec"] = round(piece.duration, 3)
        for key, rows in tables.items():
            if key in (out.get("output") or {}):
                out["output"][key]["rows"] = len(rows)
        if "frames" in out and "frames" in tables:
            out["frames"]["n_frames"] = len(tables["frames"])
        _recount(out, tables, producer_rows or {}, packed)
    block = {"rule": CARRY_RULE, "unit": old_unit, "stimulus_id": piece.old_stimulus_id,
             "root": old_root, "shift_sec": round(piece.shift, 3), "sidecar_sha256": old_sha256,
             "unchanged": unchanged}
    if packed:
        block["packed_summaries"] = packed
    if notes:
        block.update(notes)
    out["carried_from"] = block
    return out


def _recount(out: dict, tables: Mapping[str, Sequence[Row]], producer_rows: Mapping[str, int],
             packed: list[str]) -> None:
    """Per-unit counts a fresh run would report, from the carried tables."""
    models = out.get("models") or {}
    for name, meta in models.items():
        for key, producer in (("n_turns", "diarize"), ("n_words", "transcribe")):
            if key in meta and meta.get("derived_from") == producer:
                if producer in producer_rows:
                    meta[key] = producer_rows[producer]
                else:
                    packed.append(f"models.{name}.{key}")
        if "n_gated_windows" in meta and meta.get("gated_columns") and "frames" in tables:
            gated = meta["gated_columns"]
            meta["n_gated_windows"] = sum(all(r.get(c, "") == "" for c in gated) for r in tables["frames"])
        for key in meta:
            if key.startswith("n_") and key.endswith("_windows") and key != "n_gated_windows":
                packed.append(f"models.{name}.{key}")
    if "diarization" in out and "speakers" in tables:
        turns = tables["speakers"]
        d = out["diarization"]
        time: dict[str, float] = {}
        for r in turns:
            time[r["speaker"]] = time.get(r["speaker"], 0.0) + float(r["offset"]) - float(r["onset"])
        d["n_turns"] = len(turns)
        d["n_speakers"] = len(time)
        d["speaker_time_sec"] = {k: round(v, 3) for k, v in sorted(time.items())}
    if "transcription" in out:
        packed += [f"transcription.{k}" for k in ("n_speech_segments", "speech_timing")
                   if k in out["transcription"]]
    inp = out.get("input") or {}
    if out.get("extractor") == "word2psy" and "words" in tables and "chunks" in tables:
        inp["n_words"] = len(tables["words"])
        inp["n_chunks"] = len(tables["chunks"])
        if "n_sentences" in inp:
            inp["n_sentences"] = len({r.get("sentence_idx") for r in tables["words"]})
