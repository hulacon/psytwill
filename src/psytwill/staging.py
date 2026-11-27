"""Staging short-file corpora into extraction units (psytwill-space).

The ``*2psy`` extractors and the fit-corpus driver work one *unit* per
cell: one media input, one feature family, one sidecar. Speech and music
corpora arrive as thousands of short files (a LibriSpeech utterance is
~12 s, a music clip 30 s, a spoken word 0.5 s), so they are **packed** into
units: files are concatenated in a fixed order with a silence gap, every
file starting on the shared frame grid, and a *segments table* records
where each original file sits (``onset``/``offset`` in the unit's own
seconds) together with any curated text. Downstream pooling schemas
recover the original files from that table; the extractors never see it.

This module holds the pure logic (packing, grid alignment, text
normalisation); ``scripts/fitcorpus/stage.py`` does the IO per corpus.

Invariants (tested):

- a *group* (a chapter, an episode, a track) is never split across units;
- every segment starts at a multiple of ``hop`` and is followed by at
  least ``gap`` seconds of silence before the next one;
- packing is deterministic given the input order (no shuffling here —
  adapters that subsample do so with a seeded RNG *before* packing);
- a unit exceeds ``target`` seconds only when a single group already does.

**Per item** (:func:`item_units`; E5's "re-extract per item", DECIDED
2026-11-13 for the packed short-item corpora): every group is its own unit,
so no window can read another item and no item mask is needed. The unit ends
on the first grid point at or after its last file (no trailing gap: there is
no next item to keep apart).

**Per recording** (psytwill-space 0.3.0, rule C; DECIDED 2026-11-25): one
extraction file holds exactly one recording (the staging group: a clip, a
chapter, an episode, a track, a movement, a word, a film). Packing across
recordings is not allowed, so windowed members are clamped at the recording's
own edges. This is :func:`item_units` applied to every corpus. A root staged
per item is per recording under its older name (:data:`ONE_RECORDING_LAYOUTS`).

:func:`split_recordings` re-stages one packed unit as per-recording units,
laid out exactly as :func:`item_units` would lay out the same group, so a
frame of the old unit at time ``t`` is the new unit's frame at ``t - shift``.
That is what lets unwindowed features carry from a packed extraction onto the
new units without re-extracting them. A unit holding one group is already one
recording and stays as it is.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

#: Staging layouts in which every unit is exactly one recording. ``per-item``
#: is the 0.2.x name (E5) for the same thing.
ONE_RECORDING_LAYOUTS = ("per-recording", "per-item")

__all__ = [
    "ONE_RECORDING_LAYOUTS",
    "RecordingPiece",
    "Segment",
    "Unit",
    "item_units",
    "split_recordings",
    "pack_groups",
    "layout_unit",
    "normalize_gigaspeech_text",
    "normalize_librispeech_text",
]


@dataclass(frozen=True)
class Segment:
    """One source file inside a unit."""

    segment: str  # native id (utterance / segment / track id), [a-z0-9-]
    group: str  # packing group (chapter, episode, track, word)
    duration: float  # seconds of audio
    text: str = ""  # curated text, "" when the corpus has none
    extra: dict = field(default_factory=dict)  # per-corpus columns


@dataclass
class Unit:
    """A packed unit: its segments in order, with grid-aligned onsets."""

    unit: str  # native unit id, [a-z0-9-]
    segments: list[Segment]
    onsets: list[float]  # per segment, seconds into the unit
    duration: float  # total seconds including the trailing gap

    @property
    def offsets(self) -> list[float]:
        return [o + s.duration for o, s in zip(self.onsets, self.segments)]


def _grid_ceil(t: float, hop: float) -> float:
    """Smallest multiple of ``hop`` that is >= ``t`` (float-noise tolerant)."""
    return math.ceil(t / hop - 1e-9) * hop


def layout_unit(unit: str, segments: list[Segment], *, hop: float = 0.5,
                gap: float = 0.5, trailing_gap: float | None = None) -> Unit:
    """Assign grid-aligned onsets to ``segments`` in order.

    Segment ``k`` starts at the first grid point at or after
    ``offset(k-1) + gap``; the unit ends ``trailing_gap`` (default: ``gap``)
    after the last offset, rounded up to the grid so the unit length is a
    whole number of frames.
    """
    trailing = gap if trailing_gap is None else trailing_gap
    if gap < 0 or trailing < 0 or hop <= 0:
        raise ValueError(f"need gap, trailing_gap >= 0 and hop > 0, got gap={gap}, "
                         f"trailing_gap={trailing_gap}, hop={hop}")
    onsets: list[float] = []
    t = 0.0
    for seg in segments:
        onset = _grid_ceil(t, hop)
        onsets.append(onset)
        t = onset + seg.duration + gap
    end = t - gap + trailing
    return Unit(unit=unit, segments=list(segments), onsets=onsets,
                duration=_grid_ceil(end, hop) if segments else 0.0)


def pack_groups(groups: list[tuple[str, list[Segment]]], *, target: float,
                hop: float = 0.5, gap: float = 0.5,
                unit_prefix: str = "u") -> list[Unit]:
    """Greedy first-fit-in-order packing of whole groups into units.

    ``groups`` is an ordered list of ``(group_key, segments)``; groups are
    laid into the current unit until the next group would push its
    duration past ``target``, at which point a new unit starts. A group
    longer than ``target`` on its own gets a unit of its own. Units are
    named ``<unit_prefix><k:03d>`` in creation order, so the caller
    controls the id namespace (e.g. ``"spk1034-"`` for per-speaker
    packing) and ids stay ``[a-z0-9-]``.
    """
    if target <= 0:
        raise ValueError(f"target must be positive, got {target}")
    units: list[Unit] = []
    current: list[Segment] = []

    def flush() -> None:
        if current:
            units.append(layout_unit(f"{unit_prefix}{len(units):03d}", current,
                                     hop=hop, gap=gap))
            current.clear()

    for key, segs in groups:
        if not segs:
            continue
        for s in segs:
            if s.group != key:
                raise ValueError(f"segment {s.segment!r} carries group {s.group!r}, "
                                 f"listed under {key!r}")
        trial = layout_unit("trial", current + list(segs), hop=hop, gap=gap)
        if current and trial.duration > target:
            flush()
        current.extend(segs)
    flush()
    return units


_UNIT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def item_units(groups: list[tuple[str, list[Segment]]], *, hop: float = 0.5,
               gap: float = 0.5, unit_prefix: str = "u",
               unit_id: Callable[[str], str] | None = None) -> list[Unit]:
    """One unit per group: E5's "re-extract per item" (DECIDED 2026-11-13).

    A group (a word, a track, a movement) becomes its own unit
    ``<unit_prefix><group>``, so a windowed member computed on the unit can
    read nothing but that item. Files of one group keep ``gap`` between them;
    the unit has no trailing gap and ends on the first grid point at or after
    its last file. Ids must be native ids (``[a-z0-9-]``) and distinct.
    ``unit_id`` maps a group key to its unit id (default: ``unit_prefix + key``).
    """
    units: list[Unit] = []
    seen: set[str] = set()
    for key, segs in groups:
        if not segs:
            continue
        for s in segs:
            if s.group != key:
                raise ValueError(f"segment {s.segment!r} carries group {s.group!r}, listed under {key!r}")
        uid = unit_id(key) if unit_id is not None else f"{unit_prefix}{key}"
        if not _UNIT_ID_RE.match(uid):
            raise ValueError(f"unit id {uid!r} is not a native id ([a-z0-9-]); slug the group key "
                             "in the corpus adapter")
        if uid in seen:
            raise ValueError(f"unit id {uid!r} occurs twice; a group must be listed once per prefix")
        seen.add(uid)
        units.append(layout_unit(uid, list(segs), hop=hop, gap=gap, trailing_gap=0.0))
    return units


@dataclass(frozen=True)
class RecordingPiece:
    """One recording of a packed unit, as its own unit.

    ``shift`` is where the recording starts in the old unit (a grid point):
    old time ``t`` is new time ``t - shift``. ``unchanged`` marks a unit that
    already held one recording, kept as it is (same id, wav and tables).
    """

    old_unit: str
    new_unit: str
    group: str
    shift: float
    duration: float  # the new unit's length (the old unit's, when unchanged)
    segments: tuple[str, ...]
    onsets: tuple[float, ...]  # per segment, in the new unit's seconds
    offsets: tuple[float, ...]
    unchanged: bool = False


def split_recordings(old_unit: str, rows: Sequence[dict], *, unit_id: Callable[[str], str],
                     old_duration: float, hop: float = 0.5, gap: float = 0.5,
                     keep_single: bool = True) -> list[RecordingPiece]:
    """Re-stage one packed unit one recording per unit.

    ``rows`` are the unit's segments table rows (``segment``, ``group``,
    ``onset``, ``offset``). Each group must be contiguous in onset order and
    start on the grid. Each new unit is checked to be laid out exactly as
    :func:`item_units` lays out the same files (same onsets after the shift),
    so carried frames line up; anything else is a ``ValueError``.

    ``keep_single`` (default) keeps a one-group unit as it is. Pass False
    when the unit id itself must change (a clip corpus, whose unit id is the
    clip's own id); the unit is then re-laid out like any other.
    """
    ordered = sorted(rows, key=lambda r: float(r["onset"]))
    groups: list[tuple[str, list[dict]]] = []
    for r in ordered:
        g = str(r["group"])
        if groups and groups[-1][0] == g:
            groups[-1][1].append(r)
        elif any(k == g for k, _ in groups):
            raise ValueError(f"{old_unit}: group {g!r} is not contiguous in onset order")
        else:
            groups.append((g, [r]))
    if not groups:
        raise ValueError(f"{old_unit}: no segments")
    if len(groups) == 1 and keep_single:
        g, rs = groups[0]
        return [RecordingPiece(old_unit, old_unit, g, 0.0, float(old_duration),
                               tuple(str(r["segment"]) for r in rs), tuple(float(r["onset"]) for r in rs),
                               tuple(float(r["offset"]) for r in rs), unchanged=True)]
    out, seen = [], set()
    for g, rs in groups:
        shift = float(rs[0]["onset"])
        if abs(shift / hop - round(shift / hop)) > 1e-6:
            raise ValueError(f"{old_unit}: group {g!r} starts at {shift}, off the {hop} s grid")
        rel_on = [round(float(r["onset"]) - shift, 6) for r in rs]
        rel_off = [round(float(r["offset"]) - shift, 6) for r in rs]
        fresh = layout_unit("check", [Segment(str(r["segment"]), g, off - on) for r, on, off in
                                      zip(rs, rel_on, rel_off)], hop=hop, gap=gap, trailing_gap=0.0)
        if any(abs(a - b) > 1e-6 for a, b in zip(fresh.onsets, rel_on)):
            raise ValueError(f"{old_unit}: group {g!r} is not laid out as a per-recording unit would be "
                             f"(onsets {rel_on[:3]} vs {fresh.onsets[:3]}); its frames cannot carry")
        uid = unit_id(g)
        if not _UNIT_ID_RE.match(uid):
            raise ValueError(f"unit id {uid!r} is not a native id ([a-z0-9-])")
        if uid in seen:
            raise ValueError(f"{old_unit}: unit id {uid!r} occurs twice")
        seen.add(uid)
        out.append(RecordingPiece(old_unit, uid, g, shift, fresh.duration, tuple(str(r["segment"]) for r in rs),
                                  tuple(rel_on), tuple(rel_off)))
    return out


_GIGA_PUNCT = {
    "<COMMA>": ",",
    "<PERIOD>": ".",
    "<QUESTIONMARK>": "?",
    "<EXCLAMATIONPOINT>": "!",
}
_GIGA_DROP = re.compile(r"<(SIL|MUSIC|NOISE|OTHER)>")


def normalize_gigaspeech_text(text: str) -> str:
    """GigaSpeech transcripts are upper-case with punctuation as tokens
    (``<COMMA>`` ...) and non-speech tags; render them as ordinary
    lower-case prose so text models see in-distribution input."""
    out = _GIGA_DROP.sub(" ", text)
    for tok, ch in _GIGA_PUNCT.items():
        out = out.replace(" " + tok, ch).replace(tok, ch)
    out = re.sub(r"\s+", " ", out).strip().lower()
    return out


def normalize_librispeech_text(text: str) -> str:
    """LibriSpeech transcripts are upper-case and unpunctuated; lower-case
    them (punctuation cannot be restored and is not invented)."""
    return re.sub(r"\s+", " ", text).strip().lower()
