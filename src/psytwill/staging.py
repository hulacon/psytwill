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
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

__all__ = [
    "Segment",
    "Unit",
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
                gap: float = 0.5) -> Unit:
    """Assign grid-aligned onsets to ``segments`` in order.

    Segment ``k`` starts at the first grid point at or after
    ``offset(k-1) + gap``; the unit ends one ``gap`` after the last
    offset, rounded up to the grid so the unit length is a whole number
    of frames.
    """
    if gap < 0 or hop <= 0:
        raise ValueError(f"need gap >= 0 and hop > 0, got gap={gap}, hop={hop}")
    onsets: list[float] = []
    t = 0.0
    for seg in segments:
        onset = _grid_ceil(t, hop)
        onsets.append(onset)
        t = onset + seg.duration + gap
    return Unit(unit=unit, segments=list(segments), onsets=onsets,
                duration=_grid_ceil(t, hop) if segments else 0.0)


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
