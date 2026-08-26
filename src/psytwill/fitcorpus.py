"""External fit-corpus identity — the ``ext-<corpus>-<native>`` namespace.

psytwill-space block weights are fit on non-mmmdata corpora and frozen
(fit-on-external-then-freeze decision; see the psytwill-space charter in
mmmdata-agents ``docs/workbench/psytwill-space/``). Feature tables extracted
over those corpora carry a namespaced ``stimulus_id`` so they can never
collide with mmmdata stimulus ids: registry ids (``shared####_nsd#####``,
``twp####_...``, movie slugs) never start with ``ext-``.

The corpus registry below is the single source of truth for the namespace.
Parsing is longest-match over registered keys, so corpus keys may themselves
contain hyphens (``ext-peoples-speech-chunk0042`` splits at the registered
key, not the first hyphen). Two invariants keep that unambiguous, enforced
by ``_validate_registry`` at import time:

- keys are lowercase ``[a-z0-9-]``, and
- no key is another key plus a ``-``-joined suffix.

Acquisition tiers (licensing decision, 2026-08-25):

- ``local``  — already on disk beside mmmdata; nothing to fetch.
- ``a``      — CC/public-domain: usable, shippable, scripted fetch.
- ``b``      — nameable-and-obtainable commercial media (CNeuroMod stimuli):
  fit corpus is named, never redistributed; acquisition is a manual,
  documented procedure.
- ``validation`` — external-brain validation targets, not fit stimuli.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from psytwill.exceptions import CorpusError

EXT_PREFIX = "ext-"

_KEY_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_NATIVE_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass(frozen=True)
class CorpusSpec:
    """One external corpus in the fit-corpus ledger."""

    key: str
    tier: str  # local | a | b | validation
    serves: str  # which block(s) / role, free text for humans
    id_width: int | None = None  # zero-pad width for integer native ids
    notes: str = field(default="")


# Keys mirror the corpus ledger in mmmdata-agents
# docs/workbench/psytwill-space/out/fit-corpus-pipeline.md §1.
CORPORA: dict[str, CorpusSpec] = {
    spec.key: spec
    for spec in (
        CorpusSpec(
            "nsd",
            tier="local",
            serves="V private; V<->L (COCO captions ride these ids)",
            id_width=6,
            notes="73k-image NSD hdf5 read in place; ids are 1-based nsdId",
        ),
        CorpusSpec(
            "twp-unpresented",
            tier="local",
            serves="A private (domain-matched voice/word audio)",
            notes="the 3,000 twp recordings frozen out of twp1000",
        ),
        CorpusSpec(
            "librispeech",
            tier="a",
            serves="A private; A<->L (read speech + transcripts)",
        ),
        CorpusSpec(
            "gigaspeech",
            tier="a",
            serves="A private; A<->L (conversational speech)",
        ),
        CorpusSpec(
            "peoples-speech",
            tier="a",
            serves="A private; A<->L (conversational speech)",
        ),
        CorpusSpec(
            "narratives",
            tier="a",
            serves="A<->L (spoken stories); external brains (aud/lang)",
        ),
        CorpusSpec("jamendo", tier="a", serves="A private (music arm)"),
        CorpusSpec("fma", tier="a", serves="A private (music arm)"),
        CorpusSpec(
            "friends",
            tier="b",
            serves="V<->A residual shared axes (weight movie10 up)",
        ),
        CorpusSpec(
            "movie10",
            tier="b",
            serves="V<->A residual shared axes; external brains (AV)",
        ),
        CorpusSpec(
            "things",
            tier="validation",
            serves="external-brain validation for the V block",
            notes="CNeuroMod-THINGS images + CC0 betas",
        ),
    )
}


def _validate_registry(corpora: dict[str, CorpusSpec]) -> None:
    keys = sorted(corpora)
    for key in keys:
        if not _KEY_RE.match(key):
            raise CorpusError(
                f"corpus key {key!r} must be lowercase [a-z0-9-] and "
                "start with a letter"
            )
    for short in keys:
        for long in keys:
            if long != short and long.startswith(short + "-"):
                raise CorpusError(
                    f"corpus keys {short!r} and {long!r} are parse-ambiguous: "
                    "no key may be another key plus a '-'-joined suffix"
                )


_validate_registry(CORPORA)


def ext_id(corpus: str, native_id: str | int) -> str:
    """Build the namespaced stimulus_id ``ext-<corpus>-<native>``.

    Integer native ids are zero-padded to the corpus's ``id_width``
    (``ext_id("nsd", 21384)`` -> ``"ext-nsd-021384"``).
    """
    spec = CORPORA.get(corpus)
    if spec is None:
        raise CorpusError(
            f"unknown fit corpus {corpus!r}; registered: {sorted(CORPORA)}. "
            "New corpora enter via the ledger "
            "(fit-corpus-pipeline.md) and this registry together."
        )
    if isinstance(native_id, int):
        if native_id < 0:
            raise CorpusError(f"native id must be non-negative, got {native_id}")
        native = f"{native_id:0{spec.id_width or 0}d}"
    else:
        native = native_id
    if not _NATIVE_RE.match(native):
        raise CorpusError(
            f"native id {native!r} must be lowercase [a-z0-9-] and start "
            "alphanumeric; normalize upstream in the corpus adapter"
        )
    return f"{EXT_PREFIX}{corpus}-{native}"


def parse_ext_id(stimulus_id: str) -> tuple[str, str]:
    """Split ``ext-<corpus>-<native>`` -> (corpus, native), longest-match."""
    if not stimulus_id.startswith(EXT_PREFIX):
        raise CorpusError(
            f"{stimulus_id!r} is not an external stimulus id "
            f"(no {EXT_PREFIX!r} prefix)"
        )
    rest = stimulus_id[len(EXT_PREFIX):]
    matches = [
        key for key in CORPORA
        if rest == key or rest.startswith(key + "-")
    ]
    if not matches:
        raise CorpusError(
            f"{stimulus_id!r} names no registered corpus; "
            f"registered: {sorted(CORPORA)}"
        )
    corpus = max(matches, key=len)
    native = rest[len(corpus) + 1:]
    if not native:
        raise CorpusError(f"{stimulus_id!r} has an empty native id")
    return corpus, native


def is_external(stimulus_id: str) -> bool:
    """True if the id lives in the fit-corpus namespace."""
    return stimulus_id.startswith(EXT_PREFIX)
