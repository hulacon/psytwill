"""Feature-space detection and registry.

A *space* is a named group of columns in a scores CSV together with a
default metric. Spaces come from two tiers:

1. **Generic embedding detection** — any column group matching
   ``{prefix}_{i:03d}`` (the sibling convention: ``minilm_000``,
   ``clip_text_511``, viz2psy's ``clip_000``). The greedy prefix match
   keeps ``clip_text_###`` and ``clip_###`` apart, generalizing the
   two-regex trick in word2psy's ``crossmodal.py``. A group must have
   >= 2 columns with contiguous indices starting at 000.
2. **Named-profile registry** — explicit column patterns for scalar
   feature families that form meaningful profile vectors (emotion,
   sentiment, readability, word-aggregate means). Ported from word2psy's
   ``viz/feature_config.py``.

Index/passthrough columns (``chunk_idx``, ``filename``, condition
columns, ...) are never swept into a space: embeddings by the strict
``_###`` suffix requirement, profiles by explicit patterns.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Iterable, Literal

EMBEDDING_RE = re.compile(r"^(?P<prefix>.+)_(?P<index>\d{3})$")

# Row-identifier columns from the sibling CSV conventions. Excluded from
# embedding detection defensively (none currently match the _### suffix).
INDEX_COLUMNS = {
    "chunk_idx",
    "chunk_label",
    "n_words",
    "word",
    "word_idx",
    "sentence_idx",
    "onset",
    "offset",
    "filename",
    "filepath",
    "image_idx",
    "time",
}

# Cross-modal space pairs that live in one shared representational space
# (word2psy clip_text and viz2psy clip share an OpenCLIP ViT-B-32
# checkpoint). Checked by equal dimensionality at compute time.
COMPATIBLE_SPACES = [("clip_text", "clip"), ("clip", "clip_text")]


@dataclass
class SpaceConfig:
    """One feature space resolved against a concrete CSV's columns."""

    name: str
    kind: Literal["embedding", "profile"]
    columns: list[str]  # resolved, ordered column names
    default_metric: str
    source: Literal["detected", "registry"]
    description: str = ""

    @property
    def n_dims(self) -> int:
        return len(self.columns)


# --- named-profile registry (tier 2) ------------------------------------

_NORM_FEATURES = [
    "concreteness",
    "valence",
    "arousal",
    "dominance",
    "age_of_acquisition",
    "imageability",
    "familiarity",
    "semantic_size",
    "gender_association",
    "socialness",
    "body_object_interaction",
    "sensorimotor_touch",
    "sensorimotor_hearing",
    "sensorimotor_smell",
    "sensorimotor_taste",
    "sensorimotor_vision",
    "sensorimotor_interoception",
    "sensorimotor_mouth",
    "sensorimotor_hand",
    "sensorimotor_foot",
    "sensorimotor_head",
    "sensorimotor_torso",
    "zipf_frequency",
]

_WORDFORM_FEATURES = ["length", "n_syllables", "n_phonemes", "old20"]

# name -> (column patterns, description). Patterns are fnmatch-style.
PROFILE_REGISTRY: dict[str, tuple[list[str], str]] = {
    "emotion": (["emotion_*"], "GoEmotions probability profile"),
    "sentiment": (["sentiment_*"], "Negative/neutral/positive profile"),
    "readability": (["readability_*"], "Classic readability metrics"),
    # Per-chunk means of word-level features (word2psy aggregates).
    # Mean only: mixing mean/sd/min/max in one profile would mix scales.
    "word_aggregates": (
        [
            f"{feat}_mean"
            for feat in _NORM_FEATURES + _WORDFORM_FEATURES + ["gpt2_surprisal"]
        ],
        "Per-chunk means of word-level norms and surprisal",
    ),
}

DEFAULT_METRICS = {"embedding": "cosine", "profile": "correlation"}


def detect_embedding_spaces(columns: Iterable[str]) -> dict[str, SpaceConfig]:
    """Tier 1: group ``{prefix}_{i:03d}`` columns by prefix."""
    groups: dict[str, list[tuple[int, str]]] = {}
    for col in columns:
        if col in INDEX_COLUMNS:
            continue
        m = EMBEDDING_RE.match(col)
        if m:
            groups.setdefault(m["prefix"], []).append((int(m["index"]), col))

    spaces = {}
    for prefix, indexed in groups.items():
        indexed.sort()
        indices = [i for i, _ in indexed]
        # Require the sibling convention exactly: >= 2 dims, contiguous
        # from 000 — a stray passthrough column like "topic_001" is not
        # an embedding.
        if len(indexed) < 2 or indices != list(range(len(indexed))):
            continue
        spaces[prefix] = SpaceConfig(
            name=prefix,
            kind="embedding",
            columns=[c for _, c in indexed],
            default_metric=DEFAULT_METRICS["embedding"],
            source="detected",
            description=f"{len(indexed)}-dim {prefix} embeddings",
        )
    return spaces


def detect_profile_spaces(columns: Iterable[str]) -> dict[str, SpaceConfig]:
    """Tier 2: match the named-profile registry against columns."""
    columns = list(columns)
    spaces = {}
    for name, (patterns, description) in PROFILE_REGISTRY.items():
        matched: list[str] = []
        for pattern in patterns:
            for col in columns:
                if fnmatch.fnmatch(col, pattern) and col not in matched:
                    matched.append(col)
        if len(matched) >= 2:
            spaces[name] = SpaceConfig(
                name=name,
                kind="profile",
                columns=matched,
                default_metric=DEFAULT_METRICS["profile"],
                source="registry",
                description=description,
            )
    return spaces


def detect_spaces(columns: Iterable[str]) -> dict[str, SpaceConfig]:
    """All spaces available in a CSV's columns (embeddings first)."""
    columns = list(columns)
    spaces = detect_embedding_spaces(columns)
    for name, config in detect_profile_spaces(columns).items():
        spaces.setdefault(name, config)
    return spaces


def match_spaces(
    spaces_a: dict[str, SpaceConfig],
    spaces_b: dict[str, SpaceConfig],
) -> list[tuple[SpaceConfig, SpaceConfig]]:
    """Space pairs comparable across two CSVs (cross mode).

    Same-name spaces pair directly; declared-compatible cross-modal
    pairs (clip_text x clip) are added as distinct comparisons.
    """
    pairs = [
        (sa, spaces_b[name]) for name, sa in spaces_a.items() if name in spaces_b
    ]
    for name_a, name_b in COMPATIBLE_SPACES:
        if name_a in spaces_a and name_b in spaces_b:
            pairs.append((spaces_a[name_a], spaces_b[name_b]))
    return pairs
