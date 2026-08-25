"""Feature-space detection and registry.

A *space* is a named group of columns in a scores CSV together with a
default metric. Spaces come from two tiers:

1. **Generic embedding detection** — any column group matching
   ``{prefix}_{index}`` with a fixed-width, >= 3-digit index (the sibling
   convention: ``minilm_000``, ``clip_text_511``, viz2psy's ``clip_000``,
   and 4-digit for >999-d spaces like ``ebind_0000..ebind_1023``). The
   index must be *all* trailing digits, which keeps ``clip_text_###`` and
   ``clip_###`` apart (generalizing the two-regex trick in word2psy's
   ``crossmodal.py``) and stops a 4-digit ``ebind_1023`` from being
   misread as prefix ``ebind_1``. A group must have >= 2 columns with
   contiguous indices starting at 0.
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

# Lazy prefix + unbounded index: the index captures ALL trailing digits
# (>= 3), so 4-digit columns in >999-d spaces parse with the right prefix.
EMBEDDING_RE = re.compile(r"^(?P<prefix>.+?)_(?P<index>\d{3,})$")

# Row-identifier columns from the sibling CSV conventions. Excluded from
# embedding detection defensively (none currently match the _### suffix).
# Mirrors the reserved-column registry in constellation-contracts.md §4.1.
INDEX_COLUMNS = {
    "stimulus_id",
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
    "voice",
    "speaker",
    "turn_idx",
}

# Cross-modal space pairs that live in one shared representational space
# (word2psy clip_text and viz2psy clip share an OpenCLIP ViT-B-32
# checkpoint; word2psy clap_text and aud2psy clap share a LAION-CLAP
# checkpoint; viz2psy ebind, word2psy ebind_text, and aud2psy ebind_audio
# all share the EBind encord-team/ebind-full checkpoint). Checked by equal
# dimensionality at compute time.
COMPATIBLE_SPACES = [
    ("clip_text", "clip"), ("clip", "clip_text"),
    ("clap_text", "clap"), ("clap", "clap_text"),
    ("ebind", "ebind_text"), ("ebind_text", "ebind"),
    ("ebind", "ebind_audio"), ("ebind_audio", "ebind"),
    ("ebind_text", "ebind_audio"), ("ebind_audio", "ebind_text"),
]


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

# word2psy 0.4.0 prefixed these (Contract B §4.1); they were bare before,
# which left psytwill unable to attribute any of them to a model.
_NORM_FEATURES = [
    "lexical_norms_concreteness",
    "lexical_norms_valence",
    "lexical_norms_arousal",
    "lexical_norms_dominance",
    "lexical_norms_age_of_acquisition",
    "lexical_norms_imageability",
    "lexical_norms_familiarity",
    "lexical_norms_semantic_size",
    "lexical_norms_gender_association",
    "lexical_norms_socialness",
    "lexical_norms_body_object_interaction",
    "lexical_norms_sensorimotor_touch",
    "lexical_norms_sensorimotor_hearing",
    "lexical_norms_sensorimotor_smell",
    "lexical_norms_sensorimotor_taste",
    "lexical_norms_sensorimotor_vision",
    "lexical_norms_sensorimotor_interoception",
    "lexical_norms_sensorimotor_mouth",
    "lexical_norms_sensorimotor_hand",
    "lexical_norms_sensorimotor_foot",
    "lexical_norms_sensorimotor_head",
    "lexical_norms_sensorimotor_torso",
    "lexical_norms_zipf_frequency",
]

_WORDFORM_FEATURES = [
    "wordform_length",
    "wordform_n_syllables",
    "wordform_n_phonemes",
    "wordform_old20",
]

# aud2psy frame-level features (model-prefixed, one value per timepoint).
_ACOUSTIC_PATTERNS = [
    "loudness_*", "pitch_*", "spectral_*", "onsets_*",
    "tonal_*", "rhythm_*", "speech_prob",
]

# name -> (column patterns, description). Patterns are fnmatch-style.
PROFILE_REGISTRY: dict[str, tuple[list[str], str]] = {
    "emotion": (["emotion_*"], "GoEmotions probability profile"),
    "sentiment": (["sentiment_*"], "Negative/neutral/positive profile"),
    "readability": (["readability_*"], "Classic readability metrics"),
    "interaction": (["interaction_*"], "Social-interaction rates (word2psy)"),
    # aud2psy per-model profiles (speech_prob is a single column, so it
    # only appears inside the combined acoustic profile).
    "loudness": (["loudness_*"], "RMS energy and dB level (aud2psy)"),
    "pitch": (["pitch_*"], "pYIN f0 and voicing probability (aud2psy)"),
    "spectral": (["spectral_*"], "Spectral shape features (aud2psy)"),
    "onsets": (["onsets_*"], "Onset strength/rate and local tempo (aud2psy)"),
    "tonal": (["tonal_*"], "Key clarity, majorness, chroma entropy (aud2psy)"),
    "rhythm": (["rhythm_*"], "Pulse clarity, beat strength, novelty (aud2psy)"),
    # Learned affective judgment — deliberately NOT part of the combined
    # signal-level `acoustic` profile (the word_aggregates precedent).
    "music_emotion": (["music_emotion_*"], "DEAM-trained musical valence/arousal (aud2psy)"),
    "speech_emotion": (["speech_emotion_*"], "MSP-Podcast vocal arousal/dominance/valence (aud2psy)"),
    # Raw openSMILE LLDs — its own 25-d profile, not folded into `acoustic`
    # (parallel algorithms to the librosa set; would also swamp the 19-d mix).
    "egemaps": (["egemaps_*"], "eGeMAPS prosody/voice-quality LLDs (aud2psy)"),
    # Diarization-derived speaker/turn structure — who-speaks-when, not
    # signal acoustics; stays out of `acoustic` (the music_emotion
    # precedent: derived structure does not join the signal-level mix).
    "conversation": (["conversation_*"], "Conversation structure from diarization (aud2psy)"),
    # Optical-flow statistics — viz2psy's temporal arm, video only.
    "motion": (["motion_*"], "Optical-flow motion statistics (viz2psy)"),
    # Face count/size/configuration — social-visual structure beyond
    # yolo's person count.
    "faces": (["faces_*"], "Face count/size/configuration (viz2psy)"),
    "acoustic": (_ACOUSTIC_PATTERNS, "All aud2psy frame-level acoustic features"),
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
