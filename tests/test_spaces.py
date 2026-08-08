"""Space detection: generic embeddings, profile registry, cross matching."""

import pandas as pd

from psyquilt.spaces import (
    detect_embedding_spaces,
    detect_profile_spaces,
    detect_spaces,
    match_spaces,
)


def _cols(*groups, extra=()):
    cols = list(extra)
    for prefix, n in groups:
        cols += [f"{prefix}_{i:03d}" for i in range(n)]
    return cols


class TestEmbeddingDetection:
    def test_groups_by_prefix(self):
        spaces = detect_embedding_spaces(_cols(("minilm", 4), ("fasttext", 3)))
        assert set(spaces) == {"minilm", "fasttext"}
        assert spaces["minilm"].n_dims == 4
        assert spaces["minilm"].columns == [
            "minilm_000", "minilm_001", "minilm_002", "minilm_003"
        ]
        assert spaces["minilm"].default_metric == "cosine"

    def test_clip_vs_clip_text_disambiguation(self):
        # The word2psy crossmodal regex guarantee, generalized
        spaces = detect_embedding_spaces(_cols(("clip", 4), ("clip_text", 4)))
        assert set(spaces) == {"clip", "clip_text"}
        assert spaces["clip"].columns[0] == "clip_000"
        assert spaces["clip_text"].columns[0] == "clip_text_000"

    def test_unordered_columns_are_sorted(self):
        cols = [f"minilm_{i:03d}" for i in (2, 0, 1)]
        spaces = detect_embedding_spaces(cols)
        assert spaces["minilm"].columns == ["minilm_000", "minilm_001", "minilm_002"]

    def test_singleton_group_rejected(self):
        # A stray passthrough column like "topic_001" is not an embedding
        assert detect_embedding_spaces(["topic_001"]) == {}

    def test_noncontiguous_group_rejected(self):
        assert detect_embedding_spaces(["x_000", "x_002"]) == {}
        assert detect_embedding_spaces(["x_001", "x_002"]) == {}

    def test_index_and_passthrough_columns_ignored(self):
        cols = _cols(("minilm", 2), extra=["chunk_idx", "chunk_label", "n_words", "condition"])
        assert set(detect_embedding_spaces(cols)) == {"minilm"}


class TestProfileDetection:
    def test_emotion_profile(self):
        cols = ["emotion_joy", "emotion_fear", "emotion_anger", "chunk_idx"]
        spaces = detect_profile_spaces(cols)
        assert set(spaces) == {"emotion"}
        assert spaces["emotion"].kind == "profile"
        assert spaces["emotion"].default_metric == "correlation"
        assert spaces["emotion"].n_dims == 3

    def test_word_aggregates_mean_only(self):
        cols = [
            "concreteness_mean", "concreteness_sd", "concreteness_min",
            "valence_mean", "valence_max",
        ]
        spaces = detect_profile_spaces(cols)
        assert spaces["word_aggregates"].columns == [
            "concreteness_mean", "valence_mean"
        ]

    def test_single_column_profile_rejected(self):
        assert detect_profile_spaces(["emotion_joy"]) == {}


class TestDetectSpaces:
    def test_combined(self):
        cols = _cols(
            ("minilm", 3),
            extra=["chunk_idx", "sentiment_negative", "sentiment_neutral",
                   "sentiment_positive"],
        )
        spaces = detect_spaces(cols)
        assert list(spaces) == ["minilm", "sentiment"]


class TestMatchSpaces:
    def test_same_name_pairs(self):
        a = detect_spaces(_cols(("minilm", 3), ("fasttext", 3)))
        b = detect_spaces(_cols(("minilm", 3)))
        pairs = match_spaces(a, b)
        assert [(pa.name, pb.name) for pa, pb in pairs] == [("minilm", "minilm")]

    def test_clip_text_x_clip_compatible(self):
        a = detect_spaces(_cols(("clip_text", 4)))
        b = detect_spaces(_cols(("clip", 4)))
        pairs = match_spaces(a, b)
        assert [(pa.name, pb.name) for pa, pb in pairs] == [("clip_text", "clip")]

    def test_no_overlap(self):
        a = detect_spaces(_cols(("minilm", 3)))
        b = detect_spaces(_cols(("gist", 3)))
        assert match_spaces(a, b) == []
