"""Matrix computation: self/cross, labels, NaN policy, series."""

import numpy as np
import pandas as pd
import pytest

from psyquilt.exceptions import SpaceError
from psyquilt.matrices import (
    compute_matrix,
    diagonal_records,
    resolve_labels,
    transition_records,
)
from psyquilt.spaces import detect_spaces

from tests.conftest import embedding_frame, unit


def _space(df, name):
    return detect_spaces(df.columns)[name]


class TestResolveLabels:
    def test_chunk_label_preferred(self):
        df = pd.DataFrame({"chunk_label": ["a", "b"], "chunk_idx": [0, 1]})
        assert resolve_labels(df) == (["a", "b"], "chunk_label")

    def test_viz2psy_filename(self):
        df = pd.DataFrame({"filename": ["x.png"], "image_idx": [0]})
        assert resolve_labels(df) == (["x.png"], "filename")

    def test_fallback_row_order(self):
        df = pd.DataFrame({"foo": [1, 2]})
        assert resolve_labels(df) == (["row_0", "row_1"], "(row order)")


class TestSelfMatrix:
    def test_square_symmetric_unit_diagonal(self, two_topic_frame):
        space = _space(two_topic_frame, "minilm")
        r = compute_matrix(two_topic_frame, space)
        M = r.frame.to_numpy()
        assert M.shape == (8, 8)
        assert np.allclose(np.diag(M), 1.0)
        assert np.allclose(M, M.T)
        assert r.form == "similarity"
        assert r.key == "minilm__cosine"

    def test_block_structure(self, two_topic_frame):
        # Interleaved topics: same-topic pairs similar, cross-topic not
        space = _space(two_topic_frame, "minilm")
        M = compute_matrix(two_topic_frame, space).frame.to_numpy()
        same = [M[i, j] for i in range(8) for j in range(8)
                if i != j and i % 2 == j % 2]
        diff = [M[i, j] for i in range(8) for j in range(8) if i % 2 != j % 2]
        assert min(same) > max(diff)

    def test_distance_flag(self, two_topic_frame):
        space = _space(two_topic_frame, "minilm")
        r = compute_matrix(two_topic_frame, space, distance=True)
        assert r.form == "distance"
        assert np.allclose(np.diag(r.frame.to_numpy()), 0.0)

    def test_nan_row_kept_with_nan_entries(self):
        vecs = [unit(4, 0), [np.nan] * 4, unit(4, 1)]
        df = embedding_frame(vecs, labels=["a", "oov", "b"])
        space = _space(df, "minilm")
        r = compute_matrix(df, space)
        M = r.frame.to_numpy()
        assert M.shape == (3, 3)  # NaN row kept, shape preserved
        assert np.isnan(M[1, :]).all() and np.isnan(M[:, 1]).all()
        assert r.n_valid_a == 2
        assert r.nan_labels_a == ["oov"]

    def test_profile_zscored_for_euclidean(self):
        # One high-variance column must not dominate after z-scoring
        df = pd.DataFrame({
            "emotion_joy": [0.0, 1.0, 0.0],
            "emotion_fear": [0.0, 0.0, 1000.0],
        })
        space = _space(df, "emotion")
        r = compute_matrix(df, space, metric_name="euclidean")
        M = r.frame.to_numpy()
        # After per-column z-scoring both deviant rows are comparably far
        # from row 0; raw values would make row 2 ~1000x farther.
        assert M[0, 2] / M[0, 1] < 2.0


class TestCrossMatrix:
    def test_clip_text_x_clip(self):
        texts = embedding_frame(
            [unit(4, 0), unit(4, 1)], prefix="clip_text", labels=["cat", "dog"]
        )
        images = embedding_frame([unit(4, 1), unit(4, 0)], prefix="clip")
        images["filename"] = ["dog.png", "cat.png"]
        sa = _space(texts, "clip_text")
        sb = _space(images, "clip")
        r = compute_matrix(texts, sa, df_b=images, space_b=sb)
        assert r.key == "clip_text__x__clip__cosine"
        assert r.frame.loc["cat", "cat.png"] == pytest.approx(1.0)
        assert r.frame.loc["dog", "dog.png"] == pytest.approx(1.0)
        assert r.frame.loc["cat", "dog.png"] == pytest.approx(0.0)

    def test_dimension_mismatch_raises(self):
        a = embedding_frame([unit(4, 0)], prefix="clip_text")
        b = embedding_frame([unit(6, 0)], prefix="clip")
        with pytest.raises(SpaceError, match="Dimensions differ"):
            compute_matrix(
                a, _space(a, "clip_text"),
                df_b=b, space_b=_space(b, "clip"),
            )

    def test_rectangular(self):
        a = embedding_frame([unit(4, 0)] * 3, prefix="minilm")
        b = embedding_frame([unit(4, 0)] * 5, prefix="minilm")
        r = compute_matrix(a, _space(a, "minilm"), df_b=b, space_b=_space(b, "minilm"))
        assert r.frame.shape == (3, 5)


class TestSeries:
    def test_transitions_match_offdiagonal(self, two_topic_frame):
        space = _space(two_topic_frame, "minilm")
        r = compute_matrix(two_topic_frame, space)
        recs = transition_records(r)
        assert len(recs) == 7
        M = r.frame.to_numpy()
        for rec in recs:
            assert rec["value"] == pytest.approx(M[rec["boundary"], rec["boundary"] + 1])
        assert recs[0]["space"] == "minilm"
        assert recs[0]["from_label"] == "s0/A"
        assert recs[0]["to_label"] == "s1/B"

    def test_diagonal_requires_square(self):
        a = embedding_frame([unit(4, 0)] * 2, prefix="minilm")
        b = embedding_frame([unit(4, 0)] * 3, prefix="minilm")
        r = compute_matrix(a, _space(a, "minilm"), df_b=b, space_b=_space(b, "minilm"))
        with pytest.raises(SpaceError, match="equal row counts"):
            diagonal_records(r)

    def test_diagonal_values(self):
        vecs = [unit(4, 0), unit(4, 1)]
        a = embedding_frame(vecs, prefix="minilm", labels=["a0", "a1"])
        b = embedding_frame(vecs, prefix="minilm", labels=["b0", "b1"])
        r = compute_matrix(a, _space(a, "minilm"), df_b=b, space_b=_space(b, "minilm"))
        recs = diagonal_records(r)
        assert [rec["value"] for rec in recs] == [pytest.approx(1.0)] * 2
        assert recs[1]["label_a"] == "a1" and recs[1]["label_b"] == "b1"
