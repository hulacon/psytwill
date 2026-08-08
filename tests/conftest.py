"""Shared synthetic-frame helpers. Fully offline: no models, no downloads."""

import numpy as np
import pandas as pd
import pytest


def embedding_frame(
    vectors, prefix="minilm", labels=None, extra=None
) -> pd.DataFrame:
    """Chunk CSV-shaped frame with {prefix}_{i:03d} columns."""
    vectors = np.asarray(vectors, dtype=float)
    data = {}
    if labels is not None:
        data["chunk_idx"] = np.arange(len(vectors))
        data["chunk_label"] = labels
    for i in range(vectors.shape[1]):
        data[f"{prefix}_{i:03d}"] = vectors[:, i]
    if extra:
        data.update(extra)
    return pd.DataFrame(data)


def unit(dim, hot):
    v = np.zeros(dim)
    v[hot] = 1.0
    return v


@pytest.fixture
def two_topic_frame():
    """8 chunks alternating between two orthogonal 'topics' (A B A B...).

    Topic vectors are noisy copies of two orthogonal bases, so the
    cosine matrix should show a checkerboard/block structure.
    """
    rng = np.random.RandomState(7)
    base_a, base_b = unit(16, 0), unit(16, 8)
    rows = []
    for i in range(8):
        base = base_a if i % 2 == 0 else base_b
        rows.append(base + rng.randn(16) * 0.05)
    labels = [f"s{i}/{'A' if i % 2 == 0 else 'B'}" for i in range(8)]
    return embedding_frame(rows, prefix="minilm", labels=labels)
