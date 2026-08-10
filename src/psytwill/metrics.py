"""Similarity/distance metrics between row-vector matrices.

Every metric takes two float arrays ``X (n_a, d)`` and ``Y (n_b, d)``
and returns an ``(n_a, n_b)`` matrix. All are NaN-aware by propagation:
a row containing NaN (e.g. word2vec OOV) yields NaN in its entire
row/column of the output rather than crashing — pure numpy throughout,
since sklearn's pairwise functions reject NaN input.

Conventions inherited from word2psy's ``crossmodal.py``: zero-norm rows
are treated as if unit-norm (cosine 0 against everything) rather than
raising; the same convention covers constant rows under correlation
(Pearson undefined -> 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd

from psytwill.exceptions import MetricError


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def cosine(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Cosine similarity (re-normalizes defensively, like crossmodal.py)."""
    return _l2_normalize(X) @ _l2_normalize(Y).T


def _center_rows(X: np.ndarray) -> np.ndarray:
    return X - np.mean(X, axis=1, keepdims=True)


def correlation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson r between row vectors (cosine of row-centered vectors)."""
    return cosine(_center_rows(X), _center_rows(Y))


def _rank_rows(X: np.ndarray) -> np.ndarray:
    # pandas rank: average ties, NaN stays NaN (propagates downstream)
    return pd.DataFrame(X).rank(axis=1).to_numpy()


def spearman(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Spearman rho between row vectors (Pearson on within-row ranks)."""
    return correlation(_rank_rows(X), _rank_rows(Y))


def euclidean(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Euclidean distance (a distance, not a similarity — see form)."""
    sq = (
        (X**2).sum(axis=1)[:, None]
        + (Y**2).sum(axis=1)[None, :]
        - 2.0 * (X @ Y.T)
    )
    return np.sqrt(np.clip(sq, 0.0, None))


@dataclass
class MetricConfig:
    name: str
    func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    form: Literal["similarity", "distance"]
    description: str


METRIC_REGISTRY: dict[str, MetricConfig] = {
    "cosine": MetricConfig(
        "cosine", cosine, "similarity", "Cosine similarity (L2-normalized dot)"
    ),
    "correlation": MetricConfig(
        "correlation", correlation, "similarity", "Pearson r between rows"
    ),
    "spearman": MetricConfig(
        "spearman", spearman, "similarity", "Spearman rho between rows"
    ),
    "euclidean": MetricConfig(
        "euclidean", euclidean, "distance", "Euclidean distance between rows"
    ),
}


def get_metric(name: str) -> MetricConfig:
    if name not in METRIC_REGISTRY:
        raise MetricError(
            f"Unknown metric '{name}'. Available: "
            f"{', '.join(METRIC_REGISTRY)}."
        )
    return METRIC_REGISTRY[name]
