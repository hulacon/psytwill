"""Core matrix computation: self/cross matrices and transition series.

Row order in the input CSV is taken as narrative/stimulus order (the
siblings write chunks and frames in order); transitions are adjacent
rows in that order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from psytwill.exceptions import InputError, SpaceError
from psytwill.metrics import get_metric
from psytwill.spaces import SpaceConfig

# Row-label candidates, in preference order — the union of word2psy's
# text chain (chunk_label -> chunk_idx) and viz2psy's image chain
# (filename -> filepath -> image_idx -> time).
LABEL_COLUMNS = ["chunk_label", "filename", "filepath", "image_idx", "time", "chunk_idx"]


def resolve_labels(df: pd.DataFrame) -> tuple[list[str], str]:
    """Row labels for a scores CSV and the column they came from."""
    for col in LABEL_COLUMNS:
        if col in df.columns:
            return df[col].astype(str).tolist(), col
    return [f"row_{i}" for i in range(len(df))], "(row order)"


@dataclass
class MatrixResult:
    """One computed matrix plus what the sidecar needs to describe it."""

    key: str  # e.g. "minilm__cosine" or "clip_text__x__clip__cosine"
    frame: pd.DataFrame  # labeled rows (input A) x columns (input B or A)
    space_a: SpaceConfig
    space_b: SpaceConfig  # == space_a in self mode
    metric: str
    form: str  # "similarity" | "distance"
    n_valid_a: int
    n_valid_b: int
    nan_labels_a: list[str]
    nan_labels_b: list[str]


def _extract(df: pd.DataFrame, space: SpaceConfig) -> np.ndarray:
    try:
        return df[space.columns].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise InputError(
            f"Space '{space.name}' has non-numeric values: {exc}"
        ) from exc


def _zscore_columns(
    Xa: np.ndarray, Xb: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Z-score each feature column; pooled stats across both inputs."""
    pool = Xa if Xb is None else np.vstack([Xa, Xb])
    mean = np.nanmean(pool, axis=0)
    sd = np.nanstd(pool, axis=0)
    sd = np.where((sd == 0) | np.isnan(sd), 1.0, sd)
    za = (Xa - mean) / sd
    zb = None if Xb is None else (Xb - mean) / sd
    return za, zb


def _nan_rows(X: np.ndarray) -> np.ndarray:
    return np.isnan(X).any(axis=1)


def compute_matrix(
    df_a: pd.DataFrame,
    space_a: SpaceConfig,
    metric_name: str | None = None,
    df_b: pd.DataFrame | None = None,
    space_b: SpaceConfig | None = None,
    distance: bool = False,
) -> MatrixResult:
    """One relational matrix: self (df_b None) or cross.

    Profile spaces are z-scored per column (pooled across both inputs in
    cross mode) before scale-sensitive metrics (cosine, euclidean);
    correlation/spearman are unaffected by their own row-centering/
    ranking, so raw values are used there.
    """
    cross = df_b is not None
    space_b = space_b or space_a
    metric = get_metric(metric_name or space_a.default_metric)

    if space_a.n_dims != space_b.n_dims:
        raise SpaceError(
            f"Dimensions differ: {space_a.name} has {space_a.n_dims}, "
            f"{space_b.name} has {space_b.n_dims}. These spaces cannot "
            "be compared."
        )

    Xa = _extract(df_a, space_a)
    Xb = _extract(df_b, space_b) if cross else None

    if space_a.kind == "profile" and metric.name in ("cosine", "euclidean"):
        Xa, Xb = _zscore_columns(Xa, Xb)

    M = metric.func(Xa, Xa if Xb is None else Xb)
    form = metric.form
    if distance and form == "similarity":
        M = 1.0 - M
        form = "distance"

    labels_a, _ = resolve_labels(df_a)
    labels_b = labels_a if not cross else resolve_labels(df_b)[0]
    frame = pd.DataFrame(M, index=labels_a, columns=labels_b)

    nan_a = _nan_rows(Xa)
    nan_b = nan_a if Xb is None else _nan_rows(Xb)
    if space_a.name == space_b.name:
        key = f"{space_a.name}__{metric.name}"
    else:
        key = f"{space_a.name}__x__{space_b.name}__{metric.name}"

    return MatrixResult(
        key=key,
        frame=frame,
        space_a=space_a,
        space_b=space_b,
        metric=metric.name,
        form=form,
        n_valid_a=int((~nan_a).sum()),
        n_valid_b=int((~nan_b).sum()),
        nan_labels_a=[l for l, bad in zip(labels_a, nan_a) if bad],
        nan_labels_b=[l for l, bad in zip(labels_b, nan_b) if bad],
    )


def transition_records(result: MatrixResult) -> list[dict]:
    """Adjacent-chunk series from a self matrix (the off-diagonal band).

    One record per chunk boundary: value = relation(chunk_i, chunk_i+1).
    NaN rows yield NaN values (kept, so curves show gaps explicitly).
    """
    M = result.frame.to_numpy()
    labels = list(result.frame.index)
    return [
        {
            "boundary": i,
            "from_label": labels[i],
            "to_label": labels[i + 1],
            "space": result.key.rsplit("__", 1)[0],
            "metric": result.metric,
            "value": M[i, i + 1],
        }
        for i in range(len(labels) - 1)
    ]


def diagonal_records(result: MatrixResult) -> list[dict]:
    """Aligned-pairs series from a cross matrix with equal row counts."""
    M = result.frame.to_numpy()
    if M.shape[0] != M.shape[1]:
        raise SpaceError(
            f"Diagonal series needs equal row counts; got {M.shape[0]} "
            f"and {M.shape[1]}."
        )
    return [
        {
            "idx": i,
            "label_a": result.frame.index[i],
            "label_b": result.frame.columns[i],
            "space": result.key.rsplit("__", 1)[0],
            "metric": result.metric,
            "value": M[i, i],
        }
        for i in range(M.shape[0])
    ]
