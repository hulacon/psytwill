"""Second-order measures: how two feature *spaces* relate to each other.

Everything else in psytwill is first-order — one space at a time, producing
an N x N matrix over stimuli. These measures take two spaces defined on the
**same rows** and return a single number describing their relationship:

- :func:`ridge_predictivity` — cross-validated R^2 of A -> B. The only
  asymmetric measure, which is the point: "does EBind contain emonet" is a
  different question from "does emonet contain EBind".
- :func:`cka` — linear and RBF Centered Kernel Alignment; the standard
  comparison for spaces of unequal dimensionality.
- :func:`second_order_rsa` — Spearman over the off-diagonal entries of the
  two spaces' RDMs, which psytwill already knows how to build.
- :func:`neighbor_overlap` — shared k-nearest-neighbour fraction; catches
  nonlinear agreement the linear measures miss.
- :func:`participation_ratio` — effective dimensionality, so a 1024-d space
  is not mistaken for 1024 degrees of freedom.

Three conventions, each of which exists because getting it wrong produces a
plausible number rather than an error:

**Rows must be aligned and grouped honestly.** These measures assume row *i*
of ``X`` and row *i* of ``Y`` describe the same stimulus at the same moment.
Where rows are not independent — a 0.5 s movie grid, where adjacent frames
are near-duplicates — pass ``groups`` so folds split by clip. Random folds on
autocorrelated rows have the model predict a frame from its own neighbours,
and R^2 inflates towards 1 with no signal present.

**Nulls respect the same structure.** :func:`permutation_null` takes a
``block_size``; on temporally ordered rows, shuffle blocks rather than rows,
or the null destroys autocorrelation that the observed value keeps and every
comparison clears it.

**NaN rows are dropped, not propagated.** psytwill's first-order metrics
propagate NaN into the output matrix (word2vec OOV is legitimate). A scalar
summary has nowhere to propagate to, so rows with any NaN in either space are
dropped and the count is reported alongside the value.

Dependencies stay numpy / pandas / scikit-learn, as elsewhere in psytwill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from psytwill.exceptions import SpaceError
from psytwill.metrics import get_metric

DEFAULT_ALPHAS: tuple[float, ...] = tuple(np.logspace(-2, 8, 21))
DEFAULT_K = 20


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def _as_2d(X, name: str) -> np.ndarray:
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if A.ndim != 2:
        raise SpaceError(
            f"'{name}' must be 2-d (n_stimuli, n_features); got shape "
            f"{A.shape}. Reshape a single feature to (-1, 1)."
        )
    if A.shape[0] == 0:
        raise SpaceError(f"'{name}' has no rows.")
    return A


def _align(X, Y, groups=None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """Check row alignment, drop rows with NaN in either space."""
    A, B = _as_2d(X, "X"), _as_2d(Y, "Y")
    if A.shape[0] != B.shape[0]:
        raise SpaceError(
            f"Spaces must share a row index: X has {A.shape[0]} rows, Y has "
            f"{B.shape[0]}. Project both onto a common grain before comparing "
            "(pooling for chunk grain, interval matching for irregular text)."
        )
    keep = ~(np.isnan(A).any(axis=1) | np.isnan(B).any(axis=1))
    g = None if groups is None else np.asarray(groups)[keep]
    if g is not None and g.shape[0] != keep.sum():
        raise SpaceError("'groups' must have one entry per row of X/Y.")
    return A[keep], B[keep], g, int((~keep).sum())


def _column_center(A: np.ndarray) -> np.ndarray:
    return A - A.mean(axis=0, keepdims=True)


def _offdiag(M: np.ndarray) -> np.ndarray:
    """Upper-triangle entries, excluding the diagonal."""
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def _rank(v: np.ndarray) -> np.ndarray:
    return pd.Series(v).rank().to_numpy()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:  # a constant vector has no correlation with anything
        return 0.0
    return float(a @ b / denom)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_rank(a), _rank(b))


# --------------------------------------------------------------------------
# effective dimensionality
# --------------------------------------------------------------------------


def participation_ratio(X) -> float:
    """Effective dimensionality: ``(sum L)^2 / sum L^2`` over covariance eigenvalues.

    1.0 for a rank-1 space, ~d for an isotropic d-dimensional one. Computed
    from the SVD of the column-centered matrix, so it costs the same whether
    the space is 3-d or 1024-d.
    """
    A = _column_center(_as_2d(X, "X"))
    A = A[~np.isnan(A).any(axis=1)]
    if A.shape[0] < 2:
        raise SpaceError("Participation ratio needs at least 2 rows.")
    s = np.linalg.svd(A, compute_uv=False)
    lam = s**2
    total = lam.sum()
    if total == 0:
        return 0.0
    return float(total**2 / (lam**2).sum())


# --------------------------------------------------------------------------
# ridge predictivity (asymmetric)
# --------------------------------------------------------------------------


@dataclass
class RidgeResult:
    """Cross-validated predictivity of one space from another."""

    r2: float
    """Pooled out-of-fold R^2, variance-weighted across target dimensions."""
    r2_mean: float
    """Unweighted mean R^2 across target dimensions."""
    r2_per_fold: list[float]
    n_used: int
    n_dropped: int
    n_splits: int
    grouped: bool
    alphas_selected: list[float] = field(default_factory=list)


def ridge_predictivity(
    X,
    Y,
    *,
    groups: Sequence | None = None,
    n_splits: int = 5,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    random_state: int = 0,
) -> RidgeResult:
    """Cross-validated R^2 predicting space ``Y`` from space ``X``.

    Alpha is selected **inside** each training fold (RidgeCV's efficient
    leave-one-out GCV), never once on the whole set: the R^2 >= 0.5 style
    criterion moves with regularization strength, so a fixed alpha silently
    decides the verdict.

    Pass ``groups`` (e.g. one clip id per row) to split by group. On any
    temporally ordered grid this is not optional — see the module docstring.
    """
    A, B, g, n_dropped = _align(X, Y, groups)
    n = A.shape[0]
    if n < 3:
        raise SpaceError(f"Need at least 3 usable rows; {n} survived NaN removal.")

    if g is not None:
        n_groups = len(np.unique(g))
        if n_groups < 2:
            raise SpaceError(
                "'groups' has a single group, so no split holds a group out. "
                "Drop groups, or compare on a set spanning several clips."
            )
        splits = GroupKFold(n_splits=min(n_splits, n_groups)).split(A, B, groups=g)
        n_splits_used = min(n_splits, n_groups)
    else:
        n_splits_used = min(n_splits, n)
        splits = KFold(
            n_splits=n_splits_used, shuffle=True, random_state=random_state
        ).split(A)

    oof = np.full_like(B, np.nan, dtype=float)
    per_fold: list[float] = []
    chosen: list[float] = []
    for train, test in splits:
        scaler = StandardScaler().fit(A[train])
        model = RidgeCV(alphas=list(alphas), alpha_per_target=True)
        model.fit(scaler.transform(A[train]), B[train])
        pred = model.predict(scaler.transform(A[test]))
        oof[test] = pred.reshape(len(test), -1)
        per_fold.append(
            float(r2_score(B[test], oof[test], multioutput="variance_weighted"))
        )
        chosen.extend(np.atleast_1d(model.alpha_).astype(float).tolist())

    return RidgeResult(
        r2=float(r2_score(B, oof, multioutput="variance_weighted")),
        r2_mean=float(np.mean(r2_score(B, oof, multioutput="raw_values"))),
        r2_per_fold=per_fold,
        n_used=n,
        n_dropped=n_dropped,
        n_splits=n_splits_used,
        grouped=g is not None,
        alphas_selected=sorted(set(chosen)),
    )


# --------------------------------------------------------------------------
# CKA (symmetric)
# --------------------------------------------------------------------------


def linear_cka(X, Y) -> float:
    """Linear Centered Kernel Alignment in its efficient feature-space form.

    Invariant to orthogonal rotation and isotropic scaling of either space,
    which is what makes it comparable across spaces of unequal width.
    """
    A, B, _, _ = _align(X, Y)
    A, B = _column_center(A), _column_center(B)
    cross = np.linalg.norm(B.T @ A, ord="fro") ** 2
    denom = np.linalg.norm(A.T @ A, ord="fro") * np.linalg.norm(B.T @ B, ord="fro")
    return 0.0 if denom == 0 else float(cross / denom)


def _rbf_gram(A: np.ndarray, gamma: float | None) -> np.ndarray:
    sq = (
        (A**2).sum(axis=1)[:, None]
        + (A**2).sum(axis=1)[None, :]
        - 2.0 * (A @ A.T)
    )
    sq = np.clip(sq, 0.0, None)
    if gamma is None:  # median heuristic
        med = np.median(_offdiag(sq))
        gamma = 1.0 / med if med > 0 else 1.0
    return np.exp(-gamma * sq)


def _center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def rbf_cka(X, Y, *, gamma: float | None = None) -> float:
    """RBF-kernel CKA with a median-distance bandwidth by default."""
    A, B, _, _ = _align(X, Y)
    Kc, Lc = _center_gram(_rbf_gram(A, gamma)), _center_gram(_rbf_gram(B, gamma))
    denom = np.linalg.norm(Kc, ord="fro") * np.linalg.norm(Lc, ord="fro")
    return 0.0 if denom == 0 else float((Kc * Lc).sum() / denom)


def cka(X, Y, *, kernel: Literal["linear", "rbf"] = "linear", gamma=None) -> float:
    if kernel == "linear":
        return linear_cka(X, Y)
    if kernel == "rbf":
        return rbf_cka(X, Y, gamma=gamma)
    raise SpaceError(f"Unknown CKA kernel '{kernel}'. Available: linear, rbf.")


# --------------------------------------------------------------------------
# second-order RSA (symmetric)
# --------------------------------------------------------------------------


def second_order_rsa(X, Y, *, metric: str = "correlation") -> float:
    """Spearman rho between the two spaces' off-diagonal RDM entries.

    The RDMs are built with psytwill's own first-order metrics, so this
    measure agrees by construction with what ``psytwill matrices`` writes.
    """
    A, B, _, _ = _align(X, Y)
    if A.shape[0] < 3:
        raise SpaceError("Second-order RSA needs at least 3 rows.")
    func = get_metric(metric).func
    return _spearman(_offdiag(func(A, A)), _offdiag(func(B, B)))


# --------------------------------------------------------------------------
# neighbour overlap (symmetric)
# --------------------------------------------------------------------------


def neighbor_overlap(X, Y, *, k: int = DEFAULT_K, metric: str = "cosine") -> float:
    """Mean fraction of each row's k nearest neighbours shared by both spaces.

    Chance is roughly ``k / (n - 1)``; :func:`permutation_null` gives the
    calibrated version.
    """
    A, B, _, _ = _align(X, Y)
    n = A.shape[0]
    if n < 3:
        raise SpaceError("Neighbour overlap needs at least 3 rows.")
    k_eff = min(k, n - 1)
    cfg = get_metric(metric)
    sign = -1.0 if cfg.form == "similarity" else 1.0  # rank nearest first

    def knn(M: np.ndarray) -> np.ndarray:
        D = sign * cfg.func(M, M)
        np.fill_diagonal(D, np.inf)  # never a neighbour of itself
        return np.argsort(D, axis=1, kind="stable")[:, :k_eff]

    na, nb = knn(A), knn(B)
    shared = [len(set(na[i]) & set(nb[i])) for i in range(n)]
    return float(np.mean(shared) / k_eff)


# --------------------------------------------------------------------------
# permutation nulls
# --------------------------------------------------------------------------


def block_permutation(n: int, rng: np.random.RandomState, block_size: int | None) -> np.ndarray:
    """Row order for a null: full shuffle, or a shuffle of contiguous blocks.

    ``block_size=None`` destroys all structure, which is correct for
    exchangeable rows (an image set) and wrong for a temporal grid.
    """
    if block_size is None or block_size <= 1:
        return rng.permutation(n)
    starts = np.arange(0, n, block_size)
    blocks = [np.arange(s, min(s + block_size, n)) for s in starts]
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])


@dataclass
class NullResult:
    observed: float
    p_value: float
    null_mean: float
    null_sd: float
    n_perm: int
    block_size: int | None
    null: list[float] = field(default_factory=list)


def permutation_null(
    func: Callable[[np.ndarray, np.ndarray], float],
    X,
    Y,
    *,
    n_perm: int = 1000,
    block_size: int | None = None,
    random_state: int = 0,
) -> NullResult:
    """Calibrate any symmetric measure against permuted rows of ``Y``.

    One-sided (greater), with the observed value counted in the numerator —
    so p is never 0 and is bounded below by ``1 / (n_perm + 1)``.
    """
    A, B, _, _ = _align(X, Y)
    rng = np.random.RandomState(random_state)
    observed = float(func(A, B))
    null = [
        float(func(A, B[block_permutation(A.shape[0], rng, block_size)]))
        for _ in range(n_perm)
    ]
    arr = np.asarray(null)
    return NullResult(
        observed=observed,
        p_value=float((1 + (arr >= observed).sum()) / (n_perm + 1)),
        null_mean=float(arr.mean()),
        null_sd=float(arr.std()),
        n_perm=n_perm,
        block_size=block_size,
        null=null,
    )


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


@dataclass
class MeasureConfig:
    name: str
    func: Callable
    symmetric: bool
    description: str


MEASURE_REGISTRY: dict[str, MeasureConfig] = {
    "ridge": MeasureConfig(
        "ridge", ridge_predictivity, False, "Cross-validated R^2 of X -> Y"
    ),
    "cka_linear": MeasureConfig(
        "cka_linear", linear_cka, True, "Linear Centered Kernel Alignment"
    ),
    "cka_rbf": MeasureConfig(
        "cka_rbf", rbf_cka, True, "RBF Centered Kernel Alignment (median bandwidth)"
    ),
    "rsa": MeasureConfig(
        "rsa", second_order_rsa, True, "Spearman over off-diagonal RDM entries"
    ),
    "neighbor_overlap": MeasureConfig(
        "neighbor_overlap", neighbor_overlap, True, "Shared k-NN fraction"
    ),
}


def get_measure(name: str) -> MeasureConfig:
    if name not in MEASURE_REGISTRY:
        raise SpaceError(
            f"Unknown measure '{name}'. Available: {', '.join(MEASURE_REGISTRY)}."
        )
    return MEASURE_REGISTRY[name]


def applicability(measure: str, d_x: int, d_y: int) -> str | None:
    """Note why a measure's value needs qualifying here, or None if it does not.

    Deliberately advisory rather than an exception. A 1-d space is a perfectly
    good ridge *target*, and CKA/RSA/overlap remain computable on it — they
    just collapse to statements about a single ordering, so a scalar head can
    show near-perfect neighbour overlap with anything monotonically related to
    it. The caller records the note next to the number instead of discovering
    the collapse afterwards.
    """
    cfg = get_measure(measure)
    if cfg.name == "ridge":
        if d_x == 1:
            return "1-d source: predicts at most one direction of the target"
        return None
    if d_x == 1 or d_y == 1:
        return (
            "1-d space: this measure reduces to a statement about a single "
            "ordering, not a geometry"
        )
    return None
