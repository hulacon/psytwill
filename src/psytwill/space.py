"""Private-block fits — psytwill-space v0.1 (``psytwill space fit|project|check``).

A private block is a frozen linear map from the concatenation of a block's
member spaces (as the battery emits them, at the native grain) to ``k``
block scores. Fitting is unsupervised; the acceptance test is the
feature-space-geometry pre-registered subsumption criterion applied to
every member space on held-out rows:

    ridge R^2 (block scores -> space) >= ``r2_min``   AND
    top-``k_nn`` neighbour overlap beating its permutation null at ``alpha``

with ``k`` below the summed participation ratios of the members.

Pipeline (design on record in the psytwill-space workbench,
``out/space-fit-design.md``):

1. member spaces are aligned on their shared labels;
2. each space is z-scored per column and PCA-whitened to ``ceil(PR)``
   directions (its participation ratio), so every member contributes about
   PR unit-variance directions regardless of its nominal width;
3. PCA on the concatenation gives nested block scores;
4. retention walks a ``k`` schedule and keeps the smallest ``k`` at which
   every member passes the criterion in every outer fold (block map fitted
   on the training rows, ridge/overlap evaluated on the test rows);
5. the frozen fit is refitted on all rows at that ``k``.

Private blocks are grain-free: mean pooling commutes with a linear map, so
a block fitted at the native grain projects any pooled table (pool, then
project).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from psytwill import __version__
from psytwill.compare import (
    DEFAULT_ALPHAS,
    DEFAULT_K,
    neighbor_overlap_null,
    participation_ratio,
    ridge_predictivity,
)
from psytwill.exceptions import SpaceError
from psytwill.store import SpaceMatrix, align_spaces

DEFAULT_K_SCHEDULE: tuple[int, ...] = (8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256)
#: Members whitened below this rank are scored with Euclidean neighbours
#: instead of cosine. Cosine similarity in one dimension takes only the
#: values +/-1, so every pair ties and `argsort(kind="stable")` returns the
#: same index list for every row -- a constant graph whose overlap with
#: anything is chance by construction. MEASURED 2026-09-12 on the V block:
#: rank-1 members produced 21 distinct neighbour sets across 5,000 rows
#: (99.6 %% sharing one), rank 2 produced 576 (88.5 %% sharing one), and
#: every member of rank >= 4 produced 5,000 of 5,000. Euclidean restores a
#: well-ordered graph for those members and leaves non-degenerate ones
#: essentially unchanged.
EUCLIDEAN_RANK_BELOW: int = 3


def metric_for_rank(rank: int) -> str:
    """Neighbour metric for a member whitened to ``rank`` directions."""
    return "euclidean" if int(rank) < EUCLIDEAN_RANK_BELOW else "cosine"
SPACE_SCHEMA_VERSION = "1.2"


# --------------------------------------------------------------------------
# per-space whitening
# --------------------------------------------------------------------------


@dataclass
class SpaceWhitener:
    """Center, scale and PCA-whiten one member space to ``rank`` directions."""

    name: str
    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    components: np.ndarray  # (rank, dim)
    scales: np.ndarray  # (rank,) 1/sqrt(eigenvalue)
    participation_ratio: float
    #: feature -> raw fill value for structurally-conditional columns
    #: (frozen at fit time; see the NaN policy section)
    structural_fill: dict = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return int(self.components.shape[0])

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.structural_fill:
            X = X.copy()
            idx = {f: i for i, f in enumerate(self.features)}
            for f, v in self.structural_fill.items():
                j = idx.get(f)
                if j is not None:
                    col = X[:, j]
                    col[np.isnan(col)] = v
        Z = (X - self.mean) / self.std
        Z = np.where(np.isnan(Z), 0.0, Z)  # mean imputation of undefined entries
        return (Z @ self.components.T) * self.scales


def fit_whitener(space: SpaceMatrix, X: np.ndarray, *, rank: int | None = None,
                 rel_tol: float = 1e-8,
                 structural_fill: dict | None = None) -> SpaceWhitener:
    """Fit a whitener on ``X`` (training rows of ``space``).

    ``rank`` defaults to ``ceil(participation_ratio)`` of the training rows,
    capped by the number of non-degenerate directions.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3:
        raise SpaceError(f"'{space.name}': need at least 3 training rows to whiten.")
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(std > 0, std, 1.0)  # constant columns pass through as zeros
    Z = (X - mean) / std
    Z = np.where(np.isnan(Z), 0.0, Z)  # mean imputation (undefined -> the column mean)
    pr = float(participation_ratio(Z))
    # eigendecomposition of the covariance via SVD of Z
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    eig = (s ** 2) / max(Z.shape[0] - 1, 1)
    keep = eig > eig[0] * rel_tol if eig.size else np.zeros(0, dtype=bool)
    n_ok = int(keep.sum())
    if n_ok == 0:
        raise SpaceError(f"'{space.name}' is constant on the training rows; nothing to whiten.")
    r = min(n_ok, int(math.ceil(pr))) if rank is None else min(int(rank), n_ok)
    r = max(1, r)
    return SpaceWhitener(
        name=space.name,
        features=list(space.features),
        mean=mean,
        std=std,
        components=vt[:r],
        scales=1.0 / np.sqrt(eig[:r]),
        participation_ratio=pr,
        structural_fill=dict(structural_fill or {}),
    )


# --------------------------------------------------------------------------
# block map
# --------------------------------------------------------------------------


@dataclass
class BlockMap:
    """Whiteners for every member plus the block PCA over their concatenation."""

    members: list[str]
    whiteners: dict[str, SpaceWhitener]
    block_mean: np.ndarray  # (sum rank,)
    block_components: np.ndarray  # (k_max, sum rank)
    block_eigenvalues: np.ndarray  # (k_max,)

    @property
    def k_max(self) -> int:
        return int(self.block_components.shape[0])

    def concat(self, spaces: dict[str, SpaceMatrix], rows: np.ndarray | None = None) -> np.ndarray:
        parts = []
        for m in self.members:
            X = spaces[m].X if rows is None else spaces[m].X[rows]
            parts.append(self.whiteners[m].transform(X))
        return np.concatenate(parts, axis=1)

    def scores(self, spaces: dict[str, SpaceMatrix], rows: np.ndarray | None = None,
               k: int | None = None) -> np.ndarray:
        W = self.concat(spaces, rows) - self.block_mean
        S = W @ self.block_components.T
        return S if k is None else S[:, :k]


def fit_block_map(spaces: dict[str, SpaceMatrix], members: Sequence[str], rows: np.ndarray,
                  *, k_max: int | None = None,
                  structural_fill: dict[str, dict] | None = None) -> BlockMap:
    sf = structural_fill or {}
    whiteners = {m: fit_whitener(spaces[m], spaces[m].X[rows], structural_fill=sf.get(m))
                 for m in members}
    parts = [whiteners[m].transform(spaces[m].X[rows]) for m in members]
    W = np.concatenate(parts, axis=1)
    mean = W.mean(axis=0)
    _, s, vt = np.linalg.svd(W - mean, full_matrices=False)
    eig = (s ** 2) / max(W.shape[0] - 1, 1)
    kk = vt.shape[0] if k_max is None else min(int(k_max), vt.shape[0])
    return BlockMap(
        members=list(members),
        whiteners=whiteners,
        block_mean=mean,
        block_components=vt[:kk],
        block_eigenvalues=eig[:kk],
    )



# --------------------------------------------------------------------------
# NaN policy
# --------------------------------------------------------------------------

DEFAULT_MAX_NAN_FRAC = 0.5

# Structurally-conditional columns (DECIDED 2026-09-13, psytwill-space
# workbench): some columns have *no value* when their condition is absent --
# a formant frequency on a frame with no voice, a mean turn duration in
# instrumental music. The extractors encode this correctly (null, where a
# genuine absence-of-events is 0), and mean-imputing those nulls fabricates
# a mid-range measurement on every such frame. The rule:
#
# 1. DETECT the conditional set from the data, per column, by per-corpus
#    contrast: null in > STRUCTURAL_NULL_HIGH of one corpus's rows and
#    < STRUCTURAL_NULL_LOW of another's. The split is bimodal (measured
#    2026-09-13 across eight corpora: 0-2 of 1,635 columns in the 0.1-0.9
#    band), so the thresholds are not delicate. A hardcoded column list
#    would go stale silently when an extractor changes.
# 2. FILL those entries with one frozen constant per column, placed at
#    STRUCTURAL_FILL_Z defined-entry standard deviations below the
#    defined-entry mean. "Undefined" becomes a single point outside the
#    bulk of the measured values -- a label, not a measurement -- so the
#    member represents "no value here" as a direction instead of
#    inheriting a fabricated mean. The whitener re-standardizes over the
#    filled matrix, so the sentinel's magnitude does not dominate variance.
#    The gate itself is already visible to the block through columns that
#    are always defined (the rule adds no columns).
#
# Detection needs a per-row corpus label and runs only when the caller
# provides one (`fit_block(..., corpora=...)`); without it the policy is
# unchanged (max_nan_frac drop + mean-impute). Filled columns are exempt
# from the max_nan_frac drop by construction: the fill runs first.
STRUCTURAL_NULL_HIGH = 0.9
STRUCTURAL_NULL_LOW = 0.2
STRUCTURAL_FILL_Z = -3.0


def detect_structural_columns(space: SpaceMatrix, corpora: Sequence, *,
                              high: float = STRUCTURAL_NULL_HIGH,
                              low: float = STRUCTURAL_NULL_LOW) -> list[str]:
    """Columns whose null fraction is > ``high`` in some corpus and < ``low``
    in another. Returns [] when fewer than two corpora are represented."""
    X = np.asarray(space.X, dtype=float)
    g = np.asarray(corpora)
    if len(g) != X.shape[0]:
        raise SpaceError(f"'{space.name}': corpora must have one entry per row "
                         f"({len(g)} != {X.shape[0]})")
    keys = np.unique(g)
    if keys.size < 2:
        return []
    frac = np.stack([np.isnan(X[g == c]).mean(axis=0) for c in keys])
    hit = (frac.max(axis=0) > high) & (frac.min(axis=0) < low)
    return [f for f, h in zip(space.features, hit) if h]


def structural_fill_values(space: SpaceMatrix, columns: Sequence[str], *,
                           fill_z: float = STRUCTURAL_FILL_Z) -> dict[str, float]:
    """Frozen raw fill value per column: defined-entry mean + ``fill_z`` sd."""
    X = np.asarray(space.X, dtype=float)
    idx = {f: i for i, f in enumerate(space.features)}
    out: dict[str, float] = {}
    for f in columns:
        col = X[:, idx[f]]
        mean = float(np.nanmean(col)) if np.isfinite(col).any() else 0.0
        std = float(np.nanstd(col))
        out[f] = mean + fill_z * (std if std > 0 else 1.0)
    return out


def apply_structural_fill(space: SpaceMatrix, fills: dict | None) -> SpaceMatrix:
    """Replace nulls in the named columns with their frozen fill values."""
    if not fills:
        return space
    X = np.asarray(space.X, dtype=float).copy()
    idx = {f: i for i, f in enumerate(space.features)}
    for f, v in fills.items():
        j = idx.get(f)
        if j is not None:
            col = X[:, j]
            col[np.isnan(col)] = v
    return SpaceMatrix(name=space.name, labels=list(space.labels), X=X,
                       features=list(space.features), modality=space.modality,
                       extractor=space.extractor, n_replicates=space.n_replicates)


def prepare_member(space: SpaceMatrix, *, max_nan_frac: float = DEFAULT_MAX_NAN_FRAC) -> tuple[SpaceMatrix, list[str]]:
    """Drop columns undefined in more than ``max_nan_frac`` of rows.

    A feature that exists only for a minority of stimuli (``faces_mutual_dist``
    needs two faces; 98 % of NSD images have fewer) would otherwise force the
    whole member onto that minority. Sparse gaps in the kept columns are
    mean-imputed for the fit and dropped row-wise by the criterion.
    """
    X = np.asarray(space.X, dtype=float)
    frac = np.isnan(X).mean(axis=0) if X.size else np.zeros(X.shape[1])
    keep = frac <= max_nan_frac
    dropped = [f for f, k in zip(space.features, keep) if not k]
    if not dropped:
        return space, []
    if not keep.any():
        raise SpaceError(f"'{space.name}': every column is undefined in more than {max_nan_frac:.0%} of rows")
    out = SpaceMatrix(name=space.name, labels=list(space.labels), X=X[:, keep],
                      features=[f for f, k in zip(space.features, keep) if k],
                      modality=space.modality, extractor=space.extractor, n_replicates=space.n_replicates)
    return out, dropped


def select_features(space: SpaceMatrix, features: Sequence[str]) -> SpaceMatrix:
    """Restrict ``space`` to ``features`` in that order (a fit's kept columns)."""
    idx = {f: i for i, f in enumerate(space.features)}
    missing = [f for f in features if f not in idx]
    if missing:
        raise SpaceError(f"'{space.name}' lacks {len(missing)} feature(s) the fit needs, e.g. {missing[:3]}")
    take = [idx[f] for f in features]
    return SpaceMatrix(name=space.name, labels=list(space.labels), X=np.asarray(space.X, dtype=float)[:, take],
                      features=list(features), modality=space.modality, extractor=space.extractor,
                      n_replicates=space.n_replicates)


# --------------------------------------------------------------------------
# criterion
# --------------------------------------------------------------------------


@dataclass
class MemberCheck:
    member: str
    k: int
    fold: int
    r2: float
    overlap: float
    overlap_p: float
    null_mean: float
    n_rows: int
    metric: str = "cosine"
    eval_rows: int = 0

    def passes(self, r2_min: float, alpha: float) -> bool:
        return bool(self.r2 >= r2_min and self.overlap_p < alpha)


def check_member(scores: np.ndarray, space_X: np.ndarray, *, member: str, k: int, fold: int,
                 groups: Sequence | None = None, k_nn: int = DEFAULT_K, n_perm: int = 250,
                 eval_n: int | None = 5000, block_size: int | None = None,
                 random_state: int = 0, alphas: Sequence[float] = DEFAULT_ALPHAS,
                 metric: str = "cosine") -> MemberCheck:
    """Ridge R^2 (scores -> space) and neighbour overlap vs null on the given rows.

    ``metric`` applies to BOTH sides of the neighbour comparison. The mixed
    form (block cosine, member Euclidean) was measured to pass a member the
    block recovers at R^2 = 0.10, so it is not offered.
    """
    rr = ridge_predictivity(scores, space_X, groups=groups, alphas=alphas, random_state=random_state)
    n = scores.shape[0]
    if eval_n is not None and n > eval_n:
        rng = np.random.default_rng(random_state + fold)
        sub = np.sort(rng.choice(n, size=eval_n, replace=False))
    else:
        sub = np.arange(n)
    nr = neighbor_overlap_null(scores[sub], space_X[sub], k=k_nn, n_perm=n_perm,
                               metric=metric, block_size=block_size, random_state=random_state)
    return MemberCheck(member=member, k=k, fold=fold, r2=float(rr.r2), overlap=float(nr.observed),
                       overlap_p=float(nr.p_value), null_mean=float(nr.null_mean), n_rows=int(n),
                       metric=metric, eval_rows=int(sub.size))



def _check_perm_floor(n_perm: int, alpha: float) -> None:
    """A permutation p-value cannot go below 1/(n_perm+1); refuse a null that
    can never beat ``alpha`` instead of failing every member silently."""
    floor = 1.0 / (n_perm + 1)
    if floor >= alpha:
        raise SpaceError(
            f"n_perm={n_perm} gives a minimum p of {floor:.4f}, which cannot beat alpha={alpha}; "
            f"use n_perm >= {int(math.ceil(1.0 / alpha))}"
        )


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------


@dataclass
class BlockFit:
    block: str
    members: list[str]
    k: int
    map: BlockMap
    labels: list[str]
    curve: list[dict]  # one row per (k, member, fold)
    manifest: dict = field(default_factory=dict)

    def project(self, spaces: dict[str, SpaceMatrix]) -> tuple[np.ndarray, list[str]]:
        picked = {m: select_features(spaces[m], self.map.whiteners[m].features) for m in self.members}
        aligned, labels = align_spaces(picked)
        return self.map.scores(aligned, k=self.k), labels


def _folds(n: int, n_splits: int, groups: Sequence | None, random_state: int):
    from sklearn.model_selection import GroupKFold, KFold

    idx = np.arange(n)
    if groups is not None:
        g = np.asarray(groups)
        if len(np.unique(g)) < n_splits:
            raise SpaceError(f"{len(np.unique(g))} groups cannot make {n_splits} grouped folds.")
        return list(GroupKFold(n_splits=n_splits).split(idx, groups=g))
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(idx))


def fit_block(
    spaces: dict[str, SpaceMatrix],
    members: Sequence[str],
    *,
    block: str = "block",
    k_schedule: Sequence[int] = DEFAULT_K_SCHEDULE,
    n_splits: int = 5,
    groups: Sequence | None = None,
    r2_min: float = 0.5,
    alpha: float = 0.01,
    k_nn: int = DEFAULT_K,
    n_perm: int = 250,
    eval_n: int | None = 5000,
    block_size: int | None = None,
    random_state: int = 0,
    max_nan_frac: float = DEFAULT_MAX_NAN_FRAC,
    corpora: Sequence | None = None,
    structural_null_high: float = STRUCTURAL_NULL_HIGH,
    structural_null_low: float = STRUCTURAL_NULL_LOW,
    structural_fill_z: float = STRUCTURAL_FILL_Z,
    progress=None,
) -> BlockFit:
    """Fit one private block; see the module docstring for the pipeline.

    ``corpora`` (one label per aligned row) turns on the structural-
    missingness rule: columns null under a per-corpus gate are detected by
    corpus contrast and filled with a frozen sentinel instead of falling
    through to the mean-impute policy (see the NaN policy section).
    """
    _check_perm_floor(n_perm, alpha)
    members = list(members)
    missing = [m for m in members if m not in spaces]
    if missing:
        raise SpaceError(f"members not in the loaded spaces: {missing}; have {sorted(spaces)}")
    if len(members) < 1:
        raise SpaceError("a block needs at least one member space")
    aligned, labels = align_spaces({m: spaces[m] for m in members})
    structural_fill: dict[str, dict[str, float]] = {}
    if corpora is not None:
        if len(corpora) != len(labels):
            raise SpaceError(f"corpora must have one entry per aligned row "
                             f"({len(corpora)} != {len(labels)})")
        for m in members:
            cols = detect_structural_columns(aligned[m], corpora,
                                             high=structural_null_high,
                                             low=structural_null_low)
            if cols:
                structural_fill[m] = structural_fill_values(
                    aligned[m], cols, fill_z=structural_fill_z)
                aligned[m] = apply_structural_fill(aligned[m], structural_fill[m])
    dropped_columns: dict[str, list[str]] = {}
    for m in members:
        aligned[m], dropped = prepare_member(aligned[m], max_nan_frac=max_nan_frac)
        if dropped:
            dropped_columns[m] = dropped
    n = len(labels)
    if n < 3 * n_splits:
        raise SpaceError(f"{n} aligned rows is too few for {n_splits} folds")
    g = None
    if groups is not None:
        if len(groups) != n:
            raise SpaceError("groups must have one entry per aligned row")
        g = np.asarray(groups)

    # The walk runs up to the concatenation's own rank (sum of whitened ranks).
    # The summed participation ratio is REPORTED as the Settles-when bound, not
    # used as a cap: PR under-counts a space's rank whenever its eigenvalues are
    # unequal, so a cap at PR can make a fit infeasible by construction (a
    # 4-latent member alone has PR ~3.7).
    #
    # PR BASIS (DECIDED 2026-09-12). The bound is summed over the SAME
    # participation ratios the whiteners use -- computed on the z-scored
    # (correlation) matrix in `fit_whitener` -- not over raw-covariance PRs.
    # Before this, `fit_block` measured raw X while `fit_whitener` measured Z,
    # so `k` counted correlation-basis directions and the bound counted
    # covariance-basis ones. For a scale-skewed member the two differ wildly
    # (V's `places`: 5.07 raw vs 90.37 z-scored), which made the verdict depend
    # on which key a reader opened. The whitening is what is frozen into the
    # weights, so the bound follows it. Raw-covariance PR remains the basis of
    # the Contract B section 4.4 space manifest; `pr_basis` names which is which.
    folds = _folds(n, n_splits, g, random_state)
    fold_maps = [fit_block_map(aligned, members, train, structural_fill=structural_fill)
                 for train, _ in folds]
    k_top = min(fm.k_max for fm in fold_maps)
    schedule = sorted({int(k) for k in k_schedule if 1 <= int(k) < k_top} | {k_top})
    curve: list[dict] = []
    chosen: int | None = None
    fold_scores = [fold_maps[i].scores(aligned, test) for i, (_, test) in enumerate(folds)]
    n_steps = len(schedule) * len(folds) * len(members)
    step = 0
    for k in schedule:
        all_pass = True
        for fi, (_, test) in enumerate(folds):
            S = fold_scores[fi][:, :k]
            gt = g[test] if g is not None else None
            for m in members:
                step += 1
                if progress:
                    progress(step, n_steps, f"k={k} fold={fi} {m}")
                mc = check_member(S, aligned[m].X[test], member=m, k=k, fold=fi, groups=gt,
                                  k_nn=k_nn, n_perm=n_perm, eval_n=eval_n, block_size=block_size,
                                  random_state=random_state,
                                  metric=metric_for_rank(fold_maps[fi].whiteners[m].rank))
                row = asdict(mc)
                row["passed"] = mc.passes(r2_min, alpha)
                curve.append(row)
                if not row["passed"]:
                    all_pass = False
        if all_pass:
            chosen = k
            break
    if chosen is None:
        chosen = schedule[-1]
        subsumed = False
    else:
        subsumed = True

    final = fit_block_map(aligned, members, np.arange(n), k_max=chosen,
                          structural_fill=structural_fill)
    chosen = min(chosen, final.k_max)  # the concatenation may have fewer directions than k
    pr = {m: float(final.whiteners[m].participation_ratio) for m in members}
    pr_sum = float(sum(pr.values()))
    lam = np.asarray(final.block_eigenvalues, dtype=float)
    block_pr = float(lam.sum() ** 2 / (lam ** 2).sum()) if lam.size and lam.sum() > 0 else 0.0
    per_member = {}
    for m in members:
        rows = [r for r in curve if r["member"] == m and r["k"] == chosen]
        per_member[m] = {
            "participation_ratio": pr[m],
            "whitened_rank": final.whiteners[m].rank,
            "metric": metric_for_rank(final.whiteners[m].rank),
            "eval_rows": [r["eval_rows"] for r in rows],
            "dim": aligned[m].dim,
            "dropped_columns": dropped_columns.get(m, []),
            "structural_columns": sorted(structural_fill.get(m, {})),
            "nan_fraction_kept": float(np.isnan(aligned[m].X).mean()),
            "r2_per_fold": [r["r2"] for r in rows],
            "overlap_per_fold": [r["overlap"] for r in rows],
            "overlap_p_per_fold": [r["overlap_p"] for r in rows],
            "passed_all_folds": all(r["passed"] for r in rows) if rows else False,
        }
    manifest = {
        "space_schema_version": SPACE_SCHEMA_VERSION,
        "psytwill_version": __version__,
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "block": block,
        "members": members,
        "k": chosen,
        "subsumes_all_members": subsumed,
        "pr_sum_bound": pr_sum,
        "pr_basis": "correlation",
        "k_below_pr_bound": chosen < pr_sum,
        # Reported because the bound above has little teeth once it is on the
        # same basis as the ranks: sum(PR) is within the ceiling rounding of
        # sum(ceil(PR)), which IS the concatenation's rank, so "k < sum(PR)"
        # is close to "k < full rank". These two are the compression the block
        # actually achieves, and are what the Contract B section 4.4 row and any
        # methods section should quote.
        "block_pr": block_pr,
        "n_raw_columns": int(sum(aligned[m].dim for m in members)),
        "concat_rank": int(final.k_max),
        "n_rows": n,
        "n_splits": n_splits,
        "grouped": g is not None,
        "criterion": {"r2_min": r2_min, "alpha": alpha, "k_nn": k_nn, "n_perm": n_perm,
                      "eval_n": eval_n, "block_size": block_size, "random_state": random_state},
        "max_nan_frac": max_nan_frac,
        "structural_rule": None if corpora is None else {
            "null_high": structural_null_high,
            "null_low": structural_null_low,
            "fill_z": structural_fill_z,
            "n_corpora": int(len(set(corpora))),
            "n_structural_columns": int(sum(len(v) for v in structural_fill.values())),
        },
        "k_schedule": schedule,
        "block_eigenvalues": [float(v) for v in final.block_eigenvalues],
        "per_member": per_member,
    }
    return BlockFit(block=block, members=members, k=chosen, map=final, labels=labels,
                    curve=curve, manifest=manifest)


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def save_fit(fit: BlockFit, out_dir: str | Path, *, stem: str | None = None) -> tuple[Path, Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or f"{fit.block}_v{SPACE_SCHEMA_VERSION.split('.')[0]}"
    arrays: dict[str, np.ndarray] = {
        "block_mean": fit.map.block_mean,
        "block_components": fit.map.block_components,
        "block_eigenvalues": fit.map.block_eigenvalues,
    }
    for m in fit.members:
        w = fit.map.whiteners[m]
        arrays[f"w::{m}::mean"] = w.mean
        arrays[f"w::{m}::std"] = w.std
        arrays[f"w::{m}::components"] = w.components
        arrays[f"w::{m}::scales"] = w.scales
    npz = out / f"{stem}.npz"
    np.savez_compressed(npz, **arrays)
    meta = dict(fit.manifest)
    meta["weights"] = npz.name
    meta["member_features"] = {m: fit.map.whiteners[m].features for m in fit.members}
    meta["member_pr"] = {m: fit.map.whiteners[m].participation_ratio for m in fit.members}
    meta["member_structural_fill"] = {m: fit.map.whiteners[m].structural_fill for m in fit.members}
    manifest = out / f"{stem}.json"
    manifest.write_text(json.dumps(meta, indent=2))
    import pandas as pd

    curve = out / f"{stem}_curve.csv"
    pd.DataFrame(fit.curve).to_csv(curve, index=False)
    return npz, manifest, curve


def load_fit(manifest_path: str | Path) -> BlockFit:
    mp = Path(manifest_path)
    meta = json.loads(mp.read_text())
    data = np.load(mp.parent / meta["weights"])
    whiteners = {}
    for m in meta["members"]:
        whiteners[m] = SpaceWhitener(
            name=m,
            features=list(meta["member_features"][m]),
            mean=data[f"w::{m}::mean"],
            std=data[f"w::{m}::std"],
            components=data[f"w::{m}::components"],
            scales=data[f"w::{m}::scales"],
            participation_ratio=float(meta["member_pr"][m]),
            structural_fill=dict(meta.get("member_structural_fill", {}).get(m, {})),
        )
    bm = BlockMap(
        members=list(meta["members"]),
        whiteners=whiteners,
        block_mean=data["block_mean"],
        block_components=data["block_components"],
        block_eigenvalues=data["block_eigenvalues"],
    )
    return BlockFit(block=meta["block"], members=list(meta["members"]), k=int(meta["k"]),
                    map=bm, labels=[], curve=[], manifest=meta)


def check_fit(fit: BlockFit, spaces: dict[str, SpaceMatrix], *, groups: Sequence | None = None,
              r2_min: float = 0.5, alpha: float = 0.01, k_nn: int = DEFAULT_K, n_perm: int = 250,
              eval_n: int | None = 5000, block_size: int | None = None,
              random_state: int = 0) -> list[dict]:
    """The subsumption criterion for every member on an arbitrary table (no refit)."""
    _check_perm_floor(n_perm, alpha)
    for m in fit.members:
        if m not in spaces:
            raise SpaceError(f"member '{m}' missing from the table; have {sorted(spaces)}")
    # The frozen structural fill applies to the criterion target too:
    # without it the ridge/overlap row-drop silently removes exactly the
    # rows where the member's condition is absent (the music arm, for a
    # speech-gated member), and the check no longer covers the table.
    picked = {m: apply_structural_fill(select_features(spaces[m], fit.map.whiteners[m].features),
                                       fit.map.whiteners[m].structural_fill)
              for m in fit.members}
    aligned, labels = align_spaces(picked)
    S = fit.map.scores(aligned, k=fit.k)
    out = []
    for m in fit.members:
        mc = check_member(S, aligned[m].X, member=m, k=fit.k, fold=0, groups=groups, k_nn=k_nn,
                          n_perm=n_perm, eval_n=eval_n, block_size=block_size, random_state=random_state,
                          metric=metric_for_rank(fit.map.whiteners[m].rank))
        row = asdict(mc)
        row["passed"] = mc.passes(r2_min, alpha)
        out.append(row)
    return out
