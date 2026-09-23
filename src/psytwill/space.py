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
import warnings
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
SPACE_SCHEMA_VERSION = "1.3"


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
    #: columns declared `undefinable`: a row holding NaN in one is not projected
    undefinable: list = field(default_factory=list)

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

# CONTRACT B 1.1 (DECIDED 2026-09-22, constellation-contracts §4.1): the fit
# acts on what each producer DECLARES a null to mean, and never infers it
# from data. Per member, from its `nulls` map ({column: {"means", "when"}}):
#
#   - a NaN in a column with no entry REFUSES the fit (producer defect; a
#     1.0 input has no entries, so it refuses exactly where a NaN is present);
#   - `undefinable` (positional, e.g. a clip's trailing window): every row
#     where any member holds one is dropped from the fit and the criterion,
#     and counted per member;
#   - `undefined` (content, e.g. no voice): the frozen sentinel fill below;
#   - `missing` (extraction failed, e.g. OOV): the missing-data policy,
#     max_nan_frac drop then mean-impute, as before.
#
# The per-corpus null-contrast detector below survives only as a WARNING on
# columns declared `missing` whose nulls look gated. It no longer changes a
# fit: in its first real use it missed three gated columns (graded or rare
# gates), which is why the decision moved to the producer.
NULL_KINDS = ("undefined", "undefinable", "missing")


def null_kinds(nulls: dict | None) -> dict[str, str]:
    """``{column: kind}`` from a Contract B ``nulls`` map (None -> {})."""
    out: dict[str, str] = {}
    for col, entry in (nulls or {}).items():
        kind = entry.get("means") if isinstance(entry, dict) else None
        if kind not in NULL_KINDS:
            raise SpaceError(f"nulls entry for {col!r} has means={kind!r}; expected one of {NULL_KINDS}")
        out[col] = kind
    return out


def undeclared_nan_columns(space: SpaceMatrix, kinds: dict[str, str]) -> list[str]:
    X = np.asarray(space.X, dtype=float)
    has_nan = np.isnan(X).any(axis=0) if X.size else np.zeros(len(space.features), bool)
    return [f for f, h in zip(space.features, has_nan) if h and f not in kinds]


def undefinable_rows(spaces: dict[str, SpaceMatrix], kinds: dict[str, dict[str, str]]) -> dict[str, np.ndarray]:
    """Per member, a row mask of NaN in any of its ``undefinable`` columns (aligned rows)."""
    out = {}
    for m, sp in spaces.items():
        cols = [i for i, f in enumerate(sp.features) if kinds.get(m, {}).get(f) == "undefinable"]
        if cols:
            out[m] = np.isnan(np.asarray(sp.X, dtype=float)[:, cols]).any(axis=1)
    return out


def _take_rows(space: SpaceMatrix, keep: np.ndarray) -> SpaceMatrix:
    return SpaceMatrix(name=space.name, labels=[lab for lab, k in zip(space.labels, keep) if k],
                       X=np.asarray(space.X, dtype=float)[keep], features=list(space.features),
                       modality=space.modality, extractor=space.extractor,
                       n_replicates=space.n_replicates)

# HISTORY (superseded 2026-09-23 by the Contract B 1.1 section above; the fill
# below is still how `undefined` is represented, and the detector is kept only
# as a warning). Structurally-conditional columns (DECIDED 2026-09-13,
# psytwill-space workbench): some columns have *no value* when their condition is absent --
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
# Filled (`undefined`) columns are exempt from the max_nan_frac drop by
# construction: the fill runs first.
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
    eval_sampling: str = "all"
    # The null's spread, so a pass can be read as a margin: at the p floor an
    # observed/null_mean ratio says nothing about how close the member came to
    # alpha = .01. null_q99 is the value the observed overlap must exceed.
    null_sd: float = float("nan")
    null_q99: float = float("nan")

    def passes(self, r2_min: float, alpha: float) -> bool:
        return bool(self.r2 >= r2_min and self.overlap_p < alpha)


# A block null over m blocks has only m! distinct orderings, and a draw that
# leaves most blocks in place scores like the observed graph. Below ~10 blocks
# the null is too coarse to reach alpha = .01 honestly, so it is refused.
MIN_EVAL_BLOCKS = 10


def _eval_sampling(eval_n: int | None, block_size: int | None) -> str:
    if eval_n is None:
        return "all"
    return "blocks" if block_size is not None and block_size > 1 else "rows"


def _check_eval_blocks(eval_n: int | None, block_size: int | None) -> None:
    """Refuse a subsample too small to hold a usable block null, up front."""
    if _eval_sampling(eval_n, block_size) != "blocks":
        return
    n_take = eval_n // block_size
    if n_take < MIN_EVAL_BLOCKS:
        raise SpaceError(
            f"eval_n={eval_n} holds only {n_take} blocks of block_size={block_size}; the block "
            f"null needs at least {MIN_EVAL_BLOCKS}. Raise eval_n to >= "
            f"{MIN_EVAL_BLOCKS * block_size}, or pass eval_n=None (all rows)."
        )


def eval_subsample(n: int, eval_n: int | None, block_size: int | None,
                   rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Row positions for the kNN overlap, and how they were drawn.

    With ``block_size`` set, the subsample is ``eval_n // block_size`` whole
    blocks from a fixed grid of ``block_size`` positions (a trailing partial
    block is never drawn), so the block null permutes exactly the sampled
    blocks and each block keeps its temporal adjacency. Drawing rows at random
    and then blocking the subsample -- the behaviour before 0.18.2 -- puts
    rows from different clips in one "block" and turns the block null into a
    row null, which any temporally smooth pair of spaces beats.
    """
    if eval_n is None or n <= eval_n:
        return np.arange(n), "all"
    if block_size is None or block_size <= 1:
        return np.sort(rng.choice(n, size=eval_n, replace=False)), "rows"
    _check_eval_blocks(eval_n, block_size)
    n_take = eval_n // block_size
    starts = np.sort(rng.choice(n // block_size, size=n_take, replace=False)) * block_size
    return (starts[:, None] + np.arange(block_size)).ravel(), "blocks"


def check_member(scores: np.ndarray, space_X: np.ndarray, *, member: str, k: int, fold: int,
                 groups: Sequence | None = None, k_nn: int = DEFAULT_K, n_perm: int = 250,
                 eval_n: int | None = 5000, block_size: int | None = None,
                 random_state: int = 0, alphas: Sequence[float] = DEFAULT_ALPHAS,
                 metric: str = "cosine") -> MemberCheck:
    """Ridge R^2 (scores -> space) and neighbour overlap vs null on the given rows.

    ``metric`` applies to BOTH sides of the neighbour comparison. The mixed
    form (block cosine, member Euclidean) was measured to pass a member the
    block recovers at R^2 = 0.10, so it is not offered.

    Rows must be in temporal order when ``block_size`` is set; the overlap
    subsample is then drawn as whole blocks (see :func:`eval_subsample`).
    """
    n = scores.shape[0]
    sub, sampling = eval_subsample(n, eval_n, block_size, np.random.default_rng(random_state + fold))
    rr = ridge_predictivity(scores, space_X, groups=groups, alphas=alphas, random_state=random_state)
    nr = neighbor_overlap_null(scores[sub], space_X[sub], k=k_nn, n_perm=n_perm,
                               metric=metric, block_size=block_size, random_state=random_state)
    return MemberCheck(member=member, k=k, fold=fold, r2=float(rr.r2), overlap=float(nr.observed),
                       overlap_p=float(nr.p_value), null_mean=float(nr.null_mean), n_rows=int(n),
                       metric=metric, eval_rows=int(sub.size), eval_sampling=sampling,
                       null_sd=float(nr.null_sd), null_q99=float(np.quantile(nr.null, 0.99)))



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
        """Block scores for every aligned row that is not undefinable.

        A row holding NaN in a member's `undefinable` column has no position
        in the space (it was dropped from the fit), so it is left out rather
        than mean-filled; the returned labels say which rows were scored.
        """
        picked = {m: select_features(spaces[m], self.map.whiteners[m].features) for m in self.members}
        aligned, labels = align_spaces(picked)
        aligned, labels, _ = _drop_undefinable(self, aligned, labels)
        return self.map.scores(aligned, k=self.k), labels


def _drop_undefinable(fit: "BlockFit", aligned: dict[str, SpaceMatrix], labels: list[str]):
    kinds = {m: {f: "undefinable" for f in fit.map.whiteners[m].undefinable} for m in fit.members}
    drop = np.zeros(len(labels), dtype=bool)
    for mask in undefinable_rows(aligned, kinds).values():
        drop |= mask
    if not drop.any():
        return aligned, labels, drop
    keep = ~drop
    return ({m: _take_rows(sp, keep) for m, sp in aligned.items()},
            [lab for lab, k in zip(labels, keep) if k], drop)


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
    nulls: dict[str, dict | None] | None = None,
    structural_null_high: float = STRUCTURAL_NULL_HIGH,
    structural_null_low: float = STRUCTURAL_NULL_LOW,
    structural_fill_z: float = STRUCTURAL_FILL_Z,
    progress=None,
) -> BlockFit:
    """Fit one private block; see the module docstring for the pipeline.

    ``nulls`` maps each member to its Contract B 1.1 ``nulls`` map (None for
    a 1.0 input), and decides every null (see the NaN policy section): a NaN
    without an entry refuses the fit. ``corpora`` (one label per aligned row)
    only feeds the warning-only gated-`missing` detector.
    """
    _check_perm_floor(n_perm, alpha)
    _check_eval_blocks(eval_n, block_size)
    members = list(members)
    missing = [m for m in members if m not in spaces]
    if missing:
        raise SpaceError(f"members not in the loaded spaces: {missing}; have {sorted(spaces)}")
    if len(members) < 1:
        raise SpaceError("a block needs at least one member space")
    aligned, labels = align_spaces({m: spaces[m] for m in members})
    nulls = nulls or {}
    kinds = {m: null_kinds(nulls.get(m)) for m in members}
    # 1. refuse any undeclared NaN (per member, on the table being fitted)
    refused = {m: cols for m in members if (cols := undeclared_nan_columns(aligned[m], kinds[m]))}
    if refused:
        detail = "; ".join(f"{m}: {', '.join(c[:6])}{' ...' if len(c) > 6 else ''} ({len(c)})"
                           for m, c in refused.items())
        no_map = [m for m in refused if nulls.get(m) is None]
        raise SpaceError(
            f"NaN in column(s) with no Contract B `nulls` entry -- {detail}. An undeclared null "
            "is a producer defect (constellation-contracts §4.1), so the fit is refused rather "
            "than guessing. Fix: `<extractor> sidecar refresh` the member's producer sidecars, "
            "then `psytwill features --refresh-nulls` the group tables."
            + (f" Members with no nulls map at all (1.0 input): {no_map}." if no_map else ""))
    if corpora is not None and len(corpora) != len(labels):
        raise SpaceError(f"corpora must have one entry per aligned row "
                         f"({len(corpora)} != {len(labels)})")
    if groups is not None and len(groups) != len(labels):
        raise SpaceError("groups must have one entry per aligned row")
    # 2. drop rows any member declares undefinable
    undef_rows = undefinable_rows(aligned, kinds)
    undefinable_dropped = {m: int(mask.sum()) for m, mask in undef_rows.items()}
    drop = np.zeros(len(labels), dtype=bool)
    for mask in undef_rows.values():
        drop |= mask
    if drop.any():
        keep = ~drop
        aligned = {m: _take_rows(aligned[m], keep) for m in members}
        labels = [lab for lab, k in zip(labels, keep) if k]
        if groups is not None:
            groups = [x for x, k in zip(groups, keep) if k]
        if corpora is not None:
            corpora = [x for x, k in zip(corpora, keep) if k]
    # never-null declarations and the gated-`missing` warning read the table
    # before any fill
    declared_never_null: dict[str, list[str]] = {}
    suspected_gated: dict[str, list[str]] = {}
    for m in members:
        X = np.asarray(aligned[m].X, dtype=float)
        has_nan = dict(zip(aligned[m].features, np.isnan(X).any(axis=0))) if X.size else {}
        declared_never_null[m] = sorted(c for c in kinds[m] if c in has_nan and not has_nan[c])
        if corpora is not None:
            gated = detect_structural_columns(aligned[m], corpora, high=structural_null_high,
                                              low=structural_null_low)
            sus = [c for c in gated if kinds[m].get(c) == "missing"]
            if sus:
                suspected_gated[m] = sus
                warnings.warn(f"{m}: column(s) {sus} are declared `missing` but their nulls look "
                              "gated by corpus (null in bulk in one corpus, rare in another); the "
                              "declaration may be wrong. The fit is unchanged.", stacklevel=2)
    # 3. undefined -> the frozen sentinel, for every declared-undefined column
    #    the member carries (so a projection of new rows fills them too)
    structural_fill: dict[str, dict[str, float]] = {}
    for m in members:
        cols = [f for f in aligned[m].features if kinds[m].get(f) == "undefined"]
        if cols:
            structural_fill[m] = structural_fill_values(aligned[m], cols, fill_z=structural_fill_z)
            aligned[m] = apply_structural_fill(aligned[m], structural_fill[m])
    # 4. missing -> max_nan_frac drop + mean-impute (unchanged)
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
    for m in members:
        final.whiteners[m].undefinable = sorted(
            f for f in final.whiteners[m].features if kinds[m].get(f) == "undefinable")
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
            "nulls_declared": nulls.get(m) is not None,
            "undefinable_rows_dropped": undefinable_dropped.get(m, 0),
            "declared_never_null": declared_never_null.get(m, []),
            "suspected_gated_missing": suspected_gated.get(m, []),
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
                      "eval_n": eval_n, "block_size": block_size, "random_state": random_state,
                      "eval_sampling": _eval_sampling(eval_n, block_size)},
        "max_nan_frac": max_nan_frac,
        "null_policy": {
            "source": "contract-b-1.1 nulls",
            "fill_z": structural_fill_z,
            "n_undefined_filled_columns": int(sum(len(v) for v in structural_fill.values())),
            "n_undefinable_rows_dropped": int(drop.sum()),
            "gated_missing_detector": None if corpora is None else {
                "null_high": structural_null_high, "null_low": structural_null_low,
                "n_corpora": int(len(set(corpora))), "effect": "warning only"},
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
    meta["member_undefinable"] = {m: fit.map.whiteners[m].undefinable for m in fit.members}
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
            undefinable=list(meta.get("member_undefinable", {}).get(m, [])),
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
    _check_eval_blocks(eval_n, block_size)
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
    if groups is not None and len(groups) != len(labels):
        raise SpaceError("groups must have one entry per aligned row")
    aligned, labels, drop = _drop_undefinable(fit, aligned, labels)
    if groups is not None and drop.any():
        groups = [x for x, d in zip(groups, drop) if not d]
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
