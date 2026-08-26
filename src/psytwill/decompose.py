"""Cross-space decomposition: how many directions two spaces share.

:mod:`psytwill.compare` answers *how well* two spaces agree — one scalar per
pair. This module answers *how many and how strongly per direction*: the
question that turns a "complementary" verdict from a boolean into a dimension
count. A pair like ebind/dinov2 (ridge R^2 ~0.1, CKA ~0.7) is exactly the case
a scalar cannot explain — strong agreement about structure, weak linear
reconstruction — and the resolution is a count of shared directions against
the sizes of the two spaces.

Method: **cross-validated CCA**.

- On each training fold, each space is centered, PCA-reduced (``rank_cap``,
  eigenvalue tolerance) and whitened; the SVD of the whitened cross-covariance
  gives paired canonical directions and in-sample canonical correlations.
- Held-out rows are projected through the *training* transforms, and the
  per-component correlation of the paired held-out projections is the
  cross-validated canonical correlation. This is the quantity immune to CCA's
  in-sample optimism, which is severe: at n in the hundreds, plain CCA finds
  near-perfect correlations between independent spaces.
- The null block-permutes the held-out rows of one side's projections — no
  refit needed, permuting Y's rows only relabels its projections — and the
  shared-dimension count is the longest **prefix** of components whose mean
  held-out correlation clears the null's per-component quantile. A prefix,
  not a subset: components are ordered by the training fit, and counting a
  significant 40th component after an insignificant 30th would be reading
  noise as structure.

The same three conventions as :mod:`psytwill.compare` apply (aligned rows,
grouped folds on temporal grids, block nulls to match, NaN rows dropped and
counted). No fitted transform leaves this module: the decomposition is a
*measurement* on the stimuli it ran on. A reusable basis is a derived space
and ships as a versioned checkpoint with fit provenance, from a fit corpus —
that is :mod:`psytwill.fitcorpus` territory, deliberately not this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from psytwill import __version__
from psytwill.compare import _align, block_permutation
from psytwill.exceptions import SpaceError
from psytwill.store import SpaceMatrix

DECOMP_SCHEMA_VERSION = "1.0"
DEFAULT_RANK_CAP = 128
DEFAULT_PREFIX_ALPHA = 0.01

COMPONENT_COLUMNS: tuple[str, ...] = (
    "source",
    "target",
    "component",
    "r_train",
    "r_cv",
    "null_q",
    "p_value",
    "shared",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "source",
    "target",
    "shared_dims",
    "n_components",
    "rank_source",
    "rank_target",
    "r_cv_first",
    "n_used",
    "n_dropped",
    "n_splits",
    "grouped",
    "rank_cap",
    "n_perm",
    "block_size",
    "prefix_alpha",
)


@dataclass
class _Whitener:
    """Center + PCA-whiten fitted on training rows, applied to any rows."""

    mean: np.ndarray
    components: np.ndarray  # (d, r) columns are PC directions
    scale: np.ndarray  # per-component score sd on the training rows

    @property
    def rank(self) -> int:
        return self.components.shape[1]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) @ self.components / self.scale


def _fit_whitener(X: np.ndarray, rank_cap: int, rel_tol: float = 1e-8) -> _Whitener:
    mean = X.mean(axis=0, keepdims=True)
    # economy SVD of the centered matrix; PC score sd_j = s_j / sqrt(n - 1)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    if s[0] == 0:
        raise SpaceError(
            "Space is constant on a training fold, so it has no directions to "
            "whiten. Check the rows are real data rather than a fill value."
        )
    keep = s > rel_tol * s[0]
    r = min(int(keep.sum()), rank_cap)
    scale = s[:r] / np.sqrt(X.shape[0] - 1)
    return _Whitener(mean=mean, components=Vt[:r].T, scale=scale)


def _colwise_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r between column i of A and column i of B, vectorized."""
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(Ac, axis=0) * np.linalg.norm(Bc, axis=0)
    num = (Ac * Bc).sum(axis=0)
    out = np.zeros(A.shape[1])
    ok = denom > 0
    out[ok] = num[ok] / denom[ok]
    return out


def shared_prefix(r_cv: np.ndarray, null_q: np.ndarray) -> int:
    """Longest prefix of components whose CV correlation clears its threshold."""
    m = 0
    for r, q in zip(r_cv, null_q):
        if r > q:
            m += 1
        else:
            break
    return m


@dataclass
class CvCcaResult:
    """Per-component cross-validated canonical correlations and the count."""

    r_train: list[float]
    """In-sample canonical correlations, mean across folds (optimistic)."""
    r_cv: list[float]
    """Held-out canonical correlations, mean across folds (the real quantity)."""
    null_q: list[float]
    """Per-component null quantile at ``prefix_alpha`` (one-sided, greater)."""
    p_values: list[float]
    shared_dims: int
    """Longest prefix of components with ``r_cv`` above ``null_q``."""
    n_components: int
    rank_x: int
    rank_y: int
    n_used: int
    n_dropped: int
    n_splits: int
    grouped: bool
    rank_cap: int
    n_perm: int
    block_size: int | None
    prefix_alpha: float


def cv_cca(
    X,
    Y,
    *,
    groups: Sequence | None = None,
    n_splits: int = 5,
    rank_cap: int = DEFAULT_RANK_CAP,
    n_perm: int = 250,
    block_size: int | None = None,
    prefix_alpha: float = DEFAULT_PREFIX_ALPHA,
    random_state: int = 0,
) -> CvCcaResult:
    """Cross-validated CCA between two spaces on the same rows.

    Components are matched across folds by rank order — fold 1's second
    component and fold 2's second component are averaged as "component 2".
    That is an approximation (nearby components can swap between folds), and
    it is conservative in the safe direction: a swap depresses the mean CV
    correlation rather than inflating it.

    ``n_perm=0`` skips the null; ``shared_dims`` is then -1, because a count
    without a threshold is not a count.
    """
    A, B, g, n_dropped = _align(X, Y, groups)
    n = A.shape[0]
    if n < 6:
        raise SpaceError(f"Need at least 6 usable rows; {n} survived NaN removal.")

    if g is not None:
        n_groups = len(np.unique(g))
        if n_groups < 2:
            raise SpaceError(
                "'groups' has a single group, so no split holds a group out. "
                "Drop groups, or decompose on a set spanning several clips."
            )
        n_splits_used = min(n_splits, n_groups)
        splits = list(GroupKFold(n_splits=n_splits_used).split(A, B, groups=g))
    else:
        n_splits_used = min(n_splits, n)
        splits = list(
            KFold(n_splits=n_splits_used, shuffle=True, random_state=random_state).split(A)
        )

    # Fit per fold; the component count is the smallest rank any fold reaches,
    # so every averaged component exists in every fold.
    per_fold: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    ranks_x: list[int] = []
    ranks_y: list[int] = []
    for train, test in splits:
        wx = _fit_whitener(A[train], rank_cap)
        wy = _fit_whitener(B[train], rank_cap)
        Za, Zb = wx.transform(A[train]), wy.transform(B[train])
        U, s, Vt = np.linalg.svd(Za.T @ Zb / len(train), full_matrices=False)
        Pa = wx.transform(A[test]) @ U
        Pb = wy.transform(B[test]) @ Vt.T
        per_fold.append((np.clip(s, 0.0, 1.0), Pa, Pb))
        ranks_x.append(wx.rank)
        ranks_y.append(wy.rank)

    C = min(min(len(s), Pa.shape[1], Pb.shape[1]) for s, Pa, Pb in per_fold)
    r_train = np.mean([s[:C] for s, _, _ in per_fold], axis=0)
    r_cv = np.mean(
        [_colwise_corr(Pa[:, :C], Pb[:, :C]) for _, Pa, Pb in per_fold], axis=0
    )

    if n_perm:
        rng = np.random.RandomState(random_state)
        null = np.empty((n_perm, C))
        for p in range(n_perm):
            draws = []
            for _, Pa, Pb in per_fold:
                perm = block_permutation(Pa.shape[0], rng, block_size)
                draws.append(_colwise_corr(Pa[:, :C], Pb[perm, :C]))
            null[p] = np.mean(draws, axis=0)
        null_q = np.quantile(null, 1.0 - prefix_alpha, axis=0)
        p_values = (1 + (null >= r_cv[None, :]).sum(axis=0)) / (n_perm + 1)
        shared = shared_prefix(r_cv, null_q)
    else:
        null_q = np.full(C, np.nan)
        p_values = np.full(C, np.nan)
        shared = -1

    return CvCcaResult(
        r_train=[float(v) for v in r_train],
        r_cv=[float(v) for v in r_cv],
        null_q=[float(v) for v in null_q],
        p_values=[float(v) for v in p_values],
        shared_dims=shared,
        n_components=C,
        rank_x=min(ranks_x),
        rank_y=min(ranks_y),
        n_used=n,
        n_dropped=n_dropped,
        n_splits=n_splits_used,
        grouped=g is not None,
        rank_cap=rank_cap,
        n_perm=n_perm,
        block_size=block_size,
        prefix_alpha=prefix_alpha,
    )


# --------------------------------------------------------------------------
# driver: one source space against every other, frozen as versioned tables
# --------------------------------------------------------------------------


@dataclass
class DecompositionResult:
    components: pd.DataFrame
    summary: pd.DataFrame
    config: dict = field(default_factory=dict)


def decompose_spaces(
    spaces: Mapping[str, SpaceMatrix],
    source: str,
    *,
    groups: Sequence | None = None,
    n_splits: int = 5,
    rank_cap: int = DEFAULT_RANK_CAP,
    n_perm: int = 250,
    block_size: int | None = None,
    prefix_alpha: float = DEFAULT_PREFIX_ALPHA,
    random_state: int = 0,
    progress: Callable[[int, int, str], None] | None = None,
) -> DecompositionResult:
    """Run :func:`cv_cca` from ``source`` against every other space.

    CCA is symmetric, so one orientation covers the pair; ``source`` is
    recorded first purely so the table reads the same way as the geometry
    pair table's criterion rows.
    """
    if source not in spaces:
        raise SpaceError(
            f"Source space '{source}' is not loaded. Present: "
            f"{', '.join(sorted(spaces))}."
        )
    targets = [n for n in sorted(spaces) if n != source]
    if not targets:
        raise SpaceError("Need at least one space besides the source.")

    comp_rows: list[dict] = []
    summ_rows: list[dict] = []
    for i, tgt in enumerate(targets):
        res = cv_cca(
            spaces[source].X,
            spaces[tgt].X,
            groups=groups,
            n_splits=n_splits,
            rank_cap=rank_cap,
            n_perm=n_perm,
            block_size=block_size,
            prefix_alpha=prefix_alpha,
            random_state=random_state,
        )
        for c in range(res.n_components):
            comp_rows.append(
                {
                    "source": source,
                    "target": tgt,
                    "component": c + 1,
                    "r_train": res.r_train[c],
                    "r_cv": res.r_cv[c],
                    "null_q": res.null_q[c],
                    "p_value": res.p_values[c],
                    "shared": c < res.shared_dims,
                }
            )
        summ_rows.append(
            {
                "source": source,
                "target": tgt,
                "shared_dims": res.shared_dims,
                "n_components": res.n_components,
                "rank_source": res.rank_x,
                "rank_target": res.rank_y,
                "r_cv_first": res.r_cv[0] if res.r_cv else None,
                "n_used": res.n_used,
                "n_dropped": res.n_dropped,
                "n_splits": res.n_splits,
                "grouped": res.grouped,
                "rank_cap": res.rank_cap,
                "n_perm": res.n_perm,
                "block_size": res.block_size,
                "prefix_alpha": res.prefix_alpha,
            }
        )
        if progress is not None:
            progress(i + 1, len(targets), f"cv_cca {source} ~ {tgt}")

    config = {
        "source": source,
        "n_splits": n_splits,
        "rank_cap": rank_cap,
        "n_perm": n_perm,
        "block_size": block_size,
        "prefix_alpha": prefix_alpha,
        "random_state": random_state,
        "n_spaces": len(spaces),
        "n_targets": len(targets),
    }
    return DecompositionResult(
        components=pd.DataFrame(comp_rows, columns=list(COMPONENT_COLUMNS)),
        summary=pd.DataFrame(summ_rows, columns=list(SUMMARY_COLUMNS)),
        config=config,
    )


def write_decomposition(
    result: DecompositionResult,
    outdir: str | Path,
    *,
    name: str = "space_decomposition",
    inputs: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    extra: Mapping | None = None,
) -> dict:
    """Write the component table, the summary, and a sidecar naming both."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    components_path = out / f"{name}.components.parquet"
    summary_path = out / f"{name}.summary.parquet"
    meta_path = out / f"{name}.meta.json"

    result.components.to_parquet(components_path, index=False)
    result.summary.to_parquet(summary_path, index=False)

    meta = {
        "schema_version": DECOMP_SCHEMA_VERSION,
        "table": "space_decomposition",
        "extractor": "psytwill",
        "extractor_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": result.config,
        "outputs": {
            "components": {
                "path": str(components_path.resolve()),
                "rows": int(len(result.components)),
                "columns": list(COMPONENT_COLUMNS),
            },
            "summary": {
                "path": str(summary_path.resolve()),
                "rows": int(len(result.summary)),
                "columns": list(SUMMARY_COLUMNS),
            },
        },
        "inputs": list(inputs) if inputs else [],
        "n_labels": len(labels) if labels is not None else None,
        **(dict(extra) if extra else {}),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "components": str(components_path),
        "summary": str(summary_path),
        "meta_path": str(meta_path),
    }
