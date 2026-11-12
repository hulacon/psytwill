"""How much of an opaque target the nameable members of a battery recover, against a matched control.

The target is any table on the battery's rows: a network layer's activations
(usually reduced to its leading components by the caller), another model's
embedding, a brain response. The battery is a set of members, each a matrix
on the same rows, in the metric the caller wants shares measured in (for a
psytwill block, each member's fit-time whitened columns). Some members are
**nameable** (their columns have a label a reader can interpret). The rest,
typically foundation embeddings, are the remainder of the battery.

A read of a member set S asks two things of the same rows:

- **global**: held-out ridge R² predicting the target from S's columns
  (:func:`psytwill.compare.ridge_predictivity`, grouped CV, alpha chosen
  inside each training fold, variance-weighted over target columns);
- **local**: k-NN overlap between S's neighbour graph and the target's,
  against a block-permutation null
  (:func:`psytwill.compare.neighbor_overlap_null`).

**The control.** A share means little on its own: it grows with the number
of directions read and with how much of the battery's variance they hold.
So each read is set against random subspaces of the whole battery X (all
members' columns side by side) with the same dimensionality k and the same
captured share of X's variance as S's span
(:func:`psytwill.subspace.matched_subspaces`). Their percentile asks whether
the named directions recover more of the target than arbitrary directions
of the battery with the same capacity and variance. A random k-subspace of
S's own span would be the span itself, which is why the control draws from
all of X. The chance floor is a separate matter: held-out R² sits near 0 by
construction, and the k-NN read carries its own permutation null.

Both the named read and every control read are given the predictor as
coordinates along orthonormal directions of X (S's own columns are such
coordinates along the coordinate axes), so they reach the ridge and the
neighbour search in the same form.

**Strict and full.** ``strict`` names a subset of the nameable members (for
example, the formula-only ones by derivation label). It is read on its own
with its own control, then the full nameable set, so the two form a nested
curve. A share grows with the breadth of the battery, so neither read stands
in for the other.

**Per member.** Each nameable member is also read alone, and its unique
share is the full nameable R² minus the R² without it. Unique shares do not
sum to the total: what two members share belongs to neither.

What this measures, and what it does not: "unnamed" means *not recovered by
this battery*, never "semantic". The variance coupling of shared directions
and timing/nuisance confounds are the caller's to partition (they are
properties of the target's rows, not of this read).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from psytwill.compare import (
    DEFAULT_ALPHAS,
    DEFAULT_K,
    _overlap_from_knn,
    knn_indices,
    neighbor_overlap_null,
    ridge_predictivity,
)
from psytwill.exceptions import SpaceError
from psytwill.subspace import captured_share, matched_subspaces

#: Bump when a field of :class:`Readout`'s dict form changes meaning or goes away.
READOUT_SCHEMA = "1.0"


@dataclass
class SetRead:
    """One member set read against the target, with its matched control."""

    label: str
    """``"strict"``, ``"nameable"``, or a member name for a per-member read."""
    members: list[str]
    k: int
    """Number of the set's columns: the directions of X its span is read through (the control's dimensionality)."""
    share: float
    """The set's span's share of the battery's (centred) variance."""
    r2: float
    r2_per_fold: list[float]
    control_r2: list[float] = field(default_factory=list)
    """Held-out R² of each matched random subspace (empty when no control was drawn)."""
    r2_percentile: float | None = None
    """Fraction of control draws strictly below ``r2``."""
    control_share_range: tuple[float, float] | None = None
    gamma_bracket: tuple[float, float] | None = None
    knn_overlap: float | None = None
    knn_p: float | None = None
    knn_null_mean: float | None = None
    control_knn: list[float] = field(default_factory=list)
    knn_percentile: float | None = None
    knn_metric: str | None = None


@dataclass
class Readout:
    reads: list[SetRead]
    """The nested curve: ``strict`` (when given), then ``nameable``."""
    per_member: list[dict]
    """One row per nameable member: ``member``, ``k``, ``r2_alone``, ``r2_without``, ``r2_unique``."""
    battery: list[str]
    nameable: list[str]
    strict: list[str] | None
    n_rows: int
    n_dropped: int
    n_groups: int | None
    target_dims: int
    params: dict
    schema: str = READOUT_SCHEMA

    def read(self, label: str) -> SetRead:
        for r in self.reads:
            if r.label == label:
                return r
        raise KeyError(label)

    def to_dict(self) -> dict:
        return asdict(self)


def _as_matrix(name: str, A) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if A.ndim != 2:
        raise SpaceError(f"{name} must be 2-D (rows x columns); got shape {A.shape}")
    return A


def _percentile(observed: float, control: Sequence[float]) -> float | None:
    c = np.asarray(control, dtype=float)
    return None if c.size == 0 else float((c < observed).mean())


def readout(
    battery: dict[str, np.ndarray],
    target,
    *,
    nameable: Sequence[str],
    strict: Sequence[str] | None = None,
    groups: Sequence | None = None,
    n_control: int = 200,
    n_pool: int | None = None,
    tol: float = 0.01,
    knn: bool = True,
    k_nn: int = DEFAULT_K,
    metric: str = "euclidean",
    n_perm: int = 1000,
    block_size: int | None = None,
    per_member: bool = True,
    n_splits: int = 5,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    random_state: int = 0,
) -> Readout:
    """Read ``target`` through the nameable members of ``battery``, each read against a matched control.

    ``battery`` maps member name to a rows x columns matrix; every member and
    ``target`` hold the same rows in the same order. Rows with a non-finite
    value in any member or in the target are dropped (counted in
    ``n_dropped``), never filled. ``groups`` (one id per row, e.g. the film)
    is required whenever the rows are temporally ordered: it decides the CV
    folds.

    ``metric`` is the neighbour metric on **both** sides. Euclidean is the
    default because cosine neighbours are degenerate on a predictor whitened
    to fewer than three directions, and the per-member reads can be that
    thin. ``block_size`` keeps temporal blocks intact in the k-NN permutation
    null (rows must then be in temporal order).

    ``n_control=0`` skips the control (the reads are then unreferenced).
    ``n_pool`` defaults to ``max(20 * n_control, 2000)``.
    """
    names = list(battery)
    nameable = list(nameable)
    if not nameable:
        raise SpaceError("nameable is empty: name at least one battery member to read")
    missing = [m for m in nameable if m not in battery]
    if missing:
        raise SpaceError(f"nameable member(s) not in the battery: {', '.join(missing)}; battery has {', '.join(names)}")
    if len(set(nameable)) != len(nameable):
        raise SpaceError("nameable lists a member twice")
    if strict is not None:
        strict = list(strict)
        outside = [m for m in strict if m not in nameable]
        if outside:
            raise SpaceError(f"strict member(s) not among the nameable: {', '.join(outside)}")
        if not strict:
            raise SpaceError("strict is empty; pass None to skip the strict read")

    mats = {m: _as_matrix(f"battery[{m!r}]", battery[m]) for m in names}
    Y = _as_matrix("target", target)
    n = Y.shape[0]
    bad = [m for m, A in mats.items() if A.shape[0] != n]
    if bad:
        raise SpaceError(f"member(s) {', '.join(bad)} do not have the target's {n} rows")
    g = None if groups is None else np.asarray(groups)
    if g is not None and g.shape[0] != n:
        raise SpaceError(f"groups has {g.shape[0]} entries for {n} rows")

    X = np.hstack([mats[m] for m in names])
    keep = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    n_dropped = int((~keep).sum())
    X, Y = X[keep], Y[keep]
    g = None if g is None else g[keep]
    if X.shape[0] < 3:
        raise SpaceError(f"only {X.shape[0]} rows are finite in every member and the target")

    cols: dict[str, np.ndarray] = {}
    start = 0
    for m in names:
        w = mats[m].shape[1]
        cols[m] = np.arange(start, start + w)
        start += w
    d = X.shape[1]
    Xc = X - X.mean(0)

    knn_y = knn_indices(Y, k=k_nn, metric=metric) if knn else None
    pool = max(20 * n_control, 2000) if n_pool is None else n_pool
    rng = np.random.RandomState(random_state)

    def _r2(P):
        return ridge_predictivity(P, Y, groups=g, n_splits=n_splits, alphas=alphas, random_state=random_state)

    def read_set(label: str, members: list[str], control: bool) -> SetRead:
        idx = np.concatenate([cols[m] for m in members])
        P = Xc[:, idx]
        if not np.any(P):
            raise SpaceError(f"{label}: members {', '.join(members)} are constant on the read rows")
        # The set's columns are coordinate axes of X: an orthonormal basis of k = width directions.
        k = len(idx)
        Q = np.zeros((d, k))
        Q[idx, np.arange(k)] = 1.0
        share = captured_share(X, Q)
        rr = _r2(P)
        out = SetRead(label=label, members=list(members), k=k, share=float(share), r2=float(rr.r2),
                      r2_per_fold=[float(v) for v in rr.r2_per_fold])
        if knn:
            nr = neighbor_overlap_null(P, Y, k=k_nn, metric=metric, n_perm=n_perm, block_size=block_size,
                                       random_state=random_state, knn_y=knn_y)
            out.knn_overlap, out.knn_p, out.knn_null_mean = float(nr.observed), float(nr.p_value), float(nr.null_mean)
            out.knn_metric = metric
        if control and n_control > 0:
            if k >= d:
                raise SpaceError(
                    f"{label}: its span has rank {k} of the battery's {d} columns, so no random subspace "
                    f"of the battery differs from it; add the remaining members to the battery")
            ms = matched_subspaces(X, k, share, n=n_control, n_pool=pool, tol=tol, random_state=rng)
            out.control_share_range = ms.share_range
            out.gamma_bracket = ms.gamma_bracket
            for B in ms.bases:
                C = Xc @ B
                out.control_r2.append(float(_r2(C).r2))
                if knn:
                    out.control_knn.append(_overlap_from_knn(knn_indices(C, k=k_nn, metric=metric), knn_y))
            out.r2_percentile = _percentile(out.r2, out.control_r2)
            if knn:
                out.knn_percentile = _percentile(out.knn_overlap, out.control_knn)
        return out

    reads = []
    if strict is not None:
        reads.append(read_set("strict", strict, control=True))
    full = read_set("nameable", nameable, control=True)
    reads.append(full)

    rows = []
    if per_member:
        for m in nameable:
            alone = _r2(Xc[:, cols[m]])
            rest = [o for o in nameable if o != m]
            without = 0.0 if not rest else float(_r2(Xc[:, np.concatenate([cols[o] for o in rest])]).r2)
            rows.append({"member": m, "k": int(cols[m].size),
                         "r2_alone": float(alone.r2), "r2_without": without,
                         "r2_unique": float(full.r2 - without)})

    return Readout(
        reads=reads, per_member=rows, battery=names, nameable=nameable, strict=strict,
        n_rows=int(X.shape[0]), n_dropped=n_dropped,
        n_groups=None if g is None else int(len(np.unique(g))), target_dims=int(Y.shape[1]),
        params={"n_control": n_control, "n_pool": pool, "tol": tol, "knn": knn, "k_nn": k_nn,
                "metric": metric, "n_perm": n_perm, "block_size": block_size, "n_splits": n_splits,
                "alphas": [float(a) for a in alphas], "random_state": random_state},
    )
