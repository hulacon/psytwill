"""Shared and unique subspaces of a fitted block, and whether they reproduce across halves of its fit rows.

With ``pr`` member scaling every member reaches the block whitened, so the
block PCA over their concatenation is MAXVAR multi-set CCA. A direction two
members share at correlation r has eigenvalue 1 + r, and a direction no other
member spans has eigenvalue 1. So the eigenvalue is already a sharing measure,
and this module reads the block as two kinds of subspace rather than as
single directions:

- **shared** ``S``: the leading block directions with eigenvalue
  ``>= 1 + c`` (:func:`shared_subspace`). ``c`` is a floor on how much the
  members share each direction.
- **unique** ``U:<m>``, one per member m: m's coordinates regressed (OLS) on
  every other member's, and the directions of m that keep ``>= unique`` of
  their variance in the residual (:func:`unique_subspaces`). It is what m
  holds that no other member spans. This is the variance-partitioning sense of
  "unique", but thresholded and in m's own coordinates, so it is not a
  leave-one-out unique R². A member absent from ``>= max_absent`` of the rows
  (e.g. a faces member on faceless images) gets none and enters only through
  ``S``. A member with no direction past the cut is "shared only".
- **union** ``U``: the span of every ``U:<m>`` together.

Every object is a linear read of the block's concatenated member coordinates
(:meth:`psytwill.space.BlockMap.concat`). Each carries an orthonormalising
basis fitted on the block's fit rows (:func:`orthonormalise`), so its
:meth:`Subspace.scores` are uncorrelated and unit-variance there. Every
dimension then carries equal variance, and comparing two subspaces at matched
dimension compares them at matched variance. ``S`` is read through the block's
own placement (:meth:`~psytwill.space.BlockMap.scores_from_concat`), so a row
with an absent member is placed whenever its present members determine ``S``'s
directions. A unique object needs every member it reads present.

**Gate (reproducibility).** Each object is re-estimated on the block refitted
on each half of the fit-corpus units (:func:`unit_halves`: sha256 parity of
``"<seed>:<unit>"``). The two halves' spans are compared on common rows by
the mean squared canonical correlation (:func:`span_overlap`; 1 = one span).
An object **passes** when msc ``>= msc_min`` and msc exceeds by ``>= margin``
the mean msc of ``n_random`` pairs of random spans of the same dimension in
the block's coordinates (:func:`random_span_msc`), so overlap that comes from
dimensionality alone is not read as reproduction (:func:`reproducibility`).

``psytwill space subspaces --like <frozen manifest> -o DIR`` runs all of it
(the frozen fit's objects, both half fits, the gate) and writes them
(:func:`save_subspaces`). :func:`load_subspaces` and :func:`project_subspaces`
read the objects back and place new rows, for a downstream reader such as an
encoding harness.

Defaults: ``c = 0.5``, ``unique = 0.5``, gate ``msc >= .5`` and ``>= random
+ .25`` over 20 random pairs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from psytwill import __version__
from psytwill.exceptions import SpaceError

#: version of the files :func:`save_subspaces` writes
SUBSPACES_SCHEMA_VERSION = "1.0"
#: shared floor: S keeps directions with eigenvalue >= 1 + c
DEFAULT_SHARED_C = 0.5
#: a direction of m is unique when >= this share of its variance is residual
DEFAULT_UNIQUE = 0.5
#: a member absent from >= this share of the rows gets no unique subspace
DEFAULT_MAX_ABSENT = 0.5
#: the gate: msc >= GATE_MSC and msc >= mean random msc + GATE_MARGIN
GATE_MSC = 0.5
GATE_MARGIN = 0.25
DEFAULT_N_RANDOM = 20
KINDS = ("shared", "unique", "unique-union", "random")
#: a canonical correlation at or above this counts toward ``share_ge_90``
NEAR_ONE = 0.9


# --------------------------------------------------------------------------
# objects
# --------------------------------------------------------------------------


@dataclass
class Subspace:
    """One linear read of a block's concatenated member coordinates.

    ``shared``: the block's leading ``n_block`` directions, placed by the
    block itself. ``unique`` / ``unique-union``: ``(W - center) @ weights``,
    unplaced (NaN) on a row missing any member in ``reads``. ``basis_mean``
    and ``basis`` take those raw coordinates to orthonormal ones on the fit
    rows (None until :func:`orthonormalise`).
    """

    name: str
    kind: str
    reads: list[str]
    n_block: int | None = None
    weights: np.ndarray | None = None  # (sum rank, raw dim)
    center: np.ndarray | None = None  # (sum rank,)
    basis_mean: np.ndarray | None = None  # (raw dim,)
    basis: np.ndarray | None = None  # (raw dim, dim)
    detail: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise SpaceError(f"subspace {self.name!r}: kind must be one of {', '.join(KINDS)}; got {self.kind!r}")
        if self.kind == "shared" and self.n_block is None:
            raise SpaceError(f"subspace {self.name!r}: a shared subspace needs n_block")
        if self.kind != "shared" and (self.weights is None or self.center is None):  # unique, union, random
            raise SpaceError(f"subspace {self.name!r}: a {self.kind} subspace needs weights and center")

    @property
    def raw_dim(self) -> int:
        return int(self.n_block) if self.kind == "shared" else int(self.weights.shape[1])

    @property
    def dim(self) -> int:
        return self.raw_dim if self.basis is None else int(self.basis.shape[1])

    def raw_scores(self, block_map, W: np.ndarray) -> np.ndarray:
        """Raw coordinates of the rows of ``W`` (rows of ``block_map.concat``); NaN where unplaced."""
        W = np.asarray(W, dtype=float)
        if self.raw_dim == 0:
            return np.zeros((W.shape[0], 0))
        if self.kind == "shared":
            if self.n_block > block_map.k_max:
                raise SpaceError(f"subspace {self.name!r} reads {self.n_block} block directions; the map keeps "
                                 f"{block_map.k_max}")
            return block_map.scores_from_concat(W, self.n_block)
        cols = _read_mask(block_map, self.reads)
        out = np.full((W.shape[0], self.raw_dim), np.nan)
        ok = np.isfinite(W[:, cols]).all(axis=1)
        out[ok] = (W[np.ix_(ok, cols)] - self.center[cols]) @ self.weights[cols]
        return out

    def scores(self, block_map, W: np.ndarray) -> np.ndarray:
        """Orthonormal coordinates (uncorrelated, unit variance on the fit rows); NaN where unplaced."""
        if self.basis is None:
            raise SpaceError(f"subspace {self.name!r} has no basis yet: orthonormalise it on the fit rows first")
        R = self.raw_scores(block_map, W)
        return (R - self.basis_mean) @ self.basis


def member_slices(block_map) -> dict[str, slice]:
    """Each member's columns in the block's concatenated coordinates."""
    out, start = {}, 0
    for m in block_map.members:
        w = int(block_map.whiteners[m].rank)
        out[m] = slice(start, start + w)
        start += w
    if start != block_map.block_components.shape[1]:
        raise SpaceError(f"the members' whitened ranks sum to {start}, but the block's components have "
                         f"{block_map.block_components.shape[1]} columns")
    return out


def _read_mask(block_map, members: Sequence[str]) -> np.ndarray:
    sl = member_slices(block_map)
    unknown = [m for m in members if m not in sl]
    if unknown:
        raise SpaceError(f"not members of this block: {', '.join(unknown)}")
    mask = np.zeros(block_map.block_components.shape[1], dtype=bool)
    for m in members:
        mask[sl[m]] = True
    return mask


def _require_pr(block_map) -> None:
    bad = [m for m in block_map.members if block_map.whiteners[m].scaling != "pr"]
    if bad:
        raise SpaceError(f"shared/unique subspaces need `pr` member scaling (a direction shared at r then has "
                         f"eigenvalue 1 + r); {', '.join(bad)} are scaled otherwise. Refit with "
                         f"--member-scaling pr")


def _weights(w, n: int) -> np.ndarray:
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=float)
    if w.shape != (n,) or (w < 0).any() or w.sum() <= 0:
        raise SpaceError(f"row weights must be {n} non-negative values with a positive sum")
    return w / w.sum()


def _wmean_cov(X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = w @ X
    Xc = X - mu
    return mu, (Xc * w[:, None]).T @ Xc


def _rows(rows, n: int) -> np.ndarray:
    if rows is None:
        return np.ones(n, dtype=bool)
    r = np.asarray(rows, dtype=bool)
    if r.shape != (n,):
        raise SpaceError(f"rows must be a boolean mask over the {n} rows of W")
    return r


def shared_subspace(block_map, c: float = DEFAULT_SHARED_C) -> Subspace:
    """``S``: the block's leading directions with eigenvalue ``>= 1 + c``."""
    _require_pr(block_map)
    if c < 0:
        raise SpaceError(f"c must be >= 0 (an eigenvalue of 1 is a direction no member shares); got {c}")
    lam = np.asarray(block_map.block_eigenvalues, dtype=float)
    over = lam >= 1 + c
    n = int(np.argmin(over)) if not over.all() else int(over.size)
    if over[n:].any():
        raise SpaceError("the block's eigenvalues are not in descending order; S would not be its leading "
                         "directions")
    return Subspace(name="S", kind="shared", reads=list(block_map.members), n_block=n,
                    detail={"c": float(c), "eigenvalues": [float(v) for v in lam[:n]],
                            "next_eigenvalue": float(lam[n]) if n < lam.size else None,
                            "k_max": int(lam.size),
                            # every direction the map keeps is past the floor: S may be cut by k, not by c
                            "capped_by_k": bool(n == lam.size)})


def absent_shares(block_map, W: np.ndarray, rows=None) -> dict[str, float]:
    """Each member's share of ``rows`` on which it is absent (any NaN in its columns)."""
    W = np.asarray(W, dtype=float)
    r = _rows(rows, W.shape[0])
    if not r.any():
        raise SpaceError("no rows to read")
    return {m: float((~np.isfinite(W[np.ix_(r, np.arange(W.shape[1])[s])]).all(axis=1)).mean())
            for m, s in member_slices(block_map).items()}


def unique_subspaces(block_map, W: np.ndarray, *, unique: float = DEFAULT_UNIQUE,
                     max_absent: float = DEFAULT_MAX_ABSENT, weights=None, rows=None,
                     rel_tol: float = 1e-10) -> tuple[dict[str, Subspace], dict]:
    """``U:<m>`` per member and their union ``U``, estimated on ``rows`` of ``W``.

    ``W`` is ``block_map.concat`` of the rows; ``rows`` (default all) the
    ones to estimate on; ``weights`` (one per row of ``W``, default equal) the
    block's row weights. The regression and the residual share use the rows
    on which every read member is present.

    A direction's unique share is its residual variance over its total
    variance in m's coordinates (a generalised eigenproblem). On whitened
    coordinates whose covariance is the identity this is the eigenvalue of the
    residual's covariance, the 0.1.0 probe's rule.

    Returns ``({"U:<m>": ..., "U": ...}, info)``; ``info`` names the members
    read (``read``), those absent too often (``skipped``, with their absent
    shares), the shared-only ones and the row count. A member with no unique
    direction gets a zero-dimensional ``U:<m>``.
    """
    if not 0 < unique <= 1:
        raise SpaceError(f"unique must be in (0, 1]; got {unique}")
    W = np.asarray(W, dtype=float)
    r = _rows(rows, W.shape[0])
    sl = member_slices(block_map)
    absent = absent_shares(block_map, W, r)
    read = [m for m in block_map.members if absent[m] < max_absent]
    skipped = {m: absent[m] for m in block_map.members if m not in read}
    if len(read) < 2:
        raise SpaceError(f"a unique subspace needs >= 2 members present on most rows; only {read} are "
                         f"(absent shares {absent})")
    cols = _read_mask(block_map, read)
    complete = r & np.isfinite(W[:, cols]).all(axis=1)
    n = int(complete.sum())
    if n <= int(cols.sum()) + 1:
        raise SpaceError(f"{n} rows have every read member present: too few to regress {int(cols.sum())} "
                         f"columns. Lower max_absent, or read fewer members")
    w = _weights(None if weights is None else np.asarray(weights, dtype=float)[complete], n)
    full_center = np.zeros(W.shape[1])
    mu, _ = _wmean_cov(W[np.ix_(complete, np.flatnonzero(cols))], w)
    full_center[cols] = mu
    Xc = W[np.ix_(complete, np.flatnonzero(cols))] - mu
    sw = np.sqrt(w)[:, None]
    pos = {}  # member -> its columns inside the read columns
    start = 0
    for m in block_map.members:
        if m in read:
            width = sl[m].stop - sl[m].start
            pos[m] = np.arange(start, start + width)
            start += width
    out: dict[str, Subspace] = {}
    shared_only = []
    for m in read:
        own = pos[m]
        others = np.concatenate([pos[o] for o in read if o != m])
        Y, Z = Xc[:, own], Xc[:, others]
        B, *_ = np.linalg.lstsq(Z * sw, Y * sw, rcond=None)
        E = Y - Z @ B
        Cy = (Y * w[:, None]).T @ Y
        Ce = (E * w[:, None]).T @ E
        ey, vy = np.linalg.eigh(Cy)
        keep = ey > ey.max() * rel_tol if ey.size and ey.max() > 0 else np.zeros_like(ey, dtype=bool)
        Rm = vy[:, keep] / np.sqrt(ey[keep])  # Cy^{-1/2} on m's identified span
        share, vec = np.linalg.eigh(Rm.T @ Ce @ Rm)
        share, vec = share[::-1], vec[:, ::-1]
        hit = share >= unique
        a = Rm @ vec[:, hit]  # m-coordinate directions; their residual is the unique coordinate
        Wfull = np.zeros((W.shape[1], int(hit.sum())))
        own_cols = np.flatnonzero(cols)[own]
        oth_cols = np.flatnonzero(cols)[others]
        Wfull[own_cols] = a
        Wfull[oth_cols] = -B @ a
        if not hit.any():
            shared_only.append(m)
        out[f"U:{m}"] = Subspace(name=f"U:{m}", kind="unique", reads=list(read), weights=Wfull,
                                 center=full_center.copy(),
                                 detail={"member": m, "member_rank": int(own.size),
                                         "unique_shares": [float(v) for v in share[:max(8, int(hit.sum()))]],
                                         "unique": float(unique)})
    parts = [out[f"U:{m}"].weights for m in read if out[f"U:{m}"].raw_dim]
    union = np.hstack(parts) if parts else np.zeros((W.shape[1], 0))
    out["U"] = Subspace(name="U", kind="unique-union", reads=list(read), weights=union, center=full_center.copy(),
                        detail={"members": [m for m in read if m not in shared_only]})
    info = {"read": read, "skipped": skipped, "shared_only": shared_only, "n_rows": n,
            "max_absent": float(max_absent), "unique": float(unique)}
    return out, info


def orthonormalise(sub: Subspace, block_map, W: np.ndarray, *, weights=None, rows=None,
                   rel_tol: float = 1e-8) -> Subspace:
    """``sub`` with a basis that makes its scores uncorrelated and unit-variance on ``rows`` of ``W``.

    Directions with variance below ``rel_tol`` of the largest are dropped, so
    ``dim`` can fall below ``raw_dim`` (a union of overlapping unique
    subspaces, say); ``detail["basis"]`` records both and the row count.
    """
    W = np.asarray(W, dtype=float)
    r = _rows(rows, W.shape[0])
    R = sub.raw_scores(block_map, W[r])
    if sub.raw_dim == 0:
        return replace(sub, basis_mean=np.zeros(0), basis=np.zeros((0, 0)),
                       detail={**sub.detail, "basis": {"raw_dim": 0, "dim": 0, "n_rows": 0}})
    ok = np.isfinite(R).all(axis=1)
    n = int(ok.sum())
    if n <= sub.raw_dim:
        raise SpaceError(f"subspace {sub.name!r}: {n} placed rows cannot orthonormalise {sub.raw_dim} directions")
    w = _weights(None if weights is None else np.asarray(weights, dtype=float)[r][ok], n)
    mu, C = _wmean_cov(R[ok], w)
    ev, vec = np.linalg.eigh(C)
    ev, vec = ev[::-1], vec[:, ::-1]
    keep = ev > ev[0] * rel_tol if ev[0] > 0 else np.zeros_like(ev, dtype=bool)
    basis = vec[:, keep] / np.sqrt(ev[keep])
    return replace(sub, basis_mean=mu, basis=basis,
                   detail={**sub.detail, "basis": {"raw_dim": sub.raw_dim, "dim": int(keep.sum()), "n_rows": n}})


def block_subspaces(block_map, W: np.ndarray, *, c: float = DEFAULT_SHARED_C, unique: float = DEFAULT_UNIQUE,
                    max_absent: float = DEFAULT_MAX_ABSENT, weights=None, rows=None,
                    basis: bool = True) -> tuple[dict[str, Subspace], dict]:
    """Every object of one block: ``S``, each ``U:<m>``, then ``U``, in that order.

    With ``basis`` (default) each is orthonormalised on the same rows and
    weights it was estimated on.
    """
    objs = {"S": shared_subspace(block_map, c)}
    uniq, info = unique_subspaces(block_map, W, unique=unique, max_absent=max_absent, weights=weights, rows=rows)
    objs.update(uniq)
    if basis:
        objs = {k: orthonormalise(v, block_map, W, weights=weights, rows=rows) for k, v in objs.items()}
    return objs, info


def _haar(rng: np.random.Generator, p: int, d: int) -> np.ndarray:
    """A uniformly random (p x d) matrix with orthonormal columns."""
    Q, R = np.linalg.qr(rng.normal(size=(p, d)))
    return Q * np.sign(np.diag(R))


def draw_subspaces(objects: dict[str, Subspace], block_map, W: np.ndarray, *, n: int = DEFAULT_N_RANDOM,
                   seed: int = 0, weights=None, rows=None) -> tuple[dict[str, Subspace], dict]:
    """Random spans for reading the subspaces against each other and a reference, each orthonormal on the fit rows.

    With ``d = min(dim S, dim U)``:

    - ``S@dNNN`` / ``U@dNNN``: draw NNN, a uniformly random d-dimensional span
      inside ``S`` / ``U`` (the same draw index pairs them). A side with exactly
      d dimensions is its whole span every time.
    - ``ref@dNNN``: a random d-dimensional span of the block's coordinates
      over every member the unique objects read, orthonormalised on ``rows``
      (the reference both sides are read against).
    - ``S@<m>.dNNN``: for each member m with a unique subspace, a random
      ``dim U:<m>``-dimensional span inside ``S``.

    ``objects`` must hold orthonormalised ``S`` and ``U`` (as
    :func:`block_subspaces` returns them). Draws inside a subspace rotate its
    orthonormal coordinates, so they stay orthonormal without the fit rows.
    Returns ``(draws, info)``.
    """
    for name in ("S", "U"):
        if name not in objects or objects[name].basis is None:
            raise SpaceError(f"draws need an orthonormalised {name!r}")
    if n < 1:
        raise SpaceError(f"n must be >= 1; got {n}")
    S, U = objects["S"], objects["U"]
    d = min(S.dim, U.dim)
    rng = np.random.default_rng(seed)
    out: dict[str, Subspace] = {}

    def inside(parent: Subspace, name: str, k: int, r: int) -> Subspace:
        R = np.eye(parent.dim)[:, :k] if k == parent.dim else _haar(rng, parent.dim, k)
        return replace(parent, name=name, basis=parent.basis @ R,
                       detail={"draw_of": parent.name, "draw": r, "dim": k})

    if d:
        for r in range(n):
            out[f"S@d{r:03d}"] = inside(S, f"S@d{r:03d}", d, r)
            out[f"U@d{r:03d}"] = inside(U, f"U@d{r:03d}", d, r)
        cols = _read_mask(block_map, U.reads)
        for r in range(n):
            Wr = np.zeros((W.shape[1], d))
            Wr[cols] = _haar(rng, int(cols.sum()), d)
            ref = Subspace(name=f"ref@d{r:03d}", kind="random", reads=list(U.reads), weights=Wr,
                           center=U.center.copy(), detail={"draw": r, "dim": d})
            out[ref.name] = orthonormalise(ref, block_map, W, weights=weights, rows=rows)
    per_member = {}
    for name, o in objects.items():
        if o.kind == "unique" and o.dim:
            m = o.detail["member"]
            k = min(o.dim, S.dim)
            per_member[m] = k
            if k:
                for r in range(n):
                    out[f"S@{m}.d{r:03d}"] = inside(S, f"S@{m}.d{r:03d}", k, r)
    return out, {"n": int(n), "seed": int(seed), "d": int(d), "per_member_dim": per_member}


# --------------------------------------------------------------------------
# overlap and the gate
# --------------------------------------------------------------------------


def _orth(A: np.ndarray) -> np.ndarray:
    A = A - A.mean(axis=0)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, s > s[0] * 1e-8] if s.size and s[0] > 0 else U[:, :0]


def span_overlap(A: np.ndarray, B: np.ndarray) -> dict:
    """Canonical correlations between the column spans of ``A`` and ``B`` (same rows, finite).

    ``msc`` is their mean square over the smaller dimension (1 = one span is
    inside the other); ``share_ge_90`` the share of them ``>= .9``.
    """
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if A.shape[0] != B.shape[0]:
        raise SpaceError(f"span_overlap needs the same rows; got {A.shape[0]} and {B.shape[0]}")
    Qa, Qb = _orth(A), _orth(B)
    if Qa.shape[1] == 0 or Qb.shape[1] == 0:
        return {"dim_a": int(Qa.shape[1]), "dim_b": int(Qb.shape[1]), "msc": None, "share_ge_90": None}
    cc = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)
    return {"dim_a": int(Qa.shape[1]), "dim_b": int(Qb.shape[1]), "msc": float((cc ** 2).mean()),
            "share_ge_90": float((cc >= NEAR_ONE).mean())}


def random_span_msc(X: np.ndarray, d: int, n: int = DEFAULT_N_RANDOM, seed: int = 0) -> float | None:
    """Mean msc of ``n`` pairs of independent random ``d``-dimensional spans of ``X``'s columns."""
    X = np.asarray(X, dtype=float)
    if d <= 0:
        return None
    if n < 1:
        raise SpaceError(f"n_random must be >= 1; got {n}")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        Ra, Rb = rng.normal(size=(X.shape[1], d)), rng.normal(size=(X.shape[1], d))
        vals.append(span_overlap(X @ Ra, X @ Rb)["msc"])
    return float(np.mean(vals))


def unit_of(label: str) -> str:
    """A row label's unit: its first key part (the stimulus id)."""
    return str(label).split("|")[0]


def unit_halves(labels: Sequence[str], seed: int = 0) -> np.ndarray:
    """0 / 1 per row: the sha256 parity of ``"<seed>:<unit>"``, so a unit's rows stay in one half."""
    memo: dict[str, int] = {}
    out = np.empty(len(labels), dtype=np.int8)
    for i, lab in enumerate(labels):
        u = unit_of(lab)
        if u not in memo:
            memo[u] = hashlib.sha256(f"{seed}:{u}".encode()).digest()[0] & 1
        out[i] = memo[u]
    return out


def reproducibility(halves: Sequence[tuple], reference: np.ndarray, *, n_random: int = DEFAULT_N_RANDOM,
                    seed: int = 0, msc_min: float = GATE_MSC, margin: float = GATE_MARGIN) -> pd.DataFrame:
    """The gate: each object's two half estimates compared on common rows.

    ``halves`` is two ``(objects, block_map, W)``: a half's objects (from
    :func:`block_subspaces` on its own rows), its map, and the **same common
    rows** concatenated through that map. ``reference`` holds the block's
    coordinates on common rows for the random-span pairs. One row per object
    present in either half, in half 0's order: dims, msc, share ``>= .9``,
    the random msc at the smaller dimension, and ``passed`` (None for an
    object with no dimensions in either half).
    """
    if len(halves) != 2:
        raise SpaceError(f"the gate compares two halves; got {len(halves)}")
    (o0, m0, W0), (o1, m1, W1) = halves
    if np.asarray(W0).shape[0] != np.asarray(W1).shape[0]:
        raise SpaceError("both halves must be read on the same common rows")
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref).all(axis=1)]
    names = list(o0) + [k for k in o1 if k not in o0]
    rows = []
    for i, name in enumerate(names):
        a, b = o0.get(name), o1.get(name)
        kind = (a or b).kind
        d0, d1 = (a.raw_dim if a else 0), (b.raw_dim if b else 0)
        row = {"object": name, "kind": kind, "dim_h0": d0, "dim_h1": d1, "msc": None, "share_ge_90": None,
               "random_msc": None, "n_rows": 0, "passed": None, "reason": ""}
        if d0 == 0 and d1 == 0:
            row["reason"] = "no dimensions in either half"
        elif d0 == 0 or d1 == 0:
            row.update(passed=False, reason="dimensions in one half only")
        else:
            A, B = a.raw_scores(m0, W0), b.raw_scores(m1, W1)
            ok = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
            ov = span_overlap(A[ok], B[ok])
            d = min(ov["dim_a"], ov["dim_b"])
            rnd = random_span_msc(ref, d, n_random, seed + i)
            passed = ov["msc"] is not None and ov["msc"] >= msc_min and ov["msc"] - rnd >= margin
            row.update(msc=ov["msc"], share_ge_90=ov["share_ge_90"], random_msc=rnd, n_rows=int(ok.sum()),
                       passed=bool(passed),
                       reason="" if passed else (f"msc < {msc_min}" if ov["msc"] < msc_min
                                                 else f"msc - random < {margin}"))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# persistence and projection
# --------------------------------------------------------------------------


def save_subspaces(objects: dict[str, Subspace], meta: dict, out_dir: str | Path, *, stem: str,
                   gate: pd.DataFrame | None = None) -> tuple[Path, Path]:
    """``<stem>.npz`` (every object's arrays) and ``<stem>.json`` (meta, objects, gate); ``<stem>_gate.csv`` too."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    listing = []
    for name, s in objects.items():
        if s.basis is None:
            raise SpaceError(f"subspace {name!r} has no basis: orthonormalise before saving")
        for key in ("weights", "center", "basis_mean", "basis"):
            v = getattr(s, key)
            if v is not None:
                arrays[f"{name}::{key}"] = np.asarray(v, dtype=float)
        listing.append({"name": name, "kind": s.kind, "reads": s.reads, "n_block": s.n_block,
                        "raw_dim": s.raw_dim, "dim": s.dim, "detail": s.detail})
    npz = out / f"{stem}.npz"
    np.savez_compressed(npz, **arrays)
    doc = {"schema_version": SUBSPACES_SCHEMA_VERSION, "psytwill_version": __version__, **meta,
           "npz": npz.name, "objects": listing}
    if gate is not None:
        gate.to_csv(out / f"{stem}_gate.csv", index=False)
        doc["gate"] = json.loads(gate.to_json(orient="records"))
    js = out / f"{stem}.json"
    js.write_text(json.dumps(doc, indent=2))
    return npz, js


def load_subspaces(path: str | Path) -> tuple[dict[str, Subspace], dict]:
    """The objects and the manifest :func:`save_subspaces` wrote (pass the .json)."""
    p = Path(path)
    doc = json.loads(p.read_text())
    if str(doc.get("schema_version", "")).split(".")[0] != SUBSPACES_SCHEMA_VERSION.split(".")[0]:
        raise SpaceError(f"{p.name}: subspaces schema {doc.get('schema_version')!r}; this psytwill reads "
                         f"{SUBSPACES_SCHEMA_VERSION}")
    data = np.load(p.parent / doc["npz"])
    objs = {}
    for o in doc["objects"]:
        get = (lambda k, n=o["name"]: data[f"{n}::{k}"] if f"{n}::{k}" in data.files else None)
        objs[o["name"]] = Subspace(name=o["name"], kind=o["kind"], reads=list(o["reads"]), n_block=o["n_block"],
                                   weights=get("weights"), center=get("center"), basis_mean=get("basis_mean"),
                                   basis=get("basis"), detail=o["detail"])
    return objs, doc


def project_subspaces(objects: dict[str, Subspace], fit, spaces: dict, *, absent: dict | None = None,
                      names: Sequence[str] | None = None) -> tuple[dict[str, np.ndarray], list[str]]:
    """Orthonormal coordinates of every aligned row of ``spaces`` in each object (NaN where unplaced).

    ``fit`` is the block fit the objects were estimated on (its manifest is
    ``meta["fit_manifest"]``); rows are aligned and masked as
    :meth:`psytwill.space.BlockFit.project` does. Returns ``({name: scores},
    labels)``.
    """
    aligned, labels = fit.aligned(spaces, absent)
    W = fit.map.concat(aligned)
    want = list(objects) if names is None else list(names)
    missing = [n for n in want if n not in objects]
    if missing:
        raise SpaceError(f"no such subspace: {', '.join(missing)} (have {', '.join(objects)})")
    return {n: objects[n].scores(fit.map, W) for n in want}, labels
