"""Relations between fitted blocks: the shared directions two blocks carry.

A relation is a frozen cross-validated CCA between the score tables of two
fitted private blocks (``space project`` output), on rows that pair them —
an image and its captions, a film frame and its audio window. It answers the
question :mod:`psytwill.decompose` measures (how many directions do two
spaces share, and how strongly) and then *keeps* the answer as a versioned
map, so either side alone can be placed in the shared coordinates: an image
through the V side, a text through the L side.

Conventions (psytwill-space, DECIDED 2026-10-01):

- **k is a prefix.** The count is the leading run of components whose
  held-out correlation is at least ``r_min`` (default sqrt(.5), i.e. half the
  variance shared) and clears the permutation null. The saved directions are
  the top k, so a component past the first failure is never kept. The
  non-contiguous count is recorded beside k.
- **k = 0 is refused**, not saved: a measured absence is a fact about the
  pair, recorded with its evidence in the space release, not an empty map.
- **Inputs are pinned.** Each score table's sidecar names the block manifest
  it was projected through; the relation records that manifest's sha256, and
  ``check`` / ``project`` refuse a table projected through any other.
- **Pairing is declared, never inferred.** Rows join on named key columns.
  A side with several rows per join key must declare how they pool.
- **Subspace stability is reported, not gated** (v0.1): each fold's top-k
  map is applied to every row and compared with the full fit's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from psytwill import __version__
from psytwill.compare import block_permutation
from psytwill.decompose import (
    DEFAULT_PREFIX_ALPHA,
    CcaMap,
    _colwise_corr,
    _Whitener,
    cv_cca,
    cv_splits,
    fit_cca,
)
from psytwill.exceptions import InputError, SpaceError

#: 1.1: `fixed_k` (k given from outside, or null) and `prefix_k` (the measured
#: prefix count, which equals k unless k was fixed).
RELATE_SCHEMA_VERSION = "1.1"
DEFAULT_R_MIN = math.sqrt(0.5)
POOLS = ("none", "mean")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# score tables
# --------------------------------------------------------------------------


@dataclass
class ScoreTable:
    """A ``space project`` output: key columns + ``<block>_###`` score columns."""

    path: Path
    frame: pd.DataFrame
    block: str
    key: list[str]
    columns: list[str]
    manifest: Path
    manifest_sha256: str


def _sidecar(path: Path) -> Path:
    return path.parent / (path.name.removesuffix(path.suffix) + ".meta.json")


def read_scores(path: str | Path) -> ScoreTable:
    """Read a block score table and pin it to the block manifest it came from."""
    p = Path(path)
    side = _sidecar(p)
    if not side.exists():
        raise InputError(
            f"{p} has no sidecar {side.name}, so the block it was projected through is unknown. "
            "Write score tables with `psytwill space project`, which records it."
        )
    meta = json.loads(side.read_text())
    manifest = Path(meta["space"])
    if not manifest.exists():
        raise InputError(
            f"{side.name} names block manifest {manifest}, which does not exist. Re-project the "
            "table, or restore the manifest at that path."
        )
    frame = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    block = meta["block"]
    columns = sorted(c for c in frame.columns if c.startswith(f"{block}_") and c[len(block) + 1:].isdigit())
    if len(columns) != int(meta["k"]):
        raise InputError(f"{p}: {len(columns)} `{block}_###` columns but the sidecar says k = {meta['k']}")
    key = meta["key"].split(",")
    return ScoreTable(path=p, frame=frame, block=block, key=key, columns=columns,
                      manifest=manifest, manifest_sha256=sha256_file(manifest))


def _pooled(t: ScoreTable, join: Sequence[str], pool: str) -> tuple[pd.DataFrame, int]:
    missing = [j for j in join if j not in t.frame.columns]
    if missing:
        raise SpaceError(f"{t.path.name} has no join column(s) {missing}; its key is {t.key}")
    sizes = t.frame.groupby(list(join), sort=False).size()
    per_key = int(sizes.max())
    if per_key > 1 and pool == "none":
        raise SpaceError(
            f"{t.path.name} has up to {per_key} rows per join key {list(join)} "
            f"({int((sizes > 1).sum())} keys repeat). Declare how they pool "
            f"(`--pool-{{a,b}} mean`), or join on a finer key."
        )
    g = t.frame.groupby(list(join), sort=True)[t.columns]
    return (g.mean() if per_key > 1 else g.first()), per_key


def pair_tables(a: ScoreTable, b: ScoreTable, join: Sequence[str], *, pool_a: str = "none",
                pool_b: str = "none", exclude_ids: set[str] | None = None):
    """Rows paired on ``join`` (sorted by key), with per-side pooling applied."""
    for pool in (pool_a, pool_b):
        if pool not in POOLS:
            raise SpaceError(f"pool {pool!r} is not one of {POOLS}")
    A, per_a = _pooled(a, join, pool_a)
    B, per_b = _pooled(b, join, pool_b)
    idx = A.index.intersection(B.index).sort_values()
    if exclude_ids:
        first = idx.get_level_values(0) if isinstance(idx, pd.MultiIndex) else idx
        idx = idx[~pd.Index(first).astype(str).isin(exclude_ids)]
    if len(idx) == 0:
        raise SpaceError(f"{a.path.name} and {b.path.name} share no {list(join)} key")
    info = dict(n_a_only=int(len(A.index.difference(B.index))), n_b_only=int(len(B.index.difference(A.index))),
                rows_per_key_a=per_a, rows_per_key_b=per_b)
    return idx, A.loc[idx].to_numpy(float), B.loc[idx].to_numpy(float), info


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------


@dataclass
class RelationFit:
    name: str
    k: int
    map: CcaMap  # truncated to the top k directions
    manifest: dict = field(default_factory=dict)

    def project(self, X: np.ndarray, side: str) -> np.ndarray:
        if side == "a":
            return self.map.transform_a(X, self.k)
        if side == "b":
            return self.map.transform_b(X, self.k)
        raise SpaceError(f"side must be 'a' or 'b', not {side!r}")


def _prefix(ok: np.ndarray) -> int:
    return int(np.argmin(np.append(ok, False)))


def _subspace_overlap(P: np.ndarray, Q: np.ndarray) -> float:
    """Mean squared canonical correlation between two n x k variate sets (1 = same span)."""
    qp, _ = np.linalg.qr(P - P.mean(axis=0))
    qq, _ = np.linalg.qr(Q - Q.mean(axis=0))
    s = np.linalg.svd(qp.T @ qq, compute_uv=False)
    return float(np.mean(np.clip(s, 0.0, 1.0) ** 2))


def fit_relation(X: np.ndarray, Y: np.ndarray, *, name: str, groups: Sequence | None = None,
                 n_splits: int = 5, n_perm: int = 250, block_size: int | None = None,
                 r_min: float = DEFAULT_R_MIN, rank_cap: int | None = None,
                 prefix_alpha: float = DEFAULT_PREFIX_ALPHA, random_state: int = 0,
                 fixed_k: int | None = None) -> RelationFit:
    """Count the shared prefix by CV-CCA, then freeze the top-k map on every row.

    ``fixed_k`` freezes the map at a k given from outside (a baseline matched
    to another relation's k); the prefix count is still measured and recorded,
    as a diagnostic that does not move k.
    """
    keep = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X, Y = X[keep], Y[keep]
    g = None if groups is None else np.asarray(groups)[keep]
    rc = rank_cap or max(X.shape[1], Y.shape[1]) + 16
    res = cv_cca(X, Y, groups=g, n_splits=n_splits, rank_cap=rc, n_perm=n_perm,
                 block_size=block_size, prefix_alpha=prefix_alpha, random_state=random_state)
    r_cv = np.asarray(res.r_cv)
    ok = r_cv >= r_min
    if n_perm:
        ok &= r_cv > np.asarray(res.null_q)
    k = _prefix(ok) if fixed_k is None else int(fixed_k)
    if fixed_k is not None and not 1 <= k <= len(r_cv):
        raise SpaceError(f"{name}: fixed k {k} is outside 1..{len(r_cv)} canonical components")
    if k == 0:
        raise SpaceError(
            f"{name}: no component reaches held-out r >= {r_min:.3f} (first {r_cv[0]:.3f}), so the "
            "measured shared k is 0. A relation is not saved for an absent pair; record the absence "
            "and its evidence in the space release (`psytwill space release --absent`)."
        )
    full = fit_cca(X, Y, rc)
    top = CcaMap(wa=full.wa, wb=full.wb, U=full.U[:, :k], Vt=full.Vt[:k], r=full.r[:k])
    Pa, Pb = top.transform_a(X), top.transform_b(Y)
    top.r = _colwise_corr(Pa, Pb)  # the exact in-sample r, not the (n - 1) / n singular value

    stab = {"a": [], "b": []}
    for train, _ in cv_splits(len(X), g, n_splits, random_state):
        fm = fit_cca(X[train], Y[train], rc)
        stab["a"].append(_subspace_overlap(fm.transform_a(X, k), Pa))
        stab["b"].append(_subspace_overlap(fm.transform_b(Y, k), Pb))

    manifest = {
        "relate_schema_version": RELATE_SCHEMA_VERSION,
        "psytwill_version": __version__,
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "relation",
        "name": name,
        "k": k,
        "k_rule": ("fixed (given; the prefix count below is a diagnostic)" if fixed_k is not None else
                   f"prefix: held-out r >= r_min{' and > null quantile' if n_perm else ''}"),
        "fixed_k": None if fixed_k is None else int(fixed_k),
        "prefix_k": int(_prefix(ok)),
        "r_min": r_min,
        "count_r_ge_min": int((r_cv >= r_min).sum()),
        "n_rows": int(len(X)),
        "n_dropped_nan": int((~keep).sum()),
        "grouped": g is not None,
        "n_splits": res.n_splits,
        "rank_cap": rc,
        "rank_a": int(full.wa.rank),
        "rank_b": int(full.wb.rank),
        "n_perm": n_perm,
        "block_size": block_size,
        "prefix_alpha": prefix_alpha,
        "random_state": random_state,
        "r_insample": [float(v) for v in top.r],
        "r_cv": [float(v) for v in r_cv],
        "r_train": [float(v) for v in res.r_train],
        "null_q": [float(v) for v in res.null_q],
        "subspace_stability": {
            side: {"per_fold": [round(v, 6) for v in vals], "mean": float(np.mean(vals)),
                   "min": float(np.min(vals)),
                   "measure": "mean squared canonical correlation, fold top-k vs full top-k variates, all rows"}
            for side, vals in stab.items()
        },
    }
    return RelationFit(name=name, k=k, map=top, manifest=manifest)


def relate_tables(a: ScoreTable, b: ScoreTable, *, name: str, join: Sequence[str], pool_a: str = "none",
                  pool_b: str = "none", exclude_ids: set[str] | None = None, exclude_ids_file: str | None = None,
                  groups_from_label: bool = False, scope: str | None = None, **fit_kw) -> RelationFit:
    """Pair two block score tables, fit the relation, and pin both inputs in its manifest."""
    idx, X, Y, info = pair_tables(a, b, join, pool_a=pool_a, pool_b=pool_b, exclude_ids=exclude_ids)
    groups = None
    if groups_from_label:
        groups = (idx.get_level_values(0) if isinstance(idx, pd.MultiIndex) else idx).astype(str).to_numpy()
    fit = fit_relation(X, Y, name=name, groups=groups, **fit_kw)
    fit.manifest["sides"] = {
        side: {"block": t.block, "manifest": str(t.manifest), "manifest_sha256": t.manifest_sha256,
               "block_k": len(t.columns), "scores": str(t.path), "key": t.key, "pool": pool}
        for side, t, pool in (("a", a, pool_a), ("b", b, pool_b))
    }
    fit.manifest["join"] = list(join)
    fit.manifest["pairing"] = info
    fit.manifest["n_excluded_ids"] = len(exclude_ids or ())
    fit.manifest["exclude_ids_file"] = exclude_ids_file
    fit.manifest["scope"] = scope
    return fit


@dataclass
class RelationCheck:
    r: np.ndarray
    null_q: np.ndarray
    count: int
    prefix: int
    n: int


def check_relation(fit: RelationFit, X: np.ndarray, Y: np.ndarray, *, n_perm: int = 200,
                   block_size: int | None = None, alpha: float = 0.01, random_state: int = 0) -> RelationCheck:
    """Per-component r of the FROZEN map on paired rows the fit may never have seen."""
    keep = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    Pa, Pb = fit.project(X[keep], "a"), fit.project(Y[keep], "b")
    r = _colwise_corr(Pa, Pb)
    rng = np.random.RandomState(random_state)
    null = np.array([_colwise_corr(Pa, Pb[block_permutation(len(Pb), rng, block_size)]) for _ in range(n_perm)])
    q = np.quantile(null, 1 - alpha, axis=0) if n_perm else np.full(len(r), np.nan)
    rm = fit.manifest["r_min"]
    return RelationCheck(r=r, null_q=q, count=int((r >= rm).sum()), prefix=_prefix(r >= rm), n=int(keep.sum()))


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def save_relation(fit: RelationFit, out_dir: str | Path, *, stem: str | None = None) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or f"{fit.name}_v{RELATE_SCHEMA_VERSION.split('.')[0]}"
    m = fit.map
    npz = out / f"{stem}.npz"
    np.savez_compressed(npz, a_mean=m.wa.mean, a_components=m.wa.components, a_scale=m.wa.scale,
                        b_mean=m.wb.mean, b_components=m.wb.components, b_scale=m.wb.scale,
                        U=m.U, Vt=m.Vt, r=m.r)
    meta = dict(fit.manifest)
    meta["weights"] = npz.name
    manifest = out / f"{stem}.json"
    manifest.write_text(json.dumps(meta, indent=2))
    return npz, manifest


def load_relation(manifest_path: str | Path) -> RelationFit:
    mp = Path(manifest_path)
    meta = json.loads(mp.read_text())
    if meta.get("kind") != "relation":
        raise InputError(f"{mp} is not a relation manifest (kind = {meta.get('kind')!r})")
    d = np.load(mp.parent / meta["weights"])
    cm = CcaMap(wa=_Whitener(mean=d["a_mean"], components=d["a_components"], scale=d["a_scale"]),
                wb=_Whitener(mean=d["b_mean"], components=d["b_components"], scale=d["b_scale"]),
                U=d["U"], Vt=d["Vt"], r=d["r"])
    return RelationFit(name=meta["name"], k=int(meta["k"]), map=cm, manifest=meta)


def require_side(fit: RelationFit, table: ScoreTable, side: str) -> None:
    """Refuse a score table projected through any block manifest but the one fitted on."""
    want = fit.manifest["sides"][side]
    if table.manifest_sha256 != want["manifest_sha256"]:
        raise SpaceError(
            f"{table.path.name} was projected through {table.manifest} (block {table.block}), but "
            f"side {side!r} of relation {fit.name} was fitted on {want['manifest']} "
            f"(block {want['block']}, sha256 {want['manifest_sha256'][:12]}…). Re-project the table "
            "through that block, or refit the relation on this one."
        )
