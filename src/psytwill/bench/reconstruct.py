"""The reconstructed-member arm (R): a battery member rebuilt from a space's coordinates.

A space that subsumes a member at R^2 >= 0.5 keeps most of it, not all of it.
R measures what the rest is worth on a task: a ridge map from the space's
block scores to the member's features is fit on the block's own fit rows,
then applied to the benchmark items' coordinates, and the output is scored
exactly as the member itself is. The gap between R and the member is the
cost of compression for that member on that task.

The map is fit on rows the space was fit on and never on benchmark items;
the items only ever pass through the frozen map. Its out-of-fold R^2 on the
fit rows is recorded beside it, so the reconstruction's quality is stated,
not assumed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from psytwill.compare import DEFAULT_ALPHAS
from psytwill.exceptions import BenchError
from psytwill.store import SpaceMatrix

from .core import fit_map, folds

MIN_FIT_ROWS = 100


def _label(values) -> str:
    out = []
    for v in values:
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        out.append(str(v))
    return "|".join(out)


def scores_matrix(path: str | Path, key: Sequence[str], exclude_ids: set[str] | None = None) -> SpaceMatrix:
    """A wide ``space project`` table (key columns + ``<block>_NNN``) as a SpaceMatrix."""
    df = pd.read_parquet(path)
    missing = [k for k in key if k not in df.columns]
    if missing:
        raise BenchError(f"{path} lacks key column(s) {missing}")
    if exclude_ids:
        df = df[~df[key[0]].astype(str).isin(exclude_ids)]
    cols = [c for c in df.columns if c not in key]
    labels = [_label(r) for r in df[list(key)].itertuples(index=False)]
    if len(set(labels)) != len(labels):
        raise BenchError(f"{path}: a key repeats")
    return SpaceMatrix(name=Path(path).stem, labels=labels, X=df[cols].to_numpy(dtype=float), features=cols)


def _relabel(sm: SpaceMatrix) -> SpaceMatrix:
    """Member labels in the same spelling as `_label` (3.0 -> 3); the stimulus id is never touched."""
    labels = []
    for lab in sm.labels:
        head, *rest = lab.split("|")
        parts = [head]
        for p in rest:
            try:
                f = float(p)
                parts.append(str(int(f)) if f.is_integer() and p not in ("nan",) else p)
            except ValueError:
                parts.append(p)
        labels.append("|".join(parts))
    return SpaceMatrix(name=sm.name, labels=labels, X=sm.X, features=sm.features, modality=sm.modality,
                       extractor=sm.extractor, n_replicates=sm.n_replicates)


def fit_reconstruction(scores: SpaceMatrix, member: SpaceMatrix, *, alphas: Sequence[float] = DEFAULT_ALPHAS,
                       cv_rows: int = 20000, n_splits: int = 5, seed: int = 0):
    """Ridge from block scores to the member on their shared, complete rows; plus its out-of-fold R^2."""
    member = _relabel(member)
    pos = {lab: i for i, lab in enumerate(member.labels)}
    shared = [i for i, lab in enumerate(scores.labels) if lab in pos]
    if len(shared) < MIN_FIT_ROWS:
        raise BenchError(f"only {len(shared)} fit rows carry both the block scores and {member.name!r}; "
                         "are the two keyed the same way?")
    X = scores.X[shared]
    Y = member.X[[pos[scores.labels[i]] for i in shared]]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X, Y = X[ok], Y[ok]
    groups = np.array([scores.labels[i].split("|")[0] for i in np.array(shared)[ok]])
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(len(X), size=min(cv_rows, len(X)), replace=False))
    pred = np.full((len(sub), Y.shape[1]), np.nan)
    for tr, te in folds(groups[sub], n_splits, seed):
        pred[te] = fit_map(X[sub][tr], Y[sub][tr], alphas).predict(X[sub][te])
    resid = ((Y[sub] - pred) ** 2).sum(0)
    tot = ((Y[sub] - Y[sub].mean(0)) ** 2).sum(0)
    r2_pooled = float(1 - resid.sum() / tot.sum())  # variance-weighted over the member's columns
    lm = fit_map(X, Y, alphas)
    return lm, {"n_fit_rows": int(len(X)), "n_rows_dropped_incomplete": int((~ok).sum()), "alpha": lm.alpha,
                "cv_r2_variance_weighted": r2_pooled, "cv_rows": int(len(sub)), "cv_folds": n_splits,
                "cv_grouped_by": "first key column"}


def write_reconstruction(lm, apply: SpaceMatrix, *, name: str, member: SpaceMatrix, key: Sequence[str],
                         out: str | Path, provenance: dict) -> Path:
    """Long-form features table (one row per item x feature) + a sidecar beside it."""
    ok = np.isfinite(apply.X).all(axis=1)
    Y = np.full((apply.n, member.dim), np.nan)
    Y[ok] = lm.predict(apply.X[ok])
    feats = [f"{name}_{j:04d}" for j in range(member.dim)]
    parts = [lab.split("|") for lab in apply.labels]
    wide = pd.DataFrame(Y, columns=feats)
    for j, k in enumerate(key):
        col = [p[j] for p in parts]
        wide.insert(j, k, pd.to_numeric(col) if k != "stimulus_id" else col)
    long = wide.melt(id_vars=list(key), var_name="feature", value_name="value")
    long.insert(len(key), "model", name)
    long["value_str"] = pd.Series([None] * len(long), dtype="string")
    long["modality"] = member.modality
    long["extractor"] = "psytwill-bench-reconstruct"
    from psytwill import __version__

    long["extractor_version"] = __version__
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out, index=False)
    side = out.with_suffix(".meta.json")
    side.write_text(json.dumps({"kind": "reconstruction", "model": name, "member": member.name,
                                "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "psytwill_version": __version__,
                                "n_items": int(apply.n), "n_items_unplaced": int((~ok).sum()), **provenance},
                               indent=2, default=str) + "\n")
    return side
