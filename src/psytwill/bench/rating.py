"""Per-item ratings: predict each item's mean human rating from a model's coordinates.

Input is a ratings table with ``participant, stimulus_id`` and one column per
rated dimension (valence, arousal, ...), one row per participant per item;
an item a participant did not rate is NaN. The target is the mean over
participants per item.

- ``mapped``: a ridge from the model's coordinates to the scalar, folds over
  items (shuffled KFold), alpha chosen inside each fit. The score is Pearson r
  between the out-of-fold predictions and the ratings, over all items.
- The CI resamples items (the predictions stay out-of-fold; draws depend only
  on ``seed`` and the sorted item set, so two runs share them).
- ``reliability``: split-half between participants (random halves, Pearson r
  of the two half-means over items, Spearman-Brown corrected to the full
  panel), averaged over splits. ``r_ceiling = sqrt(reliability)`` is the
  highest r a model can expect against a mean that noisy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from psytwill.compare import DEFAULT_ALPHAS
from psytwill.exceptions import BenchError
from psytwill.store import SpaceMatrix

from .core import fit_map, folds, prototypes
from .rdm import _interval, item_draws


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _wide(ratings: pd.DataFrame, dim: str) -> tuple[list[str], np.ndarray]:
    """(participants, items) matrix of one dimension; NaN where not rated."""
    w = ratings.pivot(index="participant", columns="stimulus_id", values=dim)
    return list(w.columns), w.to_numpy(dtype=float)


def split_half_reliability(R: np.ndarray, *, n_splits: int = 100, seed: int = 0) -> dict:
    """Mean Spearman-Brown split-half reliability over random participant halves."""
    n_p = R.shape[0]
    if n_p < 4:
        raise BenchError(f"a split-half reliability needs at least 4 participants; got {n_p}")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_splits):
        perm = rng.permutation(n_p)
        h1, h2 = perm[: n_p // 2], perm[n_p // 2: n_p // 2 * 2]
        with np.errstate(invalid="ignore"):
            m1, m2 = np.nanmean(R[h1], axis=0), np.nanmean(R[h2], axis=0)
        ok = ~(np.isnan(m1) | np.isnan(m2))
        r = _pearson(m1[ok], m2[ok])
        vals.append(2 * r / (1 + r))
    v = np.array(vals)
    rel = float(np.nanmean(v))
    return {"reliability": rel, "r_ceiling": float(np.sqrt(rel)) if rel > 0 else float("nan"),
            "lo": float(np.nanquantile(v, .025)), "hi": float(np.nanquantile(v, .975)),
            "n_splits": n_splits, "half_size": n_p // 2,
            "method": "random participant halves; Pearson r of half-means; Spearman-Brown; mean over splits"}


def rating(emb: SpaceMatrix, ratings: pd.DataFrame, dimensions: Sequence[str], *, n_splits: int = 5,
           alphas: Sequence[float] = DEFAULT_ALPHAS, n_boot: int = 2000, seed: int = 0,
           n_reliability_splits: int = 100) -> tuple[pd.DataFrame, dict]:
    missing = [c for c in ("participant", "stimulus_id", *dimensions) if c not in ratings.columns]
    if missing:
        raise BenchError(f"ratings table lacks column(s) {missing}")
    r = ratings.astype({"participant": str, "stimulus_id": str})
    if r.duplicated(["participant", "stimulus_id"]).any():
        raise BenchError("ratings table repeats a (participant, stimulus_id); average repeats upstream")
    ids, P, _ = prototypes(emb)
    pos = {s: i for i, s in enumerate(ids)}
    rows, summary = [], {"n_splits": n_splits, "n_boot": n_boot, "seed": seed, "mode": "mapped",
                         "statistic": "Pearson r, out-of-fold predictions vs mean rating, over items",
                         "ci": "items resampled; predictions fixed"}
    for dim in dimensions:
        items, R = _wide(r, dim)
        y = np.nanmean(R, axis=0)
        n_raters = np.sum(~np.isnan(R), axis=0)
        keep = n_raters > 0
        items = [s for s, k in zip(items, keep) if k]
        y, n_raters, R = y[keep], n_raters[keep], R[:, keep]
        absent = [s for s in items if s not in pos]
        if absent:
            raise BenchError(f"{len(absent)} rated item(s) have no embedding (e.g. {absent[:3]})")
        X = P[[pos[s] for s in items]]
        pred = np.full(len(items), np.nan)
        fold = np.full(len(items), -1)
        alphas_used = []
        for f, (tr, te) in enumerate(folds(len(items), n_splits, seed)):
            lm = fit_map(X[tr], y[tr, None], alphas)
            pred[te] = lm.predict(X[te])[:, 0]
            fold[te] = f
            alphas_used.append(lm.alpha)
        rows.append(pd.DataFrame({"stimulus_id": items, "dimension": dim, "rating": y, "n_raters": n_raters,
                                  "prediction": pred, "fold": fold}))
        summary[dim] = _summarize_dim(y, pred, n_boot=n_boot, seed=seed) | {
            "alpha_per_fold": alphas_used,
            "ceiling": split_half_reliability(R, n_splits=n_reliability_splits, seed=seed)}
    return pd.concat(rows, ignore_index=True), summary


def _summarize_dim(y: np.ndarray, pred: np.ndarray, *, n_boot: int, seed: int) -> dict:
    n = len(y)
    boots = np.array([_pearson(pred[k], y[k]) for k in item_draws(n, n_boot, seed)])
    return {"n_items": n, "r": _interval(_pearson(pred, y), boots, n)}


def compare_rating(a: pd.DataFrame, b: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0
                   ) -> tuple[pd.DataFrame, dict]:
    """Paired r_a - r_b per dimension, on shared item draws."""
    from .compare import verdict

    key = ["stimulus_id", "dimension"]
    m = a[[*key, "rating", "prediction"]].merge(b[[*key, "rating", "prediction"]], on=key, suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        raise BenchError(f"only {len(m)} of {len(a)} / {len(b)} rated items are in both runs")
    if not np.allclose(m["rating_a"], m["rating_b"]):
        raise BenchError("the two runs carry different ratings for the same items")
    summary: dict = {"n_items": int(m["stimulus_id"].nunique()), "ci_unit": "items; predictions fixed"}
    for dim, d in m.groupby("dimension", sort=False):
        d = d.sort_values("stimulus_id")
        y, pa, pb = d["rating_a"].to_numpy(), d["prediction_a"].to_numpy(), d["prediction_b"].to_numpy()
        draws = item_draws(len(y), n_boot, seed)
        ba = np.array([_pearson(pa[k], y[k]) for k in draws])
        bb = np.array([_pearson(pb[k], y[k]) for k in draws])
        ra, rb = _pearson(pa, y), _pearson(pb, y)
        diff = _interval(ra - rb, ba - bb, len(y))
        b_int = _interval(rb, bb, len(y))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = _interval(ra / rb, ba / bb, len(y))
        summary[dim] = {"a": ra, "b": rb, "diff": diff, "verdict": verdict(diff), "ratio": ratio,
                        "ratio_defined": bool(b_int["lo"] > 0)}
    return m.rename(columns={"rating_a": "rating"}).drop(columns="rating_b"), summary
