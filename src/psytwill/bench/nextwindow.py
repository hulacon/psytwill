"""Film next-window identification: which of this film's windows comes ``horizon`` bins after this one?

Rows are one film's windows on the store's temporal grid (key
``stimulus_id|time``, binned by the loader's ``window``). For a source
window at index j the target is j + horizon; candidates are the target plus
every window of the same film more than ``gap`` bins from the target (which
also removes the source whenever ``gap >= horizon``). Two predictors:

- ``persistence``: the source window itself. Content changes slowly, so
  this is a strong floor, and the slowness differs by space — whitening a
  space changes which of its directions are slow. Read a model's
  ``ridge`` score against its own ``persistence`` score.
- ``ridge``: a map from window j to window j + horizon, fit on other films
  (grouped folds, so no film is on both sides) and applied to this one.
  Shrinkage pulls its predictions toward the mean window, so it can score
  *below* persistence even when it has learned something.
- ``delta``: persistence plus a ridge map of the change (j -> j + horizon
  minus j), same folds. With no learnable change it falls back to
  persistence, so ``delta - persistence`` isolates what is predictable
  beyond slowness.

Bootstrap CIs resample films, not windows: a film's windows are not
independent.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from psytwill.compare import DEFAULT_ALPHAS
from psytwill.exceptions import BenchError
from psytwill.store import SpaceMatrix

from .core import bootstrap_mean, fit_map, folds, identify, similarity


def _films(emb: SpaceMatrix) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    by: dict[str, list[tuple[float, int]]] = {}
    for i, lab in enumerate(emb.labels):
        parts = lab.split("|")
        if len(parts) < 2:
            raise BenchError(f"row label {lab!r} has no time; load with key stimulus_id,time and a window")
        by.setdefault(parts[0], []).append((float(parts[1]), i))
    out = {}
    for film, rows in by.items():
        rows.sort()
        out[film] = (np.array([t for t, _ in rows]), emb.X[[i for _, i in rows]])
    return out


def next_window(
    emb: SpaceMatrix,
    *,
    horizon: int = 1,
    gap: int | None = None,
    predictors: Sequence[str] = ("persistence", "ridge", "delta"),
    metric: str = "correlation",
    n_splits: int = 5,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    if horizon < 1:
        raise BenchError("horizon must be at least 1 bin")
    gap = horizon if gap is None else gap
    if gap < horizon:
        raise BenchError(f"gap {gap} < horizon {horizon} would leave the source window among the "
                         "candidates, which persistence then always picks")
    films = _films(emb)
    names = sorted(films)
    # source/target pairs per film
    pairs = {f: (films[f][1][:-horizon], films[f][1][horizon:]) for f in names if len(films[f][0]) > horizon}
    if len(pairs) < 2:
        raise BenchError(f"{len(pairs)} film(s) are longer than the horizon; need at least 2")
    preds: dict[str, dict[str, np.ndarray]] = {p: {} for p in predictors}
    unknown = set(predictors) - {"persistence", "ridge", "delta"}
    if unknown:
        raise BenchError(f"unknown predictor(s) {sorted(unknown)}; use persistence, ridge, delta")
    alpha_of: dict[str, dict[str, float]] = {"ridge": {}, "delta": {}}
    if "persistence" in predictors:
        preds["persistence"] = {f: pairs[f][0] for f in pairs}
    fl = list(pairs)
    for p in ("ridge", "delta"):
        if p not in predictors:
            continue
        for tr, te in folds(np.arange(len(fl)), min(n_splits, len(fl)), seed):
            Xtr = np.vstack([pairs[fl[i]][0] for i in tr])
            Ytr = np.vstack([pairs[fl[i]][1] for i in tr])
            lm = fit_map(Xtr, Ytr - Xtr if p == "delta" else Ytr, alphas)
            for i in te:
                src = pairs[fl[i]][0]
                preds[p][fl[i]] = src + lm.predict(src) if p == "delta" else lm.predict(src)
                alpha_of[p][fl[i]] = lm.alpha
    rows = []
    for f in pairs:
        times, X = films[f]
        n_src = len(X) - horizon
        tgt = np.arange(n_src) + horizon
        allowed = np.abs(np.arange(len(X))[None, :] - tgt[:, None]) > gap
        for p in predictors:
            S = similarity(preds[p][f], X, metric)
            r = identify(S, tgt, allowed)
            r.insert(0, "predictor", p)
            r.insert(0, "target_time", times[tgt])
            r.insert(0, "time", times[:n_src])
            r.insert(0, "stimulus_id", f)
            if p in alpha_of:
                r["alpha"] = alpha_of[p][f]
            rows.append(r)
    items = pd.concat(rows, ignore_index=True)
    items = items[items["n_candidates"] > 1].reset_index(drop=True)
    summary: dict = {"n_films": len(pairs), "horizon_bins": horizon, "gap_bins": gap, "metric": metric,
                     "chance_pct_beaten": 0.5}
    for p, d in items.groupby("predictor"):
        summary[p] = {"pct_beaten": bootstrap_mean(d["pct_beaten"], d["stimulus_id"], n_boot=n_boot, seed=seed),
                      "top1": bootstrap_mean(d["top1"].astype(float), d["stimulus_id"], n_boot=n_boot, seed=seed),
                      "n_candidates_mean": float(d["n_candidates"].mean())}
    if "persistence" in predictors:
        w = items.pivot_table(index=["stimulus_id", "time"], columns="predictor", values="pct_beaten")
        for p in ("ridge", "delta"):
            if p in predictors:
                diff = (w[p] - w["persistence"]).reset_index(name="d")
                summary[f"{p}_minus_persistence"] = bootstrap_mean(diff["d"], diff["stimulus_id"], n_boot=n_boot,
                                                                   seed=seed)
    return items, summary
