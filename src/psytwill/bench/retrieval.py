"""Cross-modal identification: find a stimulus's partner (image <-> caption) among all others.

Two sets keyed by the same stimulus id, e.g. shared1000 images (one row per
image) and their captions (several rows per image, collapsed to a
prototype). Both directions are scored.

- ``zero-shot`` ranks raw cosine between the two sides. It needs both sides
  in one trained space (EBind image vs EBind text, CLIP image vs CLIP text)
  and refuses unequal dimensions; equal dimensions do not prove a shared
  space, so the caller vouches for it.
- ``mapped`` fits a ridge map from one side to the other on training items,
  predicts the held-out items and identifies them among the held-out
  candidates. Every model can be scored this way, which makes it the
  comparable mode.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from psytwill.compare import DEFAULT_ALPHAS
from psytwill.exceptions import BenchError
from psytwill.store import SpaceMatrix

from .core import bootstrap_mean, fit_map, folds, identify, prototypes, similarity

MIN_ITEMS = 10


def retrieval(
    query: SpaceMatrix,
    target: SpaceMatrix,
    *,
    mode: str = "mapped",
    metric: str = "correlation",
    n_splits: int = 5,
    seed: int = 0,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    n_boot: int = 2000,
) -> tuple[pd.DataFrame, dict]:
    """Per-item identification rows for both directions, plus a summary with CIs."""
    q_ids, Q, q_n = prototypes(query)
    t_ids, T, t_n = prototypes(target)
    common = sorted(set(q_ids) & set(t_ids))
    if len(common) < MIN_ITEMS:
        raise BenchError(f"only {len(common)} stimulus ids are on both sides (need {MIN_ITEMS}); "
                         "are the two tables keyed on the same ids?")
    qi = {s: i for i, s in enumerate(q_ids)}
    ti = {s: i for i, s in enumerate(t_ids)}
    Q = Q[[qi[s] for s in common]]
    T = T[[ti[s] for s in common]]
    ids = np.array(common)
    n = len(ids)
    rows = []
    if mode == "zero-shot":
        if Q.shape[1] != T.shape[1]:
            raise BenchError(f"zero-shot needs one shared space, but the sides have {Q.shape[1]} and "
                             f"{T.shape[1]} dimensions; use --mode mapped")
        S = similarity(Q, T, "cosine")
        for direction, M in (("query_to_target", S), ("target_to_query", S.T)):
            r = identify(M, np.arange(n))
            r.insert(0, "direction", direction)
            r.insert(0, "stimulus_id", ids)
            r["fold"] = -1
            rows.append(r)
    elif mode == "mapped":
        for f, (tr, te) in enumerate(folds(n, n_splits, seed)):
            for direction, A, B in (("query_to_target", Q, T), ("target_to_query", T, Q)):
                lm = fit_map(A[tr], B[tr], alphas)
                S = similarity(lm.predict(A[te]), B[te], metric)
                r = identify(S, np.arange(len(te)))
                r.insert(0, "direction", direction)
                r.insert(0, "stimulus_id", ids[te])
                r["fold"] = f
                r["alpha"] = lm.alpha
                rows.append(r)
    else:
        raise BenchError(f"unknown mode {mode!r}; use zero-shot or mapped")
    items = pd.concat(rows, ignore_index=True)
    summary = {"n_items": n, "query_rows_per_id_max": int(q_n.max()), "target_rows_per_id_max": int(t_n.max()),
               "chance_pct_beaten": 0.5}
    for direction, d in items.groupby("direction"):
        summary[direction] = {
            "pct_beaten": bootstrap_mean(d["pct_beaten"], n_boot=n_boot, seed=seed),
            "top1": bootstrap_mean(d["top1"].astype(float), n_boot=n_boot, seed=seed),
            "n_candidates_mean": float(d["n_candidates"].mean()),
        }
    return items, summary
