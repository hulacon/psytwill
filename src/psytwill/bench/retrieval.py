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

``within_stimulus`` scores items that are parts of a stimulus (a film's
annotated segments, labels ``film|chunk_idx``) against the other parts of
the same stimulus only, so the task is which moment, not which film. Rows
are then items as labelled (no prototype collapse), folds are grouped by
stimulus so a film is never on both sides of a map, and every CI resamples
stimuli. In ``mapped`` mode each side is also centred within its stimulus
first (``center_within``, default on): a film's own offset in each space is
irrelevant to which of its moments is which, does not map across films, and
otherwise dominates the ridge fit. Centring uses no pairing, since all of a
film's candidates share the mean. ``zero-shot`` stays raw cosine.

Both directions score the same items, so the summary also carries
``both_directions``: each item's two scores averaged, then bootstrapped.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from psytwill.compare import DEFAULT_ALPHAS
from psytwill.exceptions import BenchError
from psytwill.store import SpaceMatrix

from .core import bootstrap_mean, fit_map, folds, identify, prototypes, similarity, stimulus_of

MIN_ITEMS = 10


def _same(groups_a: np.ndarray, groups_b: np.ndarray) -> np.ndarray:
    return groups_a[:, None] == groups_b[None, :]


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
    within_stimulus: bool = False,
    center_within: bool | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Per-item identification rows for both directions, plus a summary with CIs."""
    id_of = (lambda lab: lab) if within_stimulus else stimulus_of
    q_ids, Q, q_n = prototypes(query, id_of)
    t_ids, T, t_n = prototypes(target, id_of)
    common = sorted(set(q_ids) & set(t_ids))
    if len(common) < MIN_ITEMS:
        raise BenchError(f"only {len(common)} item ids are on both sides (need {MIN_ITEMS}); "
                         "are the two tables keyed on the same ids?")
    qi = {s: i for i, s in enumerate(q_ids)}
    ti = {s: i for i, s in enumerate(t_ids)}
    Q = Q[[qi[s] for s in common]]
    T = T[[ti[s] for s in common]]
    ids = np.array(common)
    n = len(ids)
    groups = np.array([stimulus_of(s) for s in ids]) if within_stimulus else None
    if within_stimulus:
        sizes = pd.Series(groups).value_counts()
        if (sizes < 2).any():
            raise BenchError(f"{int((sizes < 2).sum())} stimulus(es) have a single item, so nothing to "
                             f"identify it against (e.g. {list(sizes[sizes < 2].index[:3])})")
    if center_within is None:
        center_within = within_stimulus and mode == "mapped"
    if center_within:
        if not within_stimulus or mode != "mapped":
            raise BenchError("center_within applies to within-stimulus mapped scoring only")
        Q, T = Q.copy(), T.copy()
        for g in np.unique(groups):
            m = groups == g
            Q[m] -= Q[m].mean(0)
            T[m] -= T[m].mean(0)
    rows = []
    if mode == "zero-shot":
        if Q.shape[1] != T.shape[1]:
            raise BenchError(f"zero-shot needs one shared space, but the sides have {Q.shape[1]} and "
                             f"{T.shape[1]} dimensions; use --mode mapped")
        S = similarity(Q, T, "cosine")
        allowed = _same(groups, groups) if within_stimulus else None
        for direction, M in (("query_to_target", S), ("target_to_query", S.T)):
            r = identify(M, np.arange(n), allowed)
            r.insert(0, "direction", direction)
            r.insert(0, "stimulus_id", ids)
            r["fold"] = -1
            rows.append(r)
    elif mode == "mapped":
        for f, (tr, te) in enumerate(folds(groups if within_stimulus else n, n_splits, seed)):
            allowed = _same(groups[te], groups[te]) if within_stimulus else None
            for direction, A, B in (("query_to_target", Q, T), ("target_to_query", T, Q)):
                lm = fit_map(A[tr], B[tr], alphas)
                S = similarity(lm.predict(A[te]), B[te], metric)
                r = identify(S, np.arange(len(te)), allowed)
                r.insert(0, "direction", direction)
                r.insert(0, "stimulus_id", ids[te])
                r["fold"] = f
                r["alpha"] = lm.alpha
                rows.append(r)
    else:
        raise BenchError(f"unknown mode {mode!r}; use zero-shot or mapped")
    items = pd.concat(rows, ignore_index=True)
    unit = (lambda d: d["stimulus_id"].map(stimulus_of)) if within_stimulus else (lambda d: None)
    summary = {"n_items": n, "query_rows_per_id_max": int(q_n.max()), "target_rows_per_id_max": int(t_n.max()),
               "chance_pct_beaten": 0.5, "within_stimulus": within_stimulus,
               "ci_unit": "stimulus" if within_stimulus else "item", "center_within": center_within}
    if within_stimulus:
        summary["n_stimuli"] = int(len(np.unique(groups)))
    for direction, d in items.groupby("direction"):
        summary[direction] = {
            "pct_beaten": bootstrap_mean(d["pct_beaten"], unit(d), n_boot=n_boot, seed=seed),
            "top1": bootstrap_mean(d["top1"].astype(float), unit(d), n_boot=n_boot, seed=seed),
            "n_candidates_mean": float(d["n_candidates"].mean()),
        }
    both = items.assign(top1=items["top1"].astype(float)).groupby("stimulus_id", as_index=False)[
        ["pct_beaten", "top1"]].mean()
    summary["both_directions"] = {
        "pct_beaten": bootstrap_mean(both["pct_beaten"], unit(both), n_boot=n_boot, seed=seed),
        "top1": bootstrap_mean(both["top1"], unit(both), n_boot=n_boot, seed=seed),
    }
    return items, summary
