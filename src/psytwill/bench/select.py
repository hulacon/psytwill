"""The best-single-member arm (B2), chosen without looking at the items it is scored on.

Every battery member is scored on a task as its own run. Picking the member
with the best score and reporting that score would let the choice ride on the
noise of the very items it is reported on. Here the items are split into
outer folds over the task's CI unit (films, images, triplets); for each fold
the member with the best mean statistic on the OTHER folds is chosen, and
that member's rows for the fold's items become B2's rows. Every item is
therefore scored by a member chosen on data that excludes it, and B2 covers
the same items as every other arm, so ``bench compare`` pairs it with them
unchanged.

A member that lacks some of the task's items (a declared null on them) is not
a candidate: B2 must score every item, and filling a member's gap with
another member's score would make a hybrid nobody chose.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from psytwill.exceptions import BenchError

from .core import bootstrap_mean, stimulus_of


def _retrieval_item_scores(items: pd.DataFrame, statistic: str) -> pd.Series:
    d = items.assign(top1=items["top1"].astype(float))
    return d.groupby("stimulus_id")[statistic].mean()


def select_best(runs: dict[str, pd.DataFrame], task: str, *, statistic: str, within_stimulus: bool = False,
                n_splits: int = 5, seed: int = 0, n_boot: int = 2000) -> tuple[pd.DataFrame, dict]:
    """B2's items (the chosen member's rows, fold by fold) and the per-fold choices."""
    if task == "retrieval":
        scores = {m: _retrieval_item_scores(d, statistic) for m, d in runs.items()}
        item_col = "stimulus_id"
    elif task == "oddoneout":
        if statistic != "correct":
            raise BenchError("oddoneout's statistic is `correct`")
        key = ["item1", "item2", "item3", "odd"]
        ref = next(iter(runs.values()))[key].astype(str)
        for m, d in runs.items():
            if len(d) != len(ref) or not (d[key].astype(str).values == ref.values).all():
                raise BenchError(f"member {m!r} did not score the same triplets in the same order")
        scores = {m: d["correct"].astype(float).reset_index(drop=True) for m, d in runs.items()}
        item_col = None
    else:
        raise BenchError(f"no best-member selection is defined for task {task!r}")

    universe = sorted(set().union(*(set(s.index) for s in scores.values())))
    complete = {m: s for m, s in scores.items() if len(s) == len(universe) and s.notna().all()}
    excluded = sorted(set(scores) - set(complete))
    if not complete:
        raise BenchError("no member scored every item, so none can stand for B2")

    items = np.array(universe)
    units = (np.array([stimulus_of(str(i)) for i in items]) if within_stimulus else items.astype(str))
    uniq = np.unique(units)
    if len(uniq) < n_splits:
        raise BenchError(f"{len(uniq)} units cannot fill {n_splits} outer folds")
    rng = np.random.default_rng(seed)
    fold_of_unit = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % n_splits))
    fold = np.array([fold_of_unit[u] for u in units])
    S = pd.DataFrame({m: s.reindex(items).to_numpy() for m, s in complete.items()}, index=items)

    picks, rows = [], []
    for f in range(n_splits):
        out = fold != f
        means = S[out].mean()
        best = str(means.idxmax())
        picks.append({"fold": f, "member": best, "dev_mean": float(means[best]),
                      "runner_up": str(means.drop(best).idxmax()) if len(means) > 1 else None,
                      "n_items": int((~out).sum())})
        held = set(items[~out]) if item_col else set(np.flatnonzero(~out))
        d = runs[best]
        part = d[d[item_col].isin(held)] if item_col else d.iloc[sorted(held)]
        rows.append(part.assign(b2_member=best, b2_fold=f))
    out_items = pd.concat(rows).sort_index(kind="stable").reset_index(drop=True)
    # the same score summary a run of the task carries, so B2 reads like any other arm
    if item_col:
        unit = (lambda d: d["stimulus_id"].map(stimulus_of)) if within_stimulus else (lambda d: None)
        scored: dict = {"n_items": int(out_items["stimulus_id"].nunique()), "chance_pct_beaten": 0.5,
                        "within_stimulus": within_stimulus, "ci_unit": "stimulus" if within_stimulus else "item"}
        for direction, d in out_items.groupby("direction"):
            scored[direction] = {
                "pct_beaten": bootstrap_mean(d["pct_beaten"], unit(d), n_boot=n_boot, seed=seed),
                "top1": bootstrap_mean(d["top1"].astype(float), unit(d), n_boot=n_boot, seed=seed),
                "n_candidates_mean": float(d["n_candidates"].mean())}
        both = out_items.assign(top1=out_items["top1"].astype(float)).groupby("stimulus_id", as_index=False)[
            ["pct_beaten", "top1"]].mean()
        scored["both_directions"] = {st: bootstrap_mean(both[st], unit(both), n_boot=n_boot, seed=seed)
                                     for st in ("pct_beaten", "top1")}
    else:
        scored = {"n_triplets": int(len(out_items)), "chance": 1 / 3,
                  "accuracy": bootstrap_mean(out_items["correct"].astype(float), n_boot=n_boot, seed=seed)}
    summary = {**scored, "statistic": statistic, "n_splits": n_splits, "seed": seed,
               "unit": "stimulus" if within_stimulus else ("item" if item_col else "triplet"),
               "candidates": sorted(complete), "not_candidates_incomplete": excluded, "folds": picks,
               "members_chosen": sorted({p["member"] for p in picks})}
    return out_items, summary
