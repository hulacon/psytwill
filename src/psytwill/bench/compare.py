"""Paired comparison of two arms on one task: the difference, its CI, and a verdict.

Two runs of the same task (two models, or one model in two modes) score the
same items. Comparing their separate CIs answers the wrong question and is
conservative; the CI that matters is the one on the per-item difference,
resampled over the task's unit. This module joins the two items files on the
items they share, refuses a pair whose scoring parameters differ, and
bootstraps the difference ``a - b``:

- ``retrieval``: per item, the two directions' scores averaged; resampled over
  stimuli when the runs scored within stimulus, else over items.
- ``oddoneout``: per triplet, correct (1/0); resampled over triplets.
- ``congruence``: per outcome, the within-subject Spearman of each arm,
  differenced per subject and averaged over subjects; pairs resampled within
  subject with the same draw for both arms, subjects fixed.

Verdict: ``win`` when the CI lies above 0, ``loss`` below, ``tie`` otherwise.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from psytwill.exceptions import BenchError

from .core import bootstrap_mean, stimulus_of

# Parameters two runs must share for their items to be comparable. Mode and
# metric may differ (a zero-shot arm against a mapped one is a legitimate
# comparison only if the panel says so; that is the caller's call), but the
# item set, folds and grouping may not.
MUST_MATCH = {
    "retrieval": ("n_splits", "seed", "within_stimulus", "target_segments", "target_window"),
    "oddoneout": ("triplets", "split", "only_ids"),
    "congruence": ("pairs", "n_splits", "seed"),
}


def verdict(ci: dict) -> str:
    if ci["lo"] > 0:
        return "win"
    if ci["hi"] < 0:
        return "loss"
    return "tie"


def check_params(task: str, pa: dict, pb: dict) -> None:
    if task not in MUST_MATCH:
        raise BenchError(f"no paired comparison is defined for task {task!r}")
    bad = [k for k in MUST_MATCH[task] if pa.get(k) != pb.get(k)]
    if bad:
        raise BenchError(f"the two {task} runs differ in {bad}: "
                         + "; ".join(f"{k}: {pa.get(k)!r} vs {pb.get(k)!r}" for k in bad))


def _joined(a: pd.DataFrame, b: pd.DataFrame, key: Sequence[str], cols: Sequence[str]) -> pd.DataFrame:
    for name, d in (("a", a), ("b", b)):
        if d.duplicated(list(key)).any():
            raise BenchError(f"items file {name} repeats a key {list(key)}; it cannot be joined item by item")
    m = a[[*key, *cols]].merge(b[[*key, *cols]], on=list(key), suffixes=("_a", "_b"), how="inner")
    if len(m) < max(len(a), len(b)):
        raise BenchError(f"only {len(m)} of {len(a)} / {len(b)} items are in both runs; a paired comparison "
                         "needs the same items on both sides")
    return m


def compare_retrieval(a: pd.DataFrame, b: pd.DataFrame, *, within_stimulus: bool, n_boot: int = 2000,
                      seed: int = 0) -> tuple[pd.DataFrame, dict]:
    def per_item(d):
        return d.assign(top1=d["top1"].astype(float)).groupby("stimulus_id", as_index=False)[
            ["pct_beaten", "top1"]].mean()

    m = _joined(per_item(a), per_item(b), ["stimulus_id"], ["pct_beaten", "top1"])
    unit = m["stimulus_id"].map(stimulus_of) if within_stimulus else None
    summary = {"n_items": int(len(m)), "ci_unit": "stimulus" if within_stimulus else "item"}
    for stat in ("pct_beaten", "top1"):
        m[f"{stat}_diff"] = m[f"{stat}_a"] - m[f"{stat}_b"]
        ci = bootstrap_mean(m[f"{stat}_diff"], unit, n_boot=n_boot, seed=seed)
        summary[stat] = {"a": float(m[f"{stat}_a"].mean()), "b": float(m[f"{stat}_b"].mean()), "diff": ci,
                         "verdict": verdict(ci)}
    return m, summary


def compare_oddoneout(a: pd.DataFrame, b: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0
                      ) -> tuple[pd.DataFrame, dict]:
    cols = ["item1", "item2", "item3", "odd"]
    if len(a) != len(b) or not (a[cols].astype(str).values == b[cols].astype(str).values).all():
        raise BenchError("the two oddoneout items files do not list the same triplets in the same order")
    m = a[cols].copy()
    m["correct_a"] = a["correct"].astype(float).to_numpy()
    m["correct_b"] = b["correct"].astype(float).to_numpy()
    m["correct_diff"] = m["correct_a"] - m["correct_b"]
    ci = bootstrap_mean(m["correct_diff"], n_boot=n_boot, seed=seed)
    return m, {"n_items": int(len(m)), "ci_unit": "triplet",
               "accuracy": {"a": float(m["correct_a"].mean()), "b": float(m["correct_b"].mean()), "diff": ci,
                            "verdict": verdict(ci)}}


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def compare_congruence(a: pd.DataFrame, b: pd.DataFrame, outcomes: Sequence[str], *, n_boot: int = 2000,
                       seed: int = 0) -> tuple[pd.DataFrame, dict]:
    key = ["subject", "image_id", "word_id"]
    a = a.astype({k: str for k in key})
    b = b.astype({k: str for k in key})
    missing = [o for o in outcomes if o not in a.columns]
    if missing:
        raise BenchError(f"outcome column(s) {missing} are not in the items file")
    m = _joined(a[[*key, "congruence", *outcomes]], b[[*key, "congruence"]], key, ["congruence"])
    m = m.merge(a[[*key, *outcomes]], on=key, how="left")
    rng = np.random.default_rng(seed)
    summary: dict = {"n_items": int(len(m)), "ci_unit": "pairs within subject; subjects fixed"}
    for oc in outcomes:
        per_sub, boots = {}, []
        for s, d in m.groupby("subject"):
            d = d[d[oc].notna()]
            xa, xb, y = d["congruence_a"].to_numpy(), d["congruence_b"].to_numpy(), d[oc].to_numpy(dtype=float)
            ra, rb = _rho(xa, y), _rho(xb, y)
            per_sub[s] = {"rho_a": ra, "rho_b": rb, "diff": ra - rb, "n": int(len(y))}
            draws = rng.integers(0, len(y), size=(n_boot, len(y))) if len(y) >= 3 else None
            boots.append(np.array([_rho(xa[k], y[k]) - _rho(xb[k], y[k]) for k in draws]) if draws is not None
                         else np.full(n_boot, np.nan))
        mean_b = np.nanmean(np.vstack(boots), axis=0)
        diff = float(np.nanmean([v["diff"] for v in per_sub.values()]))
        ci = {"mean": diff, "lo": float(np.nanquantile(mean_b, .025)), "hi": float(np.nanquantile(mean_b, .975)),
              "n": int(len(m[m[oc].notna()])), "n_units": len(per_sub)}
        summary[oc] = {"a": float(np.nanmean([v["rho_a"] for v in per_sub.values()])),
                       "b": float(np.nanmean([v["rho_b"] for v in per_sub.values()])),
                       "diff": ci, "verdict": verdict(ci), "per_subject": per_sub}
    return m, summary
