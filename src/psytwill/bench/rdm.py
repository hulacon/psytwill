"""Representational similarity: a model's item-by-item distances against human dissimilarity judgements.

Input is a pairs table with ``participant, image_a, image_b, dissimilarity``:
one row per judged pair per participant (a multiple-arrangement RDM in long
form). The model's dissimilarity for a pair is ``1 - similarity`` of the two
items' prototypes. The task score is the mean over participants of
Spearman(model, participant), with two noise-ceiling bounds from the
judgements alone:

- ``lower``: each participant against the mean of the others (leave one out)
- ``upper``: each participant against the mean of all participants

The CI resamples **items**, not pairs: each draw takes the item set with
replacement and rebuilds every RDM from the drawn items, dropping the pairs
of an item with its own duplicate. Pairs share items, so resampling pairs
would treat 4,950 pairs among 100 items as independent and give a CI that is
too narrow. Participants are fixed; the per-participant values are reported
beside the mean.

Draws depend only on ``seed`` and the sorted item set, so two runs over the
same pairs table share them, and ``compare_rdm`` recomputes the paired
difference on those same draws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from psytwill.exceptions import BenchError
from psytwill.store import SpaceMatrix

from .core import prototypes, similarity

PAIR_COLUMNS = ("participant", "image_a", "image_b", "dissimilarity")


def _canonical(pairs: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PAIR_COLUMNS if c not in pairs.columns]
    if missing:
        raise BenchError(f"pairs table lacks column(s) {missing}; expected {list(PAIR_COLUMNS)}")
    p = pairs[list(PAIR_COLUMNS)].astype({"participant": str, "image_a": str, "image_b": str}).copy()
    if (p["image_a"] == p["image_b"]).any():
        raise BenchError(f"{int((p['image_a'] == p['image_b']).sum())} row(s) pair an item with itself")
    x, y = p["image_a"].to_numpy(dtype=str), p["image_b"].to_numpy(dtype=str)
    p["image_a"], p["image_b"] = np.where(x < y, x, y), np.where(x < y, y, x)
    dup = p.duplicated(["participant", "image_a", "image_b"])
    if dup.any():
        raise BenchError(f"{int(dup.sum())} pair(s) are judged twice by one participant (in either order); "
                         "average them upstream, or say which judgement counts")
    return p.reset_index(drop=True)


def _squares(p: pd.DataFrame, items: list[str], col: str) -> tuple[list[str], np.ndarray]:
    """One (n, n) matrix per participant, NaN where a pair was not judged."""
    pos = {s: i for i, s in enumerate(items)}
    parts = sorted(p["participant"].unique())
    D = np.full((len(parts), len(items), len(items)), np.nan)
    for k, (_, d) in enumerate(p.groupby("participant", sort=True)):
        i = d["image_a"].map(pos).to_numpy()
        j = d["image_b"].map(pos).to_numpy()
        v = d[col].to_numpy(dtype=float)
        D[k, i, j] = v
        D[k, j, i] = v
    return parts, D


def _model_square(p: pd.DataFrame, items: list[str], col: str) -> np.ndarray:
    """The model's (n, n) matrix from an items table; its value for a pair is the same for every participant."""
    u = p.drop_duplicates(["image_a", "image_b", col])
    if u.duplicated(["image_a", "image_b"]).any():
        raise BenchError(f"{col} differs between participants for the same pair; the items file is not one model's")
    pos = {s: i for i, s in enumerate(items)}
    M = np.full((len(items), len(items)), np.nan)
    i, j = u["image_a"].map(pos).to_numpy(), u["image_b"].map(pos).to_numpy()
    M[i, j] = M[j, i] = u[col].to_numpy(dtype=float)
    np.fill_diagonal(M, 0.0)
    return M


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def item_draws(n_items: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_items) item indices; shared by every run over the same item set and seed."""
    return np.random.default_rng(seed).integers(0, n_items, size=(n_boot, n_items))


def _pair_index(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upper-triangle pairs of a drawn item list, without pairs of an item with its duplicate."""
    ii, jj = np.triu_indices(len(idx), k=1)
    a, b = idx[ii], idx[jj]
    keep = a != b
    return a[keep], b[keep]


def _scores(models: list[np.ndarray], H: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """Per-participant Spearman of each model, and both ceiling bounds, on one set of pairs."""
    n_p = H.shape[0]
    h = H[:, a, b]  # (participants, pairs)
    total = np.nansum(H[:, a, b], axis=0)
    count = np.sum(~np.isnan(h), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_all = total / count
    out = {"model": [np.array([_rank_corr(M[a, b], h[k]) for k in range(n_p)]) for M in models]}
    lower, upper = np.empty(n_p), np.empty(n_p)
    for k in range(n_p):
        hk = np.nan_to_num(h[k])
        own = (~np.isnan(h[k])).astype(int)
        with np.errstate(invalid="ignore", divide="ignore"):
            others = (total - hk) / (count - own)
        lower[k] = _rank_corr(others, h[k])
        upper[k] = _rank_corr(mean_all, h[k])
    out["lower"], out["upper"] = lower, upper
    return out


def _interval(point: float, boots: np.ndarray, n_units: int) -> dict:
    b = boots[~np.isnan(boots)]
    return {"mean": float(point), "lo": float(np.quantile(b, .025)) if b.size else float("nan"),
            "hi": float(np.quantile(b, .975)) if b.size else float("nan"), "n": int(b.size), "n_units": n_units}


def _model_matrix(emb: SpaceMatrix, items: list[str], metric: str) -> np.ndarray:
    ids, P, _ = prototypes(emb)
    pos = {s: i for i, s in enumerate(ids)}
    absent = [s for s in items if s not in pos]
    if absent:
        raise BenchError(f"{len(absent)} judged item(s) have no embedding (e.g. {absent[:3]}); embed every item, "
                         "or filter the pairs first")
    X = P[[pos[s] for s in items]]
    return 1.0 - similarity(X, X, metric)


def rdm(emb: SpaceMatrix, pairs: pd.DataFrame, *, metric: str = "cosine", n_boot: int = 2000,
        seed: int = 0) -> tuple[pd.DataFrame, dict]:
    p = _canonical(pairs)
    items = sorted(set(p["image_a"]) | set(p["image_b"]))
    M = _model_matrix(emb, items, metric)
    pos = {s: i for i, s in enumerate(items)}
    p["model_dissimilarity"] = M[p["image_a"].map(pos).to_numpy(), p["image_b"].map(pos).to_numpy()]
    return p, summarize(p, n_boot=n_boot, seed=seed) | {"metric": metric}


def summarize(items: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """The run's summary, recomputed from its items file alone."""
    if "model_dissimilarity" not in items.columns:
        raise BenchError("items lack model_dissimilarity; summarize reads an rdm run's items file")
    p = items.astype({"participant": str, "image_a": str, "image_b": str})
    item_ids = sorted(set(p["image_a"].astype(str)) | set(p["image_b"].astype(str)))
    parts, H = _squares(p, item_ids, "dissimilarity")
    M = _model_square(p, item_ids, "model_dissimilarity")
    n = len(item_ids)
    full_a, full_b = np.triu_indices(n, k=1)
    point = _scores([M], H, full_a, full_b)
    draws = item_draws(n, n_boot, seed)
    boot = [_scores([M], H, *_pair_index(idx)) for idx in draws]
    rho_b = np.array([np.nanmean(s["model"][0]) for s in boot])
    low_b = np.array([np.nanmean(s["lower"]) for s in boot])
    up_b = np.array([np.nanmean(s["upper"]) for s in boot])
    rho = float(np.nanmean(point["model"][0]))
    low, up = float(np.nanmean(point["lower"])), float(np.nanmean(point["upper"]))
    per = {}
    for k, pid in enumerate(parts):
        rk = np.array([s["model"][0][k] for s in boot])
        per[pid] = {"rho": float(point["model"][0][k]), "lo": float(np.nanquantile(rk, .025)),
                    "hi": float(np.nanquantile(rk, .975)), "ceiling_lower": float(point["lower"][k]),
                    "ceiling_upper": float(point["upper"][k])}
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "n_items": n, "n_participants": len(parts), "n_pairs": int(len(p)), "n_boot": n_boot, "seed": seed,
            "statistic": "mean over participants of Spearman(model dissimilarity, participant dissimilarity)",
            "ci": "items resampled (RDMs rebuilt, duplicate self-pairs dropped); participants fixed",
            "rho": _interval(rho, rho_b, n),
            "ceiling_lower": _interval(low, low_b, n),
            "ceiling_upper": _interval(up, up_b, n),
            "share_of_ceiling_lower": _interval(rho / low, rho_b / low_b, n),
            "share_of_ceiling_upper": _interval(rho / up, rho_b / up_b, n),
            "per_participant": per,
        }


def compare_rdm(a: pd.DataFrame, b: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0
                ) -> tuple[pd.DataFrame, dict]:
    """Paired a - b on the mean participant Spearman, and the ratio a / b, on shared item draws.

    The ratio is retention when b is the reference and chance is 0. It is
    reported as undefined (``ratio_defined: false``) when b's own interval
    reaches 0, where a ratio's interval is meaningless.
    """
    key = ["participant", "image_a", "image_b"]
    a = a.astype({k: str for k in key})
    b = b.astype({k: str for k in key})
    m = a[[*key, "dissimilarity", "model_dissimilarity"]].merge(
        b[[*key, "dissimilarity", "model_dissimilarity"]], on=key, suffixes=("_a", "_b"), how="inner")
    if len(m) != len(a) or len(m) != len(b):
        raise BenchError(f"only {len(m)} of {len(a)} / {len(b)} judged pairs are in both runs; a paired "
                         "comparison needs the same pairs on both sides")
    if not np.allclose(m["dissimilarity_a"], m["dissimilarity_b"], equal_nan=True):
        raise BenchError("the two runs carry different human dissimilarities for the same pairs; they were "
                         "scored against different judgements")
    item_ids = sorted(set(m["image_a"]) | set(m["image_b"]))
    parts, H = _squares(m, item_ids, "dissimilarity_a")
    Ma = _model_square(m, item_ids, "model_dissimilarity_a")
    Mb = _model_square(m, item_ids, "model_dissimilarity_b")
    n = len(item_ids)
    fa, fb = np.triu_indices(n, k=1)
    pt = _scores([Ma, Mb], H, fa, fb)
    ra, rb = float(np.nanmean(pt["model"][0])), float(np.nanmean(pt["model"][1]))
    boot = [_scores([Ma, Mb], H, *_pair_index(idx)) for idx in item_draws(n, n_boot, seed)]
    ba = np.array([np.nanmean(s["model"][0]) for s in boot])
    bb = np.array([np.nanmean(s["model"][1]) for s in boot])
    b_int = _interval(rb, bb, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = _interval(ra / rb, ba / bb, n)
    diff = _interval(ra - rb, ba - bb, n)
    from .compare import verdict

    m = m.rename(columns={"dissimilarity_a": "dissimilarity"}).drop(columns="dissimilarity_b")
    return m, {"n_items": n, "n_pairs": int(len(m)), "n_participants": len(parts),
               "ci_unit": "items (RDMs rebuilt per draw); participants fixed",
               "rho": {"a": ra, "b": rb, "a_interval": _interval(ra, ba, n), "b_interval": b_int, "diff": diff,
                       "verdict": verdict(diff), "ratio": ratio, "ratio_defined": bool(b_int["lo"] > 0)}}
