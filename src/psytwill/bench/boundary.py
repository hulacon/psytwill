"""Event boundaries: does a film's trajectory move more at annotated boundaries than at matched non-boundaries?

Inputs, all keyed by ``stimulus_id``:

- a time grid (labels ``stimulus|time``, one row per bin of width ``window``);
- ``boundaries``: ``stimulus_id, level, onset, annotator``. A boundary is a
  segment onset > 0 at the scored level; every listed level's onsets are kept
  away from the controls;
- ``cuts``: ``stimulus_id, time``, the shot cuts to match on;
- ``durations``: ``stimulus_id, duration``. A film's grid is bins
  ``0 .. floor(duration / window) - 1``; rows past it are dropped and bins
  with no row are NaN, so a film can be scored on a declared span only.

Per-bin statistics at bin i, from the pair (i - 1, i), never across a film's
start or a NaN row (a row is placed whole or not at all):

- ``change``: cosine distance between consecutive rows, after centring every
  row on the mean over all films' rows;
- ``pe_sfa`` (given a kernel ``W`` (d, k), ``lam`` (k,)):
  ``sum_k (s_k,t - rho_k s_k,t-1)^2`` with ``s = W' x``, x centred per film, and
  ``rho = 1 - lam / 2``, the AR(1) a slow-feature kernel implies.

A boundary's value is the mean of its statistic over the bins whose start
lies within ``tol`` (1 s) of it. Its controls are same-film bins more than
``exclude`` (2 s) from every boundary at every level, whose window lies
inside the film and whose distance to the nearest cut is within a caliper of
the boundary's own (.25 s, widened to .5 s below 3 controls; otherwise the
boundary is dropped). ``delta = window(boundary) - mean(window(controls))``;
``delta_std`` divides by the SD of the film's control windows (film-SD units).

Film effect: each film's mean delta. The statistic is the mean of the
annotators' means (films nest in annotators: one annotator per film). It is
tested by an annotator-stratified sign flip (one-sided) and given an
annotator-stratified film bootstrap CI (films resampled within annotator).
Both are reported in raw and film-SD units, and again for boundaries within
and beyond ``tol`` of a cut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from psytwill.exceptions import BenchError
from psytwill.store import SpaceMatrix

from .core import stimulus_of

CALIPERS = (0.25, 0.5)
MIN_POOL = 3


def nearest(t, ref) -> np.ndarray:
    t = np.atleast_1d(np.asarray(t, float))
    if len(ref) == 0:
        return np.full(len(t), np.inf)
    ref = np.sort(np.asarray(ref, float))
    p = np.searchsorted(ref, t)
    lo = np.abs(t - ref[np.clip(p - 1, 0, len(ref) - 1)])
    hi = np.abs(ref[np.clip(p, 0, len(ref) - 1)] - t)
    return np.minimum(lo, hi)


def film_grids(sm: SpaceMatrix, durations: dict[str, float], window: float) -> dict[str, np.ndarray]:
    """stimulus -> (n_bins, d) rows on the full grid, NaN where absent; whole rows only."""
    sid = np.array([stimulus_of(lab) for lab in sm.labels])
    t = np.array([float(lab.split("|")[1]) for lab in sm.labels])
    out = {}
    missing = sorted(set(sid) - set(durations))
    if missing:
        raise BenchError(f"{len(missing)} film(s) have no duration (e.g. {missing[:3]})")
    for s in np.unique(sid):
        m = sid == s
        b = np.floor(t[m] / window + 1e-6).astype(int)
        n = int(np.floor(durations[s] / window + 1e-6))
        X = np.full((n, sm.dim), np.nan)
        keep = (b >= 0) & (b < n)
        X[b[keep]] = sm.X[m][keep]
        X[np.isnan(X).any(axis=1)] = np.nan
        out[s] = X
    return out


def change_series(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1)
    out = np.full(len(X), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = 1.0 - np.einsum("ij,ij->i", X[:-1], X[1:]) / (norm[:-1] * norm[1:])
    out[~np.isfinite(out)] = np.nan
    return out


def per_bin_stats(grids: dict[str, np.ndarray], kernel: tuple[np.ndarray, np.ndarray] | None = None
                  ) -> dict[str, dict[str, np.ndarray]]:
    mu = np.nanmean(np.vstack(list(grids.values())), axis=0)
    out: dict[str, dict[str, np.ndarray]] = {"change": {}}
    if kernel is not None:
        W, lam = kernel
        d = next(iter(grids.values())).shape[1]
        if W.shape[0] != d or W.shape[1] != len(lam):
            raise BenchError(f"kernel W is {W.shape} with {len(lam)} lambdas; the grid has {d} columns")
        rho = 1 - lam / 2
        out["pe_sfa"] = {}
    for s, X in grids.items():
        out["change"][s] = change_series(X - mu)
        if kernel is not None:
            S = (X - np.nanmean(X, axis=0)) @ W
            pe = np.full(len(X), np.nan)
            pe[1:] = np.sum((S[1:] - rho * S[:-1]) ** 2, axis=1)
            out["pe_sfa"][s] = pe
    return out


def _window_series(v: np.ndarray, half: int) -> np.ndarray:
    f = np.isfinite(v)
    k = np.ones(2 * half + 1)
    s = np.convolve(np.where(f, v, 0.0), k, mode="same")
    c = np.convolve(f.astype(float), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def _window_at(v: np.ndarray, t: float, window: float, tol: float) -> float:
    sel = np.abs(np.arange(len(v)) * window - t) <= tol + 1e-9
    x = v[sel]
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def match(series: dict[str, np.ndarray], bounds: dict[str, np.ndarray], all_bounds: dict[str, np.ndarray],
          cuts: dict[str, np.ndarray], annotator: dict[str, str], *, window: float, tol: float,
          exclude: float) -> pd.DataFrame:
    rows = []
    half = int(round(tol / window))
    for s in sorted(bounds):
        v = series.get(s)
        b = bounds[s]
        if v is None or len(b) == 0:
            continue
        win = _window_series(v, half)
        j = np.arange(half, len(v) - half)
        tj = j * window
        ok = (nearest(tj, all_bounds.get(s, [])) > exclude) & np.isfinite(win[j])
        cj, ct = j[ok], tj[ok]
        cut = cuts.get(s, np.array([]))
        dc = nearest(ct, cut)
        sd = float(np.nanstd(win[cj])) if len(cj) > 1 else np.nan
        for bt in b:
            wb = _window_at(v, bt, window, tol)
            db = float(nearest(bt, cut)[0])
            pool, cal = np.array([], int), None
            for c in CALIPERS:
                pool, cal = cj[np.abs(dc - db) <= c], c
                if len(pool) >= MIN_POOL:
                    break
            drop = len(pool) == 0 or not np.isfinite(wb)
            delta = np.nan if drop else wb - float(np.mean(win[pool]))
            rows.append({"stimulus_id": s, "annotator": annotator[s], "time": float(bt), "d_cut": db,
                         "n_pool": int(len(pool)), "caliper": cal, "dropped": bool(drop), "window_value": wb,
                         "delta": delta,
                         "delta_std": delta / sd if sd and np.isfinite(sd) else np.nan})
    return pd.DataFrame(rows)


def film_test(df: pd.DataFrame, col: str, *, n_perm: int = 10_000, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of annotator means of film means: sign-flip p (one-sided) and stratified film bootstrap CI."""
    f = df.dropna(subset=[col]).groupby("stimulus_id").agg(d=(col, "mean"), annotator=("annotator", "first"))
    if len(f) < 2:
        return {"n_films": int(len(f))}
    codes, names = pd.factorize(f["annotator"])
    na = len(names)
    d = f["d"].to_numpy()
    M = np.zeros((na, len(d)))
    M[codes, np.arange(len(d))] = 1
    M /= M.sum(1, keepdims=True)
    w = M.T @ np.full(na, 1 / na)  # per-film weight in the statistic
    t_obs = float(d @ w)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    p = float((1 + np.sum((flips * d) @ w >= t_obs)) / (1 + n_perm))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(codes == a) for a in range(na)]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = np.mean([d[rng.choice(g, size=len(g), replace=True)].mean() for g in groups])
    per_ann = f.groupby("annotator")["d"].mean()
    return {"mean": t_obs, "lo": float(np.quantile(boots, .025)), "hi": float(np.quantile(boots, .975)),
            "p_sign_flip": p, "n_films": int(len(f)), "n_boundaries": int(df[col].notna().sum()),
            "per_annotator": {k: float(v) for k, v in per_ann.items()},
            "n_annotators_positive": int((per_ann > 0).sum()), "n_annotators": na}


def boundary(sm: SpaceMatrix, boundaries: pd.DataFrame, cuts: pd.DataFrame, durations: pd.DataFrame, *,
             level: str, window: float, kernel: tuple[np.ndarray, np.ndarray] | None = None, tol: float = 1.0,
             exclude: float = 2.0, n_perm: int = 10_000, n_boot: int = 2000, seed: int = 0
             ) -> tuple[pd.DataFrame, dict]:
    for name, df, cols in (("boundaries", boundaries, ("stimulus_id", "level", "onset", "annotator")),
                           ("cuts", cuts, ("stimulus_id", "time")), ("durations", durations, ("stimulus_id", "duration"))):
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise BenchError(f"{name} table lacks column(s) {miss}")
    dur = {str(k): float(v) for k, v in zip(durations["stimulus_id"], durations["duration"])}
    bd = boundaries.astype({"stimulus_id": str, "level": str})
    bd = bd[(bd["onset"] > 0) & bd["stimulus_id"].isin(dur)]
    bd = bd[bd["onset"] <= bd["stimulus_id"].map(dur)]
    if level not in set(bd["level"]):
        raise BenchError(f"level {level!r} is not in the boundaries table (levels: {sorted(set(bd['level']))})")
    ann = bd.groupby("stimulus_id")["annotator"].agg(lambda x: sorted(set(x)))
    if (ann.map(len) > 1).any():
        raise BenchError(f"{int((ann.map(len) > 1).sum())} film(s) name more than one annotator; the film test nests "
                         "films in annotators")
    annotator = {s: a[0] for s, a in ann.items()}
    bounds = {s: np.unique(g["onset"].to_numpy(float)) for s, g in bd[bd["level"] == level].groupby("stimulus_id")}
    all_bounds = {s: np.unique(g["onset"].to_numpy(float)) for s, g in bd.groupby("stimulus_id")}
    cut_t = {str(s): g["time"].to_numpy(float) for s, g in cuts.groupby("stimulus_id")}
    grids = film_grids(sm, dur, window)
    absent = sorted(set(bounds) - set(grids))
    if absent:
        raise BenchError(f"{len(absent)} film(s) with boundaries have no grid rows (e.g. {absent[:3]})")
    stats = per_bin_stats(grids, kernel)
    items, summary = [], {"level": level, "window_sec": window, "tol_sec": tol, "exclude_sec": exclude,
                          "calipers_sec": list(CALIPERS), "min_pool": MIN_POOL, "n_films": len(bounds),
                          "statistic": "mean over annotators of film-mean (boundary window - matched control windows)",
                          "ci": "films resampled within annotator; p = annotator-stratified sign flip, one-sided"}
    for st, series in stats.items():
        df = match(series, bounds, all_bounds, cut_t, annotator, window=window, tol=tol, exclude=exclude)
        df.insert(0, "stat", st)
        items.append(df)
        kept = df[~df["dropped"]]
        near, far = df[df["d_cut"] <= tol], df[df["d_cut"] > tol]
        kw = {"n_perm": n_perm, "n_boot": n_boot, "seed": seed}
        summary[st] = {
            "n_boundaries": int(len(df)), "n_dropped": int(df["dropped"].sum()),
            "n_widened": int((kept["caliper"] == CALIPERS[1]).sum()), "median_pool": float(kept["n_pool"].median()),
            "share_within_tol_of_cut": float(np.mean(df["d_cut"] <= tol)),
            "film_sd": film_test(df, "delta_std", **kw), "raw": film_test(df, "delta", **kw),
            "far_from_cut_film_sd": film_test(far, "delta_std", **kw), "near_cut_film_sd": film_test(near, "delta_std", **kw),
            "far_from_cut_raw": film_test(far, "delta", **kw), "near_cut_raw": film_test(near, "delta", **kw),
        }
    return pd.concat(items, ignore_index=True), summary


def compare_boundary(a: pd.DataFrame, b: pd.DataFrame, *, col: str = "delta_std", n_boot: int = 2000,
                     seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Paired a - b and a / b per statistic, on boundaries kept in both runs, with shared film draws.

    Each draw resamples films within annotator once and takes both arms'
    statistic (mean over annotators of film means) on it. The ratio is
    retention against b with chance 0 (no boundary effect), flagged
    ``ratio_defined: false`` when b's own interval reaches 0.
    """
    key = ["stat", "stimulus_id", "time"]
    m = a[[*key, "annotator", col]].merge(b[[*key, col]], on=key, suffixes=("_a", "_b"))
    if len(m) == 0:
        raise BenchError("the two boundary runs share no (stat, stimulus_id, time)")
    summary: dict = {"unit": col, "ci_unit": "films within annotator; same draws for both arms",
                     "n_items": int(len(m))}
    from .compare import verdict

    for st, d in m.groupby("stat"):
        d = d.dropna(subset=[f"{col}_a", f"{col}_b"])
        f = d.groupby("stimulus_id").agg(a=(f"{col}_a", "mean"), b=(f"{col}_b", "mean"),
                                         annotator=("annotator", "first"))
        codes, names = pd.factorize(f["annotator"])
        groups = [np.flatnonzero(codes == k) for k in range(len(names))]
        va, vb = f["a"].to_numpy(), f["b"].to_numpy()

        def stat(idx_sets, v):
            return float(np.mean([v[i].mean() for i in idx_sets]))

        rng = np.random.default_rng(seed)
        ba, bb = np.empty(n_boot), np.empty(n_boot)
        for i in range(n_boot):
            draw = [rng.choice(g, size=len(g), replace=True) for g in groups]
            ba[i], bb[i] = stat(draw, va), stat(draw, vb)
        pa, pb = stat(groups, va), stat(groups, vb)

        def iv(point, draws):
            return {"mean": float(point), "lo": float(np.nanquantile(draws, .025)),
                    "hi": float(np.nanquantile(draws, .975)), "n": int(len(d)), "n_units": int(len(f))}

        diff = iv(pa - pb, ba - bb)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = iv(pa / pb, ba / bb)
        b_int = iv(pb, bb)
        summary[st] = {"a": pa, "b": pb, "a_interval": iv(pa, ba), "b_interval": b_int, "diff": diff,
                       "verdict": verdict(diff), "ratio": ratio, "ratio_defined": bool(b_int["lo"] > 0),
                       "n_boundaries_both": int(len(d)), "n_films": int(len(f))}
    return m, summary
