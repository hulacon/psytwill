"""Shared machinery for every bench task: loading, the linear map, identification, bootstrap, output.

Everything a task needs that is not the task itself lives here, so two tasks
cannot drift apart on, say, how a held-out fold is standardised or how a tie
in an identification rank is counted.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from psytwill.compare import DEFAULT_ALPHAS
from psytwill.exceptions import BenchError
from psytwill.store import SpaceMatrix, distinct_values, load_spaces

from . import BENCH_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_model(
    features: Sequence[str | Path],
    model: str,
    *,
    key: Sequence[str] = ("stimulus_id",),
    window: float | None = None,
    stimulus_ids=None,
) -> SpaceMatrix:
    """One model's rows from every table that carries it, stacked.

    Two tables contributing the same row label is refused: across tables a
    duplicate key means the same stimulus was extracted twice, and silently
    pooling it would weight that stimulus double in every score.
    """
    parts = []
    for path in features:
        if model not in distinct_values(path, "model"):
            continue
        got = load_spaces(path, key=tuple(key), models=[model], window=window, stimulus_ids=stimulus_ids)
        if model not in got:
            raise BenchError(f"model {model!r} in {path} loaded as nothing (string-valued or empty?)")
        parts.append(got[model])
    if not parts:
        raise BenchError(f"model {model!r} is in none of {[str(p) for p in features]}")
    sm = parts[0]
    if len(parts) > 1:
        if any(p.features != sm.features for p in parts[1:]):
            raise BenchError(f"model {model!r} carries different feature columns across its tables")
        labels = [lab for p in parts for lab in p.labels]
        if len(set(labels)) != len(labels):
            raise BenchError(f"model {model!r}: a row label appears in more than one table")
        sm = SpaceMatrix(name=model, labels=labels, X=np.vstack([p.X for p in parts]), features=sm.features,
                         modality=sm.modality, extractor=sm.extractor,
                         n_replicates=max(p.n_replicates for p in parts))
    return sm


def load_concat(
    features: Sequence[str | Path],
    models: Sequence[str],
    *,
    key: Sequence[str] = ("stimulus_id",),
    window: float | None = None,
) -> tuple[SpaceMatrix, dict]:
    """Several models side by side on their shared rows: the full-battery arm (B1).

    Each member's columns are z-scored over the loaded rows, so no member
    weighs by its raw units. A column holding any NaN on those rows is
    DROPPED, never filled (a member's declared nulls would otherwise have to
    be given a value, and no consumer creates one); so is a constant column.
    Both are counted per member in the returned record. Rows a member lacks
    are not invented either: the result keeps the rows every member has.
    """
    parts = {m: load_model(features, m, key=key, window=window) for m in models}
    common = set.intersection(*(set(p.labels) for p in parts.values()))
    if not common:
        raise BenchError(f"models {list(models)} share no row label")
    labels = sorted(common)
    blocks, names, info = [], [], {"members": list(models), "n_rows": len(labels), "rows_lost": {},
                                   "nan_columns_dropped": {}, "constant_columns_dropped": {}}
    for m, p in parts.items():
        pos = {lab: i for i, lab in enumerate(p.labels)}
        X = p.X[[pos[lab] for lab in labels]]
        info["rows_lost"][m] = len(p.labels) - len(labels)
        nan_col = np.isnan(X).any(axis=0)
        sd = np.nanstd(X, axis=0)
        const = ~nan_col & (sd == 0)
        keep = ~nan_col & ~const
        info["nan_columns_dropped"][m] = int(nan_col.sum())
        info["constant_columns_dropped"][m] = int(const.sum())
        if not keep.any():
            raise BenchError(f"member {m!r} keeps no column on these rows (all NaN or constant)")
        Xk = X[:, keep]
        blocks.append((Xk - Xk.mean(0)) / Xk.std(0))
        names += [f"{m}:{f}" for f, k in zip(p.features, keep) if k]
    sm = SpaceMatrix(name="+".join(models), labels=labels, X=np.hstack(blocks), features=names)
    info["n_columns"] = sm.dim
    return sm, info


def drop_nan_rows(sm: SpaceMatrix) -> tuple[SpaceMatrix, int]:
    """Rows with any NaN are dropped and counted, never filled.

    A NaN is a declared absence (Contract B ``nulls``); a consumer that
    imputes it scores a value nobody measured.
    """
    ok = ~np.isnan(sm.X).any(axis=1)
    n_bad = int((~ok).sum())
    if n_bad:
        sm = SpaceMatrix(name=sm.name, labels=[lab for lab, k in zip(sm.labels, ok) if k], X=sm.X[ok],
                         features=sm.features, modality=sm.modality, extractor=sm.extractor,
                         n_replicates=sm.n_replicates)
    return sm, n_bad


def stimulus_of(label: str) -> str:
    return label.split("|")[0]


def prototypes(sm: SpaceMatrix, id_of=stimulus_of) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Collapse replicate rows (e.g. 5 captions per image) to one per item.

    Each row is unit-normalised before averaging, so a long caption with a
    large-norm embedding does not dominate its image's prototype. ``id_of``
    maps a row label to its item: the stimulus by default; the whole label
    when each row is already one item (a film segment). Returns
    (ids, prototypes, n_rows_per_id).
    """
    ids = np.array([id_of(lab) for lab in sm.labels])
    uniq, inv, counts = np.unique(ids, return_inverse=True, return_counts=True)
    norms = np.linalg.norm(sm.X, axis=1, keepdims=True)
    Xn = sm.X / np.where(norms == 0, 1.0, norms)
    P = np.zeros((len(uniq), sm.dim))
    np.add.at(P, inv, Xn)
    P /= counts[:, None]
    return list(uniq), P, counts


def pool_segments(sm: SpaceMatrix, segments: pd.DataFrame) -> tuple[SpaceMatrix, dict]:
    """Mean-pool a time grid (labels ``stimulus|time``) into segments.

    ``segments`` has ``stimulus_id, chunk_idx, onset, offset``; a grid row
    belongs to a segment when ``onset <= time < offset``. The pooled row is
    labelled ``stimulus|chunk_idx``, the label a chunk table keyed on
    ``stimulus_id,chunk_idx`` gives the same segment, so the two sides join
    by label. A segment that no grid row falls in stops the run: dropping it
    would quietly change the candidate set its film is scored against.
    Returns the pooled matrix and the number of grid rows per segment.
    """
    need = {"stimulus_id", "chunk_idx", "onset", "offset"}
    missing = sorted(need - set(segments.columns))
    if missing:
        raise BenchError(f"segments table lacks column(s) {missing}")
    seg = segments.astype({"stimulus_id": str}).reset_index(drop=True)
    if seg.duplicated(["stimulus_id", "chunk_idx"]).any():
        raise BenchError("segments table repeats a (stimulus_id, chunk_idx)")
    if (seg["offset"] <= seg["onset"]).any():
        raise BenchError(f"{int((seg['offset'] <= seg['onset']).sum())} segment(s) end at or before they start")
    sid = np.array([stimulus_of(lab) for lab in sm.labels])
    t = np.array([float(lab.split("|")[1]) for lab in sm.labels])
    by_stim: dict[str, np.ndarray] = {}
    for s in np.unique(sid):
        by_stim[s] = np.where(sid == s)[0]
    labels, rows, counts, empty = [], [], [], []
    for r in seg.itertuples(index=False):
        idx = by_stim.get(r.stimulus_id, np.array([], int))
        hit = idx[(t[idx] >= r.onset) & (t[idx] < r.offset)]
        lab = f"{r.stimulus_id}|{int(r.chunk_idx)}"
        if hit.size == 0:
            empty.append(lab)
            continue
        labels.append(lab)
        rows.append(sm.X[hit].mean(0))
        counts.append(hit.size)
    if empty:
        raise BenchError(f"{len(empty)} segment(s) contain no grid row of {sm.name!r} (e.g. {empty[:3]}); "
                         "is the grid keyed on stimulus_id,time and loaded with the right window?")
    pooled = SpaceMatrix(name=sm.name, labels=labels, X=np.vstack(rows), features=sm.features,
                         modality=sm.modality, extractor=sm.extractor, n_replicates=sm.n_replicates)
    return pooled, {"n_segments": len(labels), "grid_rows_per_segment_min": int(min(counts)),
                    "grid_rows_per_segment_median": float(np.median(counts))}


# ---------------------------------------------------------------------------
# The linear map (the one place a fold is standardised)
# ---------------------------------------------------------------------------


@dataclass
class LinearMap:
    """Ridge from X to Y, both z-scored on the training rows only."""

    alpha: float
    mx: np.ndarray
    sx: np.ndarray
    my: np.ndarray
    sy: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mx) / self.sx
        return (Z @ self.coef.T + self.intercept) * self.sy + self.my


def fit_map(X: np.ndarray, Y: np.ndarray, alphas: Sequence[float] = DEFAULT_ALPHAS) -> LinearMap:
    """Fit on training rows only; alpha chosen inside this fit (RidgeCV GCV)."""
    from sklearn.linear_model import RidgeCV

    if X.shape[0] < 3:
        raise BenchError(f"a linear map needs at least 3 training rows; got {X.shape[0]}")
    mx, sx = X.mean(0), X.std(0)
    my, sy = Y.mean(0), Y.std(0)
    sx = np.where(sx == 0, 1.0, sx)
    sy = np.where(sy == 0, 1.0, sy)
    r = RidgeCV(alphas=list(alphas)).fit((X - mx) / sx, (Y - my) / sy)
    return LinearMap(float(r.alpha_), mx, sx, my, sy, np.atleast_2d(r.coef_), np.atleast_1d(r.intercept_))


def folds(n_or_groups, n_splits: int, seed: int):
    """Item folds (shuffled KFold) or, given a group per row, GroupKFold.

    Yields (train_idx, test_idx). Grouped folds are what keep a film's
    neighbouring windows, or one image's replicate captions, on one side.
    """
    from sklearn.model_selection import GroupKFold, KFold

    if np.ndim(n_or_groups) == 0:
        n = int(n_or_groups)
        yield from KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(n))
    else:
        g = np.asarray(n_or_groups)
        if len(np.unique(g)) < n_splits:
            raise BenchError(f"{len(np.unique(g))} groups cannot fill {n_splits} folds")
        yield from GroupKFold(n_splits=n_splits).split(np.zeros(len(g)), groups=g)


# ---------------------------------------------------------------------------
# Similarity and identification
# ---------------------------------------------------------------------------


def similarity(A: np.ndarray, B: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """Row-by-row similarity, (n_A, n_B). ``cosine`` or ``correlation``."""
    if metric == "correlation":
        A = A - A.mean(1, keepdims=True)
        B = B - B.mean(1, keepdims=True)
    elif metric != "cosine":
        raise BenchError(f"unknown metric {metric!r}; use cosine or correlation")
    na = np.linalg.norm(A, axis=1, keepdims=True)
    nb = np.linalg.norm(B, axis=1, keepdims=True)
    return (A / np.where(na == 0, 1, na)) @ (B / np.where(nb == 0, 1, nb)).T


def identify(S: np.ndarray, correct: np.ndarray, allowed: np.ndarray | None = None) -> pd.DataFrame:
    """Per-row identification of the correct column among the allowed ones.

    ``pct_beaten`` is the fraction of distractors scored below the correct
    candidate, ties counting half: it is pairwise (2-way) identification
    accuracy, chance 0.5 whatever the candidate count, so it is comparable
    across folds of different size. ``top1`` is also reported, with the
    candidate count beside it, because its chance level is 1/n_candidates.
    """
    n = S.shape[0]
    if allowed is None:
        allowed = np.ones_like(S, dtype=bool)
    allowed = allowed.copy()
    allowed[np.arange(n), correct] = True
    s_true = S[np.arange(n), correct][:, None]
    distract = allowed.copy()
    distract[np.arange(n), correct] = False
    n_d = distract.sum(1)
    above = ((S > s_true) & distract).sum(1)
    tie = ((S == s_true) & distract).sum(1)
    below = n_d - above - tie
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(n_d > 0, (below + 0.5 * tie) / n_d, np.nan)
    return pd.DataFrame({"n_candidates": n_d + 1, "rank": above, "top1": (above == 0) & (tie == 0),
                         "pct_beaten": pct})


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_mean(values, groups=None, *, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Mean with a percentile bootstrap CI; resamples whole groups when given.

    Grouped resampling is the honest unit whenever rows share a source (a
    film's windows, a subject's pairs): resampling rows would treat
    neighbouring windows as independent and give a CI that is too narrow.
    """
    v = np.asarray(values, dtype=float)
    ok = ~np.isnan(v)
    v = v[ok]
    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "n_units": 0}
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, v.size, size=(n_boot, v.size))
        boots = v[idx].mean(1)
        n_units = v.size
    else:
        g = np.asarray(groups)[ok]
        uniq, inv = np.unique(g, return_inverse=True)
        sums = np.bincount(inv, weights=v)
        cnts = np.bincount(inv).astype(float)
        pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
        boots = sums[pick].sum(1) / cnts[pick].sum(1)
        n_units = len(uniq)
    a = (1 - level) / 2
    return {"mean": float(v.mean()), "lo": float(np.quantile(boots, a)), "hi": float(np.quantile(boots, 1 - a)),
            "n": int(v.size), "n_units": int(n_units)}


# ---------------------------------------------------------------------------
# Leak guard
# ---------------------------------------------------------------------------


def refuse_overlap(item_ids, exclude_files: Sequence[str | Path] | None, *, what: str) -> dict:
    """Refuse a benchmark whose items appear in a fit's id list.

    The caller names the fit-side id files (a fit's ``inputs`` ids, or an
    explicit list); an item found in any of them stops the run. Returns the
    evidence for the sidecar. No files given is recorded as unchecked, not
    as clean.
    """
    if not exclude_files:
        return {"checked": False}
    items = set(map(str, item_ids))
    ev = []
    for f in exclude_files:
        ids = {line.strip() for line in Path(f).read_text().splitlines() if line.strip()}
        hit = sorted(items & ids)
        if hit:
            raise BenchError(f"{len(hit)} {what} item(s) appear in {f} (e.g. {hit[:3]}); a benchmark item "
                             "that a fit has seen does not score held-out behaviour. Drop them with "
                             "--drop-ids, or pick a held-out item set.")
        ev.append({"file": str(f), "n_ids": len(ids)})
    return {"checked": True, "files": ev}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _input_record(p) -> dict:
    p = Path(p)
    st = p.stat()
    return {"path": str(p.resolve()), "bytes": st.st_size, "mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                                                 time.localtime(st.st_mtime))}


def write_run(out_dir: str | Path, task: str, tag: str, items: pd.DataFrame, summary: dict, *,
              params: dict, inputs: Sequence[str | Path]) -> Path:
    """``<task>__<tag>.items.csv`` (every scored item) + ``<task>__<tag>.json`` (sidecar).

    The items file is the result; the sidecar's summary is a convenience
    recomputable from it, so a different aggregation never needs a re-run.
    """
    from psytwill import __version__

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{task}__{tag}"
    items.to_csv(out / f"{stem}.items.csv", index=False)
    side = out / f"{stem}.json"
    side.write_text(json.dumps({
        "bench_schema_version": BENCH_SCHEMA_VERSION,
        "psytwill_version": __version__,
        "task": task,
        "tag": tag,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "params": params,
        "inputs": [_input_record(p) for p in inputs],
        "items_file": f"{stem}.items.csv",
        "n_items": int(len(items)),
        "summary": summary,
    }, indent=2, default=str) + "\n")
    return side
