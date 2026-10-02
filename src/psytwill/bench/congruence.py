"""Pair congruence: how related is each studied image-word pair, and does that track what people did with it?

Input is a pairs table with ``subject, image_id, word_id`` plus one or more
per-pair outcome columns (an associability rating, later memory accuracy,
confidence). The model scores each pair's congruence; the task score is the
within-subject Spearman correlation between congruence and each outcome.
Pairings that are random per subject make each subject's pairs an
independent draw of congruence.

- ``zero-shot``: cosine between the image and the word in one trained
  space (EBind, CLIP). The caller vouches that the two sides share it.
- ``mapped``: a ridge map from image space to text space is fit on a
  separate paired set (e.g. images and their captions, embedded by the same
  text model as the words), with folds over images so a pair's own image is
  never in the map that scores it. Congruence is the similarity between the
  mapped image and the word. Every model can be scored this way.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from psytwill.compare import DEFAULT_ALPHAS
from psytwill.exceptions import BenchError
from psytwill.store import SpaceMatrix

from .core import fit_map, folds, prototypes, similarity

PAIR_COLUMNS = ("subject", "image_id", "word_id")


def _lookup(sm: SpaceMatrix) -> tuple[dict[str, int], np.ndarray]:
    ids, P, _ = prototypes(sm)
    return {s: i for i, s in enumerate(ids)}, P


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    if len(x) < 3 or np.all(y == y[0]) or np.all(x == x[0]):
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def congruence(
    image: SpaceMatrix,
    word: SpaceMatrix,
    pairs: pd.DataFrame,
    outcomes: Sequence[str],
    *,
    mode: str = "mapped",
    map_image: SpaceMatrix | None = None,
    map_text: SpaceMatrix | None = None,
    metric: str = "correlation",
    n_splits: int = 5,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    missing = [c for c in (*PAIR_COLUMNS, *outcomes) if c not in pairs.columns]
    if missing:
        raise BenchError(f"pairs table lacks column(s) {missing}")
    p = pairs.astype({"subject": str, "image_id": str, "word_id": str}).reset_index(drop=True)
    i_pos, I = _lookup(image)
    w_pos, W = _lookup(word)
    for col, pos, what in (("image_id", i_pos, "image"), ("word_id", w_pos, "word")):
        absent = sorted(set(p[col]) - set(pos))
        if absent:
            raise BenchError(f"{len(absent)} {what} id(s) in the pairs table have no embedding "
                             f"(e.g. {absent[:3]})")
    ii = p["image_id"].map(i_pos).to_numpy()
    wi = p["word_id"].map(w_pos).to_numpy()
    cong = np.full(len(p), np.nan)
    fold_of = np.full(len(p), -1)
    if mode == "zero-shot":
        if I.shape[1] != W.shape[1]:
            raise BenchError(f"zero-shot needs one shared space; image has {I.shape[1]} dims, word "
                             f"{W.shape[1]}. Use --mode mapped")
        Ii = I[ii]
        Ww = W[wi]
        cong = (Ii * Ww).sum(1) / (np.linalg.norm(Ii, axis=1) * np.linalg.norm(Ww, axis=1))
    elif mode == "mapped":
        if map_image is None or map_text is None:
            raise BenchError("mapped mode needs a paired training set: map_image and map_text")
        mi_pos, MI = _lookup(map_image)
        mt_pos, MT = _lookup(map_text)
        if MT.shape[1] != W.shape[1]:
            raise BenchError(f"map text has {MT.shape[1]} dims but the words have {W.shape[1]}; the map "
                             "must land in the words' own space (same text model)")
        train_ids = np.array(sorted(set(mi_pos) & set(mt_pos)))
        if len(train_ids) < 3 * n_splits:
            raise BenchError(f"only {len(train_ids)} ids are on both sides of the map's training set")
        # each pair image goes to the fold whose map never saw it; images
        # outside the map's set are scored by a map fit on all of it
        fold_img: dict[str, int] = {}
        for f, (_, te) in enumerate(folds(len(train_ids), n_splits, seed)):
            for s in train_ids[te]:
                fold_img[s] = f
        tr_X = MI[[mi_pos[s] for s in train_ids]]
        tr_Y = MT[[mt_pos[s] for s in train_ids]]
        fid = np.array([fold_img.get(s, -1) for s in p["image_id"]])
        for f in sorted(set(fid)):
            keep = np.array([fold_img[s] != f for s in train_ids])  # f == -1 keeps all
            lm = fit_map(tr_X[keep], tr_Y[keep], alphas)
            rows = np.where(fid == f)[0]
            pred = lm.predict(I[ii[rows]])
            S = similarity(pred, W[wi[rows]], metric)
            cong[rows] = np.diag(S)
            fold_of[rows] = f
    else:
        raise BenchError(f"unknown mode {mode!r}; use zero-shot or mapped")
    items = p.copy()
    items["congruence"] = cong
    items["fold"] = fold_of
    summary: dict = {"n_pairs": int(len(items)), "n_subjects": int(items["subject"].nunique()), "mode": mode,
                     "metric": metric, "statistic": "within-subject Spearman(congruence, outcome)"}
    rng = np.random.default_rng(seed)
    for oc in outcomes:
        per_sub = {}
        boots = []
        for s, d in items.groupby("subject"):
            d = d[d[oc].notna()]
            x, y = d["congruence"].to_numpy(), d[oc].to_numpy(dtype=float)
            rho = _spearman(x, y)
            b = np.array([_spearman(x[k], y[k]) for k in rng.integers(0, len(x), size=(n_boot, len(x)))]) \
                if len(x) >= 3 else np.full(n_boot, np.nan)
            boots.append(b)
            per_sub[s] = {"rho": rho, "lo": float(np.nanquantile(b, .025)) if len(x) >= 3 else float("nan"),
                          "hi": float(np.nanquantile(b, .975)) if len(x) >= 3 else float("nan"), "n": int(len(x))}
        mean_b = np.nanmean(np.vstack(boots), axis=0)
        rhos = [v["rho"] for v in per_sub.values()]
        summary[oc] = {"mean_rho": float(np.nanmean(rhos)), "lo": float(np.nanquantile(mean_b, .025)),
                       "hi": float(np.nanquantile(mean_b, .975)), "per_subject": per_sub,
                       "ci": "pairs resampled within subject; subjects fixed"}
    return items, summary
