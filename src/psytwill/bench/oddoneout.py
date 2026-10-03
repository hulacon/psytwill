"""Triplet odd-one-out against human choices (THINGS-style).

Input is a triplet table with columns ``item1, item2, item3, odd`` where
``odd`` names the item a person chose. The model's choice is the item left
out of the most similar pair. Per-triplet rows are written, so accuracy, its
CI and any split (train/test, repeat triplets) recompute from the output.

Only the zero-shot reading is implemented. The learned linear probe of the
human-alignment literature is a separate arm: it fits a transform on
training triplets, so it needs object-disjoint folds, and it is not this
module's to approximate.

``noise_ceiling`` is the ceiling the triplet literature reports (Hebart et
al.'s ``get_noiseceiling``): over triplets answered more than once, the share
of answers that agree with the triplet's most common answer, averaged over
triplets. A model that always names the most common answer scores exactly
this, so no model can be expected to beat it on those triplets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from psytwill.exceptions import BenchError
from psytwill.store import SpaceMatrix

from .core import bootstrap_mean, prototypes, similarity

TRIPLET_COLUMNS = ("item1", "item2", "item3", "odd")


def oddoneout(emb: SpaceMatrix, triplets: pd.DataFrame, *, metric: str = "cosine", n_boot: int = 2000,
              seed: int = 0) -> tuple[pd.DataFrame, dict]:
    missing = [c for c in TRIPLET_COLUMNS if c not in triplets.columns]
    if missing:
        raise BenchError(f"triplet table lacks column(s) {missing}; expected {list(TRIPLET_COLUMNS)}")
    tri = triplets.astype({c: str for c in TRIPLET_COLUMNS}).reset_index(drop=True)
    bad = ~((tri["odd"] == tri["item1"]) | (tri["odd"] == tri["item2"]) | (tri["odd"] == tri["item3"]))
    if bad.any():
        raise BenchError(f"{int(bad.sum())} triplet(s) name an odd item that is not one of their three")
    ids, P, _ = prototypes(emb)
    pos = {s: i for i, s in enumerate(ids)}
    need = set(tri["item1"]) | set(tri["item2"]) | set(tri["item3"])
    absent = sorted(need - set(pos))
    if absent:
        raise BenchError(f"{len(absent)} triplet item(s) have no embedding (e.g. {absent[:3]}); "
                         "embed every item, or filter the triplets first")
    S = similarity(P, P, metric)
    a = tri["item1"].map(pos).to_numpy()
    b = tri["item2"].map(pos).to_numpy()
    c = tri["item3"].map(pos).to_numpy()
    # column j = similarity of the pair that leaves out item j+1, so argmax names the odd item
    pair = np.stack([S[b, c], S[a, c], S[a, b]], axis=1)
    order = np.argsort(-pair, axis=1, kind="stable")
    tied = np.isclose(pair[np.arange(len(tri)), order[:, 0]], pair[np.arange(len(tri)), order[:, 1]])
    names = np.stack([tri["item1"], tri["item2"], tri["item3"]], axis=1)
    choice = names[np.arange(len(tri)), order[:, 0]]
    items = tri.copy()
    items["model_odd"] = choice
    items["correct"] = items["model_odd"] == items["odd"]
    items["tied"] = tied
    summary = {"n_triplets": int(len(items)), "n_items": len(need), "chance": 1 / 3, "metric": metric,
               "accuracy": bootstrap_mean(items["correct"].astype(float), n_boot=n_boot, seed=seed),
               "n_tied": int(tied.sum())}
    return items, summary


def noise_ceiling(triplets: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Ceiling from repeated triplets: mean over triplets of the modal answer's share.

    A triplet is its three items as a set, so the same triplet listed in a
    different order counts as one. Triplets answered once carry no
    information about agreement and are left out (and counted).
    """
    missing = [c for c in TRIPLET_COLUMNS if c not in triplets.columns]
    if missing:
        raise BenchError(f"triplet table lacks column(s) {missing}; expected {list(TRIPLET_COLUMNS)}")
    tri = triplets.astype({c: str for c in TRIPLET_COLUMNS})
    key = ["|".join(sorted(t)) for t in zip(tri["item1"], tri["item2"], tri["item3"])]
    d = pd.DataFrame({"triplet": key, "odd": tri["odd"].to_numpy()})
    counts = d.groupby(["triplet", "odd"]).size().unstack(fill_value=0)
    n = counts.sum(axis=1)
    rep = counts[n >= 2]
    if rep.empty:
        raise BenchError("no triplet is answered more than once; a noise ceiling needs repeats")
    consistency = rep.max(axis=1) / rep.sum(axis=1)
    return {"ceiling": bootstrap_mean(consistency.to_numpy(), n_boot=n_boot, seed=seed),
            "n_triplets_repeated": int(len(rep)), "n_triplets_single": int((n < 2).sum()),
            "answers_per_triplet_median": float(rep.sum(axis=1).median()), "unit": "triplet"}
