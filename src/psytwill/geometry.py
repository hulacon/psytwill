"""Driver: every measure over every pair of spaces, frozen as a versioned table.

:mod:`psytwill.compare` defines what two spaces' relationship *is*; this module
runs those measures across a whole inventory and writes the result as data. The
split matters because the inventory is where the expensive mistakes live: a
measure is a formula and either right or wrong, but a matrix over 35 spaces has
to decide what counts as a space, which pairs are the same pair, and what *n*
each cell ran on — and every one of those decisions is invisible in the number
it produces.

Three conventions, each recorded in the output rather than assumed:

**Symmetric measures run once per unordered pair.** ``MEASURE_REGISTRY`` carries
the flag, so CKA/RSA/overlap emit one row per pair and ridge emits two. A
symmetric measure computed twice is not wrong, but it doubles the run and
invites a reader to treat the two copies as independent evidence.

**Rows must already be aligned.** :func:`compare_spaces` refuses an inventory
whose spaces have different row counts rather than intersecting them itself —
:func:`psytwill.store.align_spaces` does that, and returns the surviving labels
for the caller to record. Silent intersection inside the driver would let a
space with thin coverage shrink the whole matrix without anyone seeing it.

**Every cell carries its own n.** Coverage is uneven (transcript spaces exist
only for clips with speech), and NaN rows are dropped per pair. An R^2 read
across cells that ran on different row counts is not comparable, so ``n_used``
and ``n_dropped`` travel with the value instead of being quoted once for the
run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from psytwill import __version__
from psytwill.compare import (
    DEFAULT_ALPHAS,
    DEFAULT_K,
    MEASURE_REGISTRY,
    applicability,
    cka,
    get_measure,
    knn_indices,
    neighbor_overlap,
    neighbor_overlap_null,
    participation_ratio,
    ridge_predictivity,
    second_order_rsa,
)
from psytwill.exceptions import SpaceError
from psytwill.store import SpaceMatrix

GEOMETRY_SCHEMA_VERSION = "1.0"

PAIR_COLUMNS: tuple[str, ...] = (
    "source",
    "target",
    "measure",
    "value",
    "symmetric",
    "n_used",
    "n_dropped",
    "note",
    # ridge only
    "r2_mean",
    "n_splits",
    "grouped",
    "alpha_min",
    "alpha_max",
    # permutation null only
    "p_value",
    "null_mean",
    "null_sd",
    "null_z",
    "n_perm",
    "block_size",
    "k",
)

MANIFEST_COLUMNS: tuple[str, ...] = (
    "space",
    "dim",
    "n",
    "participation_ratio",
    "pr_fraction",
    "modality",
    "extractor",
    "n_replicates",
    "n_nan_rows",
)


@dataclass
class GeometryResult:
    """The pair table, the space manifest, and how they were produced."""

    pairs: pd.DataFrame
    manifest: pd.DataFrame
    config: dict = field(default_factory=dict)

    def matrix(self, measure: str) -> pd.DataFrame:
        """One measure as a square space x space frame, for reading by eye.

        Symmetric measures are mirrored across the diagonal on the way out;
        they are still stored once.
        """
        rows = self.pairs[self.pairs["measure"] == measure]
        if rows.empty:
            raise SpaceError(
                f"No rows for measure '{measure}'. Present: "
                f"{', '.join(sorted(self.pairs['measure'].unique()))}."
            )
        names = list(self.manifest["space"])
        out = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
        symmetric = bool(rows["symmetric"].iloc[0])
        for r in rows.itertuples():
            out.loc[r.source, r.target] = r.value
            if symmetric:
                out.loc[r.target, r.source] = r.value
        return out


def _blank_row() -> dict:
    return {c: None for c in PAIR_COLUMNS}


def _nan_rows(X: np.ndarray) -> int:
    return int(np.isnan(X).any(axis=1).sum())


def compare_spaces(
    spaces: Mapping[str, SpaceMatrix],
    *,
    measures: Sequence[str] | None = None,
    k: int = DEFAULT_K,
    n_splits: int = 5,
    groups: Sequence | None = None,
    rsa_metric: str = "correlation",
    knn_metric: str = "cosine",
    n_permutations: int = 1000,
    block_size: int | None = None,
    random_state: int = 0,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    progress: Callable[[int, int, str], None] | None = None,
) -> GeometryResult:
    """Run ``measures`` over every pair of ``spaces``.

    ``groups`` (one entry per row, e.g. a clip id) is passed to ridge so folds
    split by group; on any temporally ordered grid it is not optional. Set
    ``block_size`` to match, so the neighbour-overlap null shuffles blocks
    rather than rows — the two settings are the same assumption stated to two
    different measures, and setting only one of them is the common error.

    ``n_permutations=0`` skips the null and reports the observed overlap alone.
    """
    names = sorted(spaces)
    if len(names) < 2:
        raise SpaceError(
            f"Need at least 2 spaces to compare; got {len(names)}. "
            "Load more group files, or widen the model filter."
        )
    wanted = list(MEASURE_REGISTRY) if measures is None else list(measures)
    for m in wanted:
        get_measure(m)  # raises naming the available measures

    row_counts = {spaces[n].n for n in names}
    if len(row_counts) != 1:
        raise SpaceError(
            f"Spaces have different row counts {sorted(row_counts)}. Call "
            "psytwill.store.align_spaces() first so every cell runs on the "
            "same rows."
        )
    n_rows = row_counts.pop()
    if groups is not None and len(groups) != n_rows:
        raise SpaceError(
            f"'groups' has {len(groups)} entries for {n_rows} rows."
        )

    # Per-space kNN graphs, built once and reused across that space's 34
    # pairings. Only valid where no rows are dropped, so a space carrying NaN
    # falls back to the per-pair path rather than indexing rows that moved.
    nan_counts = {n: _nan_rows(spaces[n].X) for n in names}
    knn_cache: dict[str, np.ndarray] = {}
    if "neighbor_overlap" in wanted:
        for n in names:
            if nan_counts[n] == 0:
                knn_cache[n] = knn_indices(spaces[n].X, k=k, metric=knn_metric)

    def cached(name: str, other: str) -> np.ndarray | None:
        if nan_counts[name] or nan_counts[other]:
            return None
        return knn_cache.get(name)

    ordered = [(a, b) for a in names for b in names if a != b]
    unordered = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    n_asym = sum(1 for m in wanted if not get_measure(m).symmetric)
    n_sym = sum(1 for m in wanted if get_measure(m).symmetric)
    total = n_asym * len(ordered) + n_sym * len(unordered)
    done = 0

    rows: list[dict] = []
    for measure in wanted:
        cfg = get_measure(measure)
        pairs = unordered if cfg.symmetric else ordered
        for a, b in pairs:
            X, Y = spaces[a], spaces[b]
            row = _blank_row()
            row.update(
                source=a,
                target=b,
                measure=measure,
                symmetric=cfg.symmetric,
                note=applicability(measure, X.dim, Y.dim),
            )

            if measure == "ridge":
                res = ridge_predictivity(
                    X.X,
                    Y.X,
                    groups=groups,
                    n_splits=n_splits,
                    alphas=alphas,
                    random_state=random_state,
                )
                row.update(
                    value=res.r2,
                    r2_mean=res.r2_mean,
                    n_used=res.n_used,
                    n_dropped=res.n_dropped,
                    n_splits=res.n_splits,
                    grouped=res.grouped,
                    alpha_min=min(res.alphas_selected) if res.alphas_selected else None,
                    alpha_max=max(res.alphas_selected) if res.alphas_selected else None,
                )
            elif measure in ("cka_linear", "cka_rbf"):
                kernel = "linear" if measure == "cka_linear" else "rbf"
                row.update(value=cka(X.X, Y.X, kernel=kernel))
            elif measure == "rsa":
                row.update(value=second_order_rsa(X.X, Y.X, metric=rsa_metric))
            elif measure == "neighbor_overlap":
                if n_permutations:
                    res = neighbor_overlap_null(
                        X.X,
                        Y.X,
                        k=k,
                        metric=knn_metric,
                        n_perm=n_permutations,
                        block_size=block_size,
                        random_state=random_state,
                        knn_x=cached(a, b),
                        knn_y=cached(b, a),
                    )
                    sd = res.null_sd
                    row.update(
                        value=res.observed,
                        p_value=res.p_value,
                        null_mean=res.null_mean,
                        null_sd=sd,
                        null_z=(res.observed - res.null_mean) / sd if sd > 0 else None,
                        n_perm=res.n_perm,
                        block_size=res.block_size,
                        k=k,
                    )
                else:
                    row.update(
                        value=neighbor_overlap(X.X, Y.X, k=k, metric=knn_metric), k=k
                    )
            else:  # a registry entry this driver has no branch for
                raise SpaceError(
                    f"Measure '{measure}' is registered but compare_spaces has "
                    "no branch for it. Add one, so its extra outputs are "
                    "recorded rather than dropped."
                )

            if row["n_used"] is None:  # every measure but ridge drops jointly
                dropped = int(
                    (np.isnan(X.X).any(axis=1) | np.isnan(Y.X).any(axis=1)).sum()
                )
                row.update(n_used=n_rows - dropped, n_dropped=dropped)

            rows.append(row)
            done += 1
            if progress is not None:
                progress(done, total, f"{measure} {a} -> {b}")

    manifest = pd.DataFrame(
        [
            {
                "space": n,
                "dim": spaces[n].dim,
                "n": spaces[n].n,
                "participation_ratio": participation_ratio(spaces[n].X),
                "pr_fraction": participation_ratio(spaces[n].X) / spaces[n].dim,
                "modality": spaces[n].modality,
                "extractor": spaces[n].extractor,
                "n_replicates": spaces[n].n_replicates,
                "n_nan_rows": nan_counts[n],
            }
            for n in names
        ],
        columns=list(MANIFEST_COLUMNS),
    )

    config = {
        "measures": wanted,
        "k": k,
        "n_splits": n_splits,
        "grouped": groups is not None,
        "rsa_metric": rsa_metric,
        "knn_metric": knn_metric,
        "n_permutations": n_permutations,
        "block_size": block_size,
        "random_state": random_state,
        "alphas": [float(a) for a in alphas],
        "n_spaces": len(names),
        "n_rows": n_rows,
        "n_pairs_ordered": len(ordered),
        "n_pairs_unordered": len(unordered),
    }
    return GeometryResult(
        pairs=pd.DataFrame(rows, columns=list(PAIR_COLUMNS)),
        manifest=manifest,
        config=config,
    )


def write_geometry(
    result: GeometryResult,
    outdir: str | Path,
    *,
    name: str = "space_geometry",
    inputs: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    extra: Mapping | None = None,
) -> dict:
    """Write the pair table, the manifest, and a sidecar naming both.

    The sidecar carries ``schema_version`` for the same reason Contract B's
    do: a consumer that reads these columns needs to fail loudly when they
    change, not silently read a renamed one as missing.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pairs_path = out / f"{name}.pairs.parquet"
    manifest_path = out / f"{name}.manifest.parquet"
    meta_path = out / f"{name}.meta.json"

    result.pairs.to_parquet(pairs_path, index=False)
    result.manifest.to_parquet(manifest_path, index=False)

    meta = {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "table": "space_geometry",
        "extractor": "psytwill",
        "extractor_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": result.config,
        "outputs": {
            "pairs": {
                "path": str(pairs_path.resolve()),
                "rows": int(len(result.pairs)),
                "columns": list(PAIR_COLUMNS),
            },
            "manifest": {
                "path": str(manifest_path.resolve()),
                "rows": int(len(result.manifest)),
                "columns": list(MANIFEST_COLUMNS),
            },
        },
        "inputs": list(inputs) if inputs else [],
        "n_labels": len(labels) if labels is not None else None,
        **(dict(extra) if extra else {}),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "pairs": str(pairs_path),
        "manifest": str(manifest_path),
        "meta_path": str(meta_path),
    }
