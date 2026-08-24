"""Read the long-form feature table back into per-space matrices.

:mod:`psytwill.features` writes the Contract B aggregate surface: one tidy
table keyed ``(stimulus_id, model, feature[, time, onset, offset, voice,
chunk_idx, word_idx])``. This module reads it back the other way, producing
the ``(n_stimuli, n_features)`` matrix per model that :mod:`psytwill.compare`
needs. ``features`` melts; ``store`` unmelts.

The three rules here are not conveniences — each one was measured against the
real store, and skipping any of them puts a wrong number in the output rather
than raising:

**Declared-string families are not spaces.** A model whose ``value`` is null
throughout and whose ``value_str`` carries text (a BLIP caption, a human
annotation) is the *input* to some other model's embedding, not a geometry.
Loaded naively it becomes an all-NaN matrix that silently drops every row.

**Replicates pool, they do not stack.** Five human captions describe one
image; five rows share one ``stimulus_id`` and differ only in ``chunk_idx``.
Pooling keeps the row index the stimulus. Stacking would quietly change what
a row *is*, and every downstream measure assumes rows are stimuli.

**Identical spaces must be deduped.** The same model is legitimately present
in more than one group file — ``ebind`` appears in both the image group and
its own group, byte-identical. Left in, a space trivially predicts itself and
the comparison matrix acquires a meaningless perfect cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from psytwill.exceptions import InputError, SpaceError

DEFAULT_KEY: tuple[str, ...] = ("stimulus_id",)
VALUE_COLUMNS = ("value", "value_str")
META_COLUMNS = ("modality", "extractor", "extractor_version")


@dataclass
class SpaceMatrix:
    """One feature space, rows aligned to a stimulus key."""

    name: str
    labels: list[str]
    X: np.ndarray
    features: list[str]
    modality: str | None = None
    extractor: str | None = None
    n_replicates: int = 1
    """Max source rows collapsed into one output row (1 = no pooling happened)."""

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def dim(self) -> int:
        return self.X.shape[1]

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"SpaceMatrix({self.name!r}, n={self.n}, dim={self.dim})"


@dataclass
class LoadReport:
    """What the loader did, so a driver can record it rather than infer it."""

    loaded: list[str] = field(default_factory=list)
    skipped_string: list[str] = field(default_factory=list)
    skipped_empty: list[str] = field(default_factory=list)
    pooled: dict[str, int] = field(default_factory=dict)
    deduped: dict[str, str] = field(default_factory=dict)
    """Dropped space -> the identical space it duplicated."""


def _read(path: Path, columns: Sequence[str], model: str | None = None) -> pd.DataFrame:
    filters = [("model", "=", model)] if model else None
    try:
        return pd.read_parquet(path, columns=list(columns), filters=filters)
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise InputError(
            f"Could not read '{path}' ({exc}). Parquet input needs pyarrow; "
            "install pyarrow, or aggregate to .csv first."
        ) from exc


def _schema_names(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema_arrow.names)


def model_inventory(path: str | Path) -> pd.DataFrame:
    """Models in a table, with feature counts and whether they are string-valued.

    Cheap: reads four columns, never the values matrix.
    """
    p = Path(path)
    df = _read(p, ["model", "feature", "value", "value_str"])
    g = df.groupby("model", dropna=False)
    out = pd.DataFrame(
        {
            "n_features": g["feature"].nunique(),
            "n_rows": g.size(),
            "n_numeric": g["value"].count(),
            "n_string": g["value_str"].count(),
        }
    ).reset_index()
    out["is_string"] = (out.n_numeric == 0) & (out.n_string > 0)
    return out.sort_values("n_features", ascending=False).reset_index(drop=True)


def load_spaces(
    path: str | Path,
    *,
    key: Sequence[str] = DEFAULT_KEY,
    models: Iterable[str] | None = None,
    pool: Literal["mean"] | None = "mean",
    prefix: str | None = None,
    report: LoadReport | None = None,
) -> dict[str, SpaceMatrix]:
    """Load each model in a long-form table as a :class:`SpaceMatrix`.

    ``key`` is the row grain: ``("stimulus_id",)`` for an image set,
    ``("stimulus_id", "time")`` for a movie grid. Rows sharing a key are
    pooled (replicate captions, repeated presentations); ``pool=None`` refuses
    them instead, which is the right setting when a duplicate key would mean
    the grain is wrong rather than replicated.

    ``prefix`` namespaces the returned keys (``"image:clip"``), so spaces from
    several group files can be merged without collision.
    """
    p = Path(path)
    if not p.exists():
        raise InputError(f"No such feature table: '{p}'.")
    rep = report if report is not None else LoadReport()

    available = _schema_names(p)
    missing = [k for k in key if k not in available]
    if missing:
        raise SpaceError(
            f"'{p.name}' has no column(s) {missing}, so it cannot be keyed that "
            f"way. Available key columns: "
            f"{sorted(available & set(('stimulus_id','voice','time','onset','offset','chunk_idx','word_idx')))}."
        )

    inv = model_inventory(p)
    wanted = set(models) if models is not None else set(inv["model"].dropna())
    spaces: dict[str, SpaceMatrix] = {}

    for row in inv.itertuples():
        if row.model not in wanted or pd.isna(row.model):
            continue
        name = f"{prefix}:{row.model}" if prefix else str(row.model)
        if row.is_string:
            rep.skipped_string.append(name)
            continue
        if row.n_numeric == 0:
            rep.skipped_empty.append(name)
            continue

        cols = list(key) + ["feature", "value"] + [
            c for c in META_COLUMNS if c in available
        ]
        df = _read(p, cols, model=str(row.model))
        wide = (
            df.groupby(list(key) + ["feature"], dropna=False)["value"]
            .mean()
            .unstack("feature")
            .sort_index()
        )
        n_rep = int(df.groupby(list(key) + ["feature"], dropna=False).size().max())
        if n_rep > 1 and pool is None:
            raise SpaceError(
                f"'{name}' has up to {n_rep} rows per key {tuple(key)}. Either "
                "pool them (pool='mean') or key at a finer grain — e.g. add "
                "'chunk_idx' to distinguish replicate captions."
            )
        if n_rep > 1:
            rep.pooled[name] = n_rep

        wide = wide.reindex(sorted(wide.columns), axis=1)
        labels = [
            str(i) if not isinstance(i, tuple) else "|".join(str(x) for x in i)
            for i in wide.index
        ]
        spaces[name] = SpaceMatrix(
            name=name,
            labels=labels,
            X=wide.to_numpy(dtype=float),
            features=[str(c) for c in wide.columns],
            modality=str(df["modality"].iloc[0]) if "modality" in df else None,
            extractor=str(df["extractor"].iloc[0]) if "extractor" in df else None,
            n_replicates=n_rep,
        )
        rep.loaded.append(name)

    return spaces


def align_spaces(
    spaces: dict[str, SpaceMatrix]
) -> tuple[dict[str, SpaceMatrix], list[str]]:
    """Restrict every space to the labels all of them share, preserving order.

    Coverage is genuinely uneven — transcript spaces exist only for clips with
    speech — so the intersection is computed rather than assumed, and the
    surviving labels are returned for the caller to record as the *n* the
    measures actually ran on.
    """
    if not spaces:
        return {}, []
    common = set.intersection(*(set(s.labels) for s in spaces.values()))
    if not common:
        raise SpaceError(
            "Spaces share no labels. Check they were keyed at the same grain "
            "and come from the same stimulus set."
        )
    labels = sorted(common)
    out = {}
    for name, s in spaces.items():
        idx = {lab: i for i, lab in enumerate(s.labels)}
        take = [idx[lab] for lab in labels]
        out[name] = SpaceMatrix(
            name=s.name,
            labels=labels,
            X=s.X[take],
            features=s.features,
            modality=s.modality,
            extractor=s.extractor,
            n_replicates=s.n_replicates,
        )
    return out, labels


def dedupe_spaces(
    spaces: dict[str, SpaceMatrix], *, report: LoadReport | None = None
) -> dict[str, SpaceMatrix]:
    """Drop spaces whose matrices are identical to one already kept.

    Same model, two group files, byte-identical values: without this the
    comparison matrix gains a cell where a space perfectly predicts itself
    under a different name.
    """
    rep = report if report is not None else LoadReport()
    kept: dict[str, SpaceMatrix] = {}
    for name, s in spaces.items():
        dup = next(
            (
                k
                for k, other in kept.items()
                if other.X.shape == s.X.shape
                and other.features == s.features
                and np.array_equal(other.X, s.X, equal_nan=True)
            ),
            None,
        )
        if dup is None:
            kept[name] = s
        else:
            rep.deduped[name] = dup
    return kept
