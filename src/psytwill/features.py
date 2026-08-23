"""Long-form feature table — Contract B's `features` surface (§4.2.4).

``psytwill features`` aggregates N extractor CSVs (plus their §4.1
sidecars) into one tidy long table, the feature surface Contract B
promises downstream consumers alongside the relational matrices:

    (stimulus_id, modality, extractor, extractor_version, model, feature
     [, time, onset, offset, voice]) -> value | value_str

Every output row carries the full fixed column set (missing keys are
null), so the schema is stable regardless of which inputs were mixed.
Numeric features land in ``value``; declared-string features (e.g.
``caption_text``) land in ``value_str`` rather than being dropped.

The table carries its own ``schema_version`` (FEATURES_SCHEMA_VERSION),
independent of the §4.1 extractor-sidecar schema the inputs declare.
Output format follows the output suffix: ``.parquet`` (preferred — §4.1's
CSV-only rule binds extractors, not this aggregate surface) or ``.csv``.
A ``<stem>.meta.json`` sidecar is always written alongside.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from psytwill import __version__
from psytwill.exceptions import InputError, SpaceError
from psytwill.matrices import resolve_labels
from psytwill.sidecar import load_sidecar, model_checkpoint
from psytwill.spaces import INDEX_COLUMNS, detect_embedding_spaces

FEATURES_SCHEMA_VERSION = "1.0"

# Reserved columns that survive as keys in the long table; the remaining
# INDEX_COLUMNS are row-identity implementation details and are dropped
# (stimulus_id is handled separately as the primary key).
CARRIED_KEYS = ["voice", "time", "onset", "offset"]

# Fixed output schema, in column order.
OUTPUT_COLUMNS = [
    "stimulus_id", "voice", "time", "onset", "offset",
    "modality", "extractor", "extractor_version", "model", "feature",
    "value", "value_str",
]

# The key that must be unique across all inputs (everything but the values).
KEY_COLUMNS = ["stimulus_id", "voice", "time", "onset", "offset", "model", "feature"]

# Extractor package -> modality of the stimuli it reads.
MODALITY_MAP = {"viz2psy": "visual", "aud2psy": "audio", "word2psy": "text"}


def _model_prefixes(sidecar: dict[str, Any] | None) -> list[tuple[str, str]]:
    """(prefix, model) pairs from the sidecar's models map, longest first.

    §4.1: the default prefix is the model's registry name; a model may
    declare additional ``prefixes`` (places' ``sunattr``). Longest-match
    keeps ``clip_text_###`` out of ``clip``.
    """
    pairs: list[tuple[str, str]] = []
    for name, entry in (sidecar or {}).get("models", {}).items():
        prefixes = entry.get("prefixes", []) if isinstance(entry, dict) else []
        for prefix in list(prefixes) + [name]:
            pairs.append((prefix, name))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def _resolve_models(
    columns: list[str],
    sidecar: dict[str, Any] | None,
) -> tuple[dict[str, str], list[str]]:
    """Map each feature column to its model; return (mapping, unattributed).

    Sidecar-declared prefixes win (longest match); columns of a detected
    embedding space fall back to the space prefix (legacy inputs). What
    remains is kept in the table with a null model, and reported.
    """
    prefixes = _model_prefixes(sidecar)
    embedded = {
        col: space.name
        for space in detect_embedding_spaces(columns).values()
        for col in space.columns
    }
    mapping: dict[str, str] = {}
    unattributed: list[str] = []
    for col in columns:
        for prefix, model in prefixes:
            if col == prefix or col.startswith(prefix + "_"):
                mapping[col] = model
                break
        else:
            if col in embedded:
                mapping[col] = embedded[col]
            else:
                unattributed.append(col)
    return mapping, unattributed


def _melt_input(
    path: str | Path,
    modality_map: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One extractor CSV -> long rows + a provenance entry for the sidecar."""
    from psytwill.pipeline import read_scores

    df = read_scores(path)
    meta = load_sidecar(path)

    if "stimulus_id" in df.columns:
        ids, label_column = df["stimulus_id"].astype(str), "stimulus_id"
    else:
        labels, label_column = resolve_labels(df)
        ids = pd.Series(labels, index=df.index)
        warnings.warn(
            f"{path} has no stimulus_id column (Contract B §4.1); using "
            f"{label_column} as the stimulus_id.",
            stacklevel=2,
        )

    feature_cols = [c for c in df.columns if c not in INDEX_COLUMNS]
    if not feature_cols:
        raise InputError(f"{path} has no feature columns (only reserved ones).")
    models, unattributed = _resolve_models(feature_cols, meta)
    if unattributed:
        warnings.warn(
            f"{path}: {len(unattributed)} feature column(s) match no "
            f"sidecar model prefix or embedding space "
            f"({', '.join(unattributed[:5])}{', ...' if len(unattributed) > 5 else ''}); "
            "kept with a null model.",
            stacklevel=2,
        )

    extractor = meta.get("extractor") if meta else None
    modality = modality_map.get(extractor) if extractor else None

    keys = pd.DataFrame({"stimulus_id": ids})
    for col in CARRIED_KEYS:
        keys[col] = df[col] if col in df.columns else None

    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    strings = [c for c in feature_cols if c not in numeric]
    parts = []
    for cols, value_column in ((numeric, "value"), (strings, "value_str")):
        if not cols:
            continue
        part = pd.concat([keys, df[cols]], axis=1).melt(
            id_vars=list(keys.columns),
            value_vars=cols,
            var_name="feature",
            value_name=value_column,
        )
        if value_column == "value_str":
            part["value_str"] = part["value_str"].astype("string")
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)

    long["model"] = long["feature"].map(models)
    long["modality"] = modality
    long["extractor"] = extractor
    long["extractor_version"] = meta.get("extractor_version") if meta else None
    for col in ("value", "value_str"):
        if col not in long.columns:
            long[col] = pd.Series(dtype="float64" if col == "value" else "string")

    entry: dict[str, Any] = {
        "path": str(Path(path).resolve()),
        "rows": len(df),
        "label_column": label_column,
        "extractor": extractor,
        "extractor_version": meta.get("extractor_version") if meta else None,
        "schema_version": meta.get("schema_version") if meta else None,
        "modality": modality,
        "n_feature_columns": len(feature_cols),
        "models": {
            name: {"checkpoint": model_checkpoint(meta, name)}
            for name in sorted(set(models.values()))
        },
    }
    if unattributed:
        entry["unattributed_columns"] = unattributed
    return long, entry


def _assert_checkpoints(entries: list[dict[str, Any]]) -> None:
    """Refuse one table mixing two checkpoints under one model name."""
    seen: dict[str, tuple[str, str]] = {}  # model -> (checkpoint, path)
    for entry in entries:
        for model, info in entry["models"].items():
            checkpoint = info.get("checkpoint")
            if not checkpoint:
                continue
            if model in seen and seen[model][0] != checkpoint:
                raise SpaceError(
                    f"Checkpoint mismatch for model {model!r}: "
                    f"{seen[model][1]} used {seen[model][0]!r}, "
                    f"{entry['path']} used {checkpoint!r}. One features "
                    "table cannot mix them; re-extract with matching "
                    "checkpoints or build separate tables."
                )
            seen.setdefault(model, (checkpoint, entry["path"]))


def _key_hash(table: pd.DataFrame) -> np.ndarray:
    """A uint64 hash per row over KEY_COLUMNS.

    The whole table never has to be resident to check key uniqueness -- 8
    bytes a row is enough to find every *candidate* collision, and the exact
    check then runs only on those. For the MMMData store that is 1.4 GB of
    hashes instead of 78 GB of melted rows.
    """
    return pd.util.hash_pandas_object(
        table[KEY_COLUMNS], index=False
    ).to_numpy(dtype="uint64", copy=False)


def _duplicate_examples(
    inputs: list[str | Path],
    per_input_hashes: list[np.ndarray],
    suspects: np.ndarray,
    modality_map: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    """Exact duplicate rows behind a set of colliding hashes.

    Only the inputs that actually carry a suspect hash are re-melted, and
    only their suspect rows are kept, so this stays small even when the
    aggregate is hundreds of millions of rows.
    """
    frames, sources = [], []
    for path, hashes in zip(inputs, per_input_hashes):
        hit = np.isin(hashes, suspects)
        if not hit.any():
            continue
        long, _ = _melt_input(path, modality_map)
        rows = long.loc[hit, KEY_COLUMNS].copy()
        rows["_input"] = str(path)
        frames.append(rows)
        sources.append(str(path))
    if not frames:
        return pd.DataFrame(columns=[*KEY_COLUMNS, "_input"]), []
    return pd.concat(frames, ignore_index=True), sources


def _assert_unique_streaming(
    inputs: list[str | Path],
    per_input_hashes: list[np.ndarray],
    modality_map: dict[str, str],
) -> None:
    """Refuse silent duplicate keys, without holding the table in memory."""
    if not per_input_hashes:
        return
    all_hashes = np.concatenate(per_input_hashes)
    uniq, counts = np.unique(all_hashes, return_counts=True)
    suspects = uniq[counts > 1]
    if not len(suspects):
        return

    # Hash collisions are possible, so confirm against the real key tuples
    # before refusing. Only suspect rows are materialised.
    rows, sources = _duplicate_examples(
        inputs, per_input_hashes, suspects, modality_map
    )
    duplicated = rows.duplicated(subset=KEY_COLUMNS, keep=False)
    if not duplicated.any():
        return
    dup = rows.loc[duplicated]
    examples = "; ".join(
        "(" + ", ".join(
            f"{k}={v!r}" for k, v in row.items()
            if k != "_input" and pd.notna(v)
        ) + ")"
        for _, row in dup[KEY_COLUMNS].head(3).iterrows()
    )
    raise InputError(
        f"{int(duplicated.sum())} duplicate feature key(s) across inputs "
        f"{sorted(dup['_input'].unique())}, e.g. {examples}. Each "
        "(stimulus_id, model, feature[, time/onset/offset/voice]) may "
        "appear once; drop the overlapping input or disambiguate upstream."
    )


def _assert_unique(table: pd.DataFrame, sources: pd.Series) -> None:
    """Refuse silent duplicate keys (same feature for one stimulus twice).

    Retained for callers holding a whole table; ``build_features`` uses the
    streaming path instead.
    """
    duplicated = table.duplicated(subset=KEY_COLUMNS, keep=False)
    if not duplicated.any():
        return
    rows = table.loc[duplicated, KEY_COLUMNS].head(3)
    examples = "; ".join(
        "(" + ", ".join(f"{k}={v!r}" for k, v in row.items() if pd.notna(v)) + ")"
        for _, row in rows.iterrows()
    )
    raise InputError(
        f"{int(duplicated.sum())} duplicate feature key(s) across inputs "
        f"{sorted(sources[duplicated].unique())}, e.g. {examples}. Each "
        "(stimulus_id, model, feature[, time/onset/offset/voice]) may "
        "appear once; drop the overlapping input or disambiguate upstream."
    )


OUTPUT_DTYPES = (
    {"time": "float64", "onset": "float64", "offset": "float64",
     "value": "float64", "value_str": "string"}
    | {c: "string" for c in ("stimulus_id", "voice", "modality",
                             "extractor", "extractor_version",
                             "model", "feature")}
)


def _coerce(long: pd.DataFrame) -> pd.DataFrame:
    """Fixed column order and dtypes, so every row group shares one schema."""
    return long[OUTPUT_COLUMNS].astype(OUTPUT_DTYPES)


def build_features(
    inputs: list[str | Path],
    output: str | Path,
    modality_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Aggregate extractor CSVs into one long-form feature table.

    Parameters
    ----------
    inputs : list of paths
        Extractor scores CSVs (§4.1 sidecars found automatically;
        sidecar-less inputs are legacy: warned, null provenance).
    output : path
        Output table; ``.parquet`` or ``.csv`` decides the format.
        ``<stem>.meta.json`` is written alongside.
    modality_map : dict or None
        Extractor-package -> modality overrides, merged over the default
        {viz2psy: visual, aud2psy: audio, word2psy: text}.
    """
    output = Path(output)
    fmt = output.suffix.lower().lstrip(".")
    if fmt not in ("parquet", "csv"):
        raise InputError(
            f"Output suffix {output.suffix!r} not supported; use "
            ".parquet (preferred) or .csv."
        )
    merged_map = {**MODALITY_MAP, **(modality_map or {})}
    inputs = list(inputs)

    if fmt == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise InputError(
                f"Writing parquet needs pyarrow ({exc}); pip install "
                "pyarrow, or use a .csv output suffix."
            ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    # Written to a sidecar path and renamed, so a refusal part-way through
    # never leaves a truncated table where a complete one used to be.
    partial = output.with_name(output.name + ".partial")

    entries: list[dict[str, Any]] = []
    per_input_hashes: list[np.ndarray] = []
    stimulus_ids: set[str] = set()
    models_seen: set[str] = set()
    n_rows = 0
    writer = None
    wrote_csv_header = False

    try:
        for path in inputs:
            long, entry = _melt_input(path, merged_map)
            entries.append(entry)
            # Incremental, so a checkpoint clash is refused at the input that
            # introduces it rather than after every input has been melted.
            _assert_checkpoints(entries)

            table = _coerce(long)
            del long
            per_input_hashes.append(_key_hash(table))
            n_rows += len(table)
            stimulus_ids.update(table["stimulus_id"].dropna().unique().tolist())
            models_seen.update(table["model"].dropna().unique().tolist())

            if fmt == "parquet":
                batch = pa.Table.from_pandas(table, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(partial, batch.schema)
                writer.write_table(batch)
                del batch
            else:
                table.to_csv(
                    partial, index=False, float_format="%.6g",
                    mode="a" if wrote_csv_header else "w",
                    header=not wrote_csv_header,
                )
                wrote_csv_header = True
            del table

        if writer is not None:
            writer.close()
            writer = None
        elif fmt == "parquet":
            # No inputs produced rows; still emit a schema-correct empty file.
            empty = _coerce(pd.DataFrame(columns=OUTPUT_COLUMNS))
            pq.write_table(
                pa.Table.from_pandas(empty, preserve_index=False), partial
            )
        elif not wrote_csv_header:
            _coerce(pd.DataFrame(columns=OUTPUT_COLUMNS)).to_csv(
                partial, index=False
            )

        _assert_unique_streaming(inputs, per_input_hashes, merged_map)
    except BaseException:
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)
        raise

    os.replace(partial, output)

    meta = {
        "schema_version": FEATURES_SCHEMA_VERSION,
        "table": "features",
        "extractor": "psytwill",
        "extractor_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": {
            "path": str(output.resolve()),
            "format": fmt,
            "rows": n_rows,
            "columns": OUTPUT_COLUMNS,
            "key_columns": KEY_COLUMNS,
        },
        "inputs": entries,
        "n_stimuli": len(stimulus_ids),
        "models": sorted(models_seen),
    }
    meta_path = output.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "output": str(output),
        "meta_path": str(meta_path),
        "rows": n_rows,
        "n_stimuli": meta["n_stimuli"],
        "models": meta["models"],
        "inputs": entries,
        "meta": meta,
    }
