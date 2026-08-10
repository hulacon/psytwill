"""Orchestration: CSV(s) in -> matrix stack + series + sidecar out.

``build_quilt`` is the single entry point the CLI wraps: read inputs,
detect spaces, compute one matrix per space (or space pair in cross
mode), derive the transition/diagonal series, write everything into the
output directory, and return a summary for display.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from psytwill.exceptions import InputError, SpaceError
from psytwill.matrices import (
    MatrixResult,
    compute_matrix,
    diagonal_records,
    resolve_labels,
    transition_records,
)
from psytwill.metadata import build_metadata
from psytwill.sidecar import load_sidecar, model_checkpoint
from psytwill.spaces import SpaceConfig, detect_spaces, match_spaces


def read_scores(path: str | Path) -> pd.DataFrame:
    """Read a sibling scores CSV/TSV."""
    path = Path(path)
    if not path.exists():
        raise InputError(f"Input file not found: {path}")
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(path, sep=sep)
    except Exception as exc:
        raise InputError(f"Could not read {path}: {exc}") from exc
    if df.empty:
        raise InputError(f"{path} contains no rows.")
    return df


def _select_pairs(
    pairs: list[tuple[SpaceConfig, SpaceConfig | None]],
    requested: dict[str, str | None] | None,
    available: list[str],
) -> list[tuple[tuple[SpaceConfig, SpaceConfig | None], str | None]]:
    """Filter pairs by requested A-side space names; attach per-space metrics."""
    if requested is None:
        return [(pair, None) for pair in pairs]
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise SpaceError(
            f"Requested space(s) not found: {', '.join(unknown)}. "
            f"Available: {', '.join(available)}."
        )
    return [
        (pair, requested[pair[0].name])
        for pair in pairs
        if pair[0].name in requested
    ]


def _assert_checkpoints(
    pairs: list[tuple[SpaceConfig, SpaceConfig | None]],
    meta_a: dict | None,
    meta_b: dict | None,
) -> None:
    """Refuse cross-input pairs whose recorded checkpoints differ.

    A shared feature space (clip_text x clip, or clip x clip from two
    runs) is only comparable if both sides used identical weights
    (Contract B §4.1). Inputs without sidecar checkpoints are legacy and
    pass unchecked.
    """
    for space_a, space_b in pairs:
        if space_b is None:
            continue
        ck_a = model_checkpoint(meta_a, space_a.name)
        ck_b = model_checkpoint(meta_b, space_b.name)
        if ck_a and ck_b and ck_a != ck_b:
            raise SpaceError(
                f"Checkpoint mismatch for {space_a.name} x {space_b.name}: "
                f"input A used {ck_a!r}, input B used {ck_b!r}. These do "
                "not live in one representational space; re-extract with "
                "matching checkpoints."
            )


def build_quilt(
    input_a: str | Path,
    input_b: str | Path | None = None,
    output_dir: str | Path = "psytwill_out",
    spaces: dict[str, str | None] | None = None,
    metric: str | None = None,
    distance: bool = False,
    diagonal: bool = False,
) -> dict[str, Any]:
    """Compute and write the full matrix stack. Returns a summary dict.

    Parameters
    ----------
    spaces : dict or None
        Space-name -> metric-override (None = default). None = all
        detected spaces. Selection is by A-side name; in cross mode a
        selected name keeps both its same-name and compatible pairs.
    metric : str or None
        Global metric override (per-space overrides win).
    distance : bool
        Convert similarity metrics to distance form (1 - sim).
    diagonal : bool
        Cross mode only: also write the aligned-pairs diagonal series
        (requires equal row counts).
    """
    df_a = read_scores(input_a)
    meta_a = load_sidecar(input_a)
    spaces_a = detect_spaces(df_a.columns)
    cross = input_b is not None

    if cross:
        df_b = read_scores(input_b)
        meta_b = load_sidecar(input_b)
        spaces_b = detect_spaces(df_b.columns)
        pairs = match_spaces(spaces_a, spaces_b)
        _assert_checkpoints(pairs, meta_a, meta_b)
        if not pairs:
            raise SpaceError(
                "No shared or compatible spaces between the two inputs. "
                f"Input A has: {', '.join(spaces_a) or '(none)'}; "
                f"input B has: {', '.join(spaces_b) or '(none)'}."
            )
    else:
        df_b, spaces_b = None, None
        if not spaces_a:
            raise SpaceError(
                f"No feature spaces detected in {input_a}. Expected "
                "embedding columns ({model}_000...) or known profile "
                "columns (emotion_*, sentiment_*, ...)."
            )
        pairs = [(sa, None) for sa in spaces_a.values()]

    available = sorted({pair[0].name for pair in pairs})
    selected = _select_pairs(pairs, spaces, available)

    results: list[MatrixResult] = []
    for (space_a, space_b), space_metric in selected:
        results.append(
            compute_matrix(
                df_a,
                space_a,
                metric_name=space_metric or metric,
                df_b=df_b,
                space_b=space_b,
                distance=distance,
            )
        )

    if diagonal and not cross:
        raise InputError("--diagonal requires two inputs (cross mode).")

    # Series: transitions (self) or optional diagonal (cross)
    series_records: list[dict] = []
    series_file = series_kind = None
    if not cross and len(df_a) >= 2:
        series_kind, series_file = "transitions", "transitions.csv"
        for r in results:
            series_records.extend(transition_records(r))
    elif cross and diagonal:
        series_kind, series_file = "diagonal", "diagonal.csv"
        for r in results:
            series_records.extend(diagonal_records(r))

    # Write everything
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_files = {}
    for r in results:
        filename = f"{r.key}.csv"
        r.frame.to_csv(output_dir / filename, index_label="label")
        matrix_files[r.key] = filename
    if series_file:
        pd.DataFrame.from_records(series_records).to_csv(
            output_dir / series_file, index=False
        )

    inputs_meta = []
    input_metas = [(input_a, df_a, meta_a)] + (
        [(input_b, df_b, meta_b)] if cross else []
    )
    for p, df, upstream in input_metas:
        entry = {
            "path": str(Path(p).resolve()),
            "rows": len(df),
            "label_column": resolve_labels(df)[1],
        }
        if upstream:
            entry["extractor"] = upstream.get("extractor")
            entry["extractor_version"] = upstream.get("extractor_version")
            entry["schema_version"] = upstream.get("schema_version")
            checkpoints = {
                name: model_checkpoint(upstream, name)
                for name in upstream.get("models", {})
                if model_checkpoint(upstream, name)
            }
            if checkpoints:
                entry["checkpoints"] = checkpoints
        inputs_meta.append(entry)
    meta = build_metadata(
        inputs_meta, results, matrix_files, series_file, series_kind,
        len(series_records),
    )
    with open(output_dir / "matrices.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "output_dir": str(output_dir),
        "mode": meta["mode"],
        "results": results,
        "matrix_files": matrix_files,
        "series_file": series_file,
        "series_kind": series_kind,
        "meta": meta,
    }
