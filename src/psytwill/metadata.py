"""Provenance sidecar (`matrices.meta.json`), viz2psy-style.

Embedding spaces are described pattern-based (``"pattern": "minilm_{NNN}"``
with a range); profile spaces list their full column set.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from psytwill import __version__
from psytwill.matrices import MatrixResult
from psytwill.spaces import SpaceConfig


def _space_entry(space: SpaceConfig) -> dict[str, Any]:
    entry: dict[str, Any] = {"kind": space.kind, "n_dims": space.n_dims}
    if space.kind == "embedding":
        entry["pattern"] = f"{space.name}_{{NNN}}"
        entry["range"] = [0, space.n_dims - 1]
    else:
        entry["columns"] = list(space.columns)
    return entry


def build_metadata(
    inputs: list[dict[str, Any]],
    results: list[MatrixResult],
    matrix_files: dict[str, str],
    series_file: str | None,
    series_kind: str | None,
    n_series_rows: int,
) -> dict[str, Any]:
    """Assemble the sidecar dict.

    Parameters
    ----------
    inputs : list of dict
        One per input CSV: ``{"path", "rows", "label_column"}``.
    results : list of MatrixResult
        Computed matrices, in output order.
    matrix_files : dict
        Matrix key -> written filename.
    series_file, series_kind : str or None
        The transitions/diagonal CSV filename and which kind it is.
    """
    cross = len(inputs) == 2
    matrices: dict[str, Any] = {}
    for r in results:
        entry: dict[str, Any] = {
            "file": matrix_files[r.key],
            "metric": r.metric,
            "form": r.form,
            "shape": [int(r.frame.shape[0]), int(r.frame.shape[1])],
            "space": _space_entry(r.space_a) | {"name": r.space_a.name},
        }
        if r.space_b.name != r.space_a.name:
            entry["space_b"] = _space_entry(r.space_b) | {"name": r.space_b.name}
        if cross:
            entry["n_valid"] = {"a": r.n_valid_a, "b": r.n_valid_b}
            if r.nan_labels_a or r.nan_labels_b:
                entry["nan_labels"] = {"a": r.nan_labels_a, "b": r.nan_labels_b}
        else:
            entry["n_valid"] = r.n_valid_a
            if r.nan_labels_a:
                entry["nan_labels"] = r.nan_labels_a
        matrices[r.key] = entry

    meta: dict[str, Any] = {
        "psytwill_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "cross" if cross else "self",
        "inputs": inputs,
        "matrices": matrices,
    }
    if series_file:
        meta[series_kind] = {"file": series_file, "rows": n_series_rows}
    return meta
