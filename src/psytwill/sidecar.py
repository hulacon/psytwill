"""Upstream sidecar (.meta.json) ingestion — Contract B convention §4.1.

Extractor sidecars carry ``schema_version``, ``extractor``, and per-model
``checkpoint`` identifiers (see mmmdata-agents
``docs/constellation-contracts.md`` §4.1). psytwill uses them to
(a) fail loudly on sidecars from an incompatible schema major version and
(b) assert checkpoint equality before pairing cross-modal
``COMPATIBLE_SPACES`` — a shared feature space is only real if both sides
were embedded with the same weights.

CSVs without a sidecar are legacy inputs: allowed with a warning, exempt
from schema and checkpoint checks.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from psytwill.exceptions import InputError

SUPPORTED_SCHEMA_MAJOR = 1

# Table suffixes of the siblings' stem-family convention (`-o scores.csv`
# writes scores_frames.csv etc. next to scores.meta.json): strip them to
# find the family sidecar. Longest-match order matters for _transcript_words.
_TABLE_SUFFIXES = (
    "_transcript_words",
    "_transcript",
    "_frames",
    "_recall",
    "_beats",
    "_speakers",
    "_words",
    "_chunks",
)


def find_sidecar(csv_path: str | Path) -> Path | None:
    """Locate the .meta.json for a scores CSV, if one exists on disk."""
    csv_path = Path(csv_path)
    candidates = [csv_path.with_suffix(".meta.json")]
    stem = csv_path.stem
    for suffix in _TABLE_SUFFIXES:
        if stem.endswith(suffix):
            candidates.append(
                csv_path.with_name(stem[: -len(suffix)] + ".meta.json")
            )
            break
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def load_sidecar(csv_path: str | Path) -> dict[str, Any] | None:
    """Load and validate the sidecar for a scores CSV.

    Returns None (with a warning) when no sidecar exists; raises
    InputError on an unreadable sidecar or an unsupported
    ``schema_version`` major (fail loudly with the version found).
    """
    path = find_sidecar(csv_path)
    if path is None:
        warnings.warn(
            f"No .meta.json sidecar found for {csv_path}; treating as a "
            "legacy input (no schema or checkpoint checks).",
            stacklevel=2,
        )
        return None
    try:
        with open(path) as f:
            meta = json.load(f)
    except Exception as exc:
        raise InputError(f"Could not read sidecar {path}: {exc}") from exc
    version = meta.get("schema_version")
    if version is not None:
        major = str(version).split(".", 1)[0]
        if not major.isdigit() or int(major) != SUPPORTED_SCHEMA_MAJOR:
            raise InputError(
                f"{path} declares schema_version {version!r}; this psytwill "
                f"understands major version {SUPPORTED_SCHEMA_MAJOR} only. "
                "Upgrade psytwill or re-extract with a matching extractor."
            )
    return meta


def model_checkpoint(sidecar: dict[str, Any] | None, space_name: str) -> str | None:
    """The checkpoint recorded for a space's model, if any.

    Space prefixes equal the extractors' model registry names, so the
    sidecar ``models`` map is keyed by the same string.
    """
    if not sidecar:
        return None
    entry = sidecar.get("models", {}).get(space_name)
    if not isinstance(entry, dict):
        return None
    checkpoint = entry.get("checkpoint")
    return checkpoint if isinstance(checkpoint, str) else None
