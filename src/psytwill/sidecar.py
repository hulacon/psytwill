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


def model_nulls(sidecar: dict[str, Any] | None, space_name: str) -> dict[str, Any] | None:
    """The Contract B 1.1 ``nulls`` map recorded for a space's model.

    ``None`` means the sidecar declares nothing (a 1.0 sidecar, or no
    sidecar): different from ``{}``, which is a producer's positive claim
    that the model never emits NaN.
    """
    if not sidecar:
        return None
    entry = sidecar.get("models", {}).get(space_name)
    if not isinstance(entry, dict) or not isinstance(entry.get("nulls"), dict):
        return None
    return entry["nulls"]


def model_window_sec(sidecar: dict[str, Any] | None, space_name: str) -> float | None:
    """The Contract B ``window_sec`` a model's entry declares, or ``None``.

    ``None`` cannot tell "each row sees only its own grid step" (the contract
    omits the key then) from a sidecar written before producers emitted it,
    so a consumer falls back to the battery registry rather than read it as
    either (see :func:`psytwill.space.resolve_member_windows`).
    """
    if not sidecar:
        return None
    entry = sidecar.get("models", {}).get(space_name)
    if not isinstance(entry, dict) or entry.get("window_sec") is None:
        return None
    return float(entry["window_sec"])


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


#: The parts of an extractor sidecar's ``input.framing`` that define the
#: framing *rule* (viz2psy >= 0.17.0, Contract B 1.3). Per-file values (the
#: crop box, sizes) legitimately differ between inputs; the rule must not.
FRAMING_RULE_KEYS = ("stage0", "stage0_version", "framing_table_sha256")
STILL_FRAME_SOURCE = "still image"
UNDECLARED = "undeclared"
#: viz2psy input types read from video: one file, or a video set (viz2psy
#: >= 0.20.0, several files under one framing rule recorded once).
VIDEO_INPUT_TYPES = ("video", "video_set")


def input_framing_rule(sidecar: dict[str, Any] | None) -> dict[str, Any] | None:
    """The framing rule one input was extracted under, or ``None`` if it has none.

    * a still-image input: ``{"frame_source": "still image", ...}`` (never
      compared: stills have no bars and square pixels);
    * a video input written before stage 0 existed (no ``input.framing.stage0``):
      ``{"stage0": "undeclared"}``, so it cannot be mixed with a declared one;
    * a video input with stage 0 (one file or a video set, whose rule is
      recorded once): ``frame_source`` + :data:`FRAMING_RULE_KEYS`;
    * anything else (no ``input``, a non-visual extractor): ``None``.
    """
    inp = (sidecar or {}).get("input")
    if not isinstance(inp, dict):
        return None
    fr = inp.get("framing") if isinstance(inp.get("framing"), dict) else {}
    if fr.get("frame_source") == STILL_FRAME_SOURCE:
        return {"frame_source": STILL_FRAME_SOURCE}
    if inp.get("type") not in VIDEO_INPUT_TYPES and "stage0" not in fr:
        return None
    if "stage0" not in fr:
        return {"frame_source": fr.get("frame_source"), "stage0": UNDECLARED}
    return {"frame_source": fr.get("frame_source"), **{k: fr.get(k) for k in FRAMING_RULE_KEYS}}


def framing_rule_key(rule: dict[str, Any] | None) -> tuple | None:
    """What two video inputs must share; ``None`` for stills and non-visual inputs."""
    if rule is None or rule.get("frame_source") == STILL_FRAME_SOURCE:
        return None
    return tuple(rule.get(k) for k in FRAMING_RULE_KEYS)


def describe_framing(rule: dict[str, Any] | None) -> str:
    if rule is None:
        return "no framing record"
    if rule.get("stage0") == UNDECLARED:
        return "no stage 0 (extracted before viz2psy 0.17.0)"
    sha = rule.get("framing_table_sha256")
    return (f"stage 0 {rule.get('stage0')!r} v{rule.get('stage0_version')}"
            + (f", table sha256 {sha[:12]}..." if sha else ""))
