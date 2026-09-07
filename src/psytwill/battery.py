"""The clamped feature battery — psytwill-space v0.1's input declaration.

psytwill-space is fit over a *battery* of extractor spaces (viz2psy, aud2psy,
word2psy). Fitting weights against a battery that keeps changing underneath
would make "the space" uncitable, so the battery is **clamped**: every model
is pinned to the exact checkpoint (§4.1's identity for learned weights), its
feature inventory, and the extractor version it was clamped against. After
the clamp, a model enters or changes only through the admission test
(``psytwill compare`` complementarity) and a battery version bump.

Sequencing decision of record (mmmdata-agents
``docs/workbench/psytwill-space/log.md``, 2026-08-25): additions first
(extractors A-G), then the two-sided declared-vs-emitted audit, then this
clamp, then the block basis and the external-corpus fits. This module is the
clamp. It is a *declaration*, deliberately dependency-free: the extractor
packages are not psytwill dependencies, so the registry cross-check
(:func:`check_registry`) takes the registry's names as an argument and the
contract tests skip it when an extractor is not importable.

Two checks, both directional (a declaration-based gate is a filter with two
sides — the stimfeat-campaign lesson):

- :func:`check_sidecar` — does an emitted sidecar (a §4.1 ``<stem>.meta.json``
  or a psytwill group ``*_features.meta.json``) name only battery models, at
  the pinned checkpoints and prefixes? Runs over an mmmdata store *and* over
  fit-corpus features: external extraction that drifts from the pin is
  refused before it can found a block.
- :func:`check_registry` — does a live extractor registry match the battery
  in both directions (nothing in the battery the package no longer ships,
  nothing shipped that the battery has not admitted or explicitly excluded)?

``dim`` is the model's native feature count — per row of the table it writes
(word-level for word2psy scalar profiles; chunk aggregates multiply that by
the four ``_mean/_sd/_min/_max`` suffixes, a consumer-side fact).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from psytwill.exceptions import BatteryError

BATTERY_VERSION = "0.1"
BATTERY_CLAMPED_ON = "2026-09-07"

#: Extractor versions the battery was clamped against. Outputs already in
#: the mmmdata store span earlier versions of the same checkpoints (aud2psy
#: 0.13.1-0.17.0, viz2psy 0.7.0-0.9.0, word2psy 0.6.0-0.9.0); their
#: changelogs record no feature-output change across those ranges, and the
#: pin that matters is the checkpoint, not the package version.
EXTRACTOR_VERSIONS: dict[str, str] = {
    "viz2psy": "0.9.0",
    "aud2psy": "0.17.0",
    "word2psy": "0.9.0",
}

#: Prefix nestings Contract B §4.1 documents as intentional and resolvable by
#: the greedy longest-match rule: (child, parent).
NESTED_PREFIX_EXCEPTIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("clip_text", "clip"),
        ("clap_text", "clap"),
        ("ebind_text", "ebind"),
        ("ebind_audio", "ebind"),
        ("speech_emotion", "speech"),
        ("speech_rate", "speech"),
    }
)

KINDS = ("embedding", "profile", "text", "events")
MODALITIES = ("visual", "audio", "text")


@dataclass(frozen=True)
class BatteryModel:
    """One battery member, pinned."""

    model: str  # registry name == default column prefix
    extractor: str  # viz2psy | aud2psy | word2psy
    modality: str  # visual | audio | text
    kind: str  # embedding | profile | text | events
    dim: int | None  # native feature count; None for non-numeric tables
    checkpoint: str | None  # None == analytic, no learned weights
    extra_prefixes: tuple[str, ...] = ()
    window_sec: float | None = None
    notes: str = field(default="")

    @property
    def prefixes(self) -> tuple[str, ...]:
        return (self.model,) + self.extra_prefixes

    @property
    def learned(self) -> bool:
        return self.checkpoint is not None


def _m(*args: Any, **kwargs: Any) -> BatteryModel:
    return BatteryModel(*args, **kwargs)


_CLAP = "laion/larger_clap_music_and_speech"
_CLIP = "ViT-B-32/laion2b_s34b_b79k"
_EBIND = "encord-team/ebind-full"
_FASTTEXT = "crawl-300d-2M-subword"

BATTERY: dict[str, BatteryModel] = {
    m.model: m
    for m in (
        # ---- viz2psy 0.9.0 (15) -------------------------------------------
        _m("aesthetics", "viz2psy", "visual", "profile", 1,
           "ViT-L-14/openai+sac+logos+ava1-l14-linearMSE"),
        _m("caption", "viz2psy", "visual", "text", None,
           "Salesforce/blip-image-captioning-large",
           notes="caption_text (dtype string); the word2psy cap: source"),
        _m("clip", "viz2psy", "visual", "embedding", 512, _CLIP),
        _m("depth", "viz2psy", "visual", "profile", 6,
           "depth-anything/Depth-Anything-V2-Small-hf",
           notes="relative depth, per-image normalized; stats only"),
        _m("dinov2", "viz2psy", "visual", "embedding", 768, "dinov2_vitb14",
           notes="sidecar pattern string reads dinov_{NNN}; columns are "
                 "dinov2_###"),
        _m("ebind", "viz2psy", "visual", "embedding", 1024, _EBIND),
        _m("emonet", "viz2psy", "visual", "profile", 20,
           "emonet-pytorch/osf-amdju"),
        _m("faces", "viz2psy", "visual", "profile", 5,
           "opencv_zoo/face_detection_yunet_2023mar",
           notes="presence-gated (NaN without a face); real-photo domain"),
        _m("gist", "viz2psy", "visual", "embedding", 512, None),
        _m("llstat", "viz2psy", "visual", "profile", 17, None),
        _m("motion", "viz2psy", "visual", "profile", 7, None,
           notes="video only (frame pairs); absent from image sources"),
        _m("places", "viz2psy", "visual", "profile", 416,
           "wideresnet18_places365", extra_prefixes=("sunattr",)),
        _m("resmem", "viz2psy", "visual", "profile", 1, "resmem-pretrained"),
        _m("saliency", "viz2psy", "visual", "embedding", 576,
           "DeepGazeIIE-pretrained", notes="saliency_{XX}_{YY} 24x24 grid"),
        _m("yolo", "viz2psy", "visual", "profile", 85, "yolov8n.pt"),
        # ---- aud2psy 0.17.0 (20) ------------------------------------------
        _m("beats", "aud2psy", "audio", "events", 1, "beat_this/final0",
           notes="point events on `time`; pooled through --window"),
        _m("clap", "aud2psy", "audio", "embedding", 512, _CLAP,
           window_sec=10.0),
        _m("conversation", "aud2psy", "audio", "profile", 6, None,
           notes="derived from the diarize turn table; presence-gated"),
        _m("diarize", "aud2psy", "audio", "events", None,
           "pyannote/speaker-diarization-community-1",
           notes="turn table (speaker, onset, offset); a pooling schema and "
                 "conversation's input, not a feature space"),
        _m("ebind_audio", "aud2psy", "audio", "embedding", 1024, _EBIND,
           window_sec=2.0,
           notes="store cells extracted at 0.13.1 predate the window_sec "
                 "sidecar key; 2.0 s is the contract value"),
        _m("egemaps", "aud2psy", "audio", "profile", 25, None,
           notes="speech-gated"),
        _m("loudness", "aud2psy", "audio", "profile", 2, None),
        _m("music_emotion", "aud2psy", "audio", "profile", 2, _CLAP,
           window_sec=10.0),
        _m("onsets", "aud2psy", "audio", "profile", 3, None),
        _m("pitch", "aud2psy", "audio", "profile", 2, None),
        _m("psychoacoustic", "aud2psy", "audio", "profile", 4, None),
        _m("rhythm", "aud2psy", "audio", "profile", 3, None),
        _m("sound_events", "aud2psy", "audio", "profile", 16, _CLAP,
           window_sec=10.0),
        _m("spectral", "aud2psy", "audio", "profile", 5, None),
        _m("speech", "aud2psy", "audio", "profile", 1, "silero_vad_v6.onnx"),
        _m("speech_emotion", "aud2psy", "audio", "profile", 3,
           "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
           window_sec=4.0, notes="speech-gated"),
        _m("speech_rate", "aud2psy", "audio", "profile", 3, None,
           notes="speech-gated; derived from transcribe word timings"),
        _m("timbre", "aud2psy", "audio", "profile", 21, None),
        _m("tonal", "aud2psy", "audio", "profile", 3, None),
        _m("transcribe", "aud2psy", "audio", "text", None, "large-v3",
           notes="Whisper transcript (+ words table); the word2psy "
                 "transcript: source"),
        # ---- word2psy 0.9.0 (13) ------------------------------------------
        _m("clap_text", "word2psy", "text", "embedding", 512, _CLAP,
           notes="backend package_version recorded as 'unknown' in sidecars"),
        _m("clip_text", "word2psy", "text", "embedding", 512, _CLIP),
        _m("ebind_text", "word2psy", "text", "embedding", 1024, _EBIND),
        _m("emotion", "word2psy", "text", "profile", 28,
           "SamLowe/roberta-base-go_emotions"),
        _m("fasttext", "word2psy", "text", "embedding", 300, _FASTTEXT),
        _m("gpt2_surprisal", "word2psy", "text", "profile", 1, "gpt2",
           notes="word-level; chunk tables carry the four aggregates"),
        _m("interaction", "word2psy", "text", "profile", 7, None),
        _m("lexical_norms", "word2psy", "text", "profile", 24,
           f"{_FASTTEXT}+ridge",
           notes="unclipped ridge extrapolations anchored to rating scales; "
                 "animacy CV r2 .719"),
        _m("minilm", "word2psy", "text", "embedding", 384,
           "sentence-transformers/all-MiniLM-L6-v2"),
        _m("readability", "word2psy", "text", "profile", 7, None),
        _m("sentiment", "word2psy", "text", "profile", 3,
           "cardiffnlp/twitter-roberta-base-sentiment-latest"),
        _m("word2vec", "word2psy", "text", "embedding", 300,
           "word2vec-google-news-300"),
        _m("wordform", "word2psy", "text", "profile", 4, None),
    )
}


def _validate(battery: Mapping[str, BatteryModel]) -> None:
    owners: dict[str, str] = {}
    for m in battery.values():
        if m.extractor not in EXTRACTOR_VERSIONS:
            raise BatteryError(f"{m.model}: unknown extractor {m.extractor!r}")
        if m.modality not in MODALITIES or m.kind not in KINDS:
            raise BatteryError(f"{m.model}: bad modality/kind")
        for p in m.prefixes:
            if p in owners:
                raise BatteryError(f"prefix {p!r} owned by both {owners[p]} and {m.model}")
            owners[p] = m.model
    for child in owners:
        for parent in owners:
            if child != parent and child.startswith(parent + "_"):
                if (child, parent) not in NESTED_PREFIX_EXCEPTIONS:
                    raise BatteryError(
                        f"prefix {child!r} nests under {parent!r} outside the "
                        "§4.1 exception set"
                    )


_validate(BATTERY)


def tie_groups(battery: Mapping[str, BatteryModel] = BATTERY) -> dict[str, tuple[str, ...]]:
    """Models sharing one checkpoint string — the cross-modal guarantees.

    A shared space is valid iff the checkpoint strings match (§4.1); these
    groups are what that guarantee currently covers.
    """
    by_ckpt: dict[str, list[str]] = {}
    for m in battery.values():
        if m.checkpoint is not None:
            by_ckpt.setdefault(m.checkpoint, []).append(m.model)
    return {k: tuple(sorted(v)) for k, v in by_ckpt.items() if len(v) > 1}


def to_records(battery: Mapping[str, BatteryModel] = BATTERY) -> list[dict[str, Any]]:
    """The battery as plain dicts (JSON/parquet-ready), sorted by extractor."""
    order = list(EXTRACTOR_VERSIONS)
    rows = []
    for m in sorted(battery.values(), key=lambda x: (order.index(x.extractor), x.model)):
        rows.append(
            {
                "battery_version": BATTERY_VERSION,
                "model": m.model,
                "extractor": m.extractor,
                "extractor_version": EXTRACTOR_VERSIONS[m.extractor],
                "modality": m.modality,
                "kind": m.kind,
                "dim": m.dim,
                "checkpoint": m.checkpoint,
                "prefixes": list(m.prefixes),
                "window_sec": m.window_sec,
                "notes": m.notes,
            }
        )
    return rows


def _norm_ckpt(value: Any) -> str | None:
    if value is None or value == "None" or value == "":
        return None
    return str(value)


def _check_one(sidecar: Mapping[str, Any], source: str, battery: Mapping[str, BatteryModel]) -> list[str]:
    out: list[str] = []
    extractor = sidecar.get("extractor")
    models = sidecar.get("models") or {}
    for name, entry in models.items():
        pinned = battery.get(name)
        if pinned is None:
            out.append(f"{source}: unknown_model {name} (not in battery {BATTERY_VERSION})")
            continue
        if extractor and extractor != pinned.extractor:
            out.append(
                f"{source}: extractor_mismatch {name} emitted by {extractor}, "
                f"pinned to {pinned.extractor}"
            )
        if not isinstance(entry, Mapping):
            continue
        found = _norm_ckpt(entry.get("checkpoint"))
        if found != pinned.checkpoint:
            out.append(
                f"{source}: checkpoint_mismatch {name} found {found!r}, "
                f"pinned {pinned.checkpoint!r}"
            )
        declared = entry.get("prefixes")
        if declared is not None and tuple(declared) != pinned.prefixes:
            out.append(
                f"{source}: prefix_mismatch {name} declared {list(declared)}, "
                f"pinned {list(pinned.prefixes)}"
            )
    return out


def check_sidecar(
    sidecar: Mapping[str, Any],
    *,
    source: str = "<sidecar>",
    battery: Mapping[str, BatteryModel] = BATTERY,
) -> list[str]:
    """Violations of the clamp in one sidecar; empty when it conforms.

    Accepts a §4.1 extractor sidecar (``models`` at the top level) or a
    psytwill group sidecar (``inputs`` list of per-cell summaries, each with
    its own ``extractor`` and ``models``). Each violation is one line naming
    the source, the kind (``unknown_model`` / ``extractor_mismatch`` /
    ``checkpoint_mismatch`` / ``prefix_mismatch``), and the model.
    """
    if "inputs" in sidecar and isinstance(sidecar["inputs"], list):
        out: list[str] = []
        for i, inp in enumerate(sidecar["inputs"]):
            if isinstance(inp, Mapping):
                out += _check_one(inp, f"{source}#{inp.get('path', i)}", battery)
        return out
    return _check_one(sidecar, source, battery)


def models_seen(sidecars: Iterable[Mapping[str, Any]]) -> set[str]:
    """Model names emitted across sidecars (both shapes) — for the
    battery-side direction: which pinned models a store never wrote."""
    seen: set[str] = set()
    for sc in sidecars:
        if "inputs" in sc and isinstance(sc["inputs"], list):
            for inp in sc["inputs"]:
                if isinstance(inp, Mapping):
                    seen.update((inp.get("models") or {}).keys())
        else:
            seen.update((sc.get("models") or {}).keys())
    return seen


def check_registry(
    extractor: str,
    registry_names: Iterable[str],
    *,
    battery: Mapping[str, BatteryModel] = BATTERY,
) -> tuple[set[str], set[str]]:
    """Two-sided registry cross-check for one extractor.

    Returns ``(missing_from_battery, missing_from_registry)``: models the
    package ships that the battery has not admitted, and battery models the
    package no longer ships. Both empty means the clamp and the package agree.
    """
    if extractor not in EXTRACTOR_VERSIONS:
        raise BatteryError(f"unknown extractor {extractor!r}; battery covers {sorted(EXTRACTOR_VERSIONS)}")
    live = set(registry_names)
    pinned = {m.model for m in battery.values() if m.extractor == extractor}
    return live - pinned, pinned - live
