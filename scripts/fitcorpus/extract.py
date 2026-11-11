#!/usr/bin/env python
"""psytwill-space fit-corpus extraction: the clamped *2psy battery over
external corpora, one Contract B §4.1 family per cell, resumably.

    cell = (corpus, source, model, unit)

`source` is what a package consumes from a corpus unit (an HDF5 image
shard, a video's frames, its soundtrack, a caption shard, a transcript);
`unit` is one CLI invocation's worth of input. Every cell writes one family
(`<stem>.csv` or `<stem>_<table>.csv`) plus exactly one `<stem>.meta.json`.
The sidecar is the done-marker **only if it passes `psytwill battery
--check`**: a cell extracted at a drifted checkpoint is not done, and is
re-run rather than silently accepted (psytwill-space clamp, 2026-09-07).

Outputs land under the durable root as
`<corpus>/features/<unit>/<prefix><model>.csv` with `stimulus_id` in the
`ext-<corpus>-<native>` namespace (`psytwill.fitcorpus`). Raw media is read
from the scratch root (or in place, for the NSD brick) and never copied.

Usage
-----
    extract.py plan [--corpus C] [--source S] [--model M] [--unit U]
    extract.py inputs                      # derived input CSVs (caption shards)
    extract.py run --corpus nsd --unit u000 [--dry-run] [--redo] [--fail-fast]
    extract.py run --corpus movie10 --unit-index 7   # array-task form
    extract.py verify                      # re-check every written sidecar
    extract.py run --corpus librispeech --unit-index 3 --skip-model psychoacoustic   # GPU job
    extract.py run --corpus librispeech --unit-index 3 --model psychoacoustic        # CPU job
    extract.py qc [--corpus C]             # transcript QC flags, <corpus>/qc/transcript_flags.parquet
    extract.py exclusions --base-rows R --base-reason WHY --out space/exclusions/<name>_exclude_rows.parquet

Short-file corpora (speech, music, words) are first packed into units by
`stage.py`; their manifest `<corpus>/inputs/units.csv` is what this driver
reads, and `<unit>_segments.csv` beside it maps unit time back to files.

Roots (flags win over env; there are no defaults -- a site names them):
    --durable-root  / $PSYTWILL_FITCORPUS_DURABLE  feature store
    --scratch-root  / $PSYTWILL_FITCORPUS_SCRATCH  staged, derived media (packed audio units, PNG frames)
    --media-root    / $PSYTWILL_FITCORPUS_MEDIA    raw source video (CNeuroMod), read in place
    --framing-table / $PSYTWILL_FRAMING_TABLE      reviewed viz2psy framing table; required by
                                                   every viz2psy cell that reads a video file
    --nsd-hdf5      / $PSYTWILL_NSD_HDF5           the 73k NSD image brick

Video paths are passed exactly as found under --media-root (symlinks not
resolved), because the framing table is keyed on that spelling; viz2psy
refuses a video the table does not hold. A video cell is done only if its
sidecar was framed with the current table (sha256), so a resumed run
re-extracts cells framed under an older table, or none.

Env: needs viz2psy, aud2psy, word2psy importable (the stimfeat env on
Talapas). The interpreter running this script runs the extractors.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from psytwill.battery import BATTERY, check_sidecar
from psytwill.fitcorpus import CORPORA, EXT_PREFIX, ext_id

PY = sys.executable
GRID_HOP = 0.5  # shared frame grid, matching the mmmdata store
NSD_UNIT = 1000  # images per NSD unit (== the shared1000 arm's proven size)
NSD_TOTAL = 73000

# Block members by extractor (psytwill-space block basis, ratified
# 2026-09-07). caption / transcribe are producers; diarize is a schema.
# Text-kind viz2psy models (caption, caption_vlm) are producers, never V
# members: excluded by kind, so a new captioner cannot leak into stills.
V_MODELS = [m.model for m in BATTERY.values() if m.extractor == "viz2psy" and m.kind != "text"]
A_MODELS = [m.model for m in BATTERY.values() if m.extractor == "aud2psy"]
L_MODELS = [m.model for m in BATTERY.values() if m.extractor == "word2psy"]

# Frame captions from a prompted VLM (viz2psy caption_vlm; psytwill-space
# 0.2.0 caption member, DECIDED 2026-11-05): video corpora only, scored as
# text by the word2psy vlmcap_ source. Greedy bf16 captions differ across GPU
# types (~12 % of frames reword between two H100s, ~34 % A100 vs H100;
# mmmdata-agents log 2026-11-09), so a caption_vlm cell runs only on a GPU
# whose name contains CAPTION_VLM_GPU ($CAPTION_VLM_GPU overrides; "" turns
# the check off), at the batch size the reference run used.
CAPTION_VLM = "caption_vlm"
CAPTION_VLM_GPU = os.environ.get("CAPTION_VLM_GPU", "H100")
VIZ_BATCH = {CAPTION_VLM: 32}  # every other viz2psy model: 64

# aud2psy models that consume another cell's table (no second inference pass).
# egemaps / speech_emotion read diarize's turns for their speech gate
# (aud2psy >= 0.20.0, where diarize is the default gate).
_AUDIO_DEPS = {"conversation": "diarize", "egemaps": "diarize",
               "speech_emotion": "diarize", "speech_rate": "transcribe"}


def _ordered_audio() -> list[str]:
    """Producers before their consumers; otherwise registry order."""
    first = [m for m in A_MODELS if m in _AUDIO_DEPS.values()]
    rest = [m for m in A_MODELS if m not in first]
    return first + rest


@dataclass
class Unit:
    id: str  # unit id (u000 | native media stem)
    out_dir: Path
    inputs: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)


@dataclass
class Source:
    corpus: str
    source: str
    package: str
    prefix: str
    models: list[str]
    what: str
    units_fn: object
    depends: str | None = None  # "corpus/source/model" that must exist first

    def units(self) -> list[Unit]:
        return self.units_fn()

    @property
    def key(self) -> str:
        return f"{self.corpus}/{self.source}"


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------

class Roots:
    def __init__(self, args: argparse.Namespace) -> None:
        self.durable = self._resolve(args.durable_root, "PSYTWILL_FITCORPUS_DURABLE", "--durable-root")
        self.scratch = self._resolve(args.scratch_root, "PSYTWILL_FITCORPUS_SCRATCH", "--scratch-root", required=False)
        self.media = self._resolve(args.media_root, "PSYTWILL_FITCORPUS_MEDIA", "--media-root", required=False)
        self.framing_table = self._resolve(args.framing_table, "PSYTWILL_FRAMING_TABLE", "--framing-table",
                                           required=False)
        self._framing_sha: str | None = None

    def framing_sha256(self) -> str:
        """The framing table's sha256, from its parameter record (viz2psy writes it)."""
        if self.framing_table is None:
            sys.exit("ERROR: a viz2psy cell reads a video file, and no --framing-table is given "
                     "($PSYTWILL_FRAMING_TABLE unset); every video is extracted through the reviewed table")
        if self._framing_sha is None:
            rec = self.framing_table.with_name(self.framing_table.name + ".json")
            if not rec.is_file():
                sys.exit(f"ERROR: framing table {self.framing_table} has no parameter record {rec}")
            self._framing_sha = json.loads(rec.read_text())["sha256"]
        return self._framing_sha
        self.nsd_hdf5 = self._resolve(args.nsd_hdf5, "PSYTWILL_NSD_HDF5", "--nsd-hdf5", required=False)

    @staticmethod
    def _resolve(flag: str | None, env: str, name: str, required: bool = True) -> Path | None:
        value = flag or os.environ.get(env)
        if not value:
            if required:
                sys.exit(f"ERROR: no {name} given and ${env} unset; this driver does not guess site paths")
            return None
        return Path(value)


ROOTS: Roots  # set in main()


# ---------------------------------------------------------------------------
# Corpus adapters
# ---------------------------------------------------------------------------

def _nsd_units() -> list[Unit]:
    if ROOTS.nsd_hdf5 is None:
        return []
    out = []
    for k in range(0, NSD_TOTAL, NSD_UNIT):
        uid = f"u{k // NSD_UNIT:03d}"
        out.append(Unit(
            id=uid,
            out_dir=ROOTS.durable / "nsd" / "features" / uid,
            inputs=[str(ROOTS.nsd_hdf5)],
            extra=["--start", str(k), "--end", str(min(k + NSD_UNIT, NSD_TOTAL)),
                   "--id-pattern", "ext-nsd-{idx1:06d}"],
        ))
    return out


def _nsd_caption_units() -> list[Unit]:
    out = []
    for k in range(0, NSD_TOTAL, NSD_UNIT):
        uid = f"u{k // NSD_UNIT:03d}"
        shard = ROOTS.durable / "nsd" / "inputs" / "captions" / f"{uid}.csv"
        out.append(Unit(
            id=uid,
            out_dir=ROOTS.durable / "nsd" / "features" / uid,
            inputs=[str(shard)],
            extra=["--text-column", "caption", "--id-column", "stimulus_id"],
        ))
    return out


def _cneuromod_media(corpus: str) -> list[Path]:
    """Present (annex content available) .mkv segments of one CNeuroMod corpus,
    under the media root, spelled as found there (the framing table's keys)."""
    if ROOTS.media is None:
        return []
    root = ROOTS.media / "cneuromod" / "algonauts_2025.competitors" / "stimuli" / "movies" / corpus
    if not root.exists():
        return []
    files = []
    for p in sorted(root.rglob("*.mkv")):
        if ".git" in p.parts:
            continue
        if p.resolve().is_file():  # annexed symlink with content
            files.append(p)
    return files


def _blender_media() -> list[Path]:
    """Blender open movies under ``<media root>/blender/``: one file per film,
    named ``<native>.<ext>`` by ``fetch.py blender`` (the framing table's keys)."""
    if ROOTS.media is None:
        return []
    root = ROOTS.media / "blender"
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES and p.is_file())


#: Video fit corpora: whole films or segments, one unit per media file, each
#: run through V (frames), A (soundtrack), Whisper transcript (L) and the
#: caption_vlm text (L). movie10/friends = CNeuroMod (tier b); blender = the
#: CC-BY open movies (tier a, the animated wedge; DECIDED 2026-08-25, joined
#: step 3 2026-11-09: the freely downloadable films only).
VIDEO_CORPORA = ("movie10", "friends", "blender")


def _video_media(corpus: str) -> list[Path]:
    return _blender_media() if corpus == "blender" else _cneuromod_media(corpus)


def _video_native(corpus: str, path: Path) -> str:
    stem = path.stem  # bourne01 | friends_s01e02a | big-buck-bunny
    if corpus == "friends" and stem.startswith("friends_"):
        stem = stem[len("friends_"):]
    return stem.lower().replace("_", "-")


def _video_units(corpus: str) -> list[Unit]:
    out = []
    for p in _video_media(corpus):
        native = _video_native(corpus, p)
        out.append(Unit(
            id=native,
            out_dir=ROOTS.durable / corpus / "features" / native,
            inputs=[str(p)],
            extra=["--stimulus-id", ext_id(corpus, native)],
        ))
    return out


def _video_transcript_units(corpus: str) -> list[Unit]:
    out = []
    for p in _video_media(corpus):
        native = _video_native(corpus, p)
        d = ROOTS.durable / corpus / "features" / native
        shard = ROOTS.durable / corpus / "inputs" / f"{native}_transcript.csv"
        out.append(Unit(
            id=native,
            out_dir=d,
            inputs=[str(shard)],
            extra=["--text-column", "text", "--id-column", "stimulus_id"],
        ))
    return out


def _video_vlmcap_units(corpus: str) -> list[Unit]:
    """word2psy over each unit's caption_vlm table (frames source output)."""
    out = []
    for p in _video_media(corpus):
        native = _video_native(corpus, p)
        d = ROOTS.durable / corpus / "features" / native
        out.append(Unit(
            id=native,
            out_dir=d,
            inputs=[str(d / f"{CAPTION_VLM}.csv")],
            extra=["--text-column", f"{CAPTION_VLM}_text", "--id-column", "stimulus_id"],
        ))
    return out


# Corpora staged by stage.py into one wav per unit (`<corpus>/inputs/units.csv`).
# Speech corpora with curated text ship their own transcript input (stage.py
# writes it); narratives has none in ds002345, so its L input is the unit's
# Whisper output, as for CNeuroMod; music has no transcript source at all.
STAGED_CURATED = {
    "librispeech": "read audiobook speech (LibriSpeech train-clean-100), packed per speaker; curated transcripts",
    "gigaspeech": "podcast speech (GigaSpeech S, podcast source only), whole episodes packed; curated transcripts",
    "twp-unpresented": "the never-presented twp recordings packed per voice; text = the word",
}
STAGED_WHISPER = {"narratives": "spoken stories (Narratives ds002345), one story per unit"}
STAGED_MUSIC = {
    "fma": "genre-balanced music clips (fma_small draw), packed per genre",
    "musopen": "public-domain orchestral recordings (Musopen DVD), movements packed",
}


def _staged_units(corpus: str) -> list[Unit]:
    manifest = ROOTS.durable / corpus / "inputs" / "units.csv"
    if not manifest.exists():
        return []
    out = []
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            out.append(Unit(
                id=row["unit"],
                out_dir=ROOTS.durable / corpus / "features" / row["unit"],
                inputs=[row["path"]],
                extra=["--stimulus-id", row["stimulus_id"]],
            ))
    return out


def _staged_transcript_units(corpus: str) -> list[Unit]:
    out = []
    for u in _staged_units(corpus):
        out.append(Unit(
            id=u.id,
            out_dir=u.out_dir,
            inputs=[str(ROOTS.durable / corpus / "inputs" / f"{u.id}_transcript.csv")],
            extra=["--text-column", "text", "--id-column", "stimulus_id"],
        ))
    return out


# Clip corpora (l-hub): audio packed with a 10 s gap by stage.py, AVCaps frames
# staged as PNG (stage.py avcaps --frames), human captions as word2psy text.
# No transcript source: what is said is not the caption's register.
STAGED_CLIPS = {
    "clotho": "Freesound clips with audio-only captions (dev + eval), packed with a 10 s gap",
    "avcaps": "AVCaps soundtracks (VidOR clips), packed with a 10 s gap",
}
CAPTION_SHARD = 5000  # caption rows per word2psy call
CAPTION_TYPES = {"clotho": ("audio",), "avcaps": ("audio", "visual", "audio_visual")}
_CAPTION_PREFIX = {"audio": "audiocap_", "visual": "viscap_", "audio_visual": "avcap_"}


def _avcaps_frame_units() -> list[Unit]:
    manifest = ROOTS.durable / "avcaps" / "inputs" / "frames_units.csv"
    if not manifest.exists():
        return []
    out = []
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            table = ROOTS.durable / "avcaps" / "inputs" / "frames" / f"{row['unit']}.csv"
            with open(table, newline="") as g:
                paths = [r["path"] for r in csv.DictReader(g)]
            out.append(Unit(id=row["unit"], out_dir=ROOTS.durable / "avcaps" / "features" / "frames" / row["unit"],
                            inputs=paths))
    return out


# AVCaps through viz2psy's video path (psytwill-space 0.2.0, DECIDED 2026-11-09):
# the framing table applies, unlike the PNG stills above, which bypass it. A
# viz2psy video set (>= 0.20.0) scores many clips per call with each model
# loaded once; batches never span clips, so clips of any size share a unit.
# Units are whole clips of one split, AVCAPS_VIDEO_UNIT per call, in native-id
# order; a cell is done only if its sidecar lists exactly the unit's clips.
AVCAPS_SPLITS = ("train", "val", "test")
AVCAPS_VIDEO_UNIT = 100  # clips per call (~50 s median -> ~10k frames)


def _avcaps_media() -> dict[str, list[Path]]:
    """AVCaps clips under ``<media root>/avcaps/<split>_videos/<split>_videos/``
    (the layout `fetch.py avcaps` unpacks), spelled as found there (the
    framing table's keys), sorted by native id."""
    if ROOTS.media is None:
        return {}
    out = {}
    for split in AVCAPS_SPLITS:
        d = ROOTS.media / "avcaps" / f"{split}_videos" / f"{split}_videos"
        if d.exists():
            out[split] = sorted((p for p in d.iterdir()
                                 if p.suffix.lower() in VIDEO_SUFFIXES and p.is_file()),
                                key=lambda p: p.stem)
    return out


def _avcaps_video_units() -> list[Unit]:
    out = []
    for split, clips in _avcaps_media().items():
        for k in range(0, len(clips), AVCAPS_VIDEO_UNIT):
            chunk = clips[k:k + AVCAPS_VIDEO_UNIT]
            for p in chunk:
                ext_id("avcaps", p.stem)  # a clip whose name is not a native id fails here
            uid = f"{split}-{k // AVCAPS_VIDEO_UNIT:03d}"
            out.append(Unit(id=uid, out_dir=ROOTS.durable / "avcaps" / "features" / "video" / uid,
                            inputs=[str(p) for p in chunk],
                            extra=["--id-pattern", f"{EXT_PREFIX}avcaps-{{stem}}"]))
    return out


def _caption_shards(corpus: str, ctype: str) -> list[Path]:
    return sorted((ROOTS.durable / corpus / "inputs" / "captions").glob(f"{ctype}-*.csv"))


def _caption_units(corpus: str, ctype: str) -> list[Unit]:
    return [Unit(id=p.stem, out_dir=ROOTS.durable / corpus / "features" / "captions" / p.stem,
                 inputs=[str(p)], extra=["--text-column", "caption", "--id-column", "stimulus_id"])
            for p in _caption_shards(corpus, ctype)]


def build_sources() -> list[Source]:
    S: list[Source] = []
    S.append(Source(
        "nsd", "image", "viz2psy", "", [m for m in V_MODELS if m != "motion"],
        f"73,000 NSD images read from the hdf5 brick in {NSD_TOTAL // NSD_UNIT} shards of {NSD_UNIT:,}; "
        "BLIP captioning skipped (COCO human captions are the L input)",
        _nsd_units,
    ))
    S.append(Source(
        "nsd", "humancap", "word2psy", "humancap_", L_MODELS,
        "COCO human captions (5-7 per image) of the same shards, scored as text",
        _nsd_caption_units,
        depends="inputs",
    ))
    for corpus in VIDEO_CORPORA:
        what = ("Blender open movies (CC-BY, whole films)" if corpus == "blender"
                else f"staged CNeuroMod {corpus} segments")
        S.append(Source(
            corpus, "frames", "viz2psy", "", V_MODELS + [CAPTION_VLM],
            f"{what}, frames every {GRID_HOP} s "
            f"({CAPTION_VLM} only on {CAPTION_VLM_GPU or 'any GPU'})",
            lambda c=corpus: _video_units(c),
        ))
        S.append(Source(
            corpus, "vlmcap", "word2psy", "vlmcap_", L_MODELS,
            f"{CAPTION_VLM} frame captions of {corpus}, scored as text",
            lambda c=corpus: _video_vlmcap_units(c),
            depends=f"{corpus}/frames/{CAPTION_VLM}",
        ))
        S.append(Source(
            corpus, "audio", "aud2psy", "", _ordered_audio(),
            f"soundtracks of the same {corpus} segments, {GRID_HOP} s hop; "
            "diarize/transcribe run before the models that read their tables",
            lambda c=corpus: _video_units(c),
        ))
        S.append(Source(
            corpus, "transcript", "word2psy", "transcript_", L_MODELS,
            f"Whisper transcript segments of {corpus} (what is said), scored as text",
            lambda c=corpus: _video_transcript_units(c),
            depends=f"{corpus}/audio/transcribe",
        ))
    for corpus, what in {**STAGED_CURATED, **STAGED_WHISPER, **STAGED_MUSIC}.items():
        S.append(Source(
            corpus, "audio", "aud2psy", "", _ordered_audio(),
            f"{what}; {GRID_HOP} s hop; diarize/transcribe run before the models that read their tables",
            lambda c=corpus: _staged_units(c),
        ))
    for corpus in STAGED_CURATED:
        S.append(Source(
            corpus, "transcript", "word2psy", "transcript_", L_MODELS,
            f"curated text of each {corpus} unit (stage.py writes it), scored as text",
            lambda c=corpus: _staged_transcript_units(c),
            depends="inputs",
        ))
    for corpus in STAGED_WHISPER:
        S.append(Source(
            corpus, "transcript", "word2psy", "transcript_", L_MODELS,
            f"Whisper transcript segments of {corpus} (what is said), scored as text",
            lambda c=corpus: _staged_transcript_units(c),
            depends=f"{corpus}/audio/transcribe",
        ))
    for corpus, what in STAGED_CLIPS.items():
        S.append(Source(
            corpus, "audio", "aud2psy", "", _ordered_audio(),
            f"{what}; {GRID_HOP} s hop; diarize/transcribe run before the models that read their tables",
            lambda c=corpus: _staged_units(c),
        ))
        for ctype in CAPTION_TYPES[corpus]:
            S.append(Source(
                corpus, f"{ctype}cap", "word2psy", _CAPTION_PREFIX[ctype], L_MODELS,
                f"{corpus} human {ctype.replace('_', '-')} captions, {CAPTION_SHARD} rows per shard, scored as text",
                lambda c=corpus, t=ctype: _caption_units(c, t),
                depends="inputs",
            ))
    S.append(Source(
        "avcaps", "frames", "viz2psy", "", [m for m in V_MODELS if m != "motion"],
        f"AVCaps frames every {GRID_HOP} s via viz2psy's video sampler, staged as PNG and scored "
        "as images ~1,000 per call (motion is video-only and belongs to D, not V); "
        "0.1.0 / l-hub, unframed: superseded by avcaps/video",
        _avcaps_frame_units,
    ))
    S.append(Source(
        "avcaps", "video", "viz2psy", "", [m for m in V_MODELS if m != "motion"],
        f"AVCaps clips through viz2psy's video path, framing table applied, frames every "
        f"{GRID_HOP} s, {AVCAPS_VIDEO_UNIT} clips per call (viz2psy >= 0.20.0 video sets; "
        "motion left out as for the stills)",
        _avcaps_video_units,
    ))
    return S


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def stem_for(src: Source, unit: Unit, model: str) -> Path:
    return unit.out_dir / f"{src.prefix}{model}.csv"


def sidecar_for(stem: Path) -> Path:
    return stem.with_suffix(".meta.json")


VIDEO_SUFFIXES = {".mkv", ".mp4", ".mov", ".avi", ".webm", ".m4v"}


def is_video_cell(src: Source, unit: Unit) -> bool:
    """A viz2psy cell whose inputs are video files (one, or a video set), not
    staged PNG frames."""
    return (src.package == "viz2psy" and bool(unit.inputs)
            and all(Path(p).suffix.lower() in VIDEO_SUFFIXES for p in unit.inputs))


def is_done(stem: Path, src: Source | None = None, unit: Unit | None = None) -> bool:
    """Sidecar present AND clean against the clamped battery AND, for a video
    cell, framed with the current framing table."""
    meta = sidecar_for(stem)
    if not meta.exists():
        return False
    try:
        sidecar = json.loads(meta.read_text())
    except (OSError, ValueError):
        return False
    if check_sidecar(sidecar, source=str(meta)):
        return False
    if src is not None and unit is not None and is_video_cell(src, unit):
        inp = sidecar.get("input") or {}
        framing = inp.get("framing") or {}
        if framing.get("framing_table_sha256") != ROOTS.framing_sha256():
            return False
        if len(unit.inputs) > 1:  # a video set: exactly this unit's clips, in order
            listed = [v.get("path") for v in inp.get("videos") or []]
            return listed == [str(Path(p).resolve()) for p in unit.inputs]
    return True


def command_for(src: Source, unit: Unit, model: str) -> list[str]:
    stem = stem_for(src, unit, model)
    if src.package == "viz2psy":
        cmd = [PY, "-m", "viz2psy.cli", model, *unit.inputs,
               "-o", str(stem), "--batch-size", str(VIZ_BATCH.get(model, 64)), "--no-viz", "--quiet"]
        if src.source in ("frames", "video"):
            cmd += ["--frame-interval", str(GRID_HOP), "--no-save-frames"]
        if is_video_cell(src, unit):
            ROOTS.framing_sha256()  # refuses without a table
            cmd += ["--framing-table", str(ROOTS.framing_table)]
        return cmd + unit.extra
    if src.package == "aud2psy":
        cmd = [PY, "-m", "aud2psy.cli", model, *unit.inputs,
               "-o", str(stem), "--hop", str(GRID_HOP), *unit.extra]
        if _AUDIO_DEPS.get(model) == "diarize":
            cmd += ["--speakers", str(unit.out_dir / f"{src.prefix}diarize_speakers.csv")]
        if model == "speech_rate":
            cmd += ["--words", str(unit.out_dir / f"{src.prefix}transcribe_transcript_words.csv")]
        return cmd
    if src.package == "word2psy":
        return [PY, "-m", "word2psy.cli", model, *unit.inputs, "-o", str(stem), *unit.extra]
    raise ValueError(src.package)


def _filtered(args: argparse.Namespace) -> list[tuple[Source, Unit, str]]:
    cells = []
    for src in build_sources():
        if args.corpus and src.corpus != args.corpus:
            continue
        if args.source and src.source != args.source:
            continue
        units = src.units()
        if args.unit_index is not None:
            if args.unit_index >= len(units):
                continue
            units = [units[args.unit_index]]
        for unit in units:
            if args.unit and unit.id != args.unit:
                continue
            for model in src.models:
                if args.model and model != args.model:
                    continue
                if model in (args.skip_model or ()):
                    continue
                cells.append((src, unit, model))
    return cells


# ---------------------------------------------------------------------------
# Verbs
# ---------------------------------------------------------------------------

def cmd_plan(args: argparse.Namespace) -> int:
    rows = []
    for src in build_sources():
        if args.corpus and src.corpus != args.corpus:
            continue
        if args.source and src.source != args.source:
            continue
        units = src.units()
        if args.unit_index is not None:
            units = units[args.unit_index:args.unit_index + 1]
        if args.unit:
            units = [u for u in units if u.id == args.unit]
        models = [m for m in src.models if (not args.model or m == args.model)
                  and m not in (args.skip_model or ())]
        n = len(units) * len(models)
        done = sum(is_done(stem_for(src, u, m), src, u) for u in units for m in models)
        rows.append((src.key, src.package, len(units), len(models), n, done, src.what, src.depends))
    print(f"fit-corpus manifest   durable: {ROOTS.durable}")
    print(f"{'corpus/source':22} {'pkg':9} {'units':>5} {'models':>6} {'cells':>7} {'done':>7} {'todo':>6}")
    print("-" * 72)
    tot = [0, 0, 0]
    for key, pkg, nu, nm, n, done, _, _ in rows:
        print(f"{key:22} {pkg:9} {nu:5d} {nm:6d} {n:7d} {done:7d} {n - done:6d}")
        tot[0] += n; tot[1] += done; tot[2] += n - done
    print("-" * 72)
    print(f"{'TOTAL':22} {'':9} {'':5} {'':6} {tot[0]:7d} {tot[1]:7d} {tot[2]:6d}")
    for key, _, _, _, _, _, what, dep in rows:
        print(f"  {key}: {what}" + (f"  [depends: {dep}]" if dep else ""))
    return 0


def cmd_inputs(args: argparse.Namespace) -> int:
    """Derived input CSVs: NSD caption shards; transcript inputs; clip-corpus caption shards."""
    import pandas as pd

    n_written = 0
    cap = ROOTS.durable / "nsd" / "inputs" / "coco_captions.parquet"
    if cap.exists():
        out_dir = ROOTS.durable / "nsd" / "inputs" / "captions"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(cap, columns=["stimulus_id", "nsd_id", "caption"])
        for k in range(0, NSD_TOTAL, NSD_UNIT):
            p = out_dir / f"u{k // NSD_UNIT:03d}.csv"
            if p.exists() and not args.force:
                continue
            shard = df[(df.nsd_id > k) & (df.nsd_id <= k + NSD_UNIT)].sort_values(["nsd_id"])
            shard[["stimulus_id", "caption"]].to_csv(p, index=False)
            n_written += 1
        print(f"nsd captions: {NSD_TOTAL // NSD_UNIT} shards under {out_dir} ({n_written} written)")
    else:
        print(f"nsd captions: {cap} missing (run fetch.py coco-captions + adapt_nsd_captions.py first)")

    # transcripts: from each unit's transcribe output (once it exists)
    for corpus in VIDEO_CORPORA:
        n = sum(_ensure_transcript_input(corpus, u, force=args.force) for u in _video_units(corpus))
        print(f"{corpus} transcripts: {n} input CSVs written")
    for corpus in STAGED_WHISPER:
        n = sum(_ensure_transcript_input(corpus, u, force=args.force) for u in _staged_units(corpus))
        print(f"{corpus} transcripts: {n} input CSVs written")

    # clip-corpus captions: stage.py's captions.csv, sharded per caption type
    for corpus, ctypes in CAPTION_TYPES.items():
        src = ROOTS.durable / corpus / "inputs" / "captions.csv"
        if not src.exists():
            print(f"{corpus} captions: {src} missing (stage.py {corpus} writes it)")
            continue
        df = pd.read_csv(src, keep_default_na=False)
        out_dir = src.parent / "captions"
        out_dir.mkdir(parents=True, exist_ok=True)
        for ctype in ctypes:
            sub = df[df.caption_type == ctype][["stimulus_id", "caption"]]
            n_new = 0
            for k in range(0, len(sub), CAPTION_SHARD):
                p = out_dir / f"{ctype}-{k // CAPTION_SHARD:03d}.csv"
                if p.exists() and not args.force:
                    continue
                sub.iloc[k:k + CAPTION_SHARD].to_csv(p, index=False)
                n_new += 1
            print(f"{corpus} {ctype} captions: {len(sub)} rows in {-(-len(sub) // CAPTION_SHARD)} shards "
                  f"({n_new} written)")
    return 0


def _ensure_transcript_input(corpus: str, unit: Unit, force: bool = False) -> bool:
    """Write `<corpus>/inputs/<unit>_transcript.csv` from the unit's Whisper
    output, if that exists. Returns True when a file was written. Text
    segments only (silent segments contribute no rows); onset/offset are the
    stimulus's own coordinates and ride along, ASR confidences do not."""
    src_csv = unit.out_dir / "transcribe_transcript.csv"
    dst = ROOTS.durable / corpus / "inputs" / f"{unit.id}_transcript.csv"
    if not src_csv.exists() or (dst.exists() and not force):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src_csv, newline="") as f, open(dst, "w", newline="") as g:
        r = csv.DictReader(f)
        w = csv.writer(g)
        w.writerow(["stimulus_id", "text", "onset", "offset"])
        for row in r:
            text = (row.get("transcribe_text") or "").strip()
            if text:
                w.writerow([row["stimulus_id"], text, row["onset"], row["offset"]])
    return True


def gpu_names() -> list[str]:
    """Names of the GPUs this process can see (nvidia-smi; [] when there is none)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [n.strip() for n in r.stdout.splitlines() if n.strip()] if r.returncode == 0 else []


def caption_vlm_gpu_problem(models: set[str], names: list[str], want: str = CAPTION_VLM_GPU) -> str | None:
    """Why caption_vlm cells may not run on these GPUs, or None when they may."""
    if CAPTION_VLM not in models or not want:
        return None
    if names and all(want in n for n in names):
        return None
    return (f"{CAPTION_VLM} cells must run on {want} GPUs (captions differ across GPU types); "
            f"this job sees {names or 'no GPU'}. Submit them on {want} nodes, run the rest with "
            f"--skip-model {CAPTION_VLM}, or set CAPTION_VLM_GPU to override.")


def cmd_run(args: argparse.Namespace) -> int:
    cells = _filtered(args)
    todo = [(s, u, m) for s, u, m in cells if args.redo or not is_done(stem_for(s, u, m), s, u)]
    print(f"{len(cells)} cells matched: {len(todo)} to run, {len(cells) - len(todo)} already done")
    if not args.dry_run:
        problem = caption_vlm_gpu_problem({m for _, _, m in todo}, gpu_names())
        if problem:
            print(f"ERROR: {problem}", file=sys.stderr)
            return 2
    n_ok = n_fail = 0
    failures: list[str] = []
    for i, (src, unit, model) in enumerate(todo, 1):
        stem = stem_for(src, unit, model)
        cmd = command_for(src, unit, model)
        label = f"{src.key}/{model}[{unit.id}]"
        if args.dry_run:
            print(f"  [{i}/{len(todo)}] {label}\n      {' '.join(cmd)}")
            continue
        if src.source == "transcript":
            # built lazily so a unit's audio and transcript run in one job
            _ensure_transcript_input(src.corpus, unit)
        missing = [p for p in unit.inputs if not Path(p).exists()]
        if missing:
            print(f"  [{i}/{len(todo)}] {label} SKIP: input missing: {missing[0]}"
                  + (f" (depends on {src.depends})" if src.depends else ""))
            n_fail += 1
            failures.append(f"{label} (missing input)")
            continue
        stem.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"  [{i}/{len(todo)}] {label} ...", flush=True)
        r = subprocess.run(cmd)
        dt = time.time() - t0
        meta = sidecar_for(stem)
        if r.returncode == 0 and meta.exists():
            bad = check_sidecar(json.loads(meta.read_text()), source=str(meta))
            if bad:
                for line in bad:
                    print(f"      VIOLATION {line}")
                n_fail += 1
                failures.append(f"{label} (battery)")
                if args.fail_fast:
                    break
            else:
                print(f"      ok  {dt:.1f}s", flush=True)
                n_ok += 1
        else:
            print(f"      FAIL rc={r.returncode} after {dt:.1f}s", flush=True)
            n_fail += 1
            failures.append(label)
            if args.fail_fast:
                break
    if not args.dry_run:
        print(f"\nDone: {n_ok} extracted, {n_fail} failed")
        for f in failures:
            print(f"  FAILED {f}")
    return 1 if n_fail else 0


def cmd_verify(args: argparse.Namespace) -> int:
    metas = sorted(ROOTS.durable.glob("*/features/*/*.meta.json"))
    n_bad = 0
    for meta in metas:
        bad = check_sidecar(json.loads(meta.read_text()), source=str(meta.relative_to(ROOTS.durable)))
        for line in bad:
            print(line)
        n_bad += bool(bad)
    print(f"{len(metas)} sidecars checked against battery pins, {n_bad} with violations")
    return 1 if n_bad else 0


# ---------------------------------------------------------------------------
# Aggregate — Contract B `features` long table, one parquet per group
# ---------------------------------------------------------------------------
# Mirrors mmmdata's stimfeat_campaign.py `aggregate`: a group is keyed
# (corpus, source, table) and holds EVERY model of that table across all
# units, written to <durable>/<corpus>/groups/<corpus>_<source>[_<table>]_
# features.parquet by psytwill.features.build_features (streaming writer).

def _dir_stems(directory: Path) -> list[str]:
    """Every cell stem any source writes into `directory`, longest first.

    Built once for the whole tree (one pass over every source's units), not
    per directory — each `units()` call globs a media directory on GPFS.
    """
    cache = _dir_stems.cache  # type: ignore[attr-defined]
    if not cache:
        by_dir: dict[Path, set[str]] = {}
        for s_ in build_sources():
            for u in s_.units():
                by_dir.setdefault(u.out_dir, set()).update(stem_for(s_, u, m).stem for m in s_.models)
        for d, stems in by_dir.items():
            cache[d] = sorted(stems, key=len, reverse=True)
    return cache.get(directory, [])


_dir_stems.cache = {}  # type: ignore[attr-defined]


def family_tables(src: Source, unit: Unit, model: str) -> dict[str, Path]:
    """{table name: csv} for one cell — `<stem>.csv` ("main") and `<stem>_<table>.csv`.

    A bare `<stem>*.csv` glob over-matches: `ebind.csv` (frames) also globs
    `ebind_audio_frames.csv` (audio), `speech.csv` globs `speech_emotion.csv`.
    Longest stem in the directory wins, as in the mmmdata campaign.
    """
    stem = stem_for(src, unit, model)
    mine = stem.stem
    others = [n for n in _dir_stems(stem.parent) if len(n) > len(mine)]
    out: dict[str, Path] = {}
    for path in sorted(stem.parent.glob(mine + "*.csv")):
        if any(path.stem == n or path.stem.startswith(n + "_") for n in others):
            continue
        name = path.stem[len(mine):].lstrip("_") or "main"
        out[name] = path
    return out


def _table_usable(path: Path) -> tuple[bool, str]:
    """Can psytwill aggregate this table? (usable, reason-if-not)."""
    from psytwill.spaces import INDEX_COLUMNS

    with open(path, newline="") as f:
        header = next(csv.reader(f), [])
    if not [c for c in header if c not in INDEX_COLUMNS]:
        return False, "no feature columns"
    if "stimulus_id" not in header:
        return False, "no stimulus_id (§4.1)"
    return True, ""


def aggregate_groups(args: argparse.Namespace) -> tuple[dict[tuple[str, str, str], list[Path]], list[tuple[Path, str]],
                                                         dict[tuple[str, str, str], dict[str, set[Path]]]]:
    """Groups, skipped tables, and per group the unit dirs each model has a usable table in."""
    groups: dict[tuple[str, str, str], list[Path]] = {}
    skipped: list[tuple[Path, str]] = []
    units_of: dict[tuple[str, str, str], dict[str, set[Path]]] = {}
    for src, unit, model in _filtered(args):
        if not is_done(stem_for(src, unit, model), src, unit):
            continue
        for table, path in family_tables(src, unit, model).items():
            usable, why = _table_usable(path)
            if not usable:
                skipped.append((path, why))
                continue
            groups.setdefault((src.corpus, src.source, table), []).append(path)
            units_of.setdefault((src.corpus, src.source, table), {}).setdefault(model, set()).add(path.parent)
    return {k: sorted(v) for k, v in sorted(groups.items())}, skipped, units_of


def agg_output(key: tuple[str, str, str]) -> Path:
    corpus, source, table = key
    name = f"{corpus}_{source}" + ("" if table == "main" else f"_{table}")
    return ROOTS.durable / corpus / "groups" / f"{name}_features.parquet"


def cmd_aggregate(args: argparse.Namespace) -> int:
    if args.model or args.skip_model:
        sys.exit("ERROR: `aggregate --model/--skip-model` would write a partial group (a group holds every model of its table); narrow with --corpus/--source instead")
    if args.unit or args.unit_index is not None:
        sys.exit("ERROR: `aggregate` is per corpus/source, not per unit")
    groups, skipped, units_of = aggregate_groups(args)
    if not groups:
        print("no aggregatable tables matched the filters")
        return 0
    by_reason: dict[str, dict[str, int]] = {}
    for path, why in skipped:
        by_reason.setdefault(why, {}).setdefault(path.name, 0)
        by_reason[why][path.name] += 1
    for why, names in sorted(by_reason.items()):
        print(f"skipped {sum(names.values()):>6} table(s): {why}")
        for name, n in sorted(names.items()):
            print(f"    {n:>5}  {name}")
    # a group holds every unit of its (corpus, source) for every model of its
    # table; incomplete ones are reported and left unbuilt unless --allow-partial
    complete: dict[tuple[str, str, str], list[Path]] = {}
    for key, paths in groups.items():
        corpus, source, table = key
        src = next(s_ for s_ in build_sources() if s_.corpus == corpus and s_.source == source)
        n_units = len(src.units())
        short = {m: len(u) for m, u in sorted(units_of[key].items()) if len(u) < n_units}
        if short and not args.allow_partial:
            done = ", ".join(f"{m} {n}/{n_units}" for m, n in short.items())
            print(f"  INCOMPLETE {corpus}/{source}/{table}: units done per model {done} — skipped")
            continue
        complete[key] = paths
    groups = complete
    if not groups:
        print("no complete groups (pass --allow-partial to build anyway)")
        return 1
    print(f"{len(groups)} group(s):")
    todo = []
    for key, paths in groups.items():
        out = agg_output(key)
        state = "done" if out.exists() and not args.redo else "todo"
        print(f"  {'/'.join(key):<34} {len(paths):>6} inputs  -> {out.relative_to(ROOTS.durable)}  [{state}]")
        if state == "todo":
            todo.append((key, paths, out))
    if args.dry_run:
        print(f"\ndry run: {len(todo)} group(s) would be built")
        return 0
    if not todo:
        print("\nnothing to do (use --redo to rebuild)")
        return 0
    from psytwill.features import build_features

    n_fail = 0
    for key, paths, out in todo:
        label = "/".join(key)
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            summary = build_features([str(p) for p in paths], output=out)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  FAIL {label}: {type(exc).__name__}: {exc}", flush=True)
            if args.fail_fast:
                return 1
            continue
        print(f"  ok   {label}: {summary['rows']:,} rows, {summary['n_stimuli']:,} stimuli, "
              f"{len(summary['models'])} models, {out.stat().st_size/1e6:.0f} MB, {time.time() - t0:.0f}s", flush=True)
        # packed short-file corpora: record the staging layout so fits mask rows whose
        # window reads two items (psytwill.windows); continuous corpora have no tables
        inputs = ROOTS.durable / key[0] / "inputs"
        if any(inputs.glob("*_segments.csv")):
            from psytwill.exceptions import InputError
            from psytwill.windows import record_table_segments

            try:
                rec = record_table_segments(out, inputs)
                print(f"       segments: {len(rec['files'])} unit(s) recorded from {inputs}", flush=True)
            except InputError as exc:  # a table keyed on other ids (e.g. a captions group)
                print(f"       segments: none recorded ({exc})", flush=True)
    print(f"\n{len(todo) - n_fail}/{len(todo)} group(s) built")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# Transcript QC (L): flags per Whisper corpus, then one fit exclusion list
# ---------------------------------------------------------------------------

def _whisper_transcript_units() -> dict[str, list[Unit]]:
    """corpus -> transcript units, for every corpus whose L text is Whisper output.
    Curated text, VLM captions and human captions are not screened."""
    out = {c: _video_transcript_units(c) for c in VIDEO_CORPORA}
    out.update({c: _staged_transcript_units(c) for c in STAGED_WHISPER})
    return out


def qc_paths(corpus: str) -> tuple[Path, Path]:
    d = ROOTS.durable / corpus / "qc"
    return d / "transcript_flags.parquet", d / "transcript_flags.meta.json"


def cmd_qc(args: argparse.Namespace) -> int:
    """Write `<corpus>/qc/transcript_flags.parquet` for each Whisper corpus."""
    import pandas as pd

    import psytwill
    from psytwill import transcript_qc as tq
    from psytwill.relate import sha256_file

    n_fail = 0
    for corpus, units in _whisper_transcript_units().items():
        if args.corpus and corpus != args.corpus:
            continue
        have = [u for u in units if Path(u.inputs[0]).exists()]
        if not have:
            print(f"{corpus}: no transcript inputs yet ({len(units)} unit(s) planned); nothing written")
            continue
        if len(have) < len(units) and not args.allow_partial:
            missing = [u.id for u in units if u not in have]
            print(f"INCOMPLETE {corpus}: {len(missing)} of {len(units)} unit(s) lack a transcript input "
                  f"(first: {missing[:3]}); run `extract.py inputs` after transcribe, or pass --allow-partial")
            n_fail += 1
            continue
        parts, shas = [], {}
        for u in have:
            p = Path(u.inputs[0])
            f = tq.flag_chunks(pd.read_csv(p, dtype={"text": str}, keep_default_na=False))
            asr = u.out_dir / "transcribe_transcript.csv"
            if asr.exists():
                f = tq.attach_asr(f, pd.read_csv(asr, dtype={"transcribe_text": str}, keep_default_na=False),
                                  text_column="transcribe_text",
                                  columns=("transcribe_asr_confidence", "transcribe_no_speech_prob"))
            f.insert(0, "unit", u.id)
            parts.append(f)
            shas[u.id] = sha256_file(p)
        flags = pd.concat(parts, ignore_index=True)
        share = tq.style_share(flags)
        flags["lower_style_share"] = flags["stimulus_id"].map(share)
        n_ex = int(flags["loop_exclude"].sum())
        print(f"{corpus}: {len(have)} unit(s), {len(flags)} chunk(s), {n_ex} loop-excluded, "
              f"{int(flags['timing_compressed'].sum())} timing-flagged, {int(flags['lower_style'].sum())} lower-style")
        if args.dry_run:
            continue
        out, meta = qc_paths(corpus)
        out.parent.mkdir(parents=True, exist_ok=True)
        flags.to_parquet(out, index=False)
        meta.write_text(json.dumps({
            "what": "per-chunk transcript QC flags (psytwill.transcript_qc); key stimulus_id,chunk_idx "
                    "= row order of each inputs/<unit>_transcript.csv",
            "corpus": corpus, "rules": tq.rules(), "psytwill_version": psytwill.__version__,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "n_units": len(have), "n_units_planned": len(units), "n_chunks": len(flags),
            "n_loop_excluded": n_ex, "sha256": sha256_file(out), "inputs_sha256": shas,
        }, indent=1) + "\n")
        print(f"       -> {out}")
    return 1 if n_fail else 0


def cmd_exclusions(args: argparse.Namespace) -> int:
    """One L exclusion list: --base-rows ∪ every Whisper corpus's loop exclusions."""
    import pandas as pd

    import psytwill
    from psytwill import transcript_qc as tq
    from psytwill.relate import sha256_file

    key = list(tq.KEY)
    base = Path(args.base_rows)
    b = pd.read_parquet(base) if base.suffix == ".parquet" else pd.read_csv(base)
    if [k for k in key if k not in b.columns]:
        sys.exit(f"ERROR: --base-rows {base} lacks {key}")
    parts = [b[key].assign(reason=args.base_reason)]
    corpora = {}
    for corpus, units in _whisper_transcript_units().items():
        flags_p, meta_p = qc_paths(corpus)
        have = [u for u in units if Path(u.inputs[0]).exists()]
        if not have:
            print(f"{corpus}: no transcript inputs; not in the list")
            continue
        if not flags_p.exists():
            sys.exit(f"ERROR: {corpus} has transcript inputs but no {flags_p}; run `extract.py qc --corpus {corpus}`")
        meta = json.loads(meta_p.read_text())
        if meta["rules"]["rules_version"] != tq.RULES_VERSION:
            sys.exit(f"ERROR: {flags_p} was written under rules {meta['rules']['rules_version']}, "
                     f"current is {tq.RULES_VERSION}; re-run `extract.py qc --corpus {corpus}`")
        if meta["sha256"] != sha256_file(flags_p):
            sys.exit(f"ERROR: {flags_p} changed since its sidecar was written; re-run `extract.py qc --corpus {corpus}`")
        stale = [u.id for u in have if meta["inputs_sha256"].get(u.id) != sha256_file(Path(u.inputs[0]))]
        if stale:
            sys.exit(f"ERROR: {corpus} flags predate {len(stale)} transcript input(s) (first: {stale[:3]}); "
                     f"re-run `extract.py qc --corpus {corpus}`")
        ex = tq.exclusion_rows(pd.read_parquet(flags_p, columns=[*key, "loop_exclude"]))
        parts.append(ex.assign(reason="transcript_loop"))
        corpora[corpus] = {"flags": str(flags_p), "flags_sha256": meta["sha256"],
                           "n_units": meta["n_units"], "n_units_planned": meta["n_units_planned"],
                           "n_rows": len(ex)}
        print(f"{corpus}: {len(ex)} loop row(s)")
    rows = pd.concat(parts, ignore_index=True)
    dup = rows.duplicated(key, keep=False)
    if dup.any():
        sys.exit(f"ERROR: {int(dup.sum())} row(s) appear under two reasons; the lists should not overlap")
    print(f"total: {len(rows)} row(s) ({len(b)} {args.base_reason} + {len(rows) - len(b)} transcript_loop)")
    out = Path(args.out)
    if args.dry_run:
        return 0
    if out.exists():
        tmp = out.with_name(out.name + ".new")
        rows.to_parquet(tmp, index=False)
        same = pd.read_parquet(out).equals(pd.read_parquet(tmp))
        tmp.unlink()
        if same:
            print(f"unchanged: {out}")
            return 0
        sys.exit(f"ERROR: {out} exists with different rows; an exclusion list a fit pins is immutable, "
                 "so name a new one")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out, index=False)
    out.with_name(out.stem + ".meta.json").write_text(json.dumps({
        "what": "L fit exclusion rows (--exclude-rows); key stimulus_id,chunk_idx; reason names the source",
        "rules": tq.rules(), "psytwill_version": psytwill.__version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base": {"path": str(base), "sha256": sha256_file(base), "reason": args.base_reason, "n_rows": len(b)},
        "transcript_loop": corpora, "n_rows": len(rows), "sha256": sha256_file(out),
    }, indent=1) + "\n")
    print(f"-> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--durable-root")
    ap.add_argument("--scratch-root")
    ap.add_argument("--media-root")
    ap.add_argument("--framing-table")
    ap.add_argument("--nsd-hdf5")
    sub = ap.add_subparsers(dest="verb", required=True)

    def filters(p: argparse.ArgumentParser) -> None:
        p.add_argument("--corpus", choices=sorted(CORPORA))
        p.add_argument("--source")
        p.add_argument("--model", choices=sorted(BATTERY))
        p.add_argument("--skip-model", action="append", choices=sorted(BATTERY), metavar="MODEL",
                       help="leave this model to another job (repeatable; e.g. the CPU-only "
                            "psychoacoustic cells run on the compute partition)")
        p.add_argument("--unit")
        p.add_argument("--unit-index", type=int, help="position in the corpus's unit list (array tasks)")

    p = sub.add_parser("plan"); filters(p); p.set_defaults(func=cmd_plan)
    p = sub.add_parser("inputs"); p.add_argument("--force", action="store_true"); p.set_defaults(func=cmd_inputs)
    p = sub.add_parser("run"); filters(p)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("verify"); p.set_defaults(func=cmd_verify)
    p = sub.add_parser("aggregate", help="build one Contract B features parquet per (corpus, source, table) group"); filters(p)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--allow-partial", action="store_true", help="write a group even if some units are not done")
    p.set_defaults(func=cmd_aggregate)
    p = sub.add_parser("qc", help="transcript QC flags per Whisper corpus (<corpus>/qc/transcript_flags.parquet)")
    p.add_argument("--corpus", choices=sorted(CORPORA))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-partial", action="store_true", help="flag a corpus some of whose units lack inputs")
    p.set_defaults(func=cmd_qc)
    p = sub.add_parser("exclusions", help="one L exclusion list: --base-rows plus every corpus's loop rows")
    p.add_argument("--base-rows", required=True, help="existing rows to keep excluding (e.g. a caption draw)")
    p.add_argument("--base-reason", required=True, help="the reason recorded for the base rows")
    p.add_argument("--out", required=True, help="the list to write (immutable once written)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_exclusions)

    args = ap.parse_args()
    global ROOTS
    ROOTS = Roots(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
