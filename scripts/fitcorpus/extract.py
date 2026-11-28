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
    extract.py centre-rule --corpus C [--table T] --out space/exclusions/<name>_exclude_rows.parquet   # rows outside every item
    extract.py carry --corpus C [--dry-run] [--unkeyed link]   # after `stage.py C --restage-from OLD`

Short-file corpora (speech, music, words) are first packed into units by
`stage.py`; their manifest `<corpus>/inputs/units.csv` is what this driver
reads, and `<unit>_segments.csv` beside it maps unit time back to files.

Per-recording corpora (`stage.py`'s layout since 0.3.0, rule C; the 0.2.x
`--per-item` roots likewise; `<corpus>/inputs/layout.json` says so) are many
short units, one recording each. Their aud2psy cells run
batched: one `aud2psy --inputs-from` call per model per `--batch-items` units,
each model loaded once, every unit still writing its own family and sidecar.
A root re-staged from a packed one (`stage.py --restage-from`) takes the
packed extraction's tables through `carry`: the diarize / transcribe tables
and the word2psy transcript families are sliced onto each recording by its
`recording_map.csv` row (`psytwill.carry`; rule C, tables only), and
sidecars gain a `carried_from` block. Every other aud2psy member is left for
`run` to extract per recording, reading the carried tables.

Producers run first; a consumer's manifest row names its unit's stored
diarize turns / transcribe words (aud2psy >= 0.24.0). `--unit-slice A:B`
takes units [A, B) of a corpus, for array tasks over items:

    extract.py run --corpus fma --source audio --unit-slice 0:500

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
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from psytwill.battery import BATTERY, check_sidecar
from psytwill.fitcorpus import CORPORA, EXT_PREFIX, ext_id
from psytwill.passmanifest import PassManifest, load_pass
from psytwill.staging import ONE_RECORDING_LAYOUTS

PY = sys.executable
#: The stage-0 version a video cell must have been framed under to count as
#: done; mirrors ``viz2psy.framing.STAGE0_VERSION`` (bump both together).
STAGE0_VERSION = "1"
#: The frame source the framing rule reads (decoded frames scored directly, D5).
FRAME_SOURCE = "raw decode"
SAR_TOL = 1e-5
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


# One aud2psy call per model per this many per-item units (--batch-items).
BATCH_ITEMS = 500


def _per_item_layout(corpus: str) -> bool:
    """True when stage.py staged ``corpus`` one unit per recording (``per-recording``, or the 0.2.x
    ``per-item``; its ``inputs/layout.json``)."""
    cache = _per_item_layout.cache  # type: ignore[attr-defined]
    if corpus not in cache:
        rec = ROOTS.durable / corpus / "inputs" / "layout.json"
        cache[corpus] = rec.exists() and json.loads(rec.read_text()).get("layout") in ONE_RECORDING_LAYOUTS
    return cache[corpus]


_per_item_layout.cache = {}  # type: ignore[attr-defined]


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
        self._framing_rows: dict[str, dict] | None = None
        self.nsd_hdf5 = self._resolve(args.nsd_hdf5, "PSYTWILL_NSD_HDF5", "--nsd-hdf5", required=False)
        self.pass_manifest = self._resolve(getattr(args, "pass_manifest", None), "PSYTWILL_PASS_MANIFEST",
                                           "--pass-manifest", required=False)
        self._pass: PassManifest | None = None

    def pass_(self) -> PassManifest | None:
        """The pass manifest, loaded once; None when none is given."""
        if self.pass_manifest is None:
            return None
        if self._pass is None:
            self._pass = load_pass(self.pass_manifest)
        return self._pass

    def framing_box(self, video: str) -> dict | None:
        """The current framing table's row for one video (keyed as viz2psy keys it:
        absolute, symlinks kept); None when the video is not in the table."""
        self.framing_sha256()  # refuses without a table or its parameter record
        if self._framing_rows is None:
            with open(self.framing_table, newline="") as fh:
                self._framing_rows = {r["path"]: r for r in csv.DictReader(fh)}
        return self._framing_rows.get(os.path.abspath(video))

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


#: AVA (a fit corpus from psytwill-space 0.3.0): one 15-min segment (15:00-30:00) per
#: film, video and audio as separate files under ``<durable>/ava/inputs/{video,audio}/``,
#: named by the YouTube id. The audio starts at 900 s on the film's clock; the video keeps
#: its original timestamps, so its first frame sits at the keyframe before 895 s.
AVA_AUDIO_START_S = 900.0


def ava_native(youtube_id: str) -> str:
    """AVA native id: 10 hex of sha1(YouTube id). YouTube ids are case-sensitive and may
    start with '-', so they cannot be lowercased into the ext-id namespace."""
    return hashlib.sha1(youtube_id.encode()).hexdigest()[:10]


def _ava_media() -> list[tuple[str, Path, Path]]:
    """(YouTube id, video, audio) for every film with both files present."""
    root = ROOTS.durable / "ava" / "inputs"
    if not (root / "video").is_dir() or not (root / "audio").is_dir():
        return []
    vids = {p.stem: p for p in (root / "video").iterdir()
            if p.suffix.lower() in VIDEO_SUFFIXES and p.is_file()}
    auds = {p.stem: p for p in (root / "audio").glob("*.flac") if p.is_file()}
    return [(y, vids[y], auds[y]) for y in sorted(vids) if y in auds]


def _ava_units(kind: str) -> list[Unit]:
    """One unit per film; ``kind`` = "video" (V) or "audio" (A, the flac)."""
    out = []
    for yid, video, audio in _ava_media():
        native = ava_native(yid)
        out.append(Unit(
            id=native,
            out_dir=ROOTS.durable / "ava" / "features" / native,
            inputs=[str(video if kind == "video" else audio)],
            extra=["--stimulus-id", ext_id("ava", native)],
        ))
    return out


def _video_start_s(path: Path) -> float:
    """The container's first presentation time (ffprobe), in seconds."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=start_time",
                        "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        sys.exit(f"ERROR: ffprobe could not read the start time of {path}: {r.stderr.strip()}")
    return float(r.stdout.strip())


def _ensure_ava_ids(force: bool = False) -> int:
    """``ava/inputs/ids.csv``: native <-> YouTube id, and each film's declared clock.

    A frame at video time t sits at t + video_start_s - audio_start_s on the audio's
    clock (V and A pair on that clock). Written once; ``--force`` rewrites it."""
    media = _ava_media()
    out = ROOTS.durable / "ava" / "inputs" / "ids.csv"
    if not media or (out.exists() and not force):
        return 0
    rows = []
    for yid, video, audio in media:
        native = ava_native(yid)
        rows.append(dict(youtube_id=yid, native=native, stimulus_id=ext_id("ava", native),
                         video=video.name, audio=audio.name,
                         video_start_s=f"{_video_start_s(video):.6f}",
                         audio_start_s=f"{AVA_AUDIO_START_S:.6f}"))
    natives = [r["native"] for r in rows]
    if len(set(natives)) != len(natives):
        sys.exit("ERROR: AVA native ids collide (sha1 prefix); widen ava_native")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


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
    "librispeech": "read audiobook speech (LibriSpeech; train-clean-100 in the fit store, a test split in a held-out root), packed per speaker; curated transcripts",
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
    S.append(Source(
        "ava", "frames", "viz2psy", "", list(V_MODELS),
        f"AVA films, minutes 15-30, frames every {GRID_HOP} s through the framing table "
        f"(motion included; no {CAPTION_VLM})",
        lambda: _ava_units("video"),
    ))
    S.append(Source(
        "ava", "audio", "aud2psy", "", _ordered_audio(),
        f"AVA soundtracks (flac, t 0 = 900 s on the film clock), {GRID_HOP} s hop; "
        "diarize/transcribe run before the models that read their tables",
        lambda: _ava_units("audio"),
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
    cell, framed as the current framing table frames each of its files."""
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
        return framing_current(sidecar, unit)
    return True


def framing_current(sidecar: dict, unit: Unit) -> bool:
    """Fit <-> projection framing parity for one video cell (spec 3a.4): the
    sidecar's stage-0 version and frame source, and each file's box (crop,
    pixel aspect, stored size) against that file's row in the current table.

    The table's version and sha256 are recorded, not compared, so a new table
    version (a new corpus's rows) does not re-extract cells whose boxes did
    not change.
    """
    inp = sidecar.get("input") or {}
    rule = inp.get("framing") or {}
    if rule.get("stage0_version") != STAGE0_VERSION or rule.get("frame_source") != FRAME_SOURCE:
        return False
    if len(unit.inputs) > 1:  # a video set: exactly this unit's clips, in order, one box each
        videos = inp.get("videos") or []
        if [v.get("path") for v in videos] != [str(Path(p).resolve()) for p in unit.inputs]:
            return False
        records = videos
    else:
        records = [rule]
    for video, rec in zip(unit.inputs, records):
        row = ROOTS.framing_box(video)
        if row is None or row.get("error"):
            return False
        crop = rec.get("crop") or {}
        if any(int(crop.get(e, -1)) != int(row[f"crop_{e}"]) for e in ("left", "right", "top", "bottom")):
            return False
        if list(rec.get("stored_wh") or []) != [int(row["width"]), int(row["height"])]:
            return False
        if rec.get("pixel_aspect") is None or abs(float(rec["pixel_aspect"]) - float(row["sar"])) > SAR_TOL:
            return False
    return True


# ---------------------------------------------------------------------------
# The pass manifest
# ---------------------------------------------------------------------------

_INSTALLED: dict[str, str] = {}


def installed_version(package: str) -> str:
    """The producer version the extracting interpreter (PY) has installed."""
    if package not in _INSTALLED:
        r = subprocess.run([PY, "-c", f"import importlib.metadata as m; print(m.version({package!r}))"],
                           capture_output=True, text=True)
        _INSTALLED[package] = r.stdout.strip() if r.returncode == 0 else "not installed"
    return _INSTALLED[package]


def pass_problem(src: Source, unit: Unit, model: str) -> str | None:
    """Why the pass manifest does not allow extracting this cell now; None = allowed
    (or no manifest given)."""
    pm = ROOTS.pass_()
    if pm is None:
        return None
    cell = pm.cell_for(src.corpus, src.source, model)
    if cell is None:
        return f"not in pass {pm.name} (a later cell needs a dated deviation in the manifest)"
    ids = cell.unit_ids()
    if ids is not None and unit.id not in ids:
        return f"unit {unit.id} is outside the pinned subset of pass cell {cell.key} ({cell.units})"
    want = pm.version_for(cell, src.package)
    if want is None:
        return f"pass {pm.name} pins no {src.package} version"
    have = installed_version(src.package)
    if have != want:
        return f"pass {pm.name} runs {src.package} {want}; this interpreter has {have}"
    return None


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


def _unit_slice(units: list[Unit], spec: str | None) -> list[Unit]:
    """``units[A:B]`` for ``--unit-slice A:B`` (either end may be empty)."""
    if not spec:
        return units
    a, sep, b = spec.partition(":")
    if not sep:
        sys.exit(f"ERROR: --unit-slice takes START:END, got {spec!r}")
    return units[int(a or 0):int(b) if b else len(units)]


#: Attached models (battery ``attached=True``) ship beside the space, extracted on
#: the release's stimuli only: they are in no block, so the fit corpora never need
#: them. `run` / `plan` leave them out unless `--with-attached` or `--model` names one.
ATTACHED = frozenset(m for m, b in BATTERY.items() if b.attached)


def _wanted(model: str, args: argparse.Namespace) -> bool:
    if args.model:
        return model == args.model
    if model in (args.skip_model or ()):
        return False
    return model not in ATTACHED or bool(getattr(args, "with_attached", False))


def _filtered(args: argparse.Namespace) -> list[tuple[Source, Unit, str]]:
    cells = []
    for src in build_sources():
        if args.corpus and src.corpus != args.corpus:
            continue
        if args.source and src.source != args.source:
            continue
        units = _unit_slice(src.units(), getattr(args, "unit_slice", None))
        if args.unit_index is not None:
            if args.unit_index >= len(units):
                continue
            units = [units[args.unit_index]]
        for unit in units:
            if args.unit and unit.id != args.unit:
                continue
            for model in src.models:
                if _wanted(model, args):
                    cells.append((src, unit, model))
    return cells


# ---------------------------------------------------------------------------
# Verbs
# ---------------------------------------------------------------------------

def cmd_plan(args: argparse.Namespace) -> int:
    rows = []
    outside: dict[str, list[str]] = {}
    for src in build_sources():
        if args.corpus and src.corpus != args.corpus:
            continue
        if args.source and src.source != args.source:
            continue
        units = _unit_slice(src.units(), getattr(args, "unit_slice", None))
        if args.unit_index is not None:
            units = units[args.unit_index:args.unit_index + 1]
        if args.unit:
            units = [u for u in units if u.id == args.unit]
        models = [m for m in src.models if _wanted(m, args)]
        n = len(units) * len(models)
        done = sum(is_done(stem_for(src, u, m), src, u) for u in units for m in models)
        rows.append((src.key, src.package, len(units), len(models), n, done, src.what, src.depends))
        pm = ROOTS.pass_()
        if pm is not None:
            outside[src.key] = sorted(m for m in models if pm.cell_for(src.corpus, src.source, m) is None)
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
    pm = ROOTS.pass_()
    if pm is not None:
        _plan_pass(pm, rows, outside, args)
    return 0


def _plan_pass(pm: PassManifest, rows: list, outside: dict[str, list[str]], args: argparse.Namespace) -> None:
    """The pass manifest against the driver: cells with no driver source yet, and
    driver models the pass does not cover."""
    driver = {key: models for key, models in
              ((s.key, set(s.models)) for s in build_sources())}
    print(f"\npass {pm.name} ({pm.path}, sha256 {pm.sha256[:12]}): {len(pm.cells)} cells"
          + (f", frozen {pm.frozen}" if pm.frozen else ", not frozen")
          + "; producers " + ", ".join(f"{k} {v}" for k, v in sorted(pm.producers.items())))
    for c in pm.cells:
        if args.corpus and c.corpus != args.corpus:
            continue
        if c.key not in driver:
            state = "NO DRIVER SOURCE"
        else:
            unknown = sorted(set(c.models) - driver[c.key] - {"*"})
            state = f"UNKNOWN MODEL(S) {', '.join(unknown)}" if unknown else "ok"
        print(f"  {c.stage:6} {c.key:28} {','.join(c.models)[:40]:40} {state}"
              + (f"  [deviation: {c.deviation}]" if c.deviation else ""))
    for key, models in sorted(outside.items()):
        if models:
            print(f"  not in pass (carried if done; new extraction refused): {key} {', '.join(models)}")


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

    n = _ensure_ava_ids(force=args.force)
    print(f"ava ids + clocks: {n} rows written" if n else "ava ids + clocks: present (or no media)")

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


def _batchable(src: Source, unit: Unit) -> bool:
    """An aud2psy cell of a per-item corpus: run in a batch, not one call per cell."""
    return src.package == "aud2psy" and len(unit.inputs) == 1 and _per_item_layout(src.corpus)


def _producer_table(src: Source, unit: Unit, model: str) -> tuple[str, Path] | None:
    """(manifest column, the unit's stored producer table) for a model that reads one."""
    if _AUDIO_DEPS.get(model) == "diarize":
        return "speakers", unit.out_dir / f"{src.prefix}diarize_speakers.csv"
    if model == "speech_rate":
        return "words", unit.out_dir / f"{src.prefix}transcribe_transcript_words.csv"
    return None


def _run_batches(src: Source, model: str, units: list[Unit], args: argparse.Namespace) -> tuple[int, int, list[str]]:
    """One model over many per-item units, ``--batch-items`` units per aud2psy call.

    Each unit keeps its own family + sidecar (the manifest's `output`), so
    done-marking, verify and aggregate see ordinary cells.
    """
    features = ROOTS.durable / src.corpus / "features"
    n_ok = n_fail = 0
    failures: list[str] = []
    ready: list[tuple[Unit, dict]] = []
    for unit in units:
        label = f"{src.key}/{model}[{unit.id}]"
        stem = stem_for(src, unit, model)
        row = {"path": unit.inputs[0], "stimulus_id": unit.extra[unit.extra.index("--stimulus-id") + 1],
               "output": str(stem.relative_to(features))}
        need = [Path(unit.inputs[0])]
        dep = _producer_table(src, unit, model)
        if dep:
            row[dep[0]] = str(dep[1])
            need.append(dep[1])
        missing = [p for p in need if not p.exists()]
        if missing and not args.dry_run:
            print(f"  {label} SKIP: input missing: {missing[0]}")
            n_fail += 1
            failures.append(f"{label} (missing input)")
            continue
        ready.append((unit, row))
    size = max(1, int(getattr(args, "batch_items", None) or BATCH_ITEMS))
    for k in range(0, len(ready), size):
        chunk = ready[k:k + size]
        manifest = ROOTS.durable / src.corpus / "inputs" / "_batch" / f"{src.source}_{src.prefix}{model}_{chunk[0][0].id}.csv"
        cmd = [PY, "-m", "aud2psy.cli", model, "--inputs-from", str(manifest), "-o", str(features),
               "--hop", str(GRID_HOP)]
        head = f"{src.key}/{model}[{chunk[0][0].id} .. {chunk[-1][0].id}]"
        if args.dry_run:
            print(f"  [batch] {head}: {len(chunk)} units in one call\n      {' '.join(cmd)}")
            continue
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(chunk[0][1]))
            w.writeheader()
            w.writerows(r for _, r in chunk)
        print(f"  [batch] {head}: {len(chunk)} units, one model load ...", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd)
        dt = time.time() - t0
        ok_here = 0
        for unit, _ in chunk:
            label = f"{src.key}/{model}[{unit.id}]"
            meta = sidecar_for(stem_for(src, unit, model))
            if not meta.exists():
                n_fail += 1
                failures.append(label)
                continue
            bad = check_sidecar(json.loads(meta.read_text()), source=str(meta))
            if bad:
                for line in bad:
                    print(f"      VIOLATION {line}")
                n_fail += 1
                failures.append(f"{label} (battery)")
                continue
            n_ok += 1
            ok_here += 1
        print(f"      {'ok' if ok_here == len(chunk) else 'PARTIAL'}  {dt:.1f}s, {dt / len(chunk):.2f}s/unit "
              f"({ok_here}/{len(chunk)} done, rc={r.returncode})", flush=True)
        if args.fail_fast and ok_here < len(chunk):
            break
    return n_ok, n_fail, failures


def cmd_run(args: argparse.Namespace) -> int:
    cells = _filtered(args)
    todo = [(s, u, m) for s, u, m in cells if args.redo or not is_done(stem_for(s, u, m), s, u)]
    print(f"{len(cells)} cells matched: {len(todo)} to run, {len(cells) - len(todo)} already done")
    pm = ROOTS.pass_()
    refused: list[str] = []
    if pm is None:
        print("  no pass manifest (--pass-manifest / $PSYTWILL_PASS_MANIFEST): cells are not checked against a pass")
    else:
        allowed = []
        for s_, u_, m_ in todo:
            why = pass_problem(s_, u_, m_)
            if why:
                refused.append(f"{s_.key}/{m_}[{u_.id}]: {why}")
            else:
                allowed.append((s_, u_, m_))
        todo = allowed
        print(f"  pass {pm.name} ({pm.path}): {len(todo)} allowed, {len(refused)} refused")
        for line in refused[:20]:
            print(f"    REFUSED {line}")
        if len(refused) > 20:
            print(f"    ... and {len(refused) - 20} more")
    if not args.dry_run:
        problem = caption_vlm_gpu_problem({m for _, _, m in todo}, gpu_names())
        if problem:
            print(f"ERROR: {problem}", file=sys.stderr)
            return 2
    n_ok = n_fail = 0
    failures: list[str] = []
    # per-item corpora: one batch per (source, model), in the source's model order
    # (producers before the models that read their tables)
    groups: dict[tuple[str, str], tuple[Source, str, list[Unit]]] = {}
    for src, unit, model in todo:
        if _batchable(src, unit):
            groups.setdefault((src.key, model), (src, model, []))[2].append(unit)
    todo = [(s, u, m) for s, u, m in todo if not _batchable(s, u)]
    for src, model, units in sorted(groups.values(), key=lambda g: (g[0].key, g[0].models.index(g[1]))):
        ok, fail, names = _run_batches(src, model, units, args)
        n_ok, n_fail = n_ok + ok, n_fail + fail
        failures += names
        if args.fail_fast and fail:
            todo = []
            break
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
        print(f"\nDone: {n_ok} extracted, {n_fail} failed" + (f", {len(refused)} refused by the pass" if refused else ""))
        for f in failures:
            print(f"  FAILED {f}")
    return 1 if n_fail or refused else 0


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


CENTRE_RULE = ("a row of a staged short-file corpus is kept only if its centre (bin start + hop / 2) falls "
               "inside an item's [onset, offset); rows past an item's end (grid padding of a per-item clip) "
               "and rows in the silence between packed items are excluded")


def _frames_table(corpus: str) -> Path:
    return ROOTS.durable / corpus / "groups" / f"{corpus}_audio_frames_features.parquet"


def _table_layout(table: Path) -> str:
    """``per-recording``, ``per-item`` or ``packed``: the staging record beside the table's corpus
    (absent = packed)."""
    rec = table.parent.parent / "inputs" / "layout.json"
    return json.loads(rec.read_text()).get("layout", "packed") if rec.exists() else "packed"


def cmd_centre_rule(args: argparse.Namespace) -> int:
    """One A exclusion list: the rows of staged corpora whose centre lies outside every item."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    import psytwill
    from psytwill.exceptions import InputError
    from psytwill.relate import sha256_file
    from psytwill.windows import outside_items, table_segments

    tables = [(c, _frames_table(c)) for c in (args.corpus or [])]
    tables += [(Path(t).resolve().parent.parent.name, Path(t).resolve()) for t in (getattr(args, "table", None) or [])]
    if not tables:
        sys.exit("ERROR: name the corpora: --corpus C (under --durable-root) and/or --table PATH "
                 "(a `<corpus>_audio_frames_features.parquet`, e.g. one in another root)")
    names = [c for c, _ in tables]
    if len(set(names)) != len(names):
        sys.exit(f"ERROR: a corpus is named twice ({sorted({c for c in names if names.count(c) > 1})}); "
                 "its rows would be listed twice")
    key, hop = ["stimulus_id", "time"], GRID_HOP
    parts, record = [], {}
    for corpus, table in tables:
        if not table.exists():
            sys.exit(f"ERROR: {table} does not exist; run `extract.py aggregate --corpus {corpus}` first")
        layout = _table_layout(table)
        try:
            segs = table_segments(table)
        except InputError as exc:
            sys.exit(f"ERROR: {exc}")
        if not segs:
            sys.exit(f"ERROR: {table} records no segments; re-run `extract.py aggregate --corpus {corpus} "
                     "--redo`, or `psytwill features --refresh-segments <inputs dir>`")
        many = {sid: s["group"].nunique() for sid, s in segs.items() if s["group"].nunique() > 1}
        if layout in ONE_RECORDING_LAYOUTS and many:
            sid, n = next(iter(many.items()))
            sys.exit(f"ERROR: {corpus}: {len(many)} unit(s) hold more than one item (e.g. {sid}: {n} groups); "
                     f"a {layout} unit holds one")
        seen = []  # the table is long (a row per feature): de-duplicate the key batch by batch
        for batch in pq.ParquetFile(table).iter_batches(columns=key, batch_size=1 << 22):
            d = batch.to_pandas()
            d["time"] = np.floor(d["time"].to_numpy(float) / hop + 1e-9) * hop  # as `space fit --window`
            seen.append(d.drop_duplicates())
        df = (pd.concat(seen, ignore_index=True).drop_duplicates()
              .sort_values(key, kind="stable").reset_index(drop=True))
        missing = sorted(set(df["stimulus_id"]) - set(segs))
        if missing:
            sys.exit(f"ERROR: {corpus}: {len(missing)} unit(s) in {table.name} have no segments table "
                     f"(e.g. {missing[0]}); re-run `extract.py aggregate --corpus {corpus} --redo`")
        labels = [f"{s}|{t}" for s, t in zip(df["stimulus_id"], df["time"])]
        hit = outside_items(labels, segs, hop)
        parts.append(df.loc[hit, key].assign(reason="centre_outside_item"))
        record[corpus] = {"table": str(table), "table_sha256": sha256_file(table), "layout": layout,
                          "n_units": int(df["stimulus_id"].nunique()), "n_rows": len(df),
                          "n_excluded": int(hit.sum())}
        print(f"{corpus} ({layout}): {record[corpus]['n_units']} unit(s), {len(df)} row(s), "
              f"{int(hit.sum())} outside every item")
    rows = pd.concat(parts, ignore_index=True)
    print(f"total: {len(rows)} row(s) excluded")
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
        "what": "A fit exclusion rows (--exclude-rows / space project, with --window 0.5); key stimulus_id,time "
                "on the binned grid; reason centre_outside_item",
        "rule": CENTRE_RULE, "hop": hop, "key": key, "psytwill_version": psytwill.__version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "corpora": record, "n_rows": len(rows), "sha256": sha256_file(out),
    }, indent=1) + "\n")
    print(f"-> {out}")
    return 0


# ---------------------------------------------------------------------------
# carry (psytwill-space 0.3.0 rule C: packed families onto per-recording units)
# ---------------------------------------------------------------------------

#: Old-root entries that are not keyed by a packed unit and are valid as they are
#: (clip-keyed captions, frame units, video sets): `carry --unkeyed link` hard-links them.
_UNKEYED_DIRS = ("features", "inputs")


def _csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def _write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.rename(path)


def _carry_unit(m: dict, *, old: Path, new_in: Path, new_wav: str, force: bool,
                dry_run: bool, tally: dict) -> None:
    """Carry one recording-map row's families (see psytwill.carry)."""
    from psytwill import carry as cr

    piece = cr.Piece(m["old_stimulus_id"], m["new_stimulus_id"], float(m["shift_sec"]),
                     float(m["duration_sec"]))
    unchanged = m["unchanged"] == "true"
    old_dir = old / "features" / m["old_unit"]
    new_dir = new_in.parent / "features" / m["new_unit"]
    metas = sorted(old_dir.glob("*.meta.json"))
    if not metas:
        tally["units without features"] += 1
        return
    for meta in metas:
        raw = meta.read_bytes()
        sc = json.loads(raw)
        model = ",".join(sc.get("models") or {}) or meta.name
        ok, why = (True, "unchanged unit") if unchanged else cr.verdict(sc)
        if not ok:
            tally[f"not carried: {model} ({why})"] += 1
            continue
        target = new_dir / meta.name
        if target.exists() and not force:
            tally["already carried"] += 1
            continue
        outputs = sc.get("output") or {}
        recorded = {k: Path(o["path"]) for k, o in outputs.items()}
        paths = {str(p.parent): str(new_dir) for p in recorded.values()}
        inp = (sc.get("input") or {}).get("path")
        if inp and sc.get("extractor") == "aud2psy":
            paths[inp] = new_wav
        elif inp and sc.get("extractor") == "word2psy":
            paths[inp] = str(new_in / f"{m['new_unit']}_transcript.csv")
        tables: dict[str, list[dict]] = {}
        fields: dict[str, list[str]] = {}
        notes: dict = {}
        if not unchanged:
            read = {k: _csv_rows(old_dir / p.name) for k, p in recorded.items()}
            fields = {k: f for k, (f, _) in read.items()}
            try:
                tables, notes = cr.slice_family(sc, {k: r for k, (_, r) in read.items()}, piece)
            except ValueError as exc:
                sys.exit(f"ERROR: {meta}: {exc}")
            for k, v in notes.items():
                tally[k.replace("_", " ")] += v
            if sc.get("extractor") == "word2psy":
                tr = new_in / f"{m['new_unit']}_transcript.csv"
                want = _csv_rows(tr)[1] if tr.exists() else []
                if [round(float(r["onset"]), 3) for r in want] != \
                        [round(float(c["onset"]), 3) for c in tables["chunks"]]:
                    sys.exit(f"ERROR: {meta}: carried chunks do not match {tr} row for row")
        new = cr.carried_sidecar(sc, piece=piece, paths=paths, tables=tables, old_unit=m["old_unit"],
                                 old_root=str(old), old_sha256=hashlib.sha256(raw).hexdigest(),
                                 unchanged=unchanged, notes=notes)
        problems = check_sidecar(new, source=str(target))
        if problems:
            sys.exit("ERROR: carried sidecar fails the battery check:\n  " + "\n  ".join(problems))
        tally[f"carried: {model}"] += 1
        if dry_run:
            continue
        new_dir.mkdir(parents=True, exist_ok=True)
        for k, p in recorded.items():
            dst = new_dir / p.name
            if unchanged:
                tmp = dst.with_name(dst.name + ".part")
                tmp.write_bytes((old_dir / p.name).read_bytes())
                tmp.rename(dst)
            else:
                _write_rows(dst, fields[k], tables[k])
        tmp = target.with_name(target.name + ".part")
        tmp.write_text(json.dumps(new, indent=2) + "\n")
        tmp.rename(target)  # the sidecar last: it is the done-marker


def _unkeyed(old: Path, new: Path, old_units: set[str]) -> list[tuple[Path, Path]]:
    """(old, new) entries of the old corpus dir that no packed unit owns and
    the new root lacks: clip-keyed captions, frame units, video sets."""
    out = []
    for sub in _UNKEYED_DIRS:
        for p in sorted((old / sub).iterdir()) if (old / sub).is_dir() else []:
            if sub == "features" and p.name in old_units:
                continue
            if sub == "inputs" and (p.name == "units.csv" or p.name.endswith(("_segments.csv", "_transcript.csv"))):
                continue
            if not (new / sub / p.name).exists():
                out.append((p, new / sub / p.name))
    return out


def _link_tree(src: Path, dst: Path) -> int:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
        return 1
    n = 0
    for p in sorted(src.rglob("*")):
        if p.is_file():
            q = dst / p.relative_to(src)
            q.parent.mkdir(parents=True, exist_ok=True)
            os.link(p, q)
            n += 1
    return n


def cmd_carry(args: argparse.Namespace) -> int:
    """Carry a re-staged corpus's packed feature families onto its per-recording units."""
    from collections import Counter

    corpus = args.corpus
    new = ROOTS.durable / corpus
    layout_file = new / "inputs" / "layout.json"
    layout = json.loads(layout_file.read_text()) if layout_file.exists() else {}
    if "restaged_from" not in layout:
        sys.exit(f"ERROR: {layout_file} records no restaged_from; carry follows `stage.py --restage-from`")
    old = Path(layout["restaged_from"])
    if not (old / "features").is_dir():
        sys.exit(f"ERROR: the packed root {old} has no features/")
    if old.resolve() == new.resolve():
        sys.exit(f"ERROR: {old} is the re-staged root itself")
    mrows = _csv_rows(new / "inputs" / layout.get("recording_map", "recording_map.csv"))[1]
    wavs = {u.id: u.inputs[0] for u in _staged_units(corpus)}
    if args.unit_slice:
        a, b = (int(x) if x else None for x in args.unit_slice.split(":"))
        mrows = mrows[a:b]
    tally: Counter = Counter()
    for m in mrows:
        if m["new_unit"] not in wavs:
            sys.exit(f"ERROR: {m['new_unit']} is in the recording map but not in {new / 'inputs' / 'units.csv'}")
        _carry_unit(m, old=old, new_in=new / "inputs", new_wav=wavs[m["new_unit"]],
                    force=args.force, dry_run=args.dry_run, tally=tally)
    print(f"{corpus}: {len(mrows)} recording(s) from {old}" + (" (dry run: nothing written)" if args.dry_run else ""))
    for k in sorted(tally):
        print(f"  {k}: {tally[k]}")
    old_units = {u["unit"] for u in _csv_rows(old / "inputs" / "units.csv")[1]}
    loose = _unkeyed(old, new, old_units)
    if loose:
        if args.unkeyed == "link" and not args.dry_run:
            n = sum(_link_tree(s, d) for s, d in loose)
            print(f"  hard-linked {n} file(s) under {len(loose)} unkeyed entr(ies): "
                  + ", ".join(f"{s.parent.name}/{s.name}" for s, _ in loose))
        else:
            print(f"  {len(loose)} unkeyed entr(ies) not carried (valid as they are; --unkeyed link "
                  "hard-links them): " + ", ".join(f"{s.parent.name}/{s.name}" for s, _ in loose))
    stale = sorted(p.name for p in old.iterdir() if p.is_dir() and p.name not in _UNKEYED_DIRS)
    if stale:
        print(f"  not carried, rebuild from the carried units: {', '.join(stale)} "
              "(e.g. groups/ by `extract.py aggregate`)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--durable-root")
    ap.add_argument("--scratch-root")
    ap.add_argument("--media-root")
    ap.add_argument("--framing-table")
    ap.add_argument("--nsd-hdf5")
    ap.add_argument("--pass-manifest", help="the release's pass manifest (psytwill.passmanifest): `run` "
                                            "refuses cells outside it; `plan` reports coverage")
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
        p.add_argument("--unit-slice", metavar="START:END",
                       help="units [START, END) of each matched source (array tasks over per-item corpora)")
        p.add_argument("--with-attached", action="store_true",
                       help=f"include the attached models ({', '.join(sorted(ATTACHED))}); they are in no "
                            "block, so the fit corpora skip them by default")

    p = sub.add_parser("plan"); filters(p); p.set_defaults(func=cmd_plan)
    p = sub.add_parser("inputs"); p.add_argument("--force", action="store_true"); p.set_defaults(func=cmd_inputs)
    p = sub.add_parser("run"); filters(p)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--batch-items", type=int, default=BATCH_ITEMS,
                   help=f"per-item corpora: units per aud2psy call (default {BATCH_ITEMS})")
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
    p = sub.add_parser("centre-rule",
                       help="one A exclusion list: staged corpora's rows whose centre lies outside every item")
    p.add_argument("--corpus", action="append", choices=sorted(CORPORA),
                   help="a staged corpus under --durable-root, per-item or packed (repeatable)")
    p.add_argument("--table", action="append", metavar="PATH",
                   help="a `<corpus>_audio_frames_features.parquet` named by path, e.g. in another root "
                        "(repeatable; the corpus is its directory's name)")
    p.add_argument("--out", required=True, help="the list to write (immutable once written)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_centre_rule)
    p = sub.add_parser("carry", help="re-staged corpus: slice its packed families onto the per-recording "
                                     "units (rule C, psytwill.carry)")
    p.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    p.add_argument("--unit-slice", metavar="START:END", help="recording-map rows [START, END) (array tasks)")
    p.add_argument("--force", action="store_true", help="rewrite families already carried")
    p.add_argument("--dry-run", action="store_true", help="slice and check every family; write nothing")
    p.add_argument("--unkeyed", choices=("report", "link"), default="report",
                   help="old-root entries no packed unit owns (clip captions, frame units, video sets): "
                        "report them, or hard-link them into the new root")
    p.set_defaults(func=cmd_carry)

    args = ap.parse_args()
    global ROOTS
    ROOTS = Roots(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
