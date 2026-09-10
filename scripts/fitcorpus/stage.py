#!/usr/bin/env python
"""Stage short-file audio corpora into extraction units (psytwill-space).

The fit-corpus driver (`extract.py`) runs the aud2psy battery one *unit*
per cell — one media file in, one feature family out. Speech and music
corpora arrive as thousands of short files, so this verb packs them into
grid-aligned units (`psytwill.staging`): files concatenated in a fixed
order with a silence gap, each starting on the 0.5 s frame grid, and a
segments table saying where every original file sits. The extractors see
one wav per unit and nothing else; downstream pooling recovers the files
from the table.

    stage.py librispeech      --scratch-root R --durable-root D
    stage.py gigaspeech       ... [--budget-hours 40]
    stage.py fma              ... [--per-genre 250]
    stage.py musopen          ... [--budget-hours 8]
    stage.py narratives       ...   (stories are units already; no packing)
    stage.py twp-unpresented  ... --twp-root <bids>/stimuli

Writes (idempotent; a unit whose wav and tables exist is skipped unless
--force):

    <scratch>/<corpus>/units/<unit>.wav            mono PCM16, corpus rate
    <durable>/<corpus>/inputs/<unit>_segments.csv  stimulus_id, segment,
                                                   group, onset, offset,
                                                   duration, text, ...
    <durable>/<corpus>/inputs/<unit>_transcript.csv  (speech corpora only:
                                                   stimulus_id,text,onset,
                                                   offset — the curated text,
                                                   the word2psy input)
    <durable>/<corpus>/inputs/units.csv            the unit manifest the
                                                   driver reads

`stimulus_id` is the unit's `ext-<corpus>-<unit>` id throughout: audio
frames and transcript rows share it, and the segments table is the join
to the original files.

Roots as in extract.py (flags win over env; nothing is guessed):
    --durable-root / $PSYTWILL_FITCORPUS_DURABLE
    --scratch-root / $PSYTWILL_FITCORPUS_SCRATCH
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from psytwill.fitcorpus import ext_id
from psytwill.staging import (
    Segment,
    Unit,
    layout_unit,
    normalize_gigaspeech_text,
    normalize_librispeech_text,
    pack_groups,
)

HOP = 0.5
GAP = 0.5
TARGET_MIN = 25.0  # minutes of audio per packed unit (≈ one CNeuroMod segment ×2)

# A packed unit carries the corpus's native rate; speech corpora are 16 kHz
# (LibriSpeech, GigaSpeech), music 44.1 kHz, the twp recordings 24 kHz.
RATES = {"librispeech": 16000, "gigaspeech": 16000, "fma": 44100,
         "musopen": 44100, "twp-unpresented": 24000}


def slug(s: str) -> str:
    """Lower-case [a-z0-9-] native id, as psytwill.fitcorpus requires."""
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "x"


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def ffmpeg_decode(path: Path, sr: int) -> np.ndarray:
    """Any file -> mono float32 at ``sr`` (ffmpeg; the aud2psy route)."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1",
           "-ar", str(sr), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {r.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(r.stdout, dtype=np.float32)


def sf_decode(src, sr: int) -> np.ndarray:
    """wav/flac (path or bytes) -> mono float32 at ``sr`` via soundfile."""
    import soundfile as sf

    y, fs = sf.read(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else str(src),
                    dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if fs != sr:
        import librosa

        y = librosa.resample(y, orig_sr=fs, target_sr=sr)
    return y


def media_duration(path: Path) -> float:
    import soundfile as sf

    try:
        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001 — a format libsndfile lacks; ask ffprobe
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
        return float(r.stdout.strip())


# ---------------------------------------------------------------------------
# Corpus adapters: each returns (sample_rate, [(unit_prefix, groups)], decode)
#   groups = [(group_key, [Segment])] in packing order; Segment.extra carries
#   what the decoder needs ("path" or "bytes") plus the columns to record.
# ---------------------------------------------------------------------------

def adapt_librispeech(args) -> tuple[int, list[tuple[str, list]], object]:
    root = args.scratch_root / "librispeech" / "train-clean-100" / "LibriSpeech" / "train-clean-100"
    if not root.is_dir():
        sys.exit(f"ERROR: {root} missing — fetch.py librispeech, then untar the archive in place")
    plans = []
    for spk in sorted(root.iterdir(), key=lambda p: int(p.name)):
        groups = []
        for chap in sorted(spk.iterdir(), key=lambda p: int(p.name)):
            trans = chap / f"{spk.name}-{chap.name}.trans.txt"
            text = {}
            for line in trans.read_text().splitlines():
                uid, _, t = line.partition(" ")
                text[uid] = normalize_librispeech_text(t)
            segs = []
            for flac in sorted(chap.glob("*.flac")):
                uid = flac.stem
                segs.append(Segment(segment=slug(uid), group=chap.name,
                                    duration=media_duration(flac), text=text[uid],
                                    extra={"path": str(flac), "speaker": spk.name,
                                           "chapter": chap.name}))
            groups.append((chap.name, segs))
        plans.append((f"spk{spk.name}-", groups))
    return RATES["librispeech"], plans, lambda seg, sr: sf_decode(seg.extra["path"], sr)


def adapt_gigaspeech(args) -> tuple[int, list[tuple[str, list]], object]:
    import pyarrow.parquet as pq

    shards = sorted((args.scratch_root / "gigaspeech" / "parquet-data" / "s").glob("train-*.parquet"))
    if not shards:
        sys.exit("ERROR: no GigaSpeech S parquet shards under the scratch root — fetch.py gigaspeech first")
    meta_cols = ["segment_id", "speaker", "text", "begin_time", "end_time",
                 "audio_id", "title", "source", "category"]
    PODCAST = 1  # class_label order in the dataset card: audiobook, podcast, youtube
    episodes: dict[str, list] = {}
    for shard in shards:
        d = pq.read_table(shard, columns=meta_cols).to_pydict()
        for i in range(len(d["segment_id"])):
            if d["source"][i] != PODCAST:
                continue
            episodes.setdefault(d["audio_id"][i], []).append(
                (d["begin_time"][i], d["segment_id"][i], d["end_time"][i], d["text"][i],
                 d["speaker"][i], d["title"][i], d["category"][i]))
    budget = args.budget_hours * 3600.0
    total = 0.0
    groups = []
    wanted: set[str] = set()
    for aid in sorted(episodes):  # deterministic: episode id order, whole episodes
        rows = sorted(episodes[aid])
        dur = sum(e - b for b, _, e, *_ in rows)
        if total + dur > budget:
            continue  # first-fit keeps whole episodes; a later short one may still fit
        total += dur
        segs = [Segment(segment=slug(sid), group=aid, duration=float(e - b),
                        text=normalize_gigaspeech_text(t),
                        extra={"segment_id": sid, "episode": aid, "speaker": spk,
                               "title": title, "category": cat,
                               "source_begin": b, "source_end": e})
                for b, sid, e, t, spk, title, cat in rows]
        groups.append((aid, segs))
        wanted.update(s.extra["segment_id"] for s in segs)
    print(f"gigaspeech: {len(episodes)} podcast episodes in S, {len(groups)} taken, "
          f"{total / 3600:.1f} h of {args.budget_hours} h budget, {len(wanted)} segments")
    audio: dict[str, bytes] = {}
    if not args.dry_run:
        for shard in shards:
            t = pq.read_table(shard, columns=["segment_id", "audio"],
                              filters=[("segment_id", "in", list(wanted))])
            for sid, a in zip(t.column("segment_id").to_pylist(), t.column("audio").to_pylist()):
                audio[sid] = a["bytes"]
        missing = wanted - set(audio)
        if missing:
            sys.exit(f"ERROR: {len(missing)} selected segments have no audio rows (e.g. {sorted(missing)[:3]})")
    return RATES["gigaspeech"], [("pod-", groups)], lambda seg, sr: sf_decode(audio[seg.extra["segment_id"]], sr)


def adapt_fma(args) -> tuple[int, list[tuple[str, list]], object]:
    import pandas as pd

    root = args.scratch_root / "fma"
    tracks_csv = root / "fma_metadata" / "tracks.csv"
    audio_root = root / "fma_small"
    if not tracks_csv.exists() or not audio_root.is_dir():
        sys.exit(f"ERROR: need {tracks_csv} and {audio_root} — fetch.py fma, then unzip both archives")
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    small = tracks[tracks[("set", "subset")] == "small"]
    genre = small[("track", "genre_top")]
    rng = np.random.default_rng(args.seed)
    plans = []
    for g in sorted(genre.dropna().unique()):
        ids = np.array(sorted(genre[genre == g].index))
        take = np.sort(rng.choice(ids, size=min(args.per_genre, len(ids)), replace=False))
        groups = []
        for tid in take:
            path = audio_root / f"{tid // 1000:03d}" / f"{tid:06d}.mp3"
            if not path.exists():
                print(f"  fma: track {tid} listed but {path.name} missing — skipped")
                continue
            try:
                dur = media_duration(path)
            except Exception as exc:  # noqa: BLE001 — a handful of fma_small clips are corrupt
                print(f"  fma: {path.name} unreadable ({exc}) — skipped")
                continue
            groups.append((str(tid), [Segment(segment=f"{tid:06d}", group=str(tid), duration=dur,
                                               extra={"path": str(path), "genre": g, "track_id": int(tid)})]))
        plans.append((f"{slug(g)}-", groups))
    return RATES["fma"], plans, lambda seg, sr: ffmpeg_decode(Path(seg.extra["path"]), sr)


def adapt_musopen(args) -> tuple[int, list[tuple[str, list]], object]:
    root = args.scratch_root / "musopen" / "Musopen DVD"
    if not root.is_dir():
        sys.exit(f"ERROR: {root} missing — fetch.py musopen, then unzip the archive")
    # Whole works in directory order until the hour budget is spent (the DVD
    # holds ~17 h; the music arm was sized at 15-25 h *with* fma_small).
    budget = args.budget_hours * 3600.0
    total = 0.0
    groups = []
    n_works = 0
    for work_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        movements = sorted(work_dir.glob("*.mp3"))
        if not movements:
            continue
        durs = [media_duration(m) for m in movements]
        if total + sum(durs) > budget:
            continue
        total += sum(durs)
        n_works += 1
        for mp3, dur in zip(movements, durs):
            sid = slug(f"{work_dir.name}-{mp3.stem}")
            groups.append((sid, [Segment(segment=sid, group=sid, duration=dur,
                                         extra={"path": str(mp3), "work": work_dir.name,
                                                "movement": mp3.stem})]))
    print(f"musopen: {n_works} works taken, {total / 3600:.2f} h of {args.budget_hours} h budget")
    return RATES["musopen"], [("dvd-", groups)], lambda seg, sr: ffmpeg_decode(Path(seg.extra["path"]), sr)


def adapt_twp(args) -> tuple[int, list[tuple[str, list]], object]:
    import pandas as pd

    if args.twp_root is None:
        sys.exit("ERROR: --twp-root <bids>/stimuli is required for twp-unpresented")
    reg = args.twp_root / "stimulus_registry" / "twp1000.tsv"
    media = args.twp_root / "twp1000"
    if not reg.exists():
        sys.exit(f"ERROR: {reg} missing (build_stimulus_registry.py writes it)")
    df = pd.read_csv(reg, sep="\t")
    voices = sorted(c[len("audio_file_"):] for c in df.columns if c.startswith("audio_file_"))
    plans = []
    for v in voices:
        groups = []
        for _, row in df.sort_values("itmno").iterrows():
            if row["presented_voice"] == v:
                continue  # presented pairs stay out: the fit corpus is the never-heard 3,000
            path = media / row[f"audio_file_{v}"]
            word = str(row["stimulus_id"])
            groups.append((word, [Segment(segment=slug(word), group=word, duration=media_duration(path),
                                          text=word, extra={"path": str(path), "voice": v,
                                                            "itmno": int(row["itmno"])})]))
        plans.append((f"{v}-", groups))
    return RATES["twp-unpresented"], plans, lambda seg, sr: ffmpeg_decode(Path(seg.extra["path"]), sr)


ADAPTERS = {"librispeech": adapt_librispeech, "gigaspeech": adapt_gigaspeech, "fma": adapt_fma,
            "musopen": adapt_musopen, "twp-unpresented": adapt_twp}


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_unit(corpus: str, unit: Unit, sr: int, decode, args) -> dict:
    import soundfile as sf

    sid = ext_id(corpus, unit.unit)
    wav = args.scratch_root / corpus / "units" / f"{unit.unit}.wav"
    inputs = args.durable_root / corpus / "inputs"
    seg_csv = inputs / f"{unit.unit}_segments.csv"
    tr_csv = inputs / f"{unit.unit}_transcript.csv"
    has_text = any(s.text for s in unit.segments)
    row = {"unit": unit.unit, "stimulus_id": sid, "path": str(wav), "sr": sr,
           "duration_sec": round(unit.duration, 3), "n_segments": len(unit.segments),
           "audio_sec": round(sum(s.duration for s in unit.segments), 3)}
    if wav.exists() and seg_csv.exists() and (tr_csv.exists() or not has_text) and not args.force:
        return row
    wav.parent.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    buf = np.zeros(int(round(unit.duration * sr)), dtype=np.float32)
    for seg, onset in zip(unit.segments, unit.onsets):
        y = decode(seg, sr)
        i = int(round(onset * sr))
        n = min(len(y), len(buf) - i)
        buf[i:i + n] = y[:n]
    tmp = wav.with_suffix(".wav.part")
    sf.write(str(tmp), buf, sr, format="WAV", subtype="PCM_16")
    tmp.rename(wav)
    extra_keys = sorted({k for s in unit.segments for k in s.extra if k not in ("path", "bytes")})
    with open(seg_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stimulus_id", "segment", "group", "onset", "offset", "duration", "text", *extra_keys])
        for seg, onset in zip(unit.segments, unit.onsets):
            w.writerow([sid, seg.segment, seg.group, f"{onset:.3f}", f"{onset + seg.duration:.3f}",
                        f"{seg.duration:.3f}", seg.text, *[seg.extra.get(k, "") for k in extra_keys]])
    if has_text:
        with open(tr_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stimulus_id", "text", "onset", "offset"])
            for seg, onset in zip(unit.segments, unit.onsets):
                if seg.text:
                    w.writerow([sid, seg.text, f"{onset:.3f}", f"{onset + seg.duration:.3f}"])
    return row


def stage_packed(corpus: str, args) -> list[dict]:
    sr, plans, decode = ADAPTERS[corpus](args)
    units: list[Unit] = []
    for prefix, groups in plans:
        units.extend(pack_groups(groups, target=args.target_minutes * 60, hop=HOP, gap=GAP,
                                 unit_prefix=prefix))
    audio_h = sum(s.duration for u in units for s in u.segments) / 3600
    print(f"{corpus}: {len(units)} units, {sum(len(u.segments) for u in units)} segments, "
          f"{audio_h:.2f} h of audio, {sr} Hz, target {args.target_minutes:g} min/unit")
    if args.dry_run:
        for u in units[:5]:
            print(f"  {u.unit}: {len(u.segments)} segments, {u.duration / 60:.1f} min")
        return []
    rows = []
    for k, u in enumerate(units, 1):
        rows.append(write_unit(corpus, u, sr, decode, args))
        if k % 25 == 0 or k == len(units):
            print(f"  {k}/{len(units)} units staged", flush=True)
    return rows


def stage_narratives(args) -> list[dict]:
    """Each story is already one unit; the wav is read in place (no copy)."""
    root = args.scratch_root / "narratives"
    stories = sorted(p for p in (root / "ds002345" / "stimuli").glob("*_audio.wav") if p.resolve().is_file())
    direct = root / "direct"  # files the annex could not verify, pulled by URL
    if direct.is_dir():
        have = {p.name for p in stories}
        stories += sorted(p for p in direct.glob("*_audio.wav") if p.name not in have)
    if not stories:
        sys.exit(f"ERROR: no story audio under {root} — fetch.py narratives first")
    inputs = args.durable_root / "narratives" / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    rows = []
    for wav in sorted(stories, key=lambda p: p.name):
        uid = slug(wav.name[:-len("_audio.wav")])
        sid = ext_id("narratives", uid)
        dur = media_duration(wav)
        import soundfile as sf

        sr = sf.info(str(wav)).samplerate
        seg_csv = inputs / f"{uid}_segments.csv"
        if not seg_csv.exists() or args.force:
            with open(seg_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["stimulus_id", "segment", "group", "onset", "offset", "duration", "text", "story"])
                w.writerow([sid, uid, uid, "0.000", f"{dur:.3f}", f"{dur:.3f}", "", uid])
        rows.append({"unit": uid, "stimulus_id": sid, "path": str(wav), "sr": sr,
                     "duration_sec": round(dur, 3), "n_segments": 1, "audio_sec": round(dur, 3)})
    print(f"narratives: {len(rows)} stories, {sum(r['audio_sec'] for r in rows) / 3600:.2f} h "
          "(transcript = the unit's Whisper output, as for CNeuroMod)")
    return rows


def write_manifest(corpus: str, rows: list[dict], args) -> None:
    out = args.durable_root / corpus / "inputs" / "units.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} units, {sum(r['audio_sec'] for r in rows) / 3600:.2f} h audio)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", choices=sorted([*ADAPTERS, "narratives"]))
    ap.add_argument("--durable-root", default=os.environ.get("PSYTWILL_FITCORPUS_DURABLE"))
    ap.add_argument("--scratch-root", default=os.environ.get("PSYTWILL_FITCORPUS_SCRATCH"))
    ap.add_argument("--twp-root", default=None, help="<bids>/stimuli (twp-unpresented only)")
    ap.add_argument("--target-minutes", type=float, default=TARGET_MIN)
    ap.add_argument("--budget-hours", type=float, default=40.0, help="gigaspeech / musopen: hours of audio to take (whole episodes / works)")
    ap.add_argument("--per-genre", type=int, default=250, help="fma: clips per top genre (seeded draw)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="rewrite units that already exist")
    ap.add_argument("--dry-run", action="store_true", help="plan and report; write nothing")
    args = ap.parse_args()
    for name in ("durable_root", "scratch_root"):
        v = getattr(args, name)
        if not v:
            sys.exit(f"ERROR: --{name.replace('_', '-')} not given and its env var is unset")
        setattr(args, name, Path(v))
    if args.twp_root:
        args.twp_root = Path(args.twp_root)
    rows = stage_narratives(args) if args.corpus == "narratives" else stage_packed(args.corpus, args)
    if rows and not args.dry_run:
        write_manifest(args.corpus, rows, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
