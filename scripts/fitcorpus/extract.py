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

Roots (flags win over env; there are no defaults -- a site names them):
    --durable-root / $PSYTWILL_FITCORPUS_DURABLE   feature store
    --scratch-root / $PSYTWILL_FITCORPUS_SCRATCH   staged raw media
    --nsd-hdf5     / $PSYTWILL_NSD_HDF5            the 73k NSD image brick

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
from psytwill.fitcorpus import CORPORA, ext_id

PY = sys.executable
GRID_HOP = 0.5  # shared frame grid, matching the mmmdata store
NSD_UNIT = 1000  # images per NSD unit (== the shared1000 arm's proven size)
NSD_TOTAL = 73000

# Block members by extractor (psytwill-space block basis, ratified
# 2026-09-07). caption / transcribe are producers; diarize is a schema.
V_MODELS = [m.model for m in BATTERY.values() if m.extractor == "viz2psy" and m.model != "caption"]
A_MODELS = [m.model for m in BATTERY.values() if m.extractor == "aud2psy"]
L_MODELS = [m.model for m in BATTERY.values() if m.extractor == "word2psy"]

# aud2psy models that consume another cell's table (no second inference pass)
_AUDIO_DEPS = {"conversation": "diarize", "speech_rate": "transcribe"}


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
    """Staged (annex content present) .mkv segments of one CNeuroMod corpus."""
    if ROOTS.scratch is None:
        return []
    root = ROOTS.scratch / "cneuromod" / "algonauts_2025.competitors" / "stimuli" / "movies" / corpus
    if not root.exists():
        return []
    files = []
    for p in sorted(root.rglob("*.mkv")):
        if ".git" in p.parts:
            continue
        if p.resolve().is_file():  # annexed symlink with content
            files.append(p)
    return files


def _cneuromod_native(corpus: str, path: Path) -> str:
    stem = path.stem  # bourne01 | friends_s01e02a
    if corpus == "friends" and stem.startswith("friends_"):
        stem = stem[len("friends_"):]
    return stem.lower().replace("_", "-")


def _cneuromod_units(corpus: str) -> list[Unit]:
    out = []
    for p in _cneuromod_media(corpus):
        native = _cneuromod_native(corpus, p)
        out.append(Unit(
            id=native,
            out_dir=ROOTS.durable / corpus / "features" / native,
            inputs=[str(p)],
            extra=["--stimulus-id", ext_id(corpus, native)],
        ))
    return out


def _cneuromod_transcript_units(corpus: str) -> list[Unit]:
    out = []
    for p in _cneuromod_media(corpus):
        native = _cneuromod_native(corpus, p)
        d = ROOTS.durable / corpus / "features" / native
        shard = ROOTS.durable / corpus / "inputs" / f"{native}_transcript.csv"
        out.append(Unit(
            id=native,
            out_dir=d,
            inputs=[str(shard)],
            extra=["--text-column", "text", "--id-column", "stimulus_id"],
        ))
    return out


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
    for corpus in ("movie10", "friends"):
        S.append(Source(
            corpus, "frames", "viz2psy", "", V_MODELS,
            f"staged CNeuroMod {corpus} segments, frames every {GRID_HOP} s",
            lambda c=corpus: _cneuromod_units(c),
        ))
        S.append(Source(
            corpus, "audio", "aud2psy", "", _ordered_audio(),
            f"soundtracks of the same {corpus} segments, {GRID_HOP} s hop; "
            "diarize/transcribe run before conversation/speech_rate",
            lambda c=corpus: _cneuromod_units(c),
        ))
        S.append(Source(
            corpus, "transcript", "word2psy", "transcript_", L_MODELS,
            f"Whisper transcript segments of {corpus} (what is said), scored as text",
            lambda c=corpus: _cneuromod_transcript_units(c),
            depends=f"{corpus}/audio/transcribe",
        ))
    return S


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def stem_for(src: Source, unit: Unit, model: str) -> Path:
    return unit.out_dir / f"{src.prefix}{model}.csv"


def sidecar_for(stem: Path) -> Path:
    return stem.with_suffix(".meta.json")


def is_done(stem: Path) -> bool:
    """Sidecar present AND clean against the clamped battery."""
    meta = sidecar_for(stem)
    if not meta.exists():
        return False
    try:
        sidecar = json.loads(meta.read_text())
    except (OSError, ValueError):
        return False
    return not check_sidecar(sidecar, source=str(meta))


def command_for(src: Source, unit: Unit, model: str) -> list[str]:
    stem = stem_for(src, unit, model)
    if src.package == "viz2psy":
        cmd = [PY, "-m", "viz2psy.cli", model, *unit.inputs,
               "-o", str(stem), "--batch-size", "64", "--no-viz", "--quiet"]
        if src.source == "frames":
            cmd += ["--frame-interval", str(GRID_HOP), "--no-save-frames"]
        return cmd + unit.extra
    if src.package == "aud2psy":
        cmd = [PY, "-m", "aud2psy.cli", model, *unit.inputs,
               "-o", str(stem), "--hop", str(GRID_HOP), *unit.extra]
        if model == "conversation":
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
        models = [m for m in src.models if not args.model or m == args.model]
        n = len(units) * len(models)
        done = sum(is_done(stem_for(src, u, m)) for u in units for m in models)
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
    """Derived input CSVs: NSD caption shards; CNeuroMod transcript inputs."""
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
    for corpus in ("movie10", "friends"):
        n = sum(_ensure_transcript_input(corpus, u, force=args.force) for u in _cneuromod_units(corpus))
        print(f"{corpus} transcripts: {n} input CSVs written")
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


def cmd_run(args: argparse.Namespace) -> int:
    cells = _filtered(args)
    todo = [(s, u, m) for s, u, m in cells if args.redo or not is_done(stem_for(s, u, m))]
    print(f"{len(cells)} cells matched: {len(todo)} to run, {len(cells) - len(todo)} already done")
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


def aggregate_groups(args: argparse.Namespace) -> tuple[dict[tuple[str, str, str], list[Path]], list[tuple[Path, str]]]:
    groups: dict[tuple[str, str, str], list[Path]] = {}
    skipped: list[tuple[Path, str]] = []
    for src, unit, model in _filtered(args):
        if not is_done(stem_for(src, unit, model)):
            continue
        for table, path in family_tables(src, unit, model).items():
            usable, why = _table_usable(path)
            if not usable:
                skipped.append((path, why))
                continue
            groups.setdefault((src.corpus, src.source, table), []).append(path)
    return {k: sorted(v) for k, v in sorted(groups.items())}, skipped


def agg_output(key: tuple[str, str, str]) -> Path:
    corpus, source, table = key
    name = f"{corpus}_{source}" + ("" if table == "main" else f"_{table}")
    return ROOTS.durable / corpus / "groups" / f"{name}_features.parquet"


def cmd_aggregate(args: argparse.Namespace) -> int:
    if args.model:
        sys.exit("ERROR: `aggregate --model` would write a partial group (a group holds every model of its table); narrow with --corpus/--source instead")
    if args.unit or args.unit_index is not None:
        sys.exit("ERROR: `aggregate` is per corpus/source, not per unit")
    groups, skipped = aggregate_groups(args)
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
    # a group holds every unit of its (corpus, source); incomplete ones are
    # reported and left unbuilt unless --allow-partial
    complete: dict[tuple[str, str, str], list[Path]] = {}
    for key, paths in groups.items():
        corpus, source, table = key
        src = next(s_ for s_ in build_sources() if s_.corpus == corpus and s_.source == source)
        n_units, n_seen = len(src.units()), len({p.parent for p in paths})
        if n_seen < n_units and not args.allow_partial:
            print(f"  INCOMPLETE {corpus}/{source}/{table}: {n_seen}/{n_units} units done — skipped")
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
    print(f"\n{len(todo) - n_fail}/{len(todo)} group(s) built")
    return 1 if n_fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--durable-root")
    ap.add_argument("--scratch-root")
    ap.add_argument("--nsd-hdf5")
    sub = ap.add_subparsers(dest="verb", required=True)

    def filters(p: argparse.ArgumentParser) -> None:
        p.add_argument("--corpus", choices=sorted(CORPORA))
        p.add_argument("--source")
        p.add_argument("--model", choices=sorted(BATTERY))
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

    args = ap.parse_args()
    global ROOTS
    ROOTS = Roots(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
