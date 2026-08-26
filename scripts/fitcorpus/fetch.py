#!/usr/bin/env python3
"""Idempotent fetchers for psytwill-space fit corpora.

One verb per corpus registered in ``psytwill.fitcorpus.CORPORA``. Scripted
downloads exist for the tier-(a) corpora that need no auth or external
tooling; the rest print their documented acquisition procedure and exit
non-zero (auth-gated and tier-(b) acquisitions are deliberate manual steps).

Destinations are explicit — nothing site-specific is hard-coded:

- durable root (small, redistributable artifacts, e.g. COCO captions):
  ``--durable-root`` or ``$PSYTWILL_FITCORPUS_DURABLE``
- scratch root (bulk raw media, re-downloadable by construction):
  ``--scratch-root`` or ``$PSYTWILL_FITCORPUS_SCRATCH``

Every fetched corpus directory gets a ``MANIFEST.sha256``; a rerun verifies
it and skips the download, so fetches are safe to re-issue. Scratch is never
the sole copy of anything not re-downloadable.

Usage:
    fetch.py --list
    fetch.py coco-captions --durable-root /path/to/fit-corpora
    fetch.py librispeech --subset train-clean-100 --scratch-root /path/...
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

MANIFEST = "MANIFEST.sha256"

COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_CAPTION_MEMBERS = (
    "annotations/captions_train2017.json",
    "annotations/captions_val2017.json",
)

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/{subset}.tar.gz"
LIBRISPEECH_SUBSETS = (
    "dev-clean",
    "test-clean",
    "train-clean-100",
    "train-clean-360",
)

# Corpora whose acquisition is documented, not scripted. Keep reasons loud:
# a silent stub would read as "not yet implemented" instead of "on purpose".
MANUAL = {
    "narratives": (
        "OpenNeuro ds002345 (CC0). Scripted download needs datalad:\n"
        "  datalad clone https://github.com/OpenNeuroDatasets/ds002345.git\n"
        "  datalad get stimuli/ (audio + aligned transcripts)\n"
        "Brain data pulled the same way when the validation leg runs."
    ),
    "gigaspeech": (
        "HuggingFace-gated (speechcolab/gigaspeech): accept the license on "
        "HF, export HF_TOKEN, then pull the chosen subset via the datasets "
        "library inside the extraction job. Subset + hours are an open "
        "corpus decision — record it in the workbench before fetching."
    ),
    "peoples-speech": (
        "HuggingFace (MLCommons/peoples_speech, CC-BY subset only). Same "
        "HF_TOKEN route as gigaspeech; same open corpus decision."
    ),
    "jamendo": (
        "MTG-Jamendo ships per-split tarballs via its own downloader "
        "(mtg/mtg-jamendo-dataset). Music-arm inclusion is an open "
        "decision — do not fetch before it is made."
    ),
    "fma": (
        "Free Music Archive tarballs (fma_small/fma_medium) from the "
        "mdeff/fma release page. Same open music-arm decision as jamendo."
    ),
    "friends": (
        "Tier (b): commercial media via the CNeuroMod data agreement plus "
        "lawfully obtained copies; their repos regenerate the stimulus "
        "derivatives. Manual procedure, never redistributed, scratch only."
    ),
    "movie10": (
        "Tier (b): same procedure as friends (CNeuroMod movie10)."
    ),
    "nsd": "Already local: read the NSD stimuli hdf5 in place; nothing to fetch.",
    "twp-unpresented": "Already local beside the mmmdata stimuli; nothing to fetch.",
    "things": (
        "CNeuroMod-THINGS release (images + CC0 betas); fetched by the "
        "validation leg, not as a fit corpus."
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(corpus_dir: Path, files: list[Path]) -> None:
    lines = [
        f"{sha256(p)}  {p.relative_to(corpus_dir)}"
        for p in sorted(files)
    ]
    (corpus_dir / MANIFEST).write_text("\n".join(lines) + "\n")


def manifest_ok(corpus_dir: Path) -> bool:
    manifest = corpus_dir / MANIFEST
    if not manifest.exists():
        return False
    for line in manifest.read_text().splitlines():
        digest, _, rel = line.partition("  ")
        path = corpus_dir / rel
        if not path.exists() or sha256(path) != digest:
            return False
    return True


def download(url: str, dest: Path) -> None:
    print(f"fetching {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)


def fetch_coco_captions(durable_root: Path) -> Path:
    corpus_dir = durable_root / "coco-captions"
    if manifest_ok(corpus_dir):
        print(f"coco-captions already present and verified at {corpus_dir}")
        return corpus_dir
    corpus_dir.mkdir(parents=True, exist_ok=True)
    zip_path = corpus_dir / "annotations_trainval2017.zip"
    if not zip_path.exists():
        download(COCO_ANNOTATIONS_URL, zip_path)
    kept = []
    with zipfile.ZipFile(zip_path) as z:
        for member in COCO_CAPTION_MEMBERS:
            out = corpus_dir / Path(member).name
            with z.open(member) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            kept.append(out)
    zip_path.unlink()  # instances/keypoints not needed; the zip re-downloads
    write_manifest(corpus_dir, kept)
    print(f"coco-captions ready: {[p.name for p in kept]} in {corpus_dir}")
    return corpus_dir


def fetch_librispeech(scratch_root: Path, subset: str) -> Path:
    if subset not in LIBRISPEECH_SUBSETS:
        raise SystemExit(
            f"unknown librispeech subset {subset!r}; "
            f"choose from {LIBRISPEECH_SUBSETS}"
        )
    corpus_dir = scratch_root / "librispeech" / subset
    if manifest_ok(corpus_dir):
        print(f"librispeech/{subset} already present and verified")
        return corpus_dir
    corpus_dir.mkdir(parents=True, exist_ok=True)
    tar_path = corpus_dir / f"{subset}.tar.gz"
    download(LIBRISPEECH_URL.format(subset=subset), tar_path)
    write_manifest(corpus_dir, [tar_path])
    print(
        f"librispeech/{subset} tarball staged at {tar_path}; "
        "extraction happens at adapt time"
    )
    return corpus_dir


def resolve_root(flag: str | None, env: str, kind: str) -> Path:
    value = flag or os.environ.get(env)
    if not value:
        raise SystemExit(
            f"no {kind} root: pass --{kind}-root or set ${env}"
        )
    return Path(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("corpus", nargs="?", help="registered corpus key")
    parser.add_argument("--list", action="store_true", help="list corpora")
    parser.add_argument("--durable-root", default=None)
    parser.add_argument("--scratch-root", default=None)
    parser.add_argument(
        "--subset", default="train-clean-100",
        help="librispeech subset (default: train-clean-100)",
    )
    args = parser.parse_args(argv)

    # Import late so --list/--help work without the package installed.
    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
    from psytwill.fitcorpus import CORPORA

    if args.list or not args.corpus:
        for key, spec in sorted(CORPORA.items()):
            mode = "scripted" if key == "librispeech" else "manual"
            print(f"{key:18} tier={spec.tier:10} [{mode}]  {spec.serves}")
        print(
            "\ncoco-captions     tier=a          [scripted]  V<->L language "
            "side (rides nsd ids, so not a registry key)"
        )
        return 0

    if args.corpus == "coco-captions":
        fetch_coco_captions(resolve_root(
            args.durable_root, "PSYTWILL_FITCORPUS_DURABLE", "durable"))
        return 0
    if args.corpus == "librispeech":
        fetch_librispeech(resolve_root(
            args.scratch_root, "PSYTWILL_FITCORPUS_SCRATCH", "scratch"),
            args.subset)
        return 0
    if args.corpus in MANUAL:
        print(f"{args.corpus}: no scripted fetch —\n{MANUAL[args.corpus]}")
        return 1
    raise SystemExit(
        f"unknown corpus {args.corpus!r}; run with --list"
    )


if __name__ == "__main__":
    sys.exit(main())
