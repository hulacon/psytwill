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
    fetch.py fma | musopen | gigaspeech | narratives --scratch-root /path/...

After fetching, short-file corpora are packed into units by stage.py.
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
FMA_URLS = {
    "fma_small.zip": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
    "fma_metadata.zip": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
}
FMA_METADATA_MEMBERS = ("fma_metadata/tracks.csv", "fma_metadata/genres.csv", "fma_metadata/README.txt")
MUSOPEN_URL = "https://archive.org/download/musopen-dvd/Musopen-DVD.zip"  # PD compilation, 2012
GIGASPEECH_REPO = "speechcolab/gigaspeech"  # HF-gated; parquet-data/s/ = the S subset with audio
NARRATIVES_GIT = "https://github.com/OpenNeuroDatasets/ds002345.git"
LIBRISPEECH_SUBSETS = (
    "dev-clean",
    "test-clean",
    "train-clean-100",
    "train-clean-360",
)

# Corpora whose acquisition is documented, not scripted. Keep reasons loud:
# a silent stub would read as "not yet implemented" instead of "on purpose".
MANUAL = {
    "peoples-speech": (
        "HuggingFace (MLCommons/peoples_speech, CC-BY subset only). Same "
        "HF_TOKEN route as gigaspeech; same open corpus decision."
    ),
    "jamendo": (
        "MTG-Jamendo ships per-split tarballs via its own downloader "
        "(mtg/mtg-jamendo-dataset). Music-arm inclusion is an open "
        "decision — do not fetch before it is made."
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
    if dest.exists():
        print(f"present {dest}")
        return
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


def _unzip(zip_path: Path, dest: Path, members=None, skip_prefixes=("__MACOSX/",)) -> int:
    """Extract (idempotently: existing files are kept) and count members written."""
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for member in (members or z.namelist()):
            if member.endswith("/") or any(member.startswith(p) for p in skip_prefixes) \
                    or Path(member).name == ".DS_Store":
                continue
            out = dest / member
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def fetch_fma(scratch_root: Path) -> Path:
    corpus_dir = scratch_root / "fma"
    if manifest_ok(corpus_dir) and (corpus_dir / "fma_small").is_dir():
        print(f"fma already present and verified at {corpus_dir}")
        return corpus_dir
    corpus_dir.mkdir(parents=True, exist_ok=True)
    zips = []
    for name, url in FMA_URLS.items():
        download(url, corpus_dir / name)
        zips.append(corpus_dir / name)
    n = _unzip(corpus_dir / "fma_small.zip", corpus_dir)
    n += _unzip(corpus_dir / "fma_metadata.zip", corpus_dir, members=FMA_METADATA_MEMBERS)
    write_manifest(corpus_dir, zips)
    print(f"fma ready: {n} files extracted under {corpus_dir} (fma_small + tracks.csv)")
    return corpus_dir


def fetch_musopen(scratch_root: Path) -> Path:
    corpus_dir = scratch_root / "musopen"
    if manifest_ok(corpus_dir) and (corpus_dir / "Musopen DVD").is_dir():
        print(f"musopen already present and verified at {corpus_dir}")
        return corpus_dir
    corpus_dir.mkdir(parents=True, exist_ok=True)
    zip_path = corpus_dir / "Musopen-DVD.zip"
    download(MUSOPEN_URL, zip_path)
    n = _unzip(zip_path, corpus_dir)
    write_manifest(corpus_dir, [zip_path])
    print(f"musopen ready: {n} files extracted under {corpus_dir}")
    return corpus_dir


def fetch_gigaspeech(scratch_root: Path) -> Path:
    """The S subset as HF parquet shards (audio bytes + metadata in one table).

    Needs the gated license accepted on the Hub and a token the hub client
    can find ($HF_TOKEN or the cached login). Podcast selection and packing
    happen in stage.py; this pulls the whole subset so the draw is
    reproducible from the shards.
    """
    from huggingface_hub import HfApi, hf_hub_download

    corpus_dir = scratch_root / "gigaspeech"
    if manifest_ok(corpus_dir):
        print(f"gigaspeech already present and verified at {corpus_dir}")
        return corpus_dir
    corpus_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(GIGASPEECH_REPO)
    shards = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith("parquet-data/s/train-"))
    if not shards:
        raise SystemExit("gigaspeech: no parquet-data/s/train-* shards listed — dataset layout changed?")
    files = []
    for rel in shards:
        p = hf_hub_download(GIGASPEECH_REPO, rel, repo_type="dataset", local_dir=corpus_dir)
        files.append(Path(p))
        print(f"  {rel}")
    write_manifest(corpus_dir, files)
    print(f"gigaspeech ready: {len(files)} S-subset shards under {corpus_dir}")
    return corpus_dir


def fetch_narratives(scratch_root: Path) -> Path:
    """OpenNeuro ds002345 stimuli via datalad, with a direct-URL fallback.

    Some annexed stimuli fail datalad's content verification from the
    public S3 copy; those are pulled by their recorded URL into
    ``narratives/direct/`` and stage.py reads them from there.
    """
    import json
    import subprocess

    if shutil.which("datalad") is None or shutil.which("git-annex") is None:
        raise SystemExit("narratives: datalad and git-annex must be on PATH (a shared env carries them)")
    corpus_dir = scratch_root / "narratives"
    ds = corpus_dir / "ds002345"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    if not ds.is_dir():
        subprocess.run(["datalad", "clone", NARRATIVES_GIT, str(ds)], check=True)
    subprocess.run(["datalad", "get", "-J", "1", "stimuli"], cwd=ds, check=False)
    stories = sorted((ds / "stimuli").glob("*_audio.wav"))
    missing = [p for p in stories if not p.resolve().exists()]
    direct = corpus_dir / "direct"
    for p in missing:
        out = direct / p.name
        if out.exists():
            continue
        r = subprocess.run(["git", "annex", "whereis", "--json", str(p.relative_to(ds))],
                           cwd=ds, capture_output=True, text=True, check=True)
        urls = [u for w in json.loads(r.stdout.splitlines()[0])["whereis"] for u in w.get("urls", [])]
        if not urls:
            print(f"  {p.name}: no URL recorded — left missing")
            continue
        direct.mkdir(parents=True, exist_ok=True)
        download(urls[0], out)
    n_ok = len(stories) - len(missing) + sum((direct / p.name).exists() for p in missing)
    print(f"narratives ready: {n_ok}/{len(stories)} stories ({len(missing)} via direct URL)")
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

    scripted = {"librispeech": None, "fma": fetch_fma, "musopen": fetch_musopen,
                "gigaspeech": fetch_gigaspeech, "narratives": fetch_narratives}
    if args.list or not args.corpus:
        for key, spec in sorted(CORPORA.items()):
            mode = "scripted" if key in scripted else "manual"
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
    if scripted.get(args.corpus):
        scripted[args.corpus](resolve_root(
            args.scratch_root, "PSYTWILL_FITCORPUS_SCRATCH", "scratch"))
        return 0
    if args.corpus in MANUAL:
        print(f"{args.corpus}: no scripted fetch —\n{MANUAL[args.corpus]}")
        return 1
    raise SystemExit(
        f"unknown corpus {args.corpus!r}; run with --list"
    )


if __name__ == "__main__":
    sys.exit(main())
