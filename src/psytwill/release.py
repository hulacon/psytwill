"""Space releases: the fitted blocks and relations that together are "the space".

A block fit (``space fit``) and a relation (``space relate fit``) are each a
manifest + weights pair on disk. A release names a set of them as one
versioned object: every file pinned by sha256, every relation checked to have
been fitted on blocks in the same release, and every pair that was measured
and found to share nothing recorded as an **absence** with a pointer to its
evidence. It is what a consumer cites ("psytwill-space 0.1.0") instead of a
list of loose paths, and :func:`load_release` refuses it if any file it names
has changed since.

A release file is immutable: writing over an existing one is refused, and a
change is a new version.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from psytwill import __version__
from psytwill.exceptions import InputError, SpaceError
from psytwill.relate import RelationFit, load_relation, sha256_file

RELEASE_SCHEMA_VERSION = "1.0"


def _pinned(manifest: Path) -> tuple[dict, dict]:
    if not manifest.exists():
        raise InputError(f"{manifest} does not exist")
    meta = json.loads(manifest.read_text())
    weights = manifest.parent / meta["weights"]
    if not weights.exists():
        raise InputError(f"{manifest.name} names weights {weights.name}, which is not beside it")
    return meta, {"manifest": str(manifest.resolve()), "manifest_sha256": sha256_file(manifest),
                  "weights": str(weights.resolve()), "weights_sha256": sha256_file(weights)}


def parse_absent(spec: str) -> tuple[list[str], str]:
    """``"V,A: evidence text"`` -> (["V", "A"], "evidence text")."""
    pair, sep, evidence = spec.partition(":")
    blocks = [b.strip() for b in pair.split(",") if b.strip()]
    if not sep or len(blocks) != 2 or not evidence.strip():
        raise SpaceError(f"--absent {spec!r}: write it as 'X,Y: <where the measurement is recorded>'")
    return blocks, evidence.strip()


def build_release(*, name: str, version: str, blocks: Sequence[str | Path], relations: Sequence[str | Path] = (),
                  absent: Sequence[tuple[Sequence[str], str]] = (), note: str | None = None) -> dict:
    out: dict = {"release_schema_version": RELEASE_SCHEMA_VERSION, "name": name, "version": version,
                 "psytwill_version": __version__,
                 "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "note": note, "blocks": {}, "relations": {}, "absent": []}
    by_sha: dict[str, str] = {}
    for path in blocks:
        meta, pin = _pinned(Path(path))
        if meta.get("kind", "block") != "block" or "block" not in meta:
            raise SpaceError(f"{path} is not a block manifest")
        b = meta["block"]
        if b in out["blocks"]:
            raise SpaceError(f"block {b!r} given twice ({out['blocks'][b]['manifest']} and {path})")
        out["blocks"][b] = {**pin, "k": int(meta["k"]), "members": list(meta["members"]),
                            "fitted_with": meta.get("psytwill_version"),
                            "space_schema_version": meta.get("space_schema_version")}
        by_sha[pin["manifest_sha256"]] = b
    pairs: set[frozenset] = set()
    for path in relations:
        meta, pin = _pinned(Path(path))
        if meta.get("kind") != "relation":
            raise SpaceError(f"{path} is not a relation manifest")
        sides = {}
        for side in ("a", "b"):
            sha = meta["sides"][side]["manifest_sha256"]
            if sha not in by_sha:
                raise SpaceError(
                    f"relation {meta['name']} side {side} was fitted on {meta['sides'][side]['manifest']} "
                    f"(block {meta['sides'][side]['block']}), which is not one of this release's blocks "
                    "(or has changed since). Release the blocks it was fitted on, or refit it.")
            sides[side] = by_sha[sha]
        if meta["name"] in out["relations"]:
            raise SpaceError(f"relation {meta['name']!r} given twice")
        out["relations"][meta["name"]] = {**pin, "k": int(meta["k"]), "a": sides["a"], "b": sides["b"],
                                          "scope": meta.get("scope"), "fitted_with": meta.get("psytwill_version")}
        pairs.add(frozenset((sides["a"], sides["b"])))
    for pair, evidence in absent:
        unknown = [b for b in pair if b not in out["blocks"]]
        if unknown:
            raise SpaceError(f"absent pair {list(pair)}: {unknown} not among this release's blocks")
        if frozenset(pair) in pairs:
            raise SpaceError(f"absent pair {list(pair)} also has a relation in this release")
        out["absent"].append({"pair": list(pair), "k": 0, "evidence": evidence})
    return out


def write_release(release: dict, out_dir: str | Path, *, stem: str | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem or release['name'] + '_' + release['version']}.json"
    if path.exists():
        raise SpaceError(f"{path} exists; a release is immutable. Give it a new --version.")
    path.write_text(json.dumps(release, indent=2))
    return path


@dataclass
class Release:
    path: Path
    meta: dict

    @property
    def label(self) -> str:
        return f"{self.meta['name']} {self.meta['version']}"

    def block(self, name: str):
        from psytwill.space import load_fit

        return load_fit(self.meta["blocks"][name]["manifest"])

    def relation(self, name: str) -> RelationFit:
        return load_relation(self.meta["relations"][name]["manifest"])


def load_release(path: str | Path) -> Release:
    """Read a release and refuse it if any file it pins has changed or moved."""
    p = Path(path)
    meta = json.loads(p.read_text())
    if "release_schema_version" not in meta:
        raise InputError(f"{p} is not a release manifest")
    bad = []
    for kind in ("blocks", "relations"):
        for nm, entry in meta[kind].items():
            for f in ("manifest", "weights"):
                fp = Path(entry[f])
                if not fp.exists():
                    bad.append(f"{kind[:-1]} {nm}: {fp} is missing")
                elif sha256_file(fp) != entry[f"{f}_sha256"]:
                    bad.append(f"{kind[:-1]} {nm}: {fp.name} has changed since the release")
    if bad:
        raise SpaceError(f"release {meta['name']} {meta['version']} does not match disk: " + "; ".join(bad)
                         + ". Restore the released files, or write a new release version.")
    return Release(path=p, meta=meta)
