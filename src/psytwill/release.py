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

Two write-time rules (DECIDED 2026-10-10, psytwill-space 0.2.0):

- **Exclusion lists live with the fits.** Every exclusion file a pinned fit
  names (`exclude_ids_file`, `exclude_rows_file`) must sit under the release's
  ``data_root`` (the fit-corpora directory), and its sha256 is pinned beside
  the fits, so a list kept in a working tree can no longer hold a release up.
- **Each member's window is declared.** A block fitted at space schema >= 1.14
  carries every member's context window; a time-keyed block whose member has
  none declared, or a block keyed without time holding a windowed member, is
  refused. Older blocks predate the field and are recorded as such.

**Attached members** (release schema 1.5). A screened candidate can ship beside
the space without being in it: an opt-in producer feature, extracted on the
release's stimuli, in no block, relation or refit. Each entry names the producer
model, its ``basis``, the gap it explores and where its evidence is recorded,
and the base feature columns it ships. ``basis`` is ``bar`` when the candidate
met the pre-registered attached bar, and ``decided`` when a decision attached it
without the bar. Keeping the two apart lets a reader tell "a gain at the
uncorrected level" from "attached as an instrument". Every sidecar of the model
under the entry's root, and every table of its family, is pinned by sha256, so
``release verify`` refuses a release whose attached tables changed.
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

#: 1.1: `model_prefix` -- the store model names a projection writes
#: (`<prefix>_v`, `<prefix>_vl`, ...). Absent = "pspace".
#: 1.2: `data_root` and `exclusions` ({path: sha256} of every exclusion file a
#: pinned fit names, all under `data_root`, checked by :func:`load_release`);
#: each block entry carries `member_windows` ({member: window_sec}, or null
#: for a block fitted before space schema 1.14).
#: 1.3: `labels` -- null, or {table, columns}: the file names (beside the
#: release) of the per-column Contract B 1.2 labels table of every pinned block
#: and its BIDS-style column dictionary, written by :func:`write_release` when
#: any block recorded labels (space schema >= 1.16).
#: 1.4: `member_windows` values may be "grid" or "event" as well as seconds
#: (Contract B 1.4 window labels, space schema 1.17). A time-keyed block whose
#: member is undeclared -- including a 1.14-1.16 manifest's registry entry with
#: no pin, which those schemas recorded as "battery" with null -- is refused.
#: 1.5: `attached` -- a list of attached members ({name, package, model, basis,
#: gap, evidence, root, columns, n_columns_declared, n_cells, extractor_versions,
#: files {path: sha256}}); see the module docstring.
RELEASE_SCHEMA_VERSION = "1.5"
#: The `basis` of an attached member: met the pre-registered bar, or attached by decision.
ATTACHED_BASES = ("bar", "decided")
#: Block / relation manifest fields that name an exclusion file.
EXCLUSION_FIELDS = ("exclude_ids_file", "exclude_rows_file")


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


def _schema_at_least(version: str | None, want: tuple[int, int]) -> bool:
    if not version:
        return False
    major, minor = (int(x) for x in str(version).split(".")[:2])
    return (major, minor) >= want


def member_windows(meta: dict) -> dict | None:
    """``{member: window_sec}`` of a block manifest, checked against its key;
    None for a block fitted before members declared windows (schema < 1.14)."""
    if not _schema_at_least(meta.get("space_schema_version"), (1, 14)):
        return None
    timed = "time" in str(meta.get("key") or "").split(",")
    out, bad = {}, []
    for m in meta["members"]:
        pm = meta["per_member"][m]
        w, src = pm.get("window_sec"), pm.get("window_source", "undeclared")
        if timed and (src == "undeclared" or w is None):
            cols = pm.get("window_undeclared_columns") or []
            what = (f"{len(cols)} column(s) without a window label, e.g. {cols[0]}" if cols else
                    "no window declared (no Contract B 1.4 window labels, no producer window_sec, no registry pin)")
            bad.append(f"{m}: {what}")
        if not timed and w is not None:
            bad.append(f"{m}: window_sec {w} in a block keyed without time ({meta.get('key')})")
        out[m] = w
    if bad:
        raise SpaceError(f"block {meta['block']}: " + "; ".join(bad))
    return out


def _exclusions(metas: Sequence[tuple[str, dict]], data_root: str | Path | None) -> dict:
    files = sorted({str(Path(meta[f]).resolve()) for _, meta in metas for f in EXCLUSION_FIELDS if meta.get(f)})
    if not files:
        return {}
    if data_root is None:
        raise SpaceError("pinned fits name exclusion files " + ", ".join(files) + "; give the release a data "
                         "root (`--data-root`, the fit-corpora directory) so they can be checked to live there")
    root = Path(data_root).resolve()
    outside = [f for f in files if not Path(f).is_relative_to(root)]
    missing = [f for f in files if not Path(f).exists()]
    if outside or missing:
        raise SpaceError(
            "exclusion files must exist under the data root " + str(root) + ": "
            + "; ".join([f"{f} is outside it" for f in outside] + [f"{f} does not exist" for f in missing])
            + ". Copy each list beside the fits (e.g. <data root>/space/exclusions/) and refit, so the "
            "fit's manifest names that copy.")
    return {f: sha256_file(f) for f in files}


def _family_tables(sidecar: Path) -> list[Path]:
    """The tables of a sidecar's family: ``<base>.csv`` and ``<base>_<table>.csv`` beside
    ``<base>.meta.json`` (Contract B §4.1). A file is claimed by the longest sidecar base
    in its directory it matches, so ``caption_x_chunks.csv`` belongs to ``caption_x``,
    not to ``caption``."""
    base = sidecar.name.removesuffix(".meta.json")
    bases = [p.name.removesuffix(".meta.json") for p in sidecar.parent.glob("*.meta.json")]

    def owner(name: str) -> str | None:
        hits = [b for b in bases if name == b + ".csv" or name.startswith(b + "_")]
        return max(hits, key=len) if hits else None

    return sorted(p for p in sidecar.parent.glob(base + "*.csv") if owner(p.name) == base)


def resolve_attached(entries: Sequence[dict], *, root: str | Path | None = None,
                     block_members: Sequence[str] = ()) -> list[dict]:
    """Check and pin each attached member (see the module docstring).

    An entry is ``{model, basis, gap, evidence}``, with optional ``name`` (default: the
    model), ``package``, ``root`` (default: ``root``), ``columns`` (base feature columns
    to ship; default: every declared one) and ``note``.
    """
    out, seen = [], set()
    for e in entries:
        unknown = set(e) - {"name", "model", "package", "basis", "gap", "evidence", "root", "columns", "note"}
        if unknown:
            raise SpaceError(f"attached entry {e.get('model')!r}: unknown field(s) {sorted(unknown)}")
        model = e.get("model")
        if not model:
            raise SpaceError(f"attached entry {e}: needs a `model` (the producer model name)")
        name = e.get("name") or model
        if e.get("basis") not in ATTACHED_BASES:
            raise SpaceError(f"attached {name}: basis must be one of {ATTACHED_BASES}, not {e.get('basis')!r}")
        for f in ("gap", "evidence"):
            if not str(e.get(f) or "").strip():
                raise SpaceError(f"attached {name}: needs `{f}` (the gap it explores / where its screen "
                                 "numbers and the decision are recorded)")
        if name in seen:
            raise SpaceError(f"attached {name!r} given twice")
        seen.add(name)
        if model in set(block_members):
            raise SpaceError(f"attached {name}: {model} is a member of a released block; an attached "
                             "member is in no block")
        r = e.get("root") or root
        if r is None:
            raise SpaceError(f"attached {name}: give a root (the stimulus feature store) to find its tables in")
        r = Path(r).resolve()
        sidecars = []
        for p in sorted(r.rglob("*.meta.json")):
            try:
                meta = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if model in (meta.get("models") or {}) and (not e.get("package") or meta.get("extractor") == e["package"]):
                sidecars.append((p, meta))
        if not sidecars:
            raise SpaceError(f"attached {name}: no sidecar under {r} records model {model!r}")
        declared, files, versions, bad = None, {}, set(), []
        for p, meta in sidecars:
            cols = list(((meta["models"][model].get("features") or {}).get("columns")) or [])
            if declared is None:
                declared = cols
            elif cols != declared:
                bad.append(f"{p} declares other columns than {sidecars[0][0]}")
            tables = _family_tables(p)
            if not tables:
                bad.append(f"{p} has no table beside it")
            for f in [p, *tables]:
                files[str(f)] = sha256_file(f)
            versions.add(str(meta.get("extractor_version")))
        if not declared:
            bad.append(f"the sidecars declare no feature columns for {model}")
        columns = list(e["columns"]) if e.get("columns") is not None else list(declared or [])
        missing = [c for c in columns if c not in set(declared or [])]
        if missing:
            bad.append(f"{len(missing)} selected column(s) not declared by the producer, e.g. {missing[0]}")
        if len(set(columns)) != len(columns) or not columns:
            bad.append("the column selection is empty or repeats a column")
        if bad:
            raise SpaceError(f"attached {name}: " + "; ".join(bad))
        out.append({"name": name, "package": sidecars[0][1].get("extractor"), "model": model,
                    "basis": e["basis"], "gap": e["gap"].strip(), "evidence": e["evidence"].strip(),
                    "note": e.get("note"), "root": str(r), "columns": columns,
                    "n_columns_declared": len(declared), "n_cells": len(sidecars),
                    "extractor_versions": sorted(versions), "files": files})
    return out


def build_release(*, name: str, version: str, blocks: Sequence[str | Path], relations: Sequence[str | Path] = (),
                  absent: Sequence[tuple[Sequence[str], str]] = (), note: str | None = None,
                  model_prefix: str = "pspace", data_root: str | Path | None = None,
                  attached: Sequence[dict] = (), attached_root: str | Path | None = None) -> dict:
    if not model_prefix.isidentifier() or model_prefix != model_prefix.lower():
        raise SpaceError(f"model prefix {model_prefix!r} must be a lower-case identifier (it names store models)")
    out: dict = {"release_schema_version": RELEASE_SCHEMA_VERSION, "name": name, "version": version,
                 "model_prefix": model_prefix, "psytwill_version": __version__,
                 "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "note": note, "blocks": {}, "relations": {}, "absent": [],
                 "data_root": None if data_root is None else str(Path(data_root).resolve()), "exclusions": {},
                 "attached": []}
    by_sha: dict[str, str] = {}
    metas: list[tuple[str, dict]] = []
    for path in blocks:
        meta, pin = _pinned(Path(path))
        if meta.get("kind", "block") != "block" or "block" not in meta:
            raise SpaceError(f"{path} is not a block manifest")
        b = meta["block"]
        if b in out["blocks"]:
            raise SpaceError(f"block {b!r} given twice ({out['blocks'][b]['manifest']} and {path})")
        out["blocks"][b] = {**pin, "k": int(meta["k"]), "members": list(meta["members"]),
                            "fitted_with": meta.get("psytwill_version"),
                            "space_schema_version": meta.get("space_schema_version"),
                            "member_windows": member_windows(meta)}
        by_sha[pin["manifest_sha256"]] = b
        metas.append((str(path), meta))
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
        metas.append((str(path), meta))
    for pair, evidence in absent:
        unknown = [b for b in pair if b not in out["blocks"]]
        if unknown:
            raise SpaceError(f"absent pair {list(pair)}: {unknown} not among this release's blocks")
        if frozenset(pair) in pairs:
            raise SpaceError(f"absent pair {list(pair)} also has a relation in this release")
        out["absent"].append({"pair": list(pair), "k": 0, "evidence": evidence})
    out["exclusions"] = _exclusions(metas, data_root)
    out["attached"] = resolve_attached(attached, root=attached_root,
                                       block_members=[m for b in out["blocks"].values() for m in b["members"]])
    return out


def write_release(release: dict, out_dir: str | Path, *, stem: str | None = None) -> Path:
    """Write the release manifest and, beside it, the labels of its blocks.

    When any pinned block recorded Contract B 1.2 labels, ``<stem>_labels.csv``
    (one stimulus-free row per block member column) and
    ``<stem>_labels.columns.json`` (a BIDS-style dictionary, one entry per
    feature column) are written too, and named under ``labels``.
    """
    from psytwill.labels import has_labels, write_labels

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or release["name"] + "_" + release["version"]
    path = out / f"{stem}.json"
    if path.exists():
        raise SpaceError(f"{path} exists; a release is immutable. Give it a new --version.")
    manifests = [json.loads(Path(e["manifest"]).read_text()) for e in release.get("blocks", {}).values()]
    release["labels"] = None
    if any(has_labels(m) for m in manifests):
        table, columns = write_labels(manifests, out / f"{stem}_labels.csv")
        release["labels"] = {"table": table.name, "columns": columns.name}
    path.write_text(json.dumps(release, indent=2))
    return path


@dataclass
class Release:
    path: Path
    meta: dict

    @property
    def model_prefix(self) -> str:
        return self.meta.get("model_prefix", "pspace")

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
    for f, sha in (meta.get("exclusions") or {}).items():  # release schema >= 1.2
        if not Path(f).exists():
            bad.append(f"exclusion list {f} is missing")
        elif sha256_file(f) != sha:
            bad.append(f"exclusion list {Path(f).name} has changed since the release")
    for a in meta.get("attached") or []:  # release schema >= 1.5
        if a.get("basis") not in ATTACHED_BASES:
            bad.append(f"attached {a.get('name')}: basis {a.get('basis')!r} is not one of {ATTACHED_BASES}")
        changed = [f for f, sha in a.get("files", {}).items() if not Path(f).exists() or sha256_file(f) != sha]
        if changed:
            bad.append(f"attached {a.get('name')}: {len(changed)} pinned file(s) missing or changed, "
                       f"e.g. {Path(changed[0]).name}")
    if bad:
        raise SpaceError(f"release {meta['name']} {meta['version']} does not match disk: " + "; ".join(bad)
                         + ". Restore the released files, or write a new release version.")
    return Release(path=p, meta=meta)
