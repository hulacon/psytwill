"""A stimulus set placed in a space release, written as a Contract B family.

``space project`` / ``space relate project`` write wide working tables for one
block or one relation side. A consumer that is not psytwill needs more: every
coordinate traceable to a pinned release, the set's own registry ids, and
proof that the stimuli it is about to analyse were not rows of the fit that
made the coordinates (contracts §4.3 item 3). This module writes that: one
§4.1 family (CSV tables + one ``.meta.json``) per stimulus set, with a model
per block (``pspace_v``, ``pspace_a``, ``pspace_l``) and per relation side
(``pspace_vl``).

Conventions (psytwill-space project-design, DECIDED 2026-10-02):

- **Through the release.** Blocks and relations are loaded from a release
  (:func:`psytwill.release.load_release`, tamper-evident), never from loose
  score files, and the family's checkpoints name the release version and the
  manifest/weights sha256.
- **Registry ids, one to one.** Source ids are mapped onto the set's registry
  (a join column + a pattern for external ids, e.g. ``ext-nsd-(\\d+)`` against
  ``nsdId``), and the mapping must be one to one and cover every registry
  stimulus. A sub-stimulus key that the source numbers differently (caption
  ``chunk_idx``) is mapped by an explicit key-map table, checked the same way.
- **Leak guard.** A block passes when the projected table is not among its
  fit's inputs, or every projected source id is in the fit's
  ``exclude_ids_file``. A relation passes when no projected source id is a
  row of either score table it was fitted on. Anything else is refused. (A
  path test cannot see the same stimulus filed under another path in a fit
  corpus; the sidecar records which rule passed.)
- **Absent members are declared, never filled.** ``--skip-member`` names a
  block member a set does not carry at all (twp1000's single-word files have
  no turn or word tables for A's ``conversation`` / ``speech_rate``). The
  member is masked in every row, exactly as an ``undefined`` member is, and
  the sidecar lists it per table.
- **Coverage.** By default the family must cover the whole registry. A source
  that legitimately lacks some stimuli (films without dialogue have no
  transcript) passes ``--partial-coverage``, and the registry ids it does not
  cover are listed in the sidecar.
- **Unplaced rows stay.** A row the block cannot place is written as NaN and
  declared ``undefined`` in ``nulls``, so every source row has a row and an
  absence is visible.
- **Releases do not overwrite each other.** Re-running the same release
  rewrites the family; a different release first moves the existing family
  into ``<stem>_<old version>/`` beside it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Sequence

import numpy as np
import pandas as pd

from psytwill import __version__
from psytwill.exceptions import InputError, SpaceError
from psytwill.relate import sha256_file
from psytwill.release import Release
from psytwill.sidecar import _TABLE_SUFFIXES

FAMILY_SCHEMA_VERSION = "1.1"  # Contract B §4.1
BLOCK_MODALITY = {"V": "visual", "A": "audio", "L": "text"}
TABLE_NAMES = tuple(s.lstrip("_") for s in _TABLE_SUFFIXES)
UNPLACED_WHEN = ("the block cannot place this row: its present members cannot determine k directions, "
                 "or (also counted here) a member holds an `undefinable` null")


def model_name(name: str) -> str:
    return f"pspace_{name.lower()}"


def _key_value(x) -> float | str:
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)


def _key_out(x):
    """A key value as written: integral numbers as int, other numbers as float, else str."""
    v = _key_value(x)
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


@dataclass
class Grain:
    """One table of the family: a block's rows at one key."""

    table: str  # "" = the family's base table, <stem>.csv
    block: str
    key: list[str]
    features: list[str]
    key_map: str | None = None
    skip_members: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.table and self.table not in TABLE_NAMES:
            raise SpaceError(
                f"table {self.table!r} is not a §4.1 table suffix psytwill can find a sidecar from "
                f"({', '.join(TABLE_NAMES)}); use one of them, or '-' for the base table")
        if self.key[0] != "stimulus_id":
            raise SpaceError(f"a grain's key must start with stimulus_id, not {self.key}")


def parse_grain(spec: Sequence[str], key_maps: dict[str, str],
                skip: dict[str, list[str]] | None = None) -> Grain:
    """``[TABLE, BLOCK, KEY, FEATURES...]`` from the CLI (TABLE '-' = base)."""
    if len(spec) < 4:
        raise SpaceError(f"--grain {' '.join(spec)}: give TABLE BLOCK KEY FEATURES... (TABLE '-' for the base table)")
    table = "" if spec[0] == "-" else spec[0]
    return Grain(table=table, block=spec[1], key=spec[2].split(","), features=list(spec[3:]),
                 key_map=key_maps.get(spec[0]), skip_members=list((skip or {}).get(spec[0], [])))


# --------------------------------------------------------------------------
# ids
# --------------------------------------------------------------------------


def registry_id_map(registry: pd.DataFrame, src_ids: Sequence[str], *, join: str | None = None,
                    pattern: str | None = None, partial: bool = False) -> dict[str, str]:
    """Source id -> registry ``stimulus_id``, refused unless one to one and covering.

    Without ``join`` the source ids must already be registry ids. With it, the
    one capture group of ``pattern`` is compared to ``registry[join]``
    (numbers as numbers, so ``002951`` matches 2951). Source ids that match no
    registry row are not part of the set and are left out.
    """
    if "stimulus_id" not in registry.columns:
        raise InputError("the registry has no stimulus_id column")
    reg_ids = registry["stimulus_id"].astype(str).tolist()
    if len(set(reg_ids)) != len(reg_ids):
        raise InputError("the registry repeats a stimulus_id")
    if join is None:
        known = set(reg_ids)
        m = {s: s for s in src_ids if s in known}
    else:
        if join not in registry.columns:
            raise InputError(f"the registry has no column {join!r}; it has {list(registry.columns)}")
        if not pattern:
            raise SpaceError("--registry-join needs --id-pattern (one capture group, e.g. 'ext-nsd-(\\d+)')")
        rx = re.compile(pattern)
        if rx.groups != 1:
            raise SpaceError(f"--id-pattern {pattern!r} must have exactly one capture group")
        by: dict = {}
        for sid, v in zip(reg_ids, registry[join]):
            k = _key_value(v)
            if k in by:
                raise InputError(f"registry column {join!r} repeats the value {v!r}")
            by[k] = sid
        m = {}
        for s in src_ids:
            hit = rx.fullmatch(s)
            if hit and _key_value(hit.group(1)) in by:
                m[s] = by[_key_value(hit.group(1))]
    _require_bijection(m, reg_ids, what="source ids", partial=partial)
    return m


def uncovered(registry: pd.DataFrame, id_map: dict[str, str]) -> list[str]:
    return sorted(set(registry["stimulus_id"].astype(str)) - set(id_map.values()))


def _require_bijection(m: dict, targets: Sequence[str], *, what: str, partial: bool = False) -> None:
    many = [t for t, c in Counter(m.values()).items() if c > 1]
    if many:
        raise SpaceError(f"{len(many)} registry stimulus_id(s) receive more than one of the {what} "
                         f"(e.g. {sorted(many)[:3]}); the mapping must be one to one")
    if not m:
        raise SpaceError(f"none of the {what} map onto the registry; check --registry-join / --id-pattern")
    missing = sorted(set(targets) - set(m.values()))
    if missing and not partial:
        raise SpaceError(f"{len(missing)} registry stimulus_id(s) have no row in the input "
                         f"(e.g. {missing[:3]}); a family covers its whole registry. If this source "
                         "legitimately lacks them, pass --partial-coverage (they are listed in the sidecar)")


def read_key_map(path: str | Path, key: Sequence[str]) -> dict[tuple, tuple]:
    """A key-map table: ``src_<k>`` and ``<k>`` columns for every key column, one to one."""
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    need = [f"src_{k}" for k in key] + list(key)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise InputError(f"key map {p.name} lacks column(s) {missing}; it needs {need}")
    src = [tuple(_key_value(v) for v in r) for r in df[[f"src_{k}" for k in key]].itertuples(index=False)]
    dst = [tuple(_key_value(v) for v in r) for r in df[list(key)].itertuples(index=False)]
    for side, rows in (("source", src), ("target", dst)):
        if len(set(rows)) != len(rows):
            raise SpaceError(f"key map {p.name} repeats a {side} key; it must be one to one")
    return dict(zip(src, dst))


# --------------------------------------------------------------------------
# leak guard
# --------------------------------------------------------------------------


def _same_file(a: str | Path, b: str | Path) -> bool:
    try:
        return Path(a).samefile(b)
    except OSError:
        return str(Path(a).resolve()) == str(Path(b).resolve())


def _read_ids(path: str | Path) -> set[str]:
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}


def block_leak_guard(manifest: dict, features: Sequence[str], src_ids: set[str]) -> dict:
    """Refuse unless no projected source id was a row of the block's fit."""
    inputs = list(manifest.get("inputs") or [])
    shared = [str(p) for p in features if any(_same_file(p, q) for q in inputs)]
    if not shared:
        return {"rule": "input not among the fit's inputs", "fit_inputs": inputs}
    ex = manifest.get("exclude_ids_file")
    if not ex or not Path(ex).exists():
        raise SpaceError(
            f"block {manifest.get('block')}: {shared} were inputs of its fit, and the fit records no "
            f"exclude_ids_file that exists ({ex!r}), so these rows may have been fitted on. Project a "
            "table the fit never read, or restore the exclusion list at that path.")
    leaked = sorted(src_ids - _read_ids(ex))
    if leaked:
        raise SpaceError(
            f"block {manifest.get('block')}: {len(leaked)} projected id(s) were rows of its fit "
            f"(e.g. {leaked[:3]}; {shared[0]} is a fit input and {Path(ex).name} does not exclude them). "
            "Their coordinates were learned from them: a leak.")
    return {"rule": "every projected id is in the fit's exclude_ids_file", "shared_inputs": shared,
            "exclude_ids_file": str(ex), "exclude_ids_sha256": sha256_file(ex), "n_ids": len(src_ids)}


def relation_leak_guard(manifest: dict, src_ids: set[str]) -> dict:
    """Refuse unless no projected source id is a row of either score table the relation was fitted on."""
    out = {"rule": "no projected id is a row of the score tables the relation was fitted on", "scores": {}}
    for side in ("a", "b"):
        p = Path(manifest["sides"][side]["scores"])
        if not p.exists():
            raise SpaceError(f"relation {manifest['name']} side {side}: its fit table {p} is missing, so its "
                             "rows cannot be checked against the projected ids. Restore it.")
        ids = pd.read_parquet(p, columns=["stimulus_id"]) if p.suffix == ".parquet" else \
            pd.read_csv(p, usecols=["stimulus_id"])
        leaked = sorted(src_ids & set(ids["stimulus_id"].astype(str)))
        if leaked:
            raise SpaceError(f"relation {manifest['name']}: {len(leaked)} projected id(s) were rows of its "
                             f"fit (side {side}, {p.name}; e.g. {leaked[:3]}): a leak.")
        out["scores"][side] = {"path": str(p), "sha256": sha256_file(p)}
    return out


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------


@dataclass
class GrainResult:
    grain: Grain
    keys: list[tuple]  # target keys, sorted
    src_ids: set[str]
    columns: dict[str, np.ndarray] = field(default_factory=dict)  # model -> n x k (NaN = unplaced)
    leak: dict = field(default_factory=dict)  # model -> guard evidence
    uncovered: list[str] = field(default_factory=list)  # registry ids with no row (--partial-coverage)


def _restrict(space, keep: set[str]):
    from psytwill.store import SpaceMatrix

    idx = [i for i, lab in enumerate(space.labels) if lab.split("|")[0] in keep]
    return SpaceMatrix(name=space.name, labels=[space.labels[i] for i in idx], X=space.X[idx],
                       features=space.features, modality=space.modality, extractor=space.extractor,
                       n_replicates=space.n_replicates)


def project_grain(release: Release, grain: Grain, spaces: dict, id_map: dict[str, str]) -> GrainResult:
    """Place one grain's rows through its release block and every relation side on that block."""
    from psytwill.store import SpaceMatrix

    fit = release.block(grain.block)
    meta = release.meta["blocks"][grain.block]
    stray = [m for m in grain.skip_members if m not in fit.members]
    if stray:
        raise SpaceError(f"--skip-member {stray}: not members of block {grain.block} ({fit.members})")
    present = [m for m in fit.members if m not in grain.skip_members]
    keep = set(id_map)
    spaces = {m: _restrict(spaces[m], keep) for m in present}
    universe = sorted({lab for sp in spaces.values() for lab in sp.labels})
    src_ids = {lab.split("|")[0] for lab in universe}
    for m in grain.skip_members:
        # absent from every row: masked exactly as an `undefined` member is, never filled
        feats = list(fit.map.whiteners[m].features)
        spaces[m] = SpaceMatrix(name=m, labels=universe, X=np.full((len(universe), len(feats)), np.nan),
                                features=feats)
    S, placed = fit.project(spaces)
    pos = {lab: i for i, lab in enumerate(universe)}
    full = np.full((len(universe), fit.k), np.nan)
    full[[pos[lab] for lab in placed]] = S

    kmap = read_key_map(grain.key_map, grain.key) if grain.key_map else None
    keys = []
    for lab in universe:
        parts = lab.split("|")
        if kmap is not None:
            src = tuple(_key_value(v) for v in parts)
            if src not in kmap:
                raise SpaceError(f"key map {Path(grain.key_map).name} has no entry for source row {parts}")
            dst = kmap[src]
            if str(dst[0]) != id_map[parts[0]]:
                raise SpaceError(f"key map sends {parts} to stimulus {dst[0]!r}, but the registry maps "
                                 f"{parts[0]!r} to {id_map[parts[0]]!r}")
            keys.append(tuple(_key_out(v) for v in dst))
        else:
            keys.append((id_map[parts[0]], *(_key_out(v) for v in parts[1:])))
    if len(set(keys)) != len(keys):
        raise SpaceError(f"table {grain.table or '(base)'}: two source rows map to one output key")
    order = sorted(range(len(keys)), key=lambda i: tuple((0, v, "") if isinstance(v, (int, float)) else (1, 0, str(v))
                                                         for v in keys[i]))
    res = GrainResult(grain=grain, keys=[keys[i] for i in order], src_ids=src_ids)
    full = full[order]
    res.columns[model_name(grain.block)] = full
    res.leak[model_name(grain.block)] = block_leak_guard(fit.manifest, grain.features, src_ids)

    ok = np.isfinite(full).all(axis=1)
    for rname, rentry in release.meta["relations"].items():
        for side in ("a", "b"):
            if rentry[side] != grain.block:
                continue
            rel = release.relation(rname)
            if rel.manifest["sides"][side]["manifest_sha256"] != meta["manifest_sha256"]:
                raise SpaceError(f"relation {rname} side {side} was not fitted on the release's block "
                                 f"{grain.block}; the release is inconsistent")
            P = np.full((len(full), rel.k), np.nan)
            P[ok] = rel.project(full[ok], side)
            res.columns[model_name(rname)] = P
            res.leak[model_name(rname)] = {"side": side, **relation_leak_guard(rel.manifest, src_ids)}
    return res


# --------------------------------------------------------------------------
# the family
# --------------------------------------------------------------------------


def _width(k: int) -> int:
    return max(3, len(str(k - 1)))


def _colnames(model: str, k: int) -> list[str]:
    w = _width(k)
    return [f"{model}_{j:0{w}d}" for j in range(k)]


def _table_path(stem: Path, table: str) -> Path:
    return stem.parent / (f"{stem.name}_{table}.csv" if table else f"{stem.name}.csv")


def _archive_previous(stem: Path, release: Release) -> str | None:
    """Move an earlier release's family out of the way (P6); same release = rewrite."""
    side = stem.parent / f"{stem.name}.meta.json"
    if not side.exists():
        return None
    old = json.loads(side.read_text())
    old_version = (old.get("input") or {}).get("release_version")
    if old_version == release.meta["version"]:
        return None
    dest = stem.parent / f"{stem.name}_{old_version}"
    if dest.exists():
        raise SpaceError(f"{dest} already exists; move it before writing release {release.meta['version']}")
    dest.mkdir()
    for f in [side, *[Path(o["path"]) for o in (old.get("output") or {}).values()]]:
        if f.exists():
            shutil.move(str(f), dest / f.name)
    return str(dest)


def write_family(release: Release, results: Sequence[GrainResult], stem: str | Path, *, set_name: str,
                 registry: str | Path, id_join: str | None, id_pattern: str | None,
                 runtime_sec: float | None = None) -> Path:
    stem = Path(stem)
    if stem.suffix:
        raise SpaceError(f"-o {stem}: give the family stem without a suffix (tables land at <stem>.csv, <stem>_<table>.csv)")
    tables = [r.grain.table for r in results]
    if len(set(tables)) != len(tables):
        raise SpaceError(f"two grains write the same table ({tables}); give each its own")
    stem.parent.mkdir(parents=True, exist_ok=True)
    archived = _archive_previous(stem, release)

    rel = release.meta
    output, models = {}, {}
    for r in results:
        g = r.grain
        df = pd.DataFrame(r.keys, columns=g.key)
        for model, X in r.columns.items():
            df = pd.concat([df, pd.DataFrame(X, columns=_colnames(model, X.shape[1]))], axis=1)
        path = _table_path(stem, g.table)
        df.to_csv(path, index=False, float_format="%.6g")
        output[g.table or "base"] = {"path": str(path), "rows": int(len(df)), "columns": list(df.columns)}
        for model, X in r.columns.items():
            is_block = model == model_name(g.block)
            src = g.block if is_block else next(n for n in rel["relations"] if model_name(n) == model)
            entry = rel["blocks"][src] if is_block else rel["relations"][src]
            cols = _colnames(model, X.shape[1])
            if model not in models:
                manifest = json.loads(Path(entry["manifest"]).read_text())
                models[model] = {
                    "package": "psytwill",
                    "package_version": __version__,
                    "checkpoint": f"{rel['name']}@{rel['version']}:{Path(entry['manifest']).stem}",
                    "modality": BLOCK_MODALITY.get(g.block, g.block) if is_block else "shared",
                    "kind": "embedding",
                    "pattern": f"{model}_{{:0{_width(X.shape[1])}d}}",
                    "range": [0, X.shape[1] - 1],
                    "count": X.shape[1],
                    "nulls": {c: {"means": "undefined", "when": UNPLACED_WHEN} for c in cols},
                    "fit": {
                        "method": "block: per-member whitening + block PCA, top k" if is_block
                        else f"relation: frozen CV-CCA between blocks {entry['a']} (side a) and {entry['b']} (side b)",
                        "k": int(entry["k"]),
                        "manifest": entry["manifest"], "manifest_sha256": entry["manifest_sha256"],
                        "weights": entry["weights"], "weights_sha256": entry["weights_sha256"],
                        "fitted_with": entry.get("fitted_with"),
                        "fit_inputs": manifest.get("inputs") if is_block
                        else {sd: manifest["sides"][sd]["scores"] for sd in ("a", "b")},
                        "scope": None if is_block else entry.get("scope"),
                    },
                    "analysis_set_in_fit": False,
                    "tables": {},
                }
            # one model may sit in several tables (a relation's two sides): per-table facts here
            models[model]["tables"][g.table or "base"] = {
                "block": g.block,
                "n_rows": int(len(X)),
                "n_unplaced": int((~np.isfinite(X).all(axis=1)).sum()),
                "members_absent": list(g.skip_members) if is_block else [],
                "leak_guard": r.leak[model],
            }
    meta = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "extractor": "psytwill",
        "extractor_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {
            "type": "psytwill-space release projection",
            "set": set_name,
            "release": str(release.path), "release_sha256": sha256_file(release.path),
            "release_name": rel["name"], "release_version": rel["version"],
            "registry": str(registry), "registry_sha256": sha256_file(registry),
            "id_map": {"join": id_join, "pattern": id_pattern},
            "grains": {
                (r.grain.table or "base"): {
                    "block": r.grain.block, "key": r.grain.key, "features": r.grain.features,
                    "key_map": None if not r.grain.key_map else
                    {"path": str(r.grain.key_map), "sha256": sha256_file(r.grain.key_map)},
                    "n_source_ids": len(r.src_ids),
                    "members_absent": list(r.grain.skip_members),
                    "registry_uncovered": r.uncovered,
                } for r in results},
            "absent": rel.get("absent", []),
            "archived_previous": archived,
        },
        "output": output,
        "models": models,
        "total_runtime_sec": runtime_sec,
    }
    side = stem.parent / f"{stem.name}.meta.json"
    side.write_text(json.dumps(meta, indent=2))
    return side
