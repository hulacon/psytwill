"""The pass manifest: one declared extraction pass per release.

A release's features come from one extraction pass. The pass manifest lists
every cell that pass may extract, each with the log entry that decided it, so
"one pass" is a check rather than a habit. A cell added after the pass froze
carries a dated deviation. The manifest is the consumer's declaration, so it
lives with its data (e.g. ``<data root>/space/passes/<name>.toml``), not in
this repo.

::

    pass_schema_version = "1.0"
    name = "<release name> <version>"
    frozen = "YYYY-MM-DD"              # optional: cells added later need a deviation

    [producers]                        # the version each package's new cells run at
    viz2psy = "X.Y.Z"
    aud2psy = "X.Y.Z"
    word2psy = "X.Y.Z"

    [[cell]]
    corpus = "<corpus>"
    source = "<driver source>"         # e.g. frames, audio, transcript
    models = ["*"]                     # or named models; "*" = every model the source declares
    stage = "pass"                     # pass | screen | top-up
    decided = "<log entry that decided it>"
    added = "YYYY-MM-DD"
    deviation = "YYYY-MM-DD: <why, where logged>"   # required when added > frozen
    units = "<file of unit ids>"       # optional pinned subset (screens), with its sha256
    units_sha256 = "<sha256>"
    version = "X.Y.Z"                  # optional: overrides [producers] for this cell

``stage``: ``pass`` is the main extraction. ``screen`` scores a candidate on
a pinned screen subset (``units`` required). ``top-up`` extracts an admitted
candidate on its block's rows after the screens.

A cell already on disk at another producer version is **carried**, not
re-extracted: a version bump alone forces nothing, and a member re-extracts
only when its output-parity check fails. The pin governs new extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from pathlib import Path
import tomllib

from psytwill.exceptions import InputError

PASS_SCHEMA_VERSION = "1.0"
STAGES = ("pass", "screen", "top-up")
PRODUCERS = ("viz2psy", "aud2psy", "word2psy")
ALL_MODELS = "*"
_CELL_FIELDS = {"corpus", "source", "models", "stage", "decided", "added", "deviation", "units",
                "units_sha256", "version", "note"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _date(value, what: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise InputError(f"{what}: {value!r} is not a YYYY-MM-DD date") from exc


@dataclass(frozen=True)
class PassCell:
    corpus: str
    source: str
    models: tuple[str, ...]
    stage: str
    decided: str
    added: date
    deviation: str = ""
    units: Path | None = None
    units_sha256: str | None = None
    version: str | None = None
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.corpus}/{self.source}"

    def covers(self, model: str) -> bool:
        return ALL_MODELS in self.models or model in self.models

    def unit_ids(self) -> frozenset[str] | None:
        """The pinned unit subset, checked against its sha256; None = every unit."""
        if self.units is None:
            return None
        if not self.units.is_file():
            raise InputError(f"pass cell {self.key}: pinned unit list {self.units} does not exist")
        if _sha256(self.units) != self.units_sha256:
            raise InputError(f"pass cell {self.key}: pinned unit list {self.units} changed since it was "
                             "pinned (sha256 differs); a new subset is a new cell with a deviation")
        return frozenset(line.strip() for line in self.units.read_text().splitlines() if line.strip())


@dataclass(frozen=True)
class PassManifest:
    path: Path
    name: str
    frozen: date | None
    producers: dict[str, str]
    cells: tuple[PassCell, ...]
    sha256: str

    def cell_for(self, corpus: str, source: str, model: str) -> PassCell | None:
        """The cell covering one (corpus, source, model): a named model wins over ``*``."""
        hits = [c for c in self.cells if c.corpus == corpus and c.source == source and c.covers(model)]
        named = [c for c in hits if model in c.models]
        return (named or hits or [None])[0]

    def version_for(self, cell: PassCell, package: str) -> str | None:
        return cell.version or self.producers.get(package)

    def record(self) -> dict:
        """What a release or a run log records about the pass."""
        return {"path": str(self.path.resolve()), "sha256": self.sha256, "name": self.name,
                "frozen": None if self.frozen is None else self.frozen.isoformat(),
                "producers": dict(self.producers), "n_cells": len(self.cells),
                "deviations": [{"cell": c.key, "models": list(c.models), "deviation": c.deviation}
                               for c in self.cells if c.deviation]}


def load_pass(path: str | Path) -> PassManifest:
    """Read and check a pass manifest. Any problem is a loud :class:`InputError`."""
    p = Path(path)
    if not p.is_file():
        raise InputError(f"pass manifest {p} does not exist")
    try:
        raw = tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise InputError(f"{p}: not valid TOML ({exc})") from exc
    if raw.get("pass_schema_version") != PASS_SCHEMA_VERSION:
        raise InputError(f"{p}: pass_schema_version must be {PASS_SCHEMA_VERSION!r}, "
                         f"not {raw.get('pass_schema_version')!r}")
    extra = set(raw) - {"pass_schema_version", "name", "frozen", "producers", "cell", "note"}
    if extra:
        raise InputError(f"{p}: unknown top-level key(s) {sorted(extra)}")
    if not str(raw.get("name") or "").strip():
        raise InputError(f"{p}: needs a `name`")
    frozen = _date(raw["frozen"], f"{p} frozen") if raw.get("frozen") else None
    producers = {str(k): str(v) for k, v in (raw.get("producers") or {}).items()}
    unknown = set(producers) - set(PRODUCERS)
    if unknown:
        raise InputError(f"{p}: unknown producer(s) {sorted(unknown)}; have {PRODUCERS}")
    cells, bad, seen = [], [], {}
    for i, c in enumerate(raw.get("cell") or []):
        where = f"cell {i} ({c.get('corpus')}/{c.get('source')})"
        if set(c) - _CELL_FIELDS:
            bad.append(f"{where}: unknown field(s) {sorted(set(c) - _CELL_FIELDS)}")
            continue
        missing = [f for f in ("corpus", "source", "models", "stage", "decided", "added") if not c.get(f)]
        if missing:
            bad.append(f"{where}: needs {', '.join(missing)}")
            continue
        models = c["models"]
        if not isinstance(models, list) or not all(isinstance(m, str) and m for m in models):
            bad.append(f"{where}: models is a list of model names, or [\"*\"]")
            continue
        if ALL_MODELS in models and len(models) > 1:
            bad.append(f"{where}: \"*\" stands alone")
        if c["stage"] not in STAGES:
            bad.append(f"{where}: stage {c['stage']!r} is not one of {STAGES}")
        try:
            added = _date(c["added"], f"{p} {where} added")
        except InputError as exc:
            bad.append(str(exc))
            continue
        if frozen is not None and added > frozen and not str(c.get("deviation") or "").strip():
            bad.append(f"{where}: added {added} after the pass froze ({frozen}) and carries no deviation")
        units = c.get("units")
        if units is not None and not c.get("units_sha256"):
            bad.append(f"{where}: a pinned unit list names its units_sha256")
        if c["stage"] == "screen" and units is None:
            bad.append(f"{where}: a screen cell runs on a pinned subset (`units`)")
        for m in models:
            ident = (c["corpus"], c["source"], m)
            if ident in seen:
                bad.append(f"{where}: {c['corpus']}/{c['source']} {m} is already declared by cell {seen[ident]}")
            seen[ident] = i
        units_path = None
        if units is not None:
            units_path = Path(units) if Path(units).is_absolute() else p.parent / units
        cells.append(PassCell(corpus=c["corpus"], source=c["source"], models=tuple(models), stage=c["stage"],
                              decided=str(c["decided"]), added=added, deviation=str(c.get("deviation") or ""),
                              units=units_path, units_sha256=c.get("units_sha256"),
                              version=None if c.get("version") is None else str(c["version"]),
                              note=str(c.get("note") or "")))
    if bad:
        raise InputError(f"{p}: " + "; ".join(bad))
    return PassManifest(path=p, name=str(raw["name"]), frozen=frozen, producers=producers,
                        cells=tuple(cells), sha256=_sha256(p))
