"""Contract B 1.2 per-column labels: read, resolve, compare, select, export.

Each extractor sidecar's ``models.<name>`` entry carries ``labels`` from
schema_version 1.2 on: what each feature column is and how its value is
computed. The block is written once per model,

    "labels": {"default": {...}, "columns": {"<column>": {...}}}

and a column's labels are ``default`` with its ``columns`` entry merged over
it key by key. After the merge every column carries ``interpretability``
(``nameable`` | ``opaque``) and ``derivation`` (one of :data:`DERIVATIONS`);
``derived_from`` (the parent model's checkpoint) is present exactly when the
derivation is ``formula-on-model``, ``probe`` or ``zero-shot``. The BIDS
column keys (``LongName``, ``Description``, ``Units``, ``Levels``,
``TermURL``) and ``source_validation`` are optional and descriptive.

A 1.0 or 1.1 sidecar has no ``labels``: its models read as *unlabelled*
(``None``), which is not an error anywhere except a label-based selection.

What psytwill does with them:

- **reads** them strictly. A malformed producer block (a value outside the
  vocabularies, ``derived_from`` where it is forbidden or missing, a
  ``columns`` key that is not a column of the model) raises
  :class:`~psytwill.exceptions.InputError` naming the sidecar and model.
- **stacks** them: two inputs of one model may be aggregated into one table
  only if every column resolves to the same ``interpretability``,
  ``derivation`` and ``derived_from`` under both (:data:`STRUCTURAL_KEYS`).
  Differences in the descriptive keys warn, and the newer sidecar's labels
  are kept (:func:`stack_labels`).
- **excludes** ``derivation: bookkeeping`` columns (they describe the
  extraction, not the stimulus) from every block fit, recorded per member.
- **selects** columns for a fit by label (:func:`parse_select`), e.g.
  ``interpretability=nameable;derivation!=probe,zero-shot``.
- **flags** circularity: a member column whose ``derived_from`` equals the
  checkpoint of another member of the same block (:func:`circular_with`).
- **exports** a stimulus-free per-column table and a BIDS-style column
  dictionary (:func:`write_labels`).
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from psytwill.exceptions import InputError, SpaceError

INTERPRETABILITY: tuple[str, ...] = ("nameable", "opaque")
DERIVATIONS: tuple[str, ...] = (
    "formula", "formula-on-model", "trained-on-labels", "probe", "zero-shot",
    "embedding", "bookkeeping",
)
#: derivations that name a parent model; ``derived_from`` is required for
#: these and forbidden for the rest
NEEDS_DERIVED_FROM: frozenset[str] = frozenset({"formula-on-model", "probe", "zero-shot"})
BOOKKEEPING = "bookkeeping"
#: keys whose disagreement refuses a stack (what the column IS)
STRUCTURAL_KEYS: tuple[str, ...] = ("interpretability", "derivation", "derived_from")
#: the BIDS tabular-sidecar column keys
BIDS_KEYS: tuple[str, ...] = ("LongName", "Description", "Units", "Levels", "TermURL")
#: keys whose disagreement only warns (how the column is DESCRIBED)
DESCRIPTIVE_KEYS: tuple[str, ...] = BIDS_KEYS + ("source_validation",)
LABEL_KEYS: tuple[str, ...] = STRUCTURAL_KEYS + DESCRIPTIVE_KEYS


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def _placeholder_regex(pattern: str) -> re.Pattern:
    """A sidecar inventory ``pattern`` (``clip_{NNN}``, ``saliency_{XX}_{YY}``,
    ``vgg19_{layer}_{NNNN}``) as a regex over column names.

    An all-``N`` placeholder is digits; any other placeholder is one or more
    name characters.
    """
    out, pos = [], 0
    for m in re.finditer(r"\{([^{}]+)\}", pattern):
        out.append(re.escape(pattern[pos:m.start()]))
        out.append(r"\d+" if set(m.group(1)) == {"N"} else r"[A-Za-z0-9_]+")
        pos = m.end()
    out.append(re.escape(pattern[pos:]))
    return re.compile("".join(out) + r"\Z")


def inventory_matcher(entry: Mapping[str, Any] | None) -> Callable[[str], bool] | None:
    """A predicate for "is this a column of the model", from the sidecar entry's
    feature inventory, or ``None`` when the entry lists none.

    The inventory may sit at the entry's top level or under ``features`` (the
    viz2psy form): an explicit ``columns`` list, or an embedding ``pattern``
    (pattern-aware, so a 1024-d model need not list its columns).
    """
    if not isinstance(entry, Mapping):
        return None
    for inv in (entry.get("features"), entry):
        if not isinstance(inv, Mapping):
            continue
        cols = inv.get("columns")
        if isinstance(cols, list) and cols:
            names = {str(c) for c in cols}
            return names.__contains__
        if isinstance(inv.get("pattern"), str):
            rx = _placeholder_regex(inv["pattern"])
            return lambda c, rx=rx: rx.match(c) is not None
    return None


def _check_value(key: str, value: Any, *, where: str) -> None:
    if key == "interpretability" and value not in INTERPRETABILITY:
        raise InputError(f"{where}: interpretability={value!r}; expected one of {INTERPRETABILITY}")
    if key == "derivation" and value not in DERIVATIONS:
        raise InputError(f"{where}: derivation={value!r}; expected one of {DERIVATIONS}")
    if key == "derived_from" and (not isinstance(value, str) or not value):
        raise InputError(f"{where}: derived_from must be a non-empty checkpoint string, got {value!r}")
    if key == "source_validation":
        if not isinstance(value, list) or not all(isinstance(v, Mapping) for v in value):
            raise InputError(f"{where}: source_validation must be a list of objects")
    if key == "Levels" and not isinstance(value, Mapping):
        raise InputError(f"{where}: Levels must be an object (BIDS), got {type(value).__name__}")


def _merge(default: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Key-wise merge of a column override over the default; an override value
    of null removes the key (so a ``formula`` column can drop a probe default's
    ``derived_from``)."""
    return {k: v for k, v in {**default, **override}.items() if v is not None}


def check_resolved(entry: Mapping[str, Any], *, where: str) -> None:
    """Refuse a column whose merged labels break the contract."""
    for key in ("interpretability", "derivation"):
        if key not in entry:
            raise InputError(f"{where}: no `{key}` after merging default and columns (required for every column)")
    needs = entry["derivation"] in NEEDS_DERIVED_FROM
    if needs and "derived_from" not in entry:
        raise InputError(f"{where}: derivation={entry['derivation']!r} requires `derived_from` "
                         "(the parent model's checkpoint)")
    if not needs and "derived_from" in entry:
        raise InputError(f"{where}: `derived_from` is only allowed for derivations "
                         f"{sorted(NEEDS_DERIVED_FROM)}, not {entry['derivation']!r}")


def validate_labels(labels: Any, *, model: str, where: str,
                    is_column: Callable[[str], bool] | None = None) -> dict:
    """Check a producer's raw ``labels`` block; returns it unchanged.

    Every value must be in its vocabulary, every ``columns`` key a column of
    the model (when the inventory says which those are), and the default
    merged with each override must resolve completely. Whether a column the
    override set does not name resolves (from ``default`` alone) is checked
    by :func:`resolve`, against the columns actually present.
    """
    at = f"{where} models.{model}.labels"
    if not isinstance(labels, Mapping):
        raise InputError(f"{at}: expected an object with `default` and/or `columns`, got {type(labels).__name__}")
    unknown = sorted(set(labels) - {"default", "columns"})
    if unknown:
        raise InputError(f"{at}: unknown key(s) {unknown}; expected `default` and `columns`")
    default = labels.get("default", {})
    columns = labels.get("columns", {})
    if not isinstance(default, Mapping) or not isinstance(columns, Mapping):
        raise InputError(f"{at}: `default` and `columns` must be objects")
    for key, value in default.items():
        _check_value(key, value, where=f"{at}.default")
    for col, entry in columns.items():
        if not isinstance(entry, Mapping):
            raise InputError(f"{at}.columns.{col}: expected an object")
        if is_column is not None and not is_column(col):
            raise InputError(f"{at}.columns: {col!r} is not a column of model {model!r} (keys are exact "
                             "column names from the model's own inventory)")
        for key, value in entry.items():
            if value is not None:  # null unsets a default key for this column
                _check_value(key, value, where=f"{at}.columns.{col}")
        check_resolved(_merge(default, entry), where=f"{at}.columns.{col}")
    return dict(labels)


def read_model_labels(sidecar: Mapping[str, Any] | None, name: str, *, where: str = "sidecar") -> dict | None:
    """The validated raw ``labels`` block of a model, or ``None`` (unlabelled:
    a 1.0/1.1 sidecar, no sidecar, or a model without the key)."""
    if not sidecar:
        return None
    entry = sidecar.get("models", {}).get(name)
    if not isinstance(entry, Mapping) or entry.get("labels") is None:
        return None
    return validate_labels(entry["labels"], model=name, where=where, is_column=inventory_matcher(entry))


def resolve(labels: Mapping[str, Any] | None, columns: Iterable[str], *, model: str = "",
            where: str = "labels") -> dict[str, dict] | None:
    """``{column: merged labels}`` for ``columns``; ``None`` when unlabelled.

    Raises :class:`InputError` on a column that does not resolve completely
    (e.g. a default without ``derivation`` and no override for the column).
    """
    if labels is None:
        return None
    default = dict(labels.get("default", {}))
    overrides = labels.get("columns", {})
    out = {}
    for col in columns:
        entry = _merge(default, overrides.get(col, {}))
        check_resolved(entry, where=f"{where} models.{model}.labels for column {col!r}")
        out[col] = entry
    return out


def labels_digest(labels: Mapping[str, Any] | None) -> str | None:
    """A short content hash of a raw labels block (canonical JSON), or None."""
    if labels is None:
        return None
    import hashlib

    blob = json.dumps(labels, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------
# stacking
# --------------------------------------------------------------------------


def diff_labels(a: Mapping[str, Any], b: Mapping[str, Any], columns: Iterable[str]) -> tuple[list[str], list[str]]:
    """(structural, descriptive) differences between two raw blocks over ``columns``.

    Each item reads ``column.key: a -> b``. Both blocks must be labelled.
    """
    ra = resolve(a, columns)
    rb = resolve(b, columns)
    structural, descriptive = [], []
    for col in ra:
        for key in sorted(set(ra[col]) | set(rb[col])):
            va, vb = ra[col].get(key), rb[col].get(key)
            if va == vb:
                continue
            item = f"{col}.{key}: {va!r} -> {vb!r}"
            (structural if key in STRUCTURAL_KEYS else descriptive).append(item)
    return structural, descriptive


def _when(ts: str | None) -> tuple[int, str]:
    """Sort key for a sidecar's ``created_at``: unparseable/missing sorts oldest."""
    if not ts:
        return (0, "")
    try:
        return (1, datetime.fromisoformat(str(ts).replace("Z", "+00:00")).isoformat())
    except ValueError:
        return (1, str(ts))


@dataclass
class LabelSource:
    """One input's claim about a model's labels, for :func:`stack_labels`."""

    labels: dict | None
    path: str
    created_at: str | None
    columns: frozenset[str]


def stack_labels(model: str, held: LabelSource | None, new: LabelSource, *, what: str = "features table",
                 fix: str = "") -> LabelSource:
    """Stack ``new`` onto ``held`` (two inputs of one model); returns the winner.

    Refused (:class:`SpaceError`) when one input is labelled and the other is
    not, or when any column of either resolves to a different
    ``interpretability``, ``derivation`` or ``derived_from`` under the two
    blocks. A descriptive difference warns; the newer sidecar (``created_at``;
    on a tie the later input) wins. The winner carries the union of columns.
    """
    if held is None:
        return new
    columns = held.columns | new.columns
    fix = fix or ("run `<extractor> sidecar refresh` on every input so they carry the same "
                  "labels, then rebuild or `psytwill features --refresh-labels` the table")
    if (held.labels is None) != (new.labels is None):
        lab, unl = (held, new) if held.labels is not None else (new, held)
        raise SpaceError(
            f"`labels` mismatch for model {model!r}: {lab.path} declares labels and {unl.path} "
            f"declares none (a 1.0/1.1 sidecar). One {what} cannot mix them; {fix}.")
    if held.labels is None:
        return LabelSource(None, new.path, new.created_at, frozenset(columns))
    structural, descriptive = diff_labels(held.labels, new.labels, sorted(columns))
    if structural:
        shown = "; ".join(structural[:4]) + (f"; ... ({len(structural)} in all)" if len(structural) > 4 else "")
        raise SpaceError(
            f"`labels` mismatch for model {model!r} between {held.path} and {new.path}: {shown}. "
            f"interpretability, derivation and derived_from must agree for every column; {fix}.")
    winner = new if _when(new.created_at) >= _when(held.created_at) else held
    if descriptive:
        loser = held if winner is new else new
        shown = "; ".join(descriptive[:3]) + (" ..." if len(descriptive) > 3 else "")
        warnings.warn(
            f"model {model!r}: descriptive labels differ between {loser.path} and {winner.path} "
            f"({shown}); keeping the newer sidecar's ({winner.path}).", stacklevel=3)
    return LabelSource(winner.labels, winner.path, winner.created_at, frozenset(columns))


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

SELECT_KEYS: dict[str, tuple[str, ...]] = {"interpretability": INTERPRETABILITY, "derivation": DERIVATIONS}

SELECT_GRAMMAR = """\
EXPR   := CLAUSE (';' CLAUSE)*        clauses are ANDed; --select may also repeat
CLAUSE := KEY OP VALUE (',' VALUE)*   a value list is a set
KEY    := interpretability | derivation
OP     := '=' (value in the set) | '!=' (value not in the set)
e.g. interpretability=nameable;derivation!=probe,zero-shot"""


@dataclass(frozen=True)
class Clause:
    key: str
    op: str  # "in" | "not_in"
    values: tuple[str, ...]

    def admits(self, entry: Mapping[str, Any]) -> bool:
        hit = entry.get(self.key) in self.values
        return hit if self.op == "in" else not hit

    def text(self) -> str:
        return f"{self.key}{'=' if self.op == 'in' else '!='}{','.join(self.values)}"


def parse_select(spec: str | Sequence[str] | None) -> list[Clause]:
    """Parse ``--select`` (one string or several, see :data:`SELECT_GRAMMAR`)."""
    if spec is None:
        return []
    parts = [spec] if isinstance(spec, str) else list(spec)
    clauses: list[Clause] = []
    for part in parts:
        for raw in str(part).split(";"):
            raw = raw.strip()
            if not raw:
                continue
            m = re.fullmatch(r"\s*([A-Za-z_]+)\s*(!=|=)\s*(.+?)\s*", raw)
            if not m:
                raise SpaceError(f"--select clause {raw!r} does not parse.\n{SELECT_GRAMMAR}")
            key, op, rhs = m.group(1), m.group(2), m.group(3)
            if key not in SELECT_KEYS:
                raise SpaceError(f"--select key {key!r} is not one of {sorted(SELECT_KEYS)}.\n{SELECT_GRAMMAR}")
            values = [v.strip() for v in rhs.split(",") if v.strip()]
            bad = [v for v in values if v not in SELECT_KEYS[key]]
            if not values or bad:
                raise SpaceError(f"--select {raw!r}: {bad or 'no values'} not in the {key} vocabulary "
                                 f"{SELECT_KEYS[key]}")
            if op == "=" and key == "derivation" and BOOKKEEPING in values:
                raise SpaceError("--select derivation=bookkeeping: bookkeeping columns are never swept into a "
                                 "feature space")
            vocab = SELECT_KEYS[key]
            ordered = tuple(sorted(set(values), key=vocab.index))
            clauses.append(Clause(key, "in" if op == "=" else "not_in", ordered))
    if not clauses and parts:
        raise SpaceError(f"--select is empty.\n{SELECT_GRAMMAR}")
    return clauses


def select_text(clauses: Sequence[Clause]) -> str:
    """The canonical expression for a parsed selection (round-trips through parse_select)."""
    return ";".join(c.text() for c in clauses)


def admits(clauses: Sequence[Clause], entry: Mapping[str, Any]) -> bool:
    return all(c.admits(entry) for c in clauses)


def circular_with(resolved: Mapping[str, Mapping[str, dict] | None], checkpoints: Mapping[str, str | None],
                  source: Mapping[str, str] | None = None) -> dict[str, dict[str, list[str]]]:
    """``{member: {column: [other members]}}`` for every column whose
    ``derived_from`` equals another member's checkpoint.

    Members read from one model (``source``: member -> model, for split
    members) share a checkpoint and never flag each other. A flag, never a
    refusal.
    """
    source = dict(source or {})
    out: dict[str, dict[str, list[str]]] = {}
    for m, cols in resolved.items():
        for col, entry in (cols or {}).items():
            parent = entry.get("derived_from")
            if not parent:
                continue
            hits = sorted(o for o, ck in checkpoints.items()
                          if o != m and ck and ck == parent and source.get(o, o) != source.get(m, m))
            if hits:
                out.setdefault(m, {})[col] = hits
    return out


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

TABLE_COLUMNS = ["block", "member", "column", "in_fit", *STRUCTURAL_KEYS, "circular_with", *DESCRIPTIVE_KEYS]


def manifest_rows(manifest: Mapping[str, Any]) -> list[dict]:
    """Per-column rows of one block manifest (empty for a fit without labels)."""
    rows = []
    feats = manifest.get("member_features", {})
    for m in manifest.get("members", []):
        pm = manifest.get("per_member", {}).get(m, {})
        cols = pm.get("labels")
        if not cols:
            continue
        circ = pm.get("circular_with") or {}
        kept = set(feats.get(m, cols))
        for col, entry in cols.items():
            row = {"block": manifest.get("block"), "member": m, "column": col, "in_fit": col in kept,
                   "circular_with": ";".join(circ.get(col, [])) or None}
            for key in LABEL_KEYS:
                v = entry.get(key)
                row[key] = json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
            rows.append(row)
    return rows


def labels_table(manifests: Sequence[Mapping[str, Any]]):
    """Stimulus-free per-column labels of one or more block manifests."""
    import pandas as pd

    rows = [r for man in manifests for r in manifest_rows(man)]
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def bids_columns(manifests: Sequence[Mapping[str, Any]]) -> dict[str, dict]:
    """A BIDS-style column dictionary: ``{column: {LongName, Description, Units,
    Levels, TermURL, interpretability, derivation, derived_from,
    source_validation}}`` (keys present only when declared).

    A column in several blocks must carry one structural label set; a
    descriptive difference warns and the most recently fitted block wins.
    """
    order = sorted(range(len(manifests)), key=lambda i: str(manifests[i].get("fitted_at") or ""))
    out: dict[str, dict] = {}
    seen: dict[str, str] = {}
    for i in order:
        man = manifests[i]
        for m in man.get("members", []):
            for col, entry in ((man.get("per_member", {}).get(m, {}) or {}).get("labels") or {}).items():
                new = {k: entry[k] for k in BIDS_KEYS if k in entry}
                new.update({k: entry[k] for k in (*STRUCTURAL_KEYS, "source_validation") if k in entry})
                if col in out:
                    old = out[col]
                    clash = [k for k in STRUCTURAL_KEYS if old.get(k) != new.get(k)]
                    if clash:
                        raise SpaceError(f"column {col!r} carries different {clash} in {seen[col]} and "
                                         f"block {man.get('block')!r}; refit on refreshed tables")
                    if old != new:
                        warnings.warn(f"column {col!r}: descriptive labels differ between {seen[col]} and "
                                      f"block {man.get('block')!r}; keeping the later fit's", stacklevel=2)
                out[col] = new
                seen[col] = f"block {man.get('block')!r}"
    return out


def write_labels(manifests: Sequence[Mapping[str, Any]], csv_path: str | Path) -> tuple[Path, Path]:
    """Write the per-column table (CSV) and, beside it, the BIDS-style column
    dictionary (``<stem>.columns.json``). Returns both paths."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table = labels_table(manifests)
    if table.empty:
        raise SpaceError("no labels to export: none of the given fits recorded any (fitted before space "
                         "schema 1.16, or on unlabelled tables)")
    table.to_csv(csv_path, index=False)
    json_path = csv_path.with_suffix(".columns.json")
    json_path.write_text(json.dumps(bids_columns(manifests), indent=2))
    return csv_path, json_path


def has_labels(manifest: Mapping[str, Any]) -> bool:
    return any((manifest.get("per_member", {}).get(m, {}) or {}).get("labels")
               for m in manifest.get("members", []))
