"""The corpus registry a release records, and the warnings ``release verify`` prints.

Two sources make one table:

- **Facts about each corpus** live in code (:data:`psytwill.fitcorpus.CORPORA`):
  roles per product and split, licence, media route, member training-set
  overlap.
- **What the consumer has done with it** lives in the consumer's usage file, a
  TOML beside the fit corpora (default ``<data root>/registry/usage.toml``): the
  corpus's overlap with the consumer's own stimuli, by grain, and the history
  of which splits have been read. Both grow with use, so they are data, not
  code. When a release names a data root, the file must exist there.

The usage file::

    usage_schema_version = "1.0"

    [corpus.<key>]
    consumer_overlap = [{grain = "recording", detail = "<which stimuli, and what was done>"}]
    read_by = [{split = "test", read = "<the read>", release = "<version>", date = "YYYY-MM-DD"}]

``grain`` is one of :data:`CONSUMER_GRAINS`. A corpus with no
``consumer_overlap`` entry has not been checked. ``read_by`` entries need a
``split`` and a ``read``. ``release`` and ``date`` are optional.

The registry is **bookkeeping, not permission**. Nothing here refuses a fit,
relation, bench read or membership pass. :func:`registry_warnings` runs the
conflict matrix over the recorded table and the uses the release's manifests
show, and every warning is printed and kept in the release. The rules that do
refuse belong to their own steps: the consumer's data in a relation fit, more
than one gate split per relation, tuning and barring on the same set.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tomllib
from typing import Mapping, Sequence

from psytwill.exceptions import CorpusError
from psytwill.fitcorpus import CORPORA, UNRECORDED, CorpusSpec

USAGE_SCHEMA_VERSION = "1.0"
#: How a corpus overlaps the consumer's stimuli: the same recording file, the
#: same stimulus (e.g. other people's judgements on it), the same item (e.g. a
#: word in another voice), or the same concept.
CONSUMER_GRAINS = ("none", "recording", "stimulus", "item", "concept")
USAGE_RELPATH = Path("registry") / "usage.toml"
#: Role kinds a release's manifests show directly (block corpus rows and
#: relation registers). Other kinds are read outside a release.
VISIBLE_KINDS = ("fit", "register")


def default_usage_path(data_root: str | Path) -> Path:
    return Path(data_root) / USAGE_RELPATH


def load_usage(path: str | Path, corpora: Mapping[str, CorpusSpec] = CORPORA) -> dict:
    """``{corpus: {"consumer_overlap": [...] | None, "read_by": [...]}}`` from a usage file."""
    p = Path(path)
    if not p.exists():
        raise CorpusError(f"registry usage file {p} does not exist; write it (see psytwill.registry) "
                          "or name another with --registry-usage")
    try:
        raw = tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise CorpusError(f"{p}: not valid TOML ({exc})") from exc
    if raw.get("usage_schema_version") != USAGE_SCHEMA_VERSION:
        raise CorpusError(f"{p}: usage_schema_version must be {USAGE_SCHEMA_VERSION!r}, "
                          f"not {raw.get('usage_schema_version')!r}")
    unknown_top = set(raw) - {"usage_schema_version", "corpus"}
    if unknown_top:
        raise CorpusError(f"{p}: unknown top-level key(s) {sorted(unknown_top)}")
    out, bad = {}, []
    for key, entry in (raw.get("corpus") or {}).items():
        if key not in corpora:
            bad.append(f"corpus {key!r} is not registered")
            continue
        extra = set(entry) - {"consumer_overlap", "read_by"}
        if extra:
            bad.append(f"corpus {key}: unknown field(s) {sorted(extra)}")
        overlap = entry.get("consumer_overlap")
        if overlap is not None:
            for o in overlap:
                if o.get("grain") not in CONSUMER_GRAINS:
                    bad.append(f"corpus {key}: consumer_overlap grain {o.get('grain')!r} is not one of "
                               f"{CONSUMER_GRAINS}")
                if o.get("grain") != "none" and not str(o.get("detail") or "").strip():
                    bad.append(f"corpus {key}: a {o.get('grain')} overlap names its `detail`")
                if set(o) - {"grain", "detail"}:
                    bad.append(f"corpus {key}: consumer_overlap fields are grain and detail")
        read_by = entry.get("read_by") or []
        for r in read_by:
            if not str(r.get("split") or "").strip() or not str(r.get("read") or "").strip():
                bad.append(f"corpus {key}: each read_by entry names its `split` and its `read`")
            if set(r) - {"split", "read", "release", "date"}:
                bad.append(f"corpus {key}: read_by fields are split, read, release and date")
        out[key] = {"consumer_overlap": None if overlap is None else [dict(o) for o in overlap],
                    "read_by": [{k: str(v) for k, v in r.items()} for r in read_by]}
    if bad:
        raise CorpusError(f"{p}: " + "; ".join(bad))
    return out


def corpus_table(corpora: Mapping[str, CorpusSpec] = CORPORA, usage: Mapping[str, dict] | None = None) -> dict:
    """The JSON-able table a release records: every registered corpus, facts + usage."""
    usage = usage or {}
    out = {}
    for key in sorted(corpora):
        s = corpora[key]
        u = usage.get(key) or {}
        out[key] = {"tier": s.tier, "licence": s.licence, "media": s.media, "dropped": s.dropped or None,
                    "roles": [asdict(r) for r in s.roles],
                    "training_overlap": [asdict(o) for o in s.training_overlap],
                    "consumer_overlap": u.get("consumer_overlap"), "read_by": list(u.get("read_by") or [])}
    return out


def block_corpora(meta: dict, corpora: Mapping[str, CorpusSpec] = CORPORA) -> list[str]:
    """The corpora a block manifest's rows came from.

    Read from the manifest's per-corpus row counts (``corpus_weights`` /
    ``criterion.corpus_rows``). A one-corpus fit records none, so its corpus is
    read from its inputs (``<corpus>/groups/<table>``).
    """
    found = set((meta.get("corpus_weights") or {}).get("corpus_rows") or {})
    found |= set((meta.get("criterion") or {}).get("corpus_rows") or {})
    if not found:
        for path in meta.get("inputs") or []:
            parts = Path(path).parts
            for i, part in enumerate(parts[:-1]):
                if parts[i + 1] == "groups" and part in corpora:
                    found.add(part)
    return sorted(found)


def release_uses(blocks: Mapping[str, dict], relations: Mapping[str, dict],
                 corpora: Mapping[str, CorpusSpec] = CORPORA) -> list[dict]:
    """Every (corpus, product, kind) use the release's manifests show.

    ``blocks`` / ``relations`` map the product name to its manifest.
    """
    uses = []
    for name, meta in blocks.items():
        for c in block_corpora(meta, corpora):
            uses.append({"corpus": c, "product": name, "kind": "fit"})
    for name, meta in relations.items():
        for c, reg in sorted((meta.get("per_register") or {}).items()):
            uses.append({"corpus": c, "product": name, "kind": "fit" if reg.get("fit", True) else "register"})
    return uses


def _splits_meet(a: str, b: str) -> bool:
    return a == b or "all" in (a, b)


def registry_warnings(table: Mapping[str, dict], uses: Sequence[dict], *,
                      block_members: Mapping[str, Sequence[str]],
                      relation_sides: Mapping[str, tuple[str, str]],
                      analytic_members: Sequence[str] = (),
                      usage_recorded: bool = True) -> list[str]:
    """The conflict matrix over a recorded table and a release's uses. Warns, never refuses.

    Each warning starts with its code (``undeclared-use: ...``) so a card can group them.
    ``analytic_members`` have no learned weights, so no training overlap to check.
    """
    w: list[str] = []
    if not usage_recorded:
        w.append("no-usage-file: no consumer usage file was recorded, so consumer overlap and read "
                 "history are unchecked")
    products = set(block_members) | set(relation_sides)
    used = {(u["corpus"], u["product"], u["kind"]) for u in uses}

    def roles(c: str, **match) -> list[dict]:
        return [r for r in table.get(c, {}).get("roles", []) if all(r.get(k) == v for k, v in match.items())]

    for c, p, k in sorted(used):
        if c not in table:
            w.append(f"unregistered: {p} {k} rows come from {c!r}, which is not a registered corpus")
            continue
        if table[c].get("dropped"):
            w.append(f"dropped-used: {c} is used as {p} {k} but is dropped ({table[c]['dropped']})")
        if not roles(c, kind=k, product=p):
            w.append(f"undeclared-use: {c} is used as {p} {k} but declares no {k} role for {p}")
    for c, entry in sorted(table.items()):
        for r in entry.get("roles", []):
            if r["kind"] in VISIBLE_KINDS and r["product"] in products and (c, r["product"], r["kind"]) not in used:
                w.append(f"declared-not-used: {c} declares {r['product']} {r['kind']} ({r['split']}), and no "
                         "artefact in this release uses it")
        if entry.get("dropped"):
            continue
        for f in ("licence", "media"):
            if entry.get(f) in (None, UNRECORDED):
                w.append(f"unrecorded: {c} has no {f} recorded")
        rs = entry.get("roles", [])
        for i, a in enumerate(rs):
            for b in rs[i + 1:]:
                if a["product"] != b["product"] or a["product"] is None or not _splits_meet(a["split"], b["split"]):
                    continue
                pair = {a["kind"], b["kind"]}
                if pair == {"dev", "bar"}:
                    w.append(f"dev-and-bar: {c} {a['product']} is declared both dev ({a['split']}) and bar "
                             f"({b['split']}) on meeting splits; tune and bar on different sets")
                elif pair in ({"fit", "gate"}, {"fit", "register"}, {"gate", "register"}):
                    w.append(f"same-rows: {c} {a['product']} declares {a['kind']} ({a['split']}) and "
                             f"{b['kind']} ({b['split']}) on meeting splits")
        for r in roles(c, kind="bar"):
            hits = [x for x in entry.get("read_by") or [] if _splits_meet(x["split"], r["split"])]
            if hits:
                w.append(f"bar-read: {c} {r['product']} bar ({r['split']}) is not unread: "
                         + "; ".join(f"{x['read']} ({x['split']}{', ' + x['release'] if x.get('release') else ''})"
                                     for x in hits))
        for r in roles(c, kind="member-training"):
            for p, members in block_members.items():
                if r["target"] in members:
                    for b in roles(c, product=p):
                        if b["kind"] in ("bar", "gate", "report"):
                            w.append(f"member-training: {c} trained {r['target']} (in {p}) and is also "
                                     f"{p} {b['kind']} ({b['split']})")
    fits: dict[str, list[str]] = {}
    for c, p, k in sorted(used):
        if c in table and k == "fit":
            fits.setdefault(c, []).append(p)
    for c, ps in sorted(fits.items()):
        overlap = table[c].get("consumer_overlap")
        if overlap is None:
            w.append(f"consumer-overlap: {c} fits {', '.join(ps)}, and its overlap with the consumer's "
                     "stimuli is unchecked")
        else:
            for o in overlap:
                if o["grain"] == "recording":
                    w.append(f"consumer-overlap: {c} fits {', '.join(ps)} and shares recordings with the "
                             f"consumer's stimuli ({o.get('detail')}); check its exclusion list is applied")
    for rel, (a, b) in sorted(relation_sides.items()):
        gates = [(c, r["split"]) for c, e in sorted(table.items()) for r in e.get("roles", [])
                 if r["kind"] == "gate" and r["product"] == rel and not e.get("dropped")]
        if not gates:
            w.append(f"no-gate: relation {rel} has no declared gate split")
        elif len(gates) > 1:
            w.append(f"gates: relation {rel} declares {len(gates)} gate splits "
                     + ", ".join(f"{c} {s}" for c, s in gates) + "; one gate split per relation")
        members = sorted((set(block_members.get(a, ())) | set(block_members.get(b, ()))) - set(analytic_members))
        fit_corpora = sorted({c for c, p, k in used if p == rel and k == "fit" and c in table})
        for c, split, kind in ([(c, "fit rows", "fit") for c in fit_corpora]
                               + [(c, s, "gate") for c, s in gates if c in table]):
            unchecked, present = [], []
            for m in members:
                o = _overlap(table[c], m)
                if o["value"] == "unchecked":
                    unchecked.append(m)
                elif o["value"] == "present" and kind == "gate" and _splits_meet(o.get("split") or "", split):
                    present.append(m)
            if unchecked:
                w.append(f"training-overlap: relation {rel} {kind} on {c}: {len(unchecked)} side member(s) "
                         f"unchecked ({', '.join(unchecked)})")
            if present:
                w.append(f"training-overlap: relation {rel} gate {c} ({split}) is in the training data of "
                         + ", ".join(present))
    return w


def _overlap(entry: dict, member: str) -> dict:
    by = {o["member"]: o for o in entry.get("training_overlap", [])}
    return by.get(member) or by.get("*") or {"member": member, "value": "unchecked", "split": None}


def record_registry(blocks: Mapping[str, dict], relations: Mapping[str, dict], *,
                    relation_sides: Mapping[str, tuple[str, str]], usage: str | Path | None = None, data_root: str | Path | None = None,
                    corpora: Mapping[str, CorpusSpec] = CORPORA) -> dict:
    """The ``registry`` entry of a release (schema >= 1.6).

    ``usage`` defaults to ``<data_root>/registry/usage.toml`` and must exist
    when the release has a data root. Without either, usage is not recorded,
    and a warning says so.
    """
    from psytwill.relate import sha256_file

    path = Path(usage) if usage is not None else (default_usage_path(data_root) if data_root is not None else None)
    loaded = load_usage(path, corpora) if path is not None else None
    table = corpus_table(corpora, loaded)
    uses = release_uses(blocks, relations, corpora)
    from psytwill.battery import BATTERY

    members = {b: list(m["members"]) for b, m in blocks.items()}
    sides = dict(relation_sides)
    analytic = sorted({m for ms in members.values() for m in ms if m in BATTERY and BATTERY[m].checkpoint is None})
    return {"usage_file": None if path is None else str(path.resolve()),
            "usage_sha256": None if path is None else sha256_file(path),
            "corpora": table, "uses": uses, "analytic_members": analytic,
            "warnings": registry_warnings(table, uses, block_members=members, relation_sides=sides,
                                          analytic_members=analytic, usage_recorded=path is not None)}


def release_warnings(release_meta: dict) -> list[str] | None:
    """Recompute a release's registry warnings from what it recorded; None before schema 1.6."""
    reg = release_meta.get("registry")
    if not reg:
        return None
    members = {b: list(e["members"]) for b, e in release_meta.get("blocks", {}).items()}
    sides = {r: (e["a"], e["b"]) for r, e in release_meta.get("relations", {}).items()}
    return registry_warnings(reg["corpora"], reg["uses"], block_members=members, relation_sides=sides,
                             analytic_members=reg.get("analytic_members") or (),
                             usage_recorded=reg.get("usage_file") is not None)
