#!/usr/bin/env python3
"""CLI for building relational matrix stacks from sibling scores CSVs.

Examples
--------
    # Self mode: every detected space -> square matrix + transitions
    psytwill matrices scores_chunks.csv -o out/

    # Cross mode: shared/compatible spaces -> rectangular matrices
    psytwill matrices text_chunks.csv frames.csv -o out/

    # Subset spaces; override one space's metric
    psytwill matrices scores_chunks.csv -o out/ --spaces minilm:euclidean,emotion

    # RDM form (1 - similarity)
    psytwill matrices scores_chunks.csv -o out/ --distance

    # Aligned-pairs series for equal-length cross inputs
    psytwill matrices text_chunks.csv frames.csv -o out/ --diagonal

    # Preview which spaces a CSV offers, without computing anything
    psytwill spaces scores_chunks.csv

    # Long-form feature table from N extractor CSVs (Contract B surface)
    psytwill features clip.csv ebind.csv caption.csv -o features.parquet

    # Compose sparse runs onto the movie grid from item stores
    psytwill compose sub-01_..._events.tsv --stores image.parquet word.parquet \\
        --registry stimulus_registry/ -o composed/run_root/

    # How the spaces in N feature tables relate to each other
    psytwill compare image.parquet:image caption.parquet:cap -o geometry/

    # Shared-dimension counts against one source space (CV-CCA)
    psytwill decompose image.parquet:image -o decomp/ --source image:ebind
"""

import argparse
import json
import sys
from pathlib import Path

from psytwill import __version__
from psytwill.compare import DEFAULT_K
from psytwill.decompose import DEFAULT_RANK_CAP
from psytwill.exceptions import InputError, PsytwillError, SpaceError


def _parse_spaces(arg: str | None) -> dict[str, str | None] | None:
    """Parse ``--spaces minilm:euclidean,emotion`` into {name: metric?}."""
    if arg is None:
        return None
    requested: dict[str, str | None] = {}
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, metric = item.partition(":")
        requested[name] = metric or None
    return requested or None


def _run_matrices(args: argparse.Namespace) -> None:
    from psytwill.pipeline import build_quilt

    if len(args.inputs) > 2:
        raise PsytwillError(
            f"Expected 1 input (self mode) or 2 (cross mode), got "
            f"{len(args.inputs)}."
        )
    input_a = args.inputs[0]
    input_b = args.inputs[1] if len(args.inputs) == 2 else None

    summary = build_quilt(
        input_a,
        input_b,
        output_dir=args.output,
        spaces=_parse_spaces(args.spaces),
        metric=args.metric,
        distance=args.distance,
        diagonal=args.diagonal,
    )

    print(f"psytwill matrices ({summary['mode']} mode) -> {summary['output_dir']}/")
    for r in summary["results"]:
        n_a, n_b = r.frame.shape
        nan_note = ""
        if r.n_valid_a < n_a or r.n_valid_b < n_b:
            nan_note = f"  [n_valid {r.n_valid_a}x{r.n_valid_b}]"
        print(
            f"  {summary['matrix_files'][r.key]:<40} {n_a}x{n_b}  "
            f"{r.form}{nan_note}"
        )
    if summary["series_file"]:
        print(f"  {summary['series_file']:<40} ({summary['series_kind']} series)")
    print("  matrices.meta.json")


def _run_spaces(args: argparse.Namespace) -> None:
    from psytwill.pipeline import read_scores
    from psytwill.spaces import detect_spaces

    df = read_scores(args.input)
    spaces = detect_spaces(df.columns)
    if not spaces:
        print(f"No feature spaces detected in {args.input}.")
        return
    print(f"{len(spaces)} space(s) detected in {args.input} ({len(df)} rows):")
    for s in spaces.values():
        cols = (
            f"{s.columns[0]}..{s.columns[-1]}"
            if s.kind == "embedding"
            else ", ".join(s.columns[:4]) + (", ..." if s.n_dims > 4 else "")
        )
        print(
            f"  {s.name:<18} {s.kind:<10} {s.n_dims:>4}d  "
            f"default={s.default_metric:<12} {cols}"
        )


def _parse_modality_map(arg: str | None) -> dict[str, str] | None:
    """Parse ``--modality-map myext=audio,other=text`` into a dict."""
    if arg is None:
        return None
    mapping = {}
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        extractor, sep, modality = item.partition("=")
        if not sep or not extractor or not modality:
            raise PsytwillError(
                f"Bad --modality-map entry {item!r}; expected "
                "EXTRACTOR=MODALITY (e.g. 'myext=audio')."
            )
        mapping[extractor] = modality
    return mapping or None


def _run_features(args: argparse.Namespace) -> None:
    from psytwill.features import build_features, refresh_group_labels, refresh_group_nulls

    if args.refresh_segments:
        from psytwill.windows import record_table_segments

        for table in args.inputs:
            rec = record_table_segments(table, args.refresh_segments)
            print(f"psytwill features --refresh-segments {table}: {len(rec['files'])} unit(s) with segments "
                  f"in {rec['dir']}")
        if not (args.refresh_nulls or args.refresh_labels):
            return
    if args.refresh_nulls or args.refresh_labels:
        for table in args.inputs:
            if args.refresh_nulls:
                got = refresh_group_nulls(table)
                n_decl = sum(v is not None for v in got.values())
                print(f"psytwill features --refresh-nulls {table}: {n_decl}/{len(got)} models declare nulls")
            if args.refresh_labels:
                got = refresh_group_labels(table)
                n_decl = sum(v is not None for v in got.values())
                print(f"psytwill features --refresh-labels {table}: {n_decl}/{len(got)} models declare labels")
        return
    if not args.output:
        raise SystemExit("psytwill features: -o/--output is required (unless --refresh-nulls/--refresh-labels)")
    summary = build_features(
        args.inputs,
        output=args.output,
        modality_map=_parse_modality_map(args.modality_map),
    )
    print(
        f"psytwill features -> {summary['output']}  "
        f"({summary['rows']} rows, {summary['n_stimuli']} stimuli, "
        f"{len(summary['models'])} models)"
    )
    for entry in summary["inputs"]:
        note = "" if entry["extractor"] else "  [legacy: no sidecar]"
        print(
            f"  {entry['path']}: {entry['rows']} rows, "
            f"{entry['n_feature_columns']} feature cols{note}"
        )
    print(f"  {summary['meta_path']}")
    if args.segments_dir:
        from psytwill.windows import record_table_segments

        rec = record_table_segments(summary["output"], args.segments_dir)
        print(f"  segments: {len(rec['files'])} unit(s) from {rec['dir']}")


def _run_timelines(args: argparse.Namespace) -> None:
    from psytwill.timelines import build_timeline

    summary = build_timeline(
        args.events,
        registry_dir=args.registry,
        output=args.output,
        features=args.features,
        models=args.models.split(",") if args.models else None,
        context_model=args.context_model,
        context_k=args.context_k,
    )
    sets = ", ".join(f"{k}={v}" for k, v in sorted(summary["sets"].items()))
    print(
        f"psytwill timelines -> {summary['output']}  "
        f"({summary['rows']} rows, {summary['presentations']} presentations; {sets})"
    )
    if summary["models"]:
        print(f"  features attached: {', '.join(summary['models'])}")
    print(f"  {summary['meta_path']}")


def _run_compose(args: argparse.Namespace) -> None:
    from psytwill.compose import build_composed

    modality_map = _parse_modality_map(args.modality_map) or None
    summary = build_composed(
        args.events,
        args.stores,
        args.output,
        registry_dir=args.registry,
        window=args.window,
        onset_column=args.onset_column,
        lead_out=args.lead_out,
        models=args.models.split(",") if args.models else None,
        modality_map=modality_map,
        sparse=args.sparse,
        min_coverage=args.min_coverage,
        comparability=args.comparability,
        family=args.family,
        force=args.force,
        dry_run=args.dry_run,
    )

    # Media renders even when the tables are up to date: adding media to an
    # existing compose is a normal second invocation, not a change of inputs.
    if args.media and not args.dry_run:
        summary["media"] = _compose_media(args)

    if args.json:
        print(json.dumps(summary, indent=2))
        return
    note = " [dry run]" if summary.get("dry_run") else (
        " [up to date]" if summary.get("up_to_date") else "")
    print(f"psytwill compose -> {summary['output_dir']}  "
          f"({len(summary['runs'])} run(s)){note}")
    for stem, info in sorted(summary["streams"].items()):
        if info.get("skipped"):
            print(f"  {stem}: up to date ({info['output']})")
        elif summary.get("dry_run"):
            print(f"  {stem}: ~{info['estimated_rows_lower_bound']:,} rows "
                  f"({len(info['models'])} models) -> {info['output']}")
        else:
            print(f"  {stem}: {info['rows']:,} rows "
                  f"({len(info['models'])} models) -> {info['output']}")
    for slug, m in (summary.get("media") or {}).items():
        print(f"  media {slug}: {m['frames']} frames, {m['audio_items']} "
              f"audio items ({m['unmapped']} unmapped)")


def _compose_media(args: argparse.Namespace) -> dict:
    from psytwill.compose import read_runs
    from psytwill.media import (
        media_map_from_registry,
        media_map_from_tsv,
        render_run_media,
    )
    from psytwill.timelines import Registry

    media_map: dict = {}
    if args.registry and args.stimuli_root:
        media_map.update(media_map_from_registry(args.registry, args.stimuli_root))
    if args.media_map:
        media_map.update(media_map_from_tsv(args.media_map))
    if not media_map:
        raise InputError(
            "--media needs a file source: --media-map map.tsv, or --registry "
            "plus --stimuli-root for registry-declared media columns."
        )
    try:
        w, h = (int(v) for v in args.screen.lower().split("x"))
    except ValueError:
        raise InputError(f"--screen must be WxH pixels, got {args.screen!r}")
    registry = Registry.from_dir(args.registry) if args.registry else None
    pres, runs = read_runs(args.events, registry,
                           onset_column=args.onset_column,
                           lead_out=args.lead_out)
    out = {}
    for slug, info in runs.items():
        out[slug] = render_run_media(
            Path(args.output) / "movies" / slug,
            pres[pres["_slug"] == slug],
            media_map,
            info["run_end"],
            window=args.window,
            screen=(w, h),
            image_frac=args.image_frac,
            bg_gray=args.bg_gray,
            scale=args.media_scale,
        )
    return out


def _run_project(args: argparse.Namespace) -> None:
    from psytwill.project import project_onto_intervals

    frame, meta = project_onto_intervals(
        args.gridded,
        args.intervals,
        window=args.window,
        models=args.models.split(",") if args.models else None,
        bins=args.bins,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    meta_path = out.parent / (out.name.removesuffix(".parquet") + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"psytwill project ({meta['bins']} bins) -> {out}  "
        f"({meta['rows']} rows, {meta['n_intervals']} intervals x "
        f"{len(meta['models'])} models, {meta['n_overlapped_bins']} "
        "overlapped bins)"
    )
    print(f"  {meta_path}")


def _parse_table_arg(arg: str) -> tuple[str, str | None]:
    """``path.parquet:prefix`` -> (path, prefix). Prefix is optional."""
    path, sep, prefix = arg.rpartition(":")
    if not sep or len(prefix) > 40 or "/" in prefix or "." in prefix:
        return arg, None
    return path, prefix


def _load_aligned(args: argparse.Namespace):
    """The loading path both geometry verbs share: load, dedupe, align, stride.

    Returns ``(spaces, labels, groups, report)``. Kept as one function so
    ``compare`` and ``decompose`` cannot drift apart in what a row is — the
    decomposition must run on exactly the rows the verdict ran on.
    """
    from psytwill.store import LoadReport, align_spaces, dedupe_spaces, load_spaces

    key = tuple(args.key.split(","))
    models = args.models.split(",") if args.models else None
    report = LoadReport()
    spaces: dict = {}
    for raw in args.inputs:
        path, prefix = _parse_table_arg(raw)
        loaded = load_spaces(
            path,
            key=key,
            models=models,
            pool="mean" if args.pool == "mean" else None,
            prefix=prefix,
            window=args.window,
            report=report,
        )
        clash = set(loaded) & set(spaces)
        if clash:
            raise InputError(
                f"Space name(s) {sorted(clash)} came from two tables. Give each "
                "input a prefix (path.parquet:image) so they cannot collide."
            )
        spaces.update(loaded)
    print(f"loaded {len(spaces)} spaces from {len(args.inputs)} table(s)")

    spaces = dedupe_spaces(spaces, report=report)
    spaces, labels = align_spaces(spaces)
    print(f"  {len(report.deduped)} deduped, {len(report.skipped_string)} string "
          f"families skipped, aligned at n={len(labels)}")
    for name, cols in sorted(report.dropped_provenance.items()):
        print(f"  dropped provenance columns from {name}: {', '.join(cols)}")

    if args.stride > 1:
        # Every Nth surviving row *within* each clip (the label prefix), so
        # the kept rows stay evenly spaced in time and no clip is favored.
        from dataclasses import replace

        idx: list[int] = []
        prev = None
        for i, lab in enumerate(labels):
            clip = lab.split(args.group_sep)[0]
            if clip != prev:
                prev, j = clip, 0
            if j % args.stride == 0:
                idx.append(i)
            j += 1
        labels = [labels[i] for i in idx]
        spaces = {
            name: replace(s, labels=labels, X=s.X[idx])
            for name, s in spaces.items()
        }
        print(f"  stride {args.stride}: kept n={len(labels)} rows")

    groups = None
    if args.group_by:
        groups = [lab.split(args.group_sep)[0] for lab in labels]
        print(f"  folds grouped by label prefix: {len(set(groups))} groups")

    return spaces, labels, groups, report


def _report_dict(report) -> dict:
    return {
        "deduped": report.deduped,
        "skipped_string": report.skipped_string,
        "skipped_empty": report.skipped_empty,
        "pooled": report.pooled,
        "dropped_provenance": report.dropped_provenance,
    }


def _print_progress(done: int, total: int, label: str) -> None:
    if done == 1 or done == total or done % max(1, total // 20) == 0:
        print(f"  [{done:>6d}/{total}] {label}", flush=True)


def _run_compare(args: argparse.Namespace) -> None:
    from psytwill.geometry import compare_spaces, write_geometry

    measures = args.measures.split(",") if args.measures else None
    spaces, labels, groups, report = _load_aligned(args)

    progress = _print_progress

    result = compare_spaces(
        spaces,
        measures=measures,
        k=args.k,
        n_splits=args.n_splits,
        groups=groups,
        n_permutations=args.permutations,
        block_size=args.block_size,
        random_state=args.seed,
        progress=progress,
    )
    paths = write_geometry(
        result,
        args.output,
        name=args.name,
        inputs=args.inputs,
        labels=labels,
        extra={"stride": args.stride, "load_report": _report_dict(report)},
    )
    print(f"wrote {len(result.pairs)} pair rows, {len(result.manifest)} spaces")
    for kind in ("pairs", "manifest", "meta_path"):
        print(f"  {paths[kind]}")


def _run_decompose(args: argparse.Namespace) -> None:
    from psytwill.decompose import decompose_spaces, write_decomposition

    spaces, labels, groups, report = _load_aligned(args)

    result = decompose_spaces(
        spaces,
        args.source,
        groups=groups,
        n_splits=args.n_splits,
        rank_cap=args.rank_cap,
        n_perm=args.permutations,
        block_size=args.block_size,
        prefix_alpha=args.prefix_alpha,
        random_state=args.seed,
        progress=_print_progress,
    )
    paths = write_decomposition(
        result,
        args.output,
        name=args.name,
        inputs=args.inputs,
        labels=labels,
        extra={"stride": args.stride, "load_report": _report_dict(report)},
    )
    print(
        f"wrote {len(result.components)} component rows, "
        f"{len(result.summary)} pair summaries"
    )
    for kind in ("components", "summary", "meta_path"):
        print(f"  {paths[kind]}")


def _run_viz_movies(args: argparse.Namespace) -> None:
    from psytwill.viz.build import build_movies_bundle
    from psytwill.viz.movies import parse_projection_spec

    page = build_movies_bundle(
        features_dir=args.features_dir,
        films_dir=args.films_dir,
        out_dir=args.output,
        registry=args.registry,
        slugs=args.slugs.split(",") if args.slugs else None,
        projections=parse_projection_spec(args.projections),
        families=not args.no_families,
    )
    print(f"psytwill viz movies -> {page}")


def _run_battery(args: argparse.Namespace) -> None:
    from psytwill.battery import (
        BATTERY,
        BATTERY_CLAMPED_ON,
        BATTERY_VERSION,
        EXTRACTOR_VERSIONS,
        check_sidecar,
        models_seen,
        to_records,
    )
    from psytwill.exceptions import BatteryError

    if args.check:
        violations: list[str] = []
        sidecars = []
        for path in args.check:
            sidecar = json.loads(Path(path).read_text())
            sidecars.append(sidecar)
            violations += check_sidecar(sidecar, source=str(path))
        for line in violations:
            print(line)
        never = sorted(set(BATTERY) - models_seen(sidecars)) if args.require_all else []
        for name in never:
            print(f"never emitted: {name}", file=sys.stderr)
        n = len(violations)
        print(
            f"battery {BATTERY_VERSION}: {len(args.check)} sidecar(s), "
            f"{n} violation{'s' if n != 1 else ''}"
            + (f", {len(never)} pinned model(s) never emitted" if args.require_all else "")
        )
        if n or never:
            raise BatteryError(
                f"{n} violation{'s' if n != 1 else ''}"
                + (f" + {len(never)} never-emitted" if never else "")
                + f" against battery {BATTERY_VERSION}"
            )
        return

    rows = to_records()
    if args.json:
        print(json.dumps(rows, indent=1))
        return
    print(
        f"psytwill battery {BATTERY_VERSION} (clamped {BATTERY_CLAMPED_ON}): "
        f"{len(rows)} models; "
        + ", ".join(f"{k} {v}" for k, v in EXTRACTOR_VERSIONS.items())
    )
    print(f"{'model':16} {'extractor':9} {'modality':8} {'kind':9} {'dim':>5}  checkpoint")
    for r in rows:
        dim = "-" if r["dim"] is None else r["dim"]
        print(
            f"{r['model']:16} {r['extractor']:9} {r['modality']:8} {r['kind']:9} "
            f"{dim:>5}  {r['checkpoint'] or '(analytic)'}"
        )



# --------------------------------------------------------------------------
# space: private-block fits (psytwill-space v0.1)
# --------------------------------------------------------------------------


def _space_members(args: argparse.Namespace, available: list[str]) -> list[str]:
    """Members from --members, else every available model whose battery pin
    has the block's modality and a numeric kind (embedding | profile)."""
    from psytwill.battery import BATTERY

    if args.members:
        members = [m.strip() for m in args.members.split(",") if m.strip()]
        attached = [m for m in members if m in BATTERY and BATTERY[m].attached]
        if attached:
            raise SpaceError(f"{attached}: attached model(s) are in no block; read them as --targets, "
                             "or ship them through a release's `attached` section")
        return members

    modality = {"V": "visual", "A": "audio", "L": "text"}.get(args.block, args.block)
    want = {name for name, m in BATTERY.items()
            if m.modality == modality and m.kind in ("embedding", "profile") and not m.attached}
    members = [name for name in available if name in want]
    if not members:
        raise SpaceError(
            f"no available model belongs to block {args.block!r} ({modality}); tables hold {sorted(available)}"
        )
    from psytwill.space import split_members

    return split_members(members)


def _key_value(x) -> float | str:
    """One key value in comparable form: numbers as float, everything else as str."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)


def _row_key(label: str) -> tuple:
    return tuple(_key_value(x) for x in label.split("|"))


def _exclude_rows(path: str | None, key: tuple[str, ...]) -> set[tuple]:
    """The --exclude-rows file as a set of comparable key tuples (empty without a file)."""
    if not path:
        return set()
    import pandas as pd

    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    missing = [k for k in key if k not in df.columns]
    if missing:
        raise SpaceError(f"--exclude-rows {path} lacks key column(s) {missing}; it needs {list(key)}")
    return {tuple(_key_value(v) for v in row) for row in df[list(key)].itertuples(index=False)}


def _assert_tables_framing(paths: list[str]) -> None:
    """Refuse member tables whose video inputs were framed under different rules.

    Each features table records the rule its video inputs share
    (``input_framing``, psytwill >= 0.50.0). Tables without video inputs
    (stills, audio, text) are ignored, as is a set where no table declares a
    stage 0 (legacy). Once one table declares one, every other video table
    must match it, and a table built before 0.50.0 must be rebuilt first.
    """
    from psytwill.features import group_framing
    from psytwill.sidecar import UNDECLARED, describe_framing, framing_rule_key

    got = {p: group_framing(p) for p in paths}
    declared = {p: r for p, r in got.items()
                if isinstance(r, dict) and framing_rule_key(r) is not None and r.get("stage0") != UNDECLARED}
    if not declared:
        return
    ref_path, ref = next(iter(declared.items()))
    for p, r in got.items():
        if r is None:
            continue
        if r == "unknown":
            raise SpaceError(
                f"{p} was built before psytwill 0.50.0 recorded input framing, and {ref_path} was "
                f"extracted with {describe_framing(ref)}; rebuild {p} with `psytwill features` so "
                "its framing can be compared.")
        if framing_rule_key(r) is not None and framing_rule_key(r) != framing_rule_key(ref):
            raise SpaceError(
                f"framing mismatch across member tables: {ref_path} has {describe_framing(ref)}, "
                f"{p} has {describe_framing(r)}. Re-extract the video corpora with the same "
                "`viz2psy --framing-table`, then rebuild the tables.")


def _space_load(args: argparse.Namespace, members: list[str] | None = None, *,
                partial_features: dict[str, list[str]] | None = None, verbose: bool = True,
                want_labels: bool = False):
    """Load member spaces one model at a time (a 300 M-row group table does
    not fit in pandas whole), dropping --exclude-ids rows. Returns
    (spaces, members, n_excluded, report).

    ``partial_features`` (member -> feature columns) marks a read of one slice
    of the stimuli (``args.include_ids``), as chunked `space project` does: a
    model with no rows in a table's slice contributes nothing from that table
    (a member with none in any comes back with zero rows, carrying these
    columns), and a model all NaN in the slice is kept as NaN rows. Both are
    what the whole-table read holds for those stimuli.

    A member is read from EVERY table that carries it and the rows are
    stacked, so a fit over several corpora (one group table per corpus) sees
    all of them. Before 0.18.1 the first table holding a model won and the
    rest were silently ignored, which would have fit the A block on one
    corpus while reporting seven. Two tables contributing the same row label
    is refused rather than pooled: on a multi-corpus fit a duplicate key
    means the same stimulus was extracted twice, not a replicate.
    """
    import numpy as np

    from psytwill.features import group_checkpoints, group_labels, group_nulls
    from psytwill.labels import LabelSource, stack_labels
    from psytwill.store import LoadReport, SpaceMatrix, distinct_values, load_spaces

    key = tuple(args.key.split(","))
    rep = LoadReport()
    where: dict[str, list[str]] = {}
    # `space fit --target-features`: tables read for outside targets only
    tables = list(args.features) + list(getattr(args, "target_features", None) or [])
    table_nulls = {str(path): group_nulls(path) for path in tables}
    # labels and checkpoints matter only to a fit (projection uses the fit's columns)
    table_labels = {str(path): group_labels(path) for path in tables} if want_labels else {}
    table_ckpts = {str(path): group_checkpoints(path) for path in tables} if want_labels else {}
    for path in tables:
        for m in sorted(distinct_values(path, "model")):
            where.setdefault(m, []).append(str(path))
    if members is None:
        members = _space_members(args, list(where))
    from psytwill.space import member_source, select_features

    source = {m: member_source(m) for m in members}
    missing = [m for m in members if source[m][0] not in where]
    if missing:
        raise SpaceError(f"member(s) {missing} not in the given tables; available {sorted(where)}")
    _assert_tables_framing(sorted({p for m in members for p in where[source[m][0]]}))
    ids: set[str] = set()
    if args.exclude_ids:
        ids = {line.strip() for line in Path(args.exclude_ids).read_text().splitlines() if line.strip()}
    drop_rows = _exclude_rows(getattr(args, "exclude_rows", None), key)
    spaces: dict = {}
    loaded: dict = {}
    for m in members:
        model, cols = source[m]
        if model in loaded:
            parts = loaded[model]
        else:
            parts = []
            for path in where[model]:
                got = load_spaces(path, key=key, models=[model], window=args.window, report=rep,
                                  stimulus_ids=getattr(args, "include_ids", None),
                                  keep_all_nan=partial_features is not None)
                if model not in got:
                    if partial_features is not None:
                        continue
                    raise SpaceError(f"model {model!r} in {path} loaded as none of {sorted(got)} "
                                     "(string-valued or empty?)")
                parts.append(got[model])
            loaded[model] = parts
        if not parts:
            feats = list(partial_features[m])
            spaces[m] = SpaceMatrix(name=m, labels=[], X=np.empty((0, len(feats))), features=feats)
            continue
        if cols is not None:
            # a split member (space.MEMBER_SPLITS): its columns of the model
            parts = [select_features(part, cols) for part in parts]
            for part in parts:
                part.name = m
        sm = parts[0]
        if len(parts) > 1:
            feats = parts[0].features
            for part, path in zip(parts[1:], where[model][1:]):
                if part.features != feats:
                    raise SpaceError(
                        f"model {m!r} has {len(part.features)} feature columns in {path} but "
                        f"{len(feats)} in {where[model][0]}; a member must carry the same columns "
                        "in every table it is stacked from")
            labels = [lab for part in parts for lab in part.labels]
            if len(set(labels)) != len(labels):
                dup = sorted({lab for lab in labels if labels.count(lab) > 1})[:5]
                raise SpaceError(
                    f"model {m!r} has {len(labels) - len(set(labels))} row label(s) present in more "
                    f"than one table (e.g. {dup}); each table must hold distinct stimuli")
            sm = SpaceMatrix(name=sm.name, labels=labels, X=np.vstack([part.X for part in parts]),
                             features=feats, modality=sm.modality, extractor=sm.extractor,
                             n_replicates=max(part.n_replicates for part in parts))
        if ids:
            keep = [i for i, lab in enumerate(sm.labels) if lab.split("|")[0] not in ids]
            if len(keep) != sm.n:
                sm = SpaceMatrix(name=sm.name, labels=[sm.labels[i] for i in keep], X=sm.X[keep],
                                 features=sm.features, modality=sm.modality, extractor=sm.extractor,
                                 n_replicates=sm.n_replicates)
        if drop_rows:
            keep = [i for i, lab in enumerate(sm.labels) if _row_key(lab) not in drop_rows]
            rep.excluded_rows[m] = sm.n - len(keep)
            if len(keep) != sm.n:
                sm = SpaceMatrix(name=sm.name, labels=[sm.labels[i] for i in keep], X=sm.X[keep],
                                 features=sm.features, modality=sm.modality, extractor=sm.extractor,
                                 n_replicates=sm.n_replicates)
        spaces[m] = sm
        declared = [table_nulls[p].get(model) for p in where[model]]
        if any(d != declared[0] for d in declared[1:]):
            raise SpaceError(
                f"model {model!r} carries different `nulls` declarations across its tables "
                f"({', '.join(where[model])}); refresh every input so they agree "
                "(`<extractor> sidecar refresh`, then `psytwill features --refresh-nulls`)")
        nulls = declared[0]
        if cols is not None and nulls is not None:
            nulls = {c: v for c, v in nulls.items() if c in cols}
        rep.nulls[m] = nulls
        # Contract B 1.2 labels, stacked across tables by the rule the
        # features aggregation uses (structural disagreement refused)
        held = None
        for p in (where[model] if want_labels else []):
            got, origin = table_labels[p]
            held = stack_labels(model, held, LabelSource(got.get(model), p,
                                                         (origin.get(model) or {}).get("created_at"),
                                                         frozenset(sm.features)),
                                what="fit", fix="`psytwill features --refresh-labels` every table")
        rep.labels[m] = held.labels if held is not None else None
        rep.label_sources[m] = list(where[model])
        rep.checkpoints[m] = next((table_ckpts[p][model] for p in where[model]
                                   if model in table_ckpts.get(p, {})), None)
        src = f" from {len(parts)} tables" if len(parts) > 1 else ""
        of = f" (split from {model})" if cols is not None else ""
        if verbose:
            print(f"  loaded {m}: {sm.n} rows x {sm.dim} features{src}{of}", flush=True)
    return spaces, members, len(ids), rep


#: what `space fit --like` copies from a manifest; passing any of these as well is refused
_LIKE_OWNED = ("features", "block", "members", "defer_members", "key", "window", "exclude_ids",
               "exclude_rows")


def _apply_like(args: argparse.Namespace) -> dict:
    """Fill a fit's settings from another fit's manifest, so a variant differs only where asked.

    The inputs, rows, members, deferrals, folds, corpus handling and criterion
    all come from the manifest; ``--member-scaling``, ``--fixed-k`` (default:
    the manifest's k), ``--k-schedule`` and ``--no-criterion`` are the
    caller's. A ``--k-schedule`` walks the variant to its OWN k under the
    source's criterion (a variant's k need not be its source's), so k is then
    not fixed. Returns the provenance recorded in the new manifest.
    """
    import hashlib
    import json

    path = Path(args.like)
    raw = path.read_bytes()
    m = json.loads(raw)
    # --key has a default, so only a non-default value counts as given
    given = [k for k in _LIKE_OWNED if getattr(args, k, None) not in (None, False)
             and not (k == "key" and args.key == "stimulus_id")]
    if given:
        raise SpaceError(f"--like {path.name} sets {', '.join('--' + g.replace('_', '-') for g in given)} "
                         "from the manifest; drop them, or fit without --like")
    crit = m["criterion"]
    args.features = list(m["inputs"])
    args.block = m["block"]
    out = [x.strip() for x in (args.leave_out or "").split(",") if x.strip()]
    unknown = [x for x in out if x not in m["members"]]
    if unknown:
        raise SpaceError(f"--leave-out {unknown}: not members of {path.name} ({', '.join(m['members'])})")
    if out and len(out) == len(m["members"]):
        raise SpaceError("--leave-out would leave the block no members")
    args.members = ",".join(x for x in m["members"] if x not in out)
    args.defer_members = ",".join(x for x in (m.get("deferred_members") or []) if x not in out)
    # a left-out member is scored as an outside target (leave-one-member-out coverage)
    args.targets = ",".join(dict.fromkeys(out + _split(args.targets)))
    args.key = m["key"]
    args.window = m["window"]
    args.exclude_ids = m.get("exclude_ids_file")
    args.exclude_rows = m.get("exclude_rows_file")
    args.n_splits = m["n_splits"]
    args.groups_from_label = bool(m["grouped"])
    # --per-corpus may ADD per-corpus scoring to a pooled source at a fixed k:
    # it is then a read beside the pooled folds and cannot move k
    added_per_corpus = bool(args.per_corpus) and crit.get("scope") != "per_corpus"
    if added_per_corpus and args.k_schedule:
        raise SpaceError("--per-corpus on a pooled --like source scores beside the source's criterion; "
                         "it cannot choose k, so drop --k-schedule")
    args.per_corpus = crit.get("scope") == "per_corpus" or added_per_corpus
    args.corpus_weights = (m.get("corpus_weights") or {}).get("scheme")
    args.corpora_from_label = bool(args.per_corpus or args.corpus_weights)
    for a, c in (("r2_min", "r2_min"), ("alpha", "alpha"), ("k_nn", "k_nn"), ("n_perm", "n_perm"),
                 ("eval_n", "eval_n"), ("block_size", "block_size"), ("seed", "random_state")):
        setattr(args, a, crit[c])
    if args.k_schedule and args.fixed_k is not None:
        raise SpaceError("--k-schedule walks to the variant's own k; --fixed-k freezes it: give one")
    if args.k_schedule and args.no_criterion:
        raise SpaceError("--k-schedule needs the criterion to choose k; drop --no-criterion")
    if args.fixed_k is None and not args.k_schedule:
        args.fixed_k = int(m["k"])
    # the label selection is part of what the source fit was (an arm is rebuilt
    # from its manifest); a --select given here replaces it
    source_sel = (m.get("selection") or {}).get("expression")
    selection_from = "caller" if args.select else "manifest" if source_sel else None
    if not args.select and source_sel:
        args.select = [source_sel]
    return {"manifest": str(path.resolve()), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "block": m["block"], "k": m["k"], "member_scaling": m.get("member_scaling", "pr"),
            "left_out": out, "added_per_corpus": added_per_corpus, "selection_from": selection_from}


def _split(arg: str | None) -> list[str]:
    return [x.strip() for x in (arg or "").split(",") if x.strip()]


def _space_fit_setup(args: argparse.Namespace, *, curve_to_disk: bool = True) -> tuple[dict, list, dict, dict]:
    """Everything `space fit` reads before it fits: ``(spaces, members, fit_block kwargs, context)``.

    ``context`` carries what the save needs afterwards (``like``, ``windows``,
    ``item_record``, ``rep``, ``n_excl``, ``partial``). With ``curve_to_disk``
    False no partial curve file is made and the walk prints nothing (a caller
    that fits without saving, e.g. `space subspaces`'s half fits).
    """
    from psytwill.space import DEFAULT_K_SCHEDULE, SPACE_SCHEMA_VERSION

    like = _apply_like(args) if args.like else None
    if not args.features or not args.block:
        raise SpaceError("space fit needs --features and --block (or --like MANIFEST)")
    if args.no_criterion and args.fixed_k is None:
        raise SpaceError("--no-criterion leaves nothing to choose k: give --fixed-k (or --like)")
    if args.leave_out and not args.like:
        raise SpaceError("--leave-out takes members out of a --like fit; without one, list --members")
    targets = _split(args.targets)
    if args.target_features and not targets:
        raise SpaceError("--target-features holds outside targets: name them with --targets")
    if args.target_rows == "block" and not targets:
        raise SpaceError("--target-rows block says how targets meet the block's rows: give --targets")
    if targets and not args.members:
        raise SpaceError("--targets needs the block's --members listed (or --like), so a target "
                         "can never be read as a member")
    if targets:
        spaces, loaded, n_excl, rep = _space_load(args, members=_split(args.members) + targets,
                                                  want_labels=True)
        members = [m for m in loaded if m not in targets]
    else:
        spaces, members, n_excl, rep = _space_load(args, want_labels=True)
    from psytwill.features import group_window_sec
    from psytwill.space import resolve_member_windows

    declared: dict = {}
    for path in args.features:
        for model, vals in group_window_sec(path).items():
            declared.setdefault(model, set()).update(vals)
    windows = resolve_member_windows(members, declared)  # refuses before the fit, not after
    absent, item_record = _item_masks_for_fit(args, spaces, members, rep)
    groups = None
    corpora = None
    schedule = tuple(int(k) for k in args.k_schedule.split(",")) if args.k_schedule else DEFAULT_K_SCHEDULE
    if args.groups_from_label or args.corpora_from_label or args.per_corpus or args.corpus_weights:
        from psytwill.store import align_spaces

        # block mode: the rows are the members' alone (targets never narrow them)
        on_rows = members if args.target_rows == "block" else members + targets
        _, labels = align_spaces({m: spaces[m] for m in on_rows})
        if args.groups_from_label:
            groups = [lab.split("|")[0] for lab in labels]
        if args.corpora_from_label or args.per_corpus or args.corpus_weights:
            from psytwill.fitcorpus import is_external, parse_ext_id

            ids = [lab.split("|")[0] for lab in labels]
            corpora = [parse_ext_id(i)[0] if is_external(i) else "internal" for i in ids]
            print(f"  corpora ({'criterion + ' if args.per_corpus else ''}gated-missing warning): "
                  f"{len(set(corpora))} corpora "
                  f"({', '.join(sorted(set(corpora)))})")

    def progress(i, n, what):
        if i == 1 or i % 25 == 0 or i == n:
            print(f"  [{i}/{n}] {what}", flush=True)

    # the walk's rows reach disk as they are scored; save_fit writes the
    # final curve and this partial file is removed once it has
    from pathlib import Path

    stem = args.stem or f"{args.block}_v{SPACE_SCHEMA_VERSION.split('.')[0]}"
    partial = Path(args.output) / f"{stem}_curve.partial.csv"
    if curve_to_disk:
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.unlink(missing_ok=True)

    def on_row(row, _cols=[]):
        import csv

        with partial.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_cols or list(row))
            if not _cols:
                _cols.extend(row)
                w.writeheader()
            w.writerow(row)

    kw = dict(block=args.block, k_schedule=schedule, n_splits=args.n_splits,
              groups=groups, corpora=corpora, nulls=rep.nulls, r2_min=args.r2_min, alpha=args.alpha, k_nn=args.k_nn,
              n_perm=args.n_perm, eval_n=args.eval_n or None, block_size=args.block_size,
              random_state=args.seed, progress=progress if curve_to_disk else None, per_corpus=args.per_corpus,
              corpus_weights=args.corpus_weights,
              on_row=on_row if curve_to_disk else None,
              defer=[m.strip() for m in (args.defer_members or "").split(",") if m.strip()],
              member_scaling=args.member_scaling, fixed_k=args.fixed_k, score=not args.no_criterion,
              targets=targets, column_labels=rep.labels, checkpoints=rep.checkpoints,
              select=args.select, label_sources=rep.label_sources, absent=absent,
              target_rows=args.target_rows)
    ctx = {"like": like, "windows": windows, "item_record": item_record, "rep": rep, "n_excl": n_excl,
           "partial": partial}
    return spaces, members, kw, ctx


def _run_space_fit(args: argparse.Namespace) -> None:
    from psytwill.space import fit_block, save_fit

    spaces, members, kw, ctx = _space_fit_setup(args)
    like, windows, item_record, rep, n_excl, partial = (ctx[k] for k in ("like", "windows", "item_record", "rep",
                                                                           "n_excl", "partial"))
    fit = fit_block(spaces, members, **kw)
    targets = kw["targets"]
    sel = fit.manifest["selection"]
    if sel:
        print(f"  selection {sel['expression']!r}"
              + (f": dropped member(s) with no matching column: {', '.join(sel['dropped_members'])}"
                 if sel["dropped_members"] else ""))
    members = list(fit.members)
    windows = {m: w for m, w in windows.items() if m in members}
    for m in members:
        pm = fit.manifest["per_member"][m]
        if pm["bookkeeping_columns"]:
            print(f"  {m}: bookkeeping column(s) left out: {', '.join(pm['bookkeeping_columns'])}")
        if pm["deselected_columns"]:
            print(f"  {m}: {len(pm['deselected_columns'])} column(s) not selected; "
                  f"{len(pm['labels'] or {})} kept")
        for col, others in pm["circular_with"].items():
            print(f"  {m}: {col} is derived_from the checkpoint of {', '.join(others)} (circular_with)")
    for m in members:
        pm = fit.manifest["per_member"][m]
        if pm["masked_columns"]:
            n_abs = fit.manifest["null_policy"]["n_member_rows_absent"][m]
            print(f"  {m}: absent from {n_abs} row(s), masked on {', '.join(pm['masked_columns'])}")
        if pm["low_support_columns"]:
            print(f"  {m}: dropped (support < {fit.manifest['min_support']} rows): "
                  f"{', '.join(pm['low_support_columns'])}")
        if pm["undefinable_rows_dropped"]:
            print(f"  {m}: {pm['undefinable_rows_dropped']} undefinable row(s) dropped")
        if pm["suspected_gated_missing"]:
            print(f"  {m}: WARNING declared `missing` but looks gated: {pm['suspected_gated_missing']}")
    from psytwill.space import member_source

    fit.manifest["member_sources"] = {m: member_source(m)[0] for m in members
                                      if member_source(m)[1] is not None}
    from psytwill.space import label_windows

    for m, w in windows.items():
        pm = fit.manifest["per_member"][m]
        lw = label_windows(pm.get("labels"), w)
        pm.update(lw if lw is not None else w)
    fit.manifest["inputs"] = [str(p) for p in args.features]
    fit.manifest["target_inputs"] = [str(p) for p in (args.target_features or [])]
    fit.manifest["item_masks"] = item_record
    for m in members:
        if fit.manifest["per_member"][m].get("item_masked_rows"):
            print(f"  {m}: absent from {fit.manifest['per_member'][m]['item_masked_rows']} row(s) whose "
                  "window reads two items (item masks)")
    fit.manifest["key"] = args.key
    fit.manifest["window"] = args.window
    fit.manifest["n_excluded_ids"] = n_excl
    fit.manifest["exclude_ids_file"] = args.exclude_ids
    fit.manifest["exclude_rows_file"] = args.exclude_rows
    fit.manifest["n_excluded_rows"] = dict(rep.excluded_rows)
    fit.manifest["like"] = like
    npz, manifest, curve = save_fit(fit, args.output, stem=args.stem)
    partial.unlink(missing_ok=True)
    verdict = ("criterion not scored" if fit.manifest["subsumes_all_members"] is None else
               "SUBSUMES all members" if fit.manifest["subsumes_all_members"] else "does NOT subsume every member")
    failing = [m for m in fit.manifest["deferred_members"]
               if fit.manifest["per_member"][m]["passed_all_folds"] is not True]
    if fit.manifest["subsumes_non_deferred"] and failing:
        verdict = f"subsumes every member except deferred {', '.join(failing)}"
    print(f"psytwill space fit [{args.block}] -> {manifest}")
    print(f"  k={fit.k} ({verdict}); PR bound {fit.manifest['pr_sum_bound']:.1f}; "
          f"n={fit.manifest['n_rows']} rows, {len(members)} members, excluded {n_excl} ids")
    for m in members:
        pm = fit.manifest["per_member"][m]
        print(f"  {m:<18} PR {pm['participation_ratio']:6.1f} rank {pm['whitened_rank']:>4}  "
              f"R2 {min(pm['r2_per_fold']) if pm['r2_per_fold'] else float('nan'):.3f}  "
              f"overlap {min(pm['overlap_per_fold']) if pm['overlap_per_fold'] else float('nan'):.3f}  "
              f"{_verdict(pm['passed_all_folds'])}"
              + ("  (R2/overlap pooled; verdict over corpora, below)" if args.per_corpus else ""))
        for c, pc in pm.get("per_corpus", {}).items():
            if pc["passed"] is None:
                print(f"      {c:<16} n {pc['n_rows']:>6}  UNSCOREABLE  ({pc['unscoreable']})")
                continue
            print(f"      {c:<16} R2 {pc['r2']:.3f}  overlap p {pc['overlap_p']:.3f}  "
                  f"n {pc['n_rows']:>6}  {_verdict(pc['passed'])}  (out of fold)")
    for t in targets:
        pt = fit.manifest["per_target"][t]
        r2s = pt["r2_per_fold"]
        print(f"  TARGET {t:<11} (outside the block) R2 {min(r2s) if r2s else float('nan'):.3f}  "
              f"{'covered' if pt['covered_at_k'] else 'UNSCOREABLE' if pt['covered_at_k'] is None else 'NOT covered'}"
              f" at k={fit.k}")
        for c, pc in pt.get("per_corpus", {}).items():
            if pc["passed"] is None:
                print(f"      {c:<16} n {pc['n_rows']:>6}  UNSCOREABLE  ({pc['unscoreable']})")
                continue
            print(f"      {c:<16} R2 {pc['r2']:.3f}  overlap p {pc['overlap_p']:.3f}  "
                  f"n {pc['n_rows']:>6}  {_verdict(pc['passed'])}  (out of fold)")
    print(f"  weights {npz}\n  curve {curve}")


def _verdict(passed: bool | None) -> str:
    return "UNSCOREABLE" if passed is None else "pass" if passed else "FAIL"


def _segments_for(args: argparse.Namespace) -> tuple[dict, dict]:
    """Segments of every input table when rows are on a time grid, and a per-table record."""
    from psytwill.windows import table_segments

    if not args.window or "time" not in str(args.key).split(","):
        return {}, {}
    segs: dict = {}
    record = {}
    for path in args.features:
        got = table_segments(path)
        record[str(path)] = len(got) if got else None
        clash = set(got) & set(segs)
        if clash:
            raise SpaceError(f"stimulus id(s) {sorted(clash)[:3]} carry segments in two tables")
        segs.update(got)
    return segs, record


def _item_masks_for_fit(args: argparse.Namespace, spaces: dict, members: list, rep) -> tuple[dict, dict]:
    """``({member: {column: row labels}}, record)``: rows whose window reads two items."""
    import numpy as np

    from psytwill.labels import resolve
    from psytwill.windows import item_masks, numeric_windows

    segs, per_table = _segments_for(args)
    record = {"hop": args.window, "segments_units_per_table": per_table,
              "note": "null = the table records no segments: each clip is one item"}
    if not segs:
        return {}, record
    absent = {}
    for m in members:
        raw = (rep.labels or {}).get(m)
        cols = resolve(raw, spaces[m].features, model=m, where=f"member {m!r}") if raw else None
        masks = item_masks(spaces[m], numeric_windows(cols), segs, args.window)
        if masks:
            lab = spaces[m].labels
            absent[m] = {c: {lab[i] for i in np.flatnonzero(v)} for c, v in masks.items()}
    return absent, record


def _read_members(fit, spaces: dict, segs: dict, hop: float | None, causal: bool) -> tuple[dict, dict, dict]:
    """Members as a projection reads them: item-masked and, with ``causal``, shifted.

    Returns ``(spaces, {member: absent row labels}, {member: record})``.
    """
    import numpy as np

    from psytwill.windows import read_absent

    if not hop or (not segs and not causal):
        return spaces, {}, {}
    out, absent, record = dict(spaces), {}, {}
    for m in fit.members:
        cols = (fit.manifest.get("per_member", {}).get(m, {}) or {}).get("labels")
        if not cols:
            continue
        sp, mask, rec = read_absent(spaces[m], cols, segs, hop, causal=causal)
        out[m] = sp
        if mask.any():
            absent[m] = {sp.labels[i] for i in np.flatnonzero(mask)}
        if rec.get("absent_rows") or rec.get("causal_shifts"):
            record[m] = rec
    return out, absent, record


def _project_frame(fit, spaces: dict, key: str, view: str, k: int | None = None,
                   absent: dict | None = None):
    import pandas as pd

    S, labels = fit.project(spaces, view=view, k=k, absent=absent)
    df = pd.DataFrame(S, columns=[f"{fit.block}_{j:03d}" for j in range(S.shape[1])])
    parts = [lab.split("|") for lab in labels]
    for j, kname in enumerate(key.split(",")):
        df.insert(j, kname, [p[j] if j < len(p) else None for p in parts])
    return df


def _subspace_frame(fit, objs: dict, spaces: dict, key: str, absent: dict | None = None):
    """Every subspace object's orthonormal coordinates on the aligned rows: ``<object>__<jjj>``
    columns (NaN where the object cannot place a row), keys first."""
    import numpy as np
    import pandas as pd

    from psytwill.shared_unique import project_subspaces

    placed, labels = project_subspaces(objs, fit, spaces, absent=absent)
    blocks = [pd.DataFrame(np.asarray(v, dtype=float), columns=[f"{n}__{j:03d}" for j in range(v.shape[1])])
              for n, v in placed.items() if v.shape[1]]
    parts = [lab.split("|") for lab in labels]
    keys = pd.DataFrame({kname: [p[j] if j < len(p) else None for p in parts]
                         for j, kname in enumerate(key.split(","))})
    return pd.concat([keys, *blocks], axis=1)  # one concat: inserting 275 objects' keys fragments


def _load_subspaces_for(args: argparse.Namespace, fit) -> tuple[dict, dict]:
    """``--subspaces`` read and checked against ``--space``; ``--objects`` keeps the named ones."""
    import hashlib

    from psytwill.shared_unique import load_subspaces

    objs, doc = load_subspaces(args.subspaces)
    sha = hashlib.sha256(Path(args.space).read_bytes()).hexdigest()
    if doc.get("fit_manifest_sha256") != sha:
        raise SpaceError(f"{args.subspaces} was estimated on {doc.get('fit_manifest')} "
                         f"(sha256 {str(doc.get('fit_manifest_sha256'))[:12]}…), not on {args.space} "
                         f"({sha[:12]}…): project through the fit the subspaces belong to")
    if args.objects:
        want = _split(args.objects)
        missing = [n for n in want if n not in objs]
        if missing:
            raise SpaceError(f"--objects: {', '.join(missing)} not in {args.subspaces}")
        objs = {n: objs[n] for n in want}
    return objs, doc


def _run_space_subspaces(args: argparse.Namespace) -> None:
    """Shared and unique subspaces of a frozen block, and their split-half gate (psytwill.shared_unique)."""
    import hashlib
    import json
    import time

    import numpy as np

    from psytwill import shared_unique as su
    from psytwill.space import _take_rows, fit_block, load_fit, select_features
    from psytwill.store import align_spaces

    t0 = time.time()
    like = Path(args.like)
    frozen = load_fit(like)
    stem = args.stem or f"{frozen.block}_subspaces"

    def say(msg):
        print(f"  [{time.time() - t0:.0f} s] {msg}", flush=True)

    # the frozen fit's own inputs and rows, exactly as `space fit --like` reads them
    fit_args = build_parser().parse_args(["space", "fit", "--like", str(like), "--no-criterion",
                                          "-o", str(args.output), "--stem", f"{stem}_refit"])
    spaces, members, kw, _ = _space_fit_setup(fit_args, curve_to_disk=False)
    if kw.get("targets"):
        raise SpaceError(f"{like.name} has outside targets; subspaces read the block's members only")
    full = fit_block(spaces, members, keep_rows=True, **kw)
    if list(full.members) != list(frozen.members):
        raise SpaceError(f"the refit keeps members {full.members}, the frozen fit {frozen.members}: the inputs "
                         f"changed since {like.name} was fitted")
    # a saved fit keeps no row labels: its row count and every member's kept columns stand in for them
    n_frozen = int(frozen.manifest["n_rows"])
    if len(full.labels) != n_frozen:
        raise SpaceError(f"the refit has {len(full.labels)} rows, the frozen fit {n_frozen}: the inputs changed "
                         f"since {like.name} was fitted")
    moved = [m for m in frozen.members
             if list(full.map.whiteners[m].features) != list(frozen.map.whiteners[m].features)]
    if moved:
        raise SpaceError(f"the refit keeps other columns than {like.name} for {', '.join(moved)}: the inputs "
                         f"changed since it was fitted")
    say(f"fit rows {len(full.labels)} over {len(full.members)} members (the frozen fit's count and columns)")

    def concat_through(fit_map):
        return fit_map.concat({m: select_features(full.rows[m], fit_map.whiteners[m].features)
                               for m in fit_map.members})

    W = concat_through(frozen.map)
    weights = full.row_weights
    objs, info = su.block_subspaces(frozen.map, W, c=args.shared_c, unique=args.unique,
                                    max_absent=args.max_absent, weights=weights)
    Sr, Ur = objs["S"].raw_scores(frozen.map, W), objs["U"].raw_scores(frozen.map, W)
    ok = np.isfinite(Sr).all(axis=1) & np.isfinite(Ur).all(axis=1)
    overlap = {**su.span_overlap(Sr[ok], Ur[ok]), "n_rows": int(ok.sum())}
    k = int(frozen.k)
    Fr, Rr = frozen.map.scores_from_concat(W, k), full.map.scores_from_concat(concat_through(full.map), k)
    ok = np.isfinite(Fr).all(axis=1) & np.isfinite(Rr).all(axis=1)
    refit = {**su.span_overlap(Fr[ok], Rr[ok]), "k": k, "n_rows": int(ok.sum())}
    say(f"S {objs['S'].dim} dims; U {objs['U'].dim} dims over {len(info['read']) - len(info['shared_only'])} "
        f"members; S/U msc {overlap['msc']}; refit vs frozen top-{k} msc {refit['msc']:.4f}")

    gate = None
    gate_meta = None
    if not args.no_gate:
        half = su.unit_halves(full.labels, args.half_seed)
        _, aligned_labels = align_spaces({m: spaces[m] for m in members})
        row_half = su.unit_halves(aligned_labels, args.half_seed)
        read_cols = np.zeros(W.shape[1], dtype=bool)
        for m in info["read"]:
            read_cols[su.member_slices(frozen.map)[m]] = True
        halves, half_info = [], []
        for h in (0, 1):
            sub = {m: _take_rows(sp, su.unit_halves(sp.labels, args.half_seed) == h) for m, sp in spaces.items()}
            kh = dict(kw)
            for key in ("groups", "corpora"):
                if kh.get(key) is not None:
                    kh[key] = [v for v, hh in zip(kh[key], row_half) if hh == h]
            fh = fit_block(sub, members, **kh)
            if list(fh.members) != list(frozen.members):
                raise SpaceError(f"half {h} keeps members {fh.members}; the block has {frozen.members}")
            Wh = concat_through(fh.map)
            oh, ih = su.block_subspaces(fh.map, Wh, c=args.shared_c, unique=args.unique,
                                        max_absent=args.max_absent, weights=weights, rows=half == h, basis=False)
            halves.append((oh, fh.map, Wh))
            half_info.append({"half": h, "n_fit_rows": int(fh.manifest["n_rows"]),
                              "n_units": int(len({su.unit_of(x) for x in fh.labels})), **ih})
            say(f"half {h}: {fh.manifest['n_rows']} rows; S {oh['S'].raw_dim} dims")
        ref = W[:, read_cols]
        gate = su.reproducibility(halves, ref[np.isfinite(ref).all(axis=1)], n_random=args.n_random,
                                  seed=args.half_seed, msc_min=args.gate_msc, margin=args.gate_margin)
        gate.insert(2, "dim", [objs[n].dim if n in objs else 0 for n in gate["object"]])
        for n, p in zip(gate["object"], gate["passed"]):
            if n in objs:
                objs[n].detail["gate_passed"] = None if p is None or p != p else bool(p)
        gate_meta = {"half_seed": args.half_seed, "n_random": args.n_random, "msc_min": args.gate_msc,
                     "margin": args.gate_margin, "halves": half_info}
    draws, draw_info = ({}, None) if not args.draws else su.draw_subspaces(
        objs, frozen.map, W, n=args.draws, seed=args.draw_seed, weights=weights)
    if draws:
        say(f"draws: {args.draws} x (S, U, ref) at d {draw_info['d']}; per-member S draws "
            f"{draw_info['per_member_dim']}")
    raw = like.read_bytes()
    meta = {"block": frozen.block, "fit_manifest": str(like.resolve()),
            "fit_manifest_sha256": hashlib.sha256(raw).hexdigest(), "k": k, "members": list(frozen.members),
            "shared_c": args.shared_c, "unique": args.unique, "max_absent": args.max_absent,
            "n_fit_rows": len(full.labels), "row_weights": (json.loads(raw).get("corpus_weights") or {}).get("scheme"),
            "unique_info": info, "overlap_S_U": overlap, "refit_vs_frozen": refit, "gate_settings": gate_meta,
            "draws": draw_info,
            "seconds": round(time.time() - t0)}
    npz, js = su.save_subspaces({**objs, **draws}, meta, args.output, stem=stem, gate=gate)
    print(f"psytwill space subspaces [{frozen.block}] -> {js}")
    rows = {} if gate is None else {r["object"]: r for r in gate.to_dict("records")}
    if draws:
        print(f"  + {len(draws)} draws (d {draw_info['d']})")
    for n, o in objs.items():
        r = rows.get(n)
        if r is None:
            verdict = ""
        elif r["n_rows"]:
            verdict = f"  msc {r['msc']:.3f} vs random {r['random_msc']:.3f}: {'PASS' if r['passed'] else 'fail'}"
        else:
            verdict = f"  {r['reason']}"
        print(f"  {n}: {o.dim} dims{verdict}")


def _run_space_project(args: argparse.Namespace) -> None:
    import pandas as pd

    from psytwill.space import load_fit
    from psytwill.store import distinct_values

    fit = load_fit(args.space)
    k = fit.k if args.k is None else args.k
    if not 1 <= k <= fit.k:
        raise SpaceError(f"--k {k} is outside {args.space}'s 1..{fit.k}; a projection can only drop components")
    subs, sub_doc = (None, None)
    if args.objects and not args.subspaces:
        raise SpaceError("--objects names subspace objects: give --subspaces")
    if args.subspaces:
        if args.k is not None or args.view != "covariance":
            raise SpaceError("--subspaces writes each object's own orthonormal coordinates: drop --k and --view")
        subs, sub_doc = _load_subspaces_for(args, fit)

    def frame(spaces, absent):
        if subs is not None:
            return _subspace_frame(fit, subs, spaces, args.key, absent)
        return _project_frame(fit, spaces, args.key, args.view, k, absent)

    if args.causal:
        if not args.window or "time" not in args.key.split(","):
            raise SpaceError("--causal shifts windows along time: it needs a time key and --window")
        pm = fit.manifest.get("per_member", {})
        if not any("window_sec" in e for m in fit.members for e in ((pm.get(m) or {}).get("labels") or {}).values()):
            raise SpaceError(f"--causal needs the fit's per-column window labels (Contract B 1.4, space schema "
                             f">= 1.17); {args.space} records none. Refit on refreshed tables.")
    segs, seg_tables = _segments_for(args)
    reads: dict = {}
    chunked = {}
    if args.chunk_size:
        # Every step after the load is row-wise (frozen whiteners, inner row
        # alignment, undefinable and unplaced rows dropped), so projecting
        # disjoint stimulus_id slices and concatenating is the whole-table
        # result; sorted ids keep the whole table's row order.
        ids = sorted(set().union(*(distinct_values(f, "stimulus_id") for f in args.features)))
        chunks = [ids[i:i + args.chunk_size] for i in range(0, len(ids), args.chunk_size)]
        feats = {m: list(fit.map.whiteners[m].features) for m in fit.members}
        frames = []
        for i, chunk in enumerate(chunks):
            ns = argparse.Namespace(**{**vars(args), "include_ids": set(chunk)})
            spaces, _, _, _ = _space_load(ns, members=fit.members, partial_features=feats, verbose=False)
            if min(sp.n for sp in spaces.values()) == 0:
                continue  # a member with no rows here: the inner alignment places none of them
            spaces, absent, rec = _read_members(fit, spaces, segs, args.window, args.causal)
            for m, r in rec.items():
                acc = reads.setdefault(m, {"item_masked_rows": 0, "absent_rows": 0})
                acc["item_masked_rows"] += r["item_masked_rows"]
                acc["absent_rows"] += r["absent_rows"]
                if "causal_shifts" in r:
                    acc["causal_shifts"] = r["causal_shifts"]
            frames.append(frame(spaces, absent))
            del spaces
            print(f"  chunk {i + 1}/{len(chunks)}: {len(frames[-1])} rows", flush=True)
        cols = args.key.split(",") + ([f"{n}__{j:03d}" for n, o in subs.items() for j in range(o.dim)]
                                      if subs is not None else [f"{fit.block}_{j:03d}" for j in range(k)])
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
        chunked = {"chunk_size": args.chunk_size, "chunks": len(chunks)}
    else:
        spaces, _, _, _ = _space_load(args, members=fit.members)
        spaces, absent, reads = _read_members(fit, spaces, segs, args.window, args.causal)
        df = frame(spaces, absent)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    # `k` is the columns written (relate.read_scores checks it); `fit_k` names the fit's own when it differs
    meta = {"space": str(args.space), "block": fit.block, "k": k, **({"fit_k": fit.k} if k != fit.k else {}),
            "inputs": [str(p) for p in args.features],
            "key": args.key, "window": args.window, "rows": int(len(df)), "view": args.view, **chunked,
            "causal": bool(args.causal), "segments_units_per_table": seg_tables, "window_reads": reads}
    if subs is not None:
        import hashlib

        meta.pop("k")
        meta["view"] = "subspace (each object's orthonormal coordinates on the fit rows)"
        meta["subspaces"] = {"path": str(Path(args.subspaces).resolve()),
                             "sha256": hashlib.sha256(Path(args.subspaces).read_bytes()).hexdigest(),
                             "schema_version": sub_doc["schema_version"],
                             "objects": {n: {"kind": o.kind, "dim": o.dim, "columns": f"{n}__000..{n}__{o.dim - 1:03d}"
                                             if o.dim else None, "gate_passed": o.detail.get("gate_passed"),
                                             **({"draw_of": o.detail["draw_of"]} if "draw_of" in o.detail else {})}
                                         for n, o in subs.items()}}
        (out.parent / (out.name.removesuffix(out.suffix) + ".meta.json")).write_text(json.dumps(meta, indent=2))
        print(f"psytwill space project [{fit.block}, {len(subs)} subspace objects] -> {out}  ({len(df)} rows)")
        return
    (out.parent / (out.name.removesuffix(out.suffix) + ".meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"psytwill space project [{fit.block}, k={k}{f' of {fit.k}' if k != fit.k else ''}] -> {out}  ({len(df)} rows)")


def _run_space_check(args: argparse.Namespace) -> None:
    from psytwill.space import check_fit, load_fit

    fit = load_fit(args.space)
    spaces, _, _, _ = _space_load(args, members=fit.members)
    groups = None
    if args.groups_from_label:
        from psytwill.store import align_spaces

        _, labels = align_spaces({m: spaces[m] for m in fit.members})
        groups = [lab.split("|")[0] for lab in labels]
    rows = check_fit(fit, spaces, groups=groups, r2_min=args.r2_min, alpha=args.alpha, k_nn=args.k_nn,
                     n_perm=args.n_perm, eval_n=args.eval_n or None, block_size=args.block_size, random_state=args.seed)
    n_pass = sum(r["passed"] is True for r in rows)
    n_fail = sum(r["passed"] is False for r in rows)
    n_unscoreable = len(rows) - n_pass - n_fail
    print(f"psytwill space check [{fit.block}, k={fit.k}] on {rows[0]['n_rows']} rows: "
          f"{n_pass}/{n_pass + n_fail} scored members pass"
          + (f"; {n_unscoreable} UNSCOREABLE on this table" if n_unscoreable else ""))
    for r in rows:
        if r["passed"] is None:
            print(f"  {r['member']:<18} n {r['n_rows']}  UNSCOREABLE  ({r['unscoreable']})")
            continue
        print(f"  {r['member']:<18} R2 {r['r2']:.3f}  overlap {r['overlap']:.3f} (null {r['null_mean']:.3f}, q99 {r['null_q99']:.3f}, p={r['overlap_p']:.3f})  "
              f"{_verdict(r['passed'])}")
    if args.output:
        import pandas as pd

        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"  {args.output}")
    if n_fail:
        raise SpaceError(f"{n_fail} member(s) not subsumed")
    if not n_pass:
        raise SpaceError("no member could be scored on this table")


def _read_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}


def _run_relate_fit(args: argparse.Namespace) -> None:
    from psytwill.relate import read_scores, relate_tables, save_relation

    a, b = read_scores(args.a), read_scores(args.b)
    fit = relate_tables(a, b, name=args.name, join=args.join.split(","), pool_a=args.pool_a, pool_b=args.pool_b,
                        exclude_ids=_read_ids(args.exclude_ids), exclude_ids_file=args.exclude_ids,
                        groups_from_label=args.groups_from_label,
                        registers_from_label=args.registers_from_label, scope=args.scope, n_splits=args.n_splits,
                        n_perm=args.n_perm, block_size=args.block_size, r_min=args.r_min,
                        rank_cap=args.rank_cap, random_state=args.seed, fixed_k=args.fixed_k)
    npz, manifest = save_relation(fit, args.output, stem=args.stem)
    m = fit.manifest
    st = m["subspace_stability"]
    print(f"psytwill space relate fit [{m['name']}: {a.block} <-> {b.block}] on {m['n_rows']} paired rows: "
          f"k = {fit.k} (prefix at r >= {m['r_min']:.3f}; {m['count_r_ge_min']} components reach it)")
    print(f"  held-out r: first {m['r_cv'][0]:.3f}, k-th {m['r_cv'][fit.k - 1]:.3f}, next {m['r_cv'][fit.k]:.3f}"
          if fit.k < len(m['r_cv']) else f"  held-out r: first {m['r_cv'][0]:.3f}")
    print(f"  subspace stability (fold vs full, mean sq. canonical r): a {st['a']['mean']:.3f} (min {st['a']['min']:.3f}), "
          f"b {st['b']['mean']:.3f} (min {st['b']['min']:.3f})")
    if m["per_register"]:
        print("  per-register r (first, k-th): " + "; ".join(
            f"{r} n={v['n']} {v['r'][0]:.3f}, {v['r'][-1]:.3f}{'' if v['full_rank'] else ' (n <= 2k)'}"
            for r, v in m["per_register"].items()))
    print(f"  {manifest}\n  {npz}")


def _relate_pair(args: argparse.Namespace):
    from psytwill.relate import load_relation, pair_tables, read_scores, require_side

    fit = load_relation(args.relation)
    a, b = read_scores(args.a), read_scores(args.b)
    require_side(fit, a, "a")
    require_side(fit, b, "b")
    sides = fit.manifest["sides"]
    idx, X, Y, _ = pair_tables(a, b, fit.manifest["join"], pool_a=sides["a"]["pool"], pool_b=sides["b"]["pool"],
                               exclude_ids=_read_ids(args.exclude_ids))
    return fit, X, Y


def _run_relate_check(args: argparse.Namespace) -> None:
    import pandas as pd

    from psytwill.relate import check_relation

    fit, X, Y = _relate_pair(args)
    res = check_relation(fit, X, Y, n_perm=args.n_perm, block_size=args.block_size, alpha=args.alpha,
                         random_state=args.seed)
    rm = fit.manifest["r_min"]
    q = f"q{100 * (1 - args.alpha):g}"
    print(f"psytwill space relate check [{fit.name}, k={fit.k}] on {res.n} paired rows: "
          f"{res.count} of {fit.k} components at r >= {rm:.3f}, prefix {res.prefix} "
          f"(fit: k = {fit.k}); first r {res.r[0]:.3f}, k-th {res.r[-1]:.3f}, null {q} max {res.null_q.max():.3f}")
    if args.output:
        cv = fit.manifest["r_cv"][: fit.k]
        pd.DataFrame({"component": range(1, fit.k + 1), "r": res.r, "null_q": res.null_q,
                      "alpha": args.alpha, "r_fit_cv": cv}).to_csv(args.output, index=False)
        print(f"  {args.output}")


def _run_relate_project(args: argparse.Namespace) -> None:
    import numpy as np

    from psytwill.relate import load_relation, read_scores, require_side

    fit = load_relation(args.relation)
    t = read_scores(args.scores)
    require_side(fit, t, args.side)
    X = t.frame[t.columns].to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    P = fit.project(X[ok], args.side)
    import pandas as pd

    df = pd.DataFrame(P, columns=[f"{fit.name}_{j:03d}" for j in range(fit.k)])
    for j, kname in enumerate(t.key):
        df.insert(j, kname, t.frame.loc[ok, kname].to_numpy())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False) if out.suffix == ".parquet" else df.to_csv(out, index=False)
    from psytwill.relate import sha256_file

    meta = {"relation": str(args.relation), "relation_sha256": sha256_file(args.relation), "name": fit.name,
            "side": args.side, "block": t.block, "k": fit.k, "r": fit.manifest["r_insample"],
            "scores": str(t.path), "view": t.view, "key": ",".join(t.key), "rows": int(len(df)),
            "n_dropped_nan": int((~ok).sum())}
    (out.parent / (out.name.removesuffix(out.suffix) + ".meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"psytwill space relate project [{fit.name} side {args.side} ({t.block}), k={fit.k}] -> {out}  ({len(df)} rows)")


def _run_space_labels(args: argparse.Namespace) -> None:
    import json

    from psytwill.labels import write_labels

    if args.release:
        from psytwill.release import load_release

        rel = load_release(args.release)
        paths = [e["manifest"] for e in rel.meta["blocks"].values()]
    else:
        paths = list(args.space)
    manifests = [json.loads(Path(p).read_text()) for p in paths]
    csv_path, json_path = write_labels(manifests, args.output)
    print(f"psytwill space labels [{len(manifests)} block(s)] -> {csv_path}\n  {json_path}")


def _run_release_write(args: argparse.Namespace) -> None:
    from psytwill.release import build_release, parse_absent, write_release

    attached, attached_root = [], None
    if args.attached:
        spec = json.loads(Path(args.attached).read_text())
        if not isinstance(spec, dict) or not isinstance(spec.get("attached"), list):
            raise InputError(f"{args.attached}: write it as {{\"root\": <store>, \"attached\": [<entry>, ...]}}")
        attached, attached_root = spec["attached"], spec.get("root")
    rel = build_release(name=args.name, version=args.version, blocks=args.blocks, relations=args.relations or [],
                        absent=[parse_absent(a) for a in (args.absent or [])], note=args.note,
                        model_prefix=args.model_prefix, data_root=args.data_root,
                        attached=attached, attached_root=attached_root)
    path = write_release(rel, args.output)
    if rel.get("labels"):
        print(f"  labels {path.parent / rel['labels']['table']}\n  columns {path.parent / rel['labels']['columns']}")
    b = ", ".join(f"{n} k={e['k']}" for n, e in rel["blocks"].items())
    r = ", ".join(f"{n} ({e['a']}<->{e['b']}) k={e['k']}" for n, e in rel["relations"].items()) or "none"
    a = ", ".join("<->".join(x["pair"]) for x in rel["absent"]) or "none"
    t = ", ".join(f"{x['name']} ({x['basis']}, {len(x['columns'])} col, {x['n_cells']} cells)"
                  for x in rel["attached"]) or "none"
    print(f"psytwill space release {rel['name']} {rel['version']}: blocks {b}; relations {r}; absent {a}; "
          f"attached {t}\n  {path}")


def _run_release_verify(args: argparse.Namespace) -> None:
    from psytwill.release import load_release

    rel = load_release(args.release)
    m = rel.meta
    print(f"psytwill space release {rel.label}: {len(m['blocks'])} blocks, {len(m['relations'])} relations, "
          f"{len(m['absent'])} absences"
          + (f", {len(m['exclusions'])} exclusion lists" if m.get("exclusions") else "")
          + (f", {len(m['attached'])} attached" if m.get("attached") else "")
          + "; every pinned file matches")


def _run_release_project(args: argparse.Namespace) -> None:
    import time

    import pandas as pd

    from psytwill.release import load_release
    from psytwill.release_project import parse_grain, project_grain, registry_id_map, uncovered, write_family
    from psytwill.store import distinct_values

    t0 = time.time()
    rel = load_release(args.release)
    key_maps = {}
    for spec in args.key_map or []:
        table, sep, path = spec.partition("=")
        if not sep:
            raise SpaceError(f"--key-map {spec!r}: write it as TABLE=FILE")
        key_maps[table] = path
    skip = {}
    for spec in args.skip_member or []:
        table, sep, members = spec.partition("=")
        if not sep or not members:
            raise SpaceError(f"--skip-member {spec!r}: write it as TABLE=MEMBER[,MEMBER]")
        skip.setdefault(table, []).extend(m for m in members.split(",") if m)
    grains = [parse_grain(g, key_maps, skip) for g in args.grain]
    stray = (set(key_maps) | set(skip)) - {g[0] for g in args.grain}
    if stray:
        raise SpaceError(f"--key-map / --skip-member for table(s) {sorted(stray)} that no --grain writes")
    reg = Path(args.registry)
    registry = pd.read_csv(reg, sep="\t" if reg.suffix in (".tsv", ".tab") else ",")
    results = []
    for g in grains:
        if g.block not in rel.meta["blocks"]:
            raise SpaceError(f"block {g.block!r} is not in {rel.label} (blocks: {sorted(rel.meta['blocks'])})")
        entry = rel.meta["blocks"][g.block]
        window = json.loads(Path(entry["manifest"]).read_text()).get("window") if "time" in g.key else None
        # map ids first (streamed), then read only the set's rows
        src = sorted(set().union(*(distinct_values(f, "stimulus_id") for f in g.features)))
        id_map = registry_id_map(registry, src, join=args.registry_join, pattern=args.id_pattern,
                                 partial=args.partial_coverage)
        ns = argparse.Namespace(features=g.features, key=",".join(g.key), window=window,
                                exclude_ids=None, exclude_rows=None, include_ids=set(id_map))
        spaces, _, _, _ = _space_load(ns, members=[m for m in entry["members"] if m not in g.skip_members])
        res = project_grain(rel, g, spaces, id_map)
        res.uncovered = uncovered(registry, id_map)
        results.append(res)
        del spaces
    side = write_family(rel, results, args.output, set_name=args.set, registry=reg, id_join=args.registry_join,
                        id_pattern=args.id_pattern, runtime_sec=round(time.time() - t0, 1))
    meta = json.loads(side.read_text())
    print(f"psytwill space release project [{rel.label} -> {args.set}]")
    for table, out in meta["output"].items():
        ms = [f"{m} k={e['count']} ({e['tables'][table]['n_unplaced']} unplaced)" for m, e in meta["models"].items()
              if table in e["tables"]]
        print(f"  {table}: {out['rows']} rows; {', '.join(ms)}\n    {out['path']}")
    for m, e in meta["models"].items():
        for table, te in e["tables"].items():
            print(f"  leak guard {m} [{table}]: {te['leak_guard']['rule']}")
    for table, gr in meta["input"]["grains"].items():
        if gr["members_absent"]:
            print(f"  {table}: members absent by declaration: {', '.join(gr['members_absent'])}")
        if gr["registry_uncovered"]:
            print(f"  {table}: {len(gr['registry_uncovered'])} registry stimuli have no row "
                  f"(--partial-coverage): {', '.join(gr['registry_uncovered'][:5])}"
                  + (" ..." if len(gr["registry_uncovered"]) > 5 else ""))
    if meta["input"]["archived_previous"]:
        print(f"  previous release's family moved to {meta['input']['archived_previous']}")
    print(f"  {side}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="psytwill",
        description=(
            "Chunk-by-chunk relational matrices (RDMs, coherence curves) "
            "from word2psy / viz2psy scores CSVs."
        ),
        epilog="Metrics: cosine, correlation, spearman, euclidean.",
    )
    parser.add_argument(
        "--version", action="version", version=f"psytwill {__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    m = sub.add_parser(
        "matrices",
        help="Build the matrix stack (1 CSV = self mode, 2 = cross mode)",
    )
    m.add_argument("inputs", nargs="+", help="1 or 2 scores CSV/TSV files")
    m.add_argument("-o", "--output", required=True, help="Output directory")
    m.add_argument(
        "--spaces",
        help="Comma-separated subset, optional per-space metric "
        "(e.g. 'minilm:euclidean,emotion')",
    )
    m.add_argument("--metric", help="Global metric override")
    m.add_argument(
        "--distance",
        action="store_true",
        help="Write similarity metrics in distance form (1 - sim)",
    )
    m.add_argument(
        "--diagonal",
        action="store_true",
        help="Cross mode: also write the aligned-pairs diagonal series",
    )
    m.set_defaults(func=_run_matrices)

    s = sub.add_parser("spaces", help="List detectable spaces in a CSV")
    s.add_argument("input", help="A scores CSV/TSV file")
    s.set_defaults(func=_run_spaces)

    f = sub.add_parser(
        "features",
        help="Aggregate N extractor CSVs into one long-form feature table",
    )
    f.add_argument("inputs", nargs="+",
                   help="Extractor scores CSV/TSV files (or, with --refresh-nulls, features tables)")
    f.add_argument(
        "-o",
        "--output",
        help="Output table (.parquet preferred, or .csv); "
        "<stem>.meta.json is written alongside",
    )
    f.add_argument(
        "--segments-dir",
        help="the corpus's staging directory of `<unit>_segments.csv` tables (packed short-file corpora): "
             "recorded in the table's sidecar with each file's sha256, so fits and projections mask rows "
             "whose window reads two items")
    f.add_argument(
        "--refresh-segments", metavar="DIR",
        help="record DIR's segments tables into the given features tables' sidecars; JSON only")
    f.add_argument(
        "--refresh-labels",
        action="store_true",
        help="patch the given features tables' sidecars with their inputs' (refreshed) Contract B 1.2 "
        "`labels` (`model_labels`); JSON only, no re-aggregation. Combines with --refresh-nulls",
    )
    f.add_argument(
        "--refresh-nulls",
        action="store_true",
        help="Instead of aggregating: re-read each input's (refreshed) extractor sidecar into "
        "the given features tables' sidecars (Contract B 1.1 `nulls`); JSON only, no re-aggregation",
    )
    f.add_argument(
        "--modality-map",
        help="Extractor->modality overrides, e.g. 'myext=audio,other=text' "
        "(defaults: viz2psy=visual, aud2psy=audio, word2psy=text)",
    )
    f.set_defaults(func=_run_features)

    def add_loading_args(p: argparse.ArgumentParser, default_name: str) -> None:
        """Args shared by the two geometry verbs, so a row means the same thing."""
        p.add_argument(
            "inputs",
            nargs="+",
            help="Long-form feature tables, optionally 'path.parquet:prefix'",
        )
        p.add_argument("-o", "--output", required=True, help="Output directory")
        p.add_argument("--name", default=default_name, help="Output file stem")
        p.add_argument(
            "--key",
            default="stimulus_id",
            help="Row grain, comma-separated (e.g. 'stimulus_id,time')",
        )
        p.add_argument("--models", help="Comma-separated model subset")
        p.add_argument(
            "--pool",
            choices=("mean", "none"),
            default="mean",
            help="Pool replicate rows sharing a key, or refuse them",
        )
        p.add_argument(
            "--window",
            type=float,
            help="Bin 'time' into windows of this many seconds before pooling; "
            "required on a temporal grid (it is the timescale axis, and it "
            "reconciles bin-start vs bin-center time stamps across groups)",
        )
        p.add_argument(
            "--stride",
            type=int,
            default=1,
            help="Keep every Nth aligned row within each clip (label prefix "
            "before --group-sep) before comparing — for running measures at "
            "a grain whose full n cannot afford them",
        )
        p.add_argument(
            "--block-size",
            type=int,
            help="Permute contiguous blocks of this size; required on a temporal grid",
        )
        p.add_argument(
            "--group-by",
            action="store_true",
            help="Group CV folds by the label prefix before --group-sep "
            "(one fold per clip on a movie grid)",
        )
        p.add_argument("--group-sep", default="|", help="Separator for --group-by")
        p.add_argument("--n-splits", type=int, default=5, help="CV folds")
        p.add_argument("--seed", type=int, default=0)

    c = sub.add_parser(
        "compare",
        help="How the spaces in N feature tables relate (Contract B geometry)",
    )
    add_loading_args(c, "space_geometry")
    c.add_argument("--measures", help="Comma-separated measure subset")
    c.add_argument("--k", type=int, default=DEFAULT_K, help="Neighbours for overlap")
    c.add_argument(
        "--permutations",
        type=int,
        default=1000,
        help="Neighbour-overlap null draws (0 skips the null)",
    )
    c.set_defaults(func=_run_compare)

    d = sub.add_parser(
        "decompose",
        help="Shared-dimension counts via cross-validated CCA (constraint-4 "
        "decomposition): one source space against every other",
    )
    add_loading_args(d, "space_decomposition")
    d.add_argument(
        "--source",
        required=True,
        help="Space every other space is decomposed against (e.g. 'frames:ebind')",
    )
    d.add_argument(
        "--rank-cap",
        type=int,
        default=DEFAULT_RANK_CAP,
        help="Max PCA rank per space before whitening",
    )
    d.add_argument(
        "--permutations",
        type=int,
        default=250,
        help="Null draws for the per-component threshold (0 skips; the "
        "shared count is then undefined)",
    )
    d.add_argument(
        "--prefix-alpha",
        type=float,
        default=0.01,
        help="Per-component null quantile a component must clear",
    )
    d.set_defaults(func=_run_decompose)

    pj = sub.add_parser(
        "project",
        help="Pool a gridded group table onto an interval index (the "
        "time-aware cross mode's irregular half)",
    )
    pj.add_argument("gridded", help="Gridded long-form group table (has 'time')")
    pj.add_argument(
        "--intervals",
        required=True,
        help="Interval-keyed group table supplying (stimulus_id, chunk_idx, "
        "onset, offset); its feature content is ignored",
    )
    pj.add_argument(
        "--window",
        type=float,
        required=True,
        help="Grid width in seconds; stamps are binned before containment, "
        "reconciling bin-start vs bin-center conventions",
    )
    pj.add_argument("--models", help="Comma-separated model subset")
    pj.add_argument(
        "--bins",
        choices=["all", "odd", "even"],
        default="all",
        help="All bins (the measurement) or one parity of the within-interval "
        "bin ranks (a split-half stability ceiling's two halves)",
    )
    pj.add_argument("-o", "--output", required=True, help="Output parquet path")
    pj.set_defaults(func=_run_project)

    b = sub.add_parser(
        "battery",
        help="The clamped feature battery (psytwill-space v0.1 input "
        "declaration): list it, or check sidecars against its pins",
    )
    b.add_argument("--json", action="store_true", help="machine-readable list")
    b.add_argument(
        "--check",
        nargs="+",
        metavar="META_JSON",
        help="§4.1 extractor sidecars and/or psytwill group sidecars to "
        "check for unknown models, checkpoint/extractor/prefix drift "
        "(exit 1 on any violation)",
    )
    b.add_argument(
        "--require-all",
        action="store_true",
        help="with --check: also fail if a pinned model appears in no sidecar",
    )
    b.set_defaults(func=_run_battery)

    tl = sub.add_parser(
        "timelines",
        help="Embed stimulus sets in experimental time: events.tsv -> registry "
        "ids -> ordered presentations with lags (contracts §4.3 item 5)",
    )
    tl.add_argument(
        "events",
        nargs="+",
        help="BIDS events.tsv files for one subject (give every session so "
        "lag_trials spans them); entities are read from the file names",
    )
    tl.add_argument(
        "--registry",
        help="stimuli/stimulus_registry/ directory (shared1000.tsv, twp1000.tsv, "
        "movies.tsv). Optional: events that carry their own stimulus_id (or "
        "BIDS stim_file) column resolve without one",
    )
    tl.add_argument(
        "--features",
        help="A `psytwill features` table (.parquet/.csv); its item-level rows "
        "are attached wide, voice-specific rows by (stimulus_id, voice)",
    )
    tl.add_argument("--models", help="Comma-separated model subset to attach")
    tl.add_argument(
        "--context-model",
        help="Model whose embedding gives ctx_<model>_k<k>_cosdist: cosine "
        "distance to the mean of the preceding k items in the run",
    )
    tl.add_argument("--context-k", type=int, default=5, help="Context width in items (default 5)")
    tl.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output table (.parquet preferred, .csv/.tsv); <stem>.meta.json alongside",
    )
    tl.set_defaults(func=_run_timelines)

    cp = sub.add_parser(
        "compose",
        help="Compose sparse runs onto a dense movie-style grid: events.tsv + "
        "item feature stores -> movie-schema tables a movie consumer loads "
        "with no special case (the dense half of contracts §4.3 item 5)",
    )
    cp.add_argument(
        "events",
        nargs="+",
        help="events.tsv files, one composed run each; BIDS names give "
        "entities, any other name composes under its stem",
    )
    cp.add_argument(
        "--stores",
        nargs="+",
        required=True,
        metavar="TABLE",
        help="`psytwill features` parquet tables holding the presented items; "
        "each model's temporal grain (untimed/gridded/chunk) picks its "
        "composition rule and target stream",
    )
    cp.add_argument(
        "--registry",
        help="stimulus_registry/ directory for events -> id resolution; "
        "optional when events carry stimulus_id or stim_file",
    )
    cp.add_argument("-o", "--output", required=True,
                    help="composed-run root; tables land in <output>/features/, "
                    "media (with --media) in <output>/movies/<slug>/")
    cp.add_argument("--window", type=float, default=0.5,
                    help="grid width in seconds (default 0.5, the movie grid)")
    cp.add_argument("--onset-column", default="onset",
                    help="events column presentations are timed by (default "
                    "onset; e.g. onset_actual where recorded)")
    cp.add_argument("--lead-out", type=float, default=0.0,
                    help="seconds the run continues past the last event "
                    "(default 0; task programs often hold the screen)")
    cp.add_argument("--models", help="comma-separated model subset to compose")
    cp.add_argument("--modality-map",
                    help="model=visual|audio|text overrides for stores whose "
                    "rows carry no usable modality, e.g. 'clip=visual'")
    cp.add_argument("--sparse", action="store_true",
                    help="emit rows only where a stimulus is on; default is "
                    "the full grid with explicit NaN rows in empty bins")
    cp.add_argument("--min-coverage", type=float, default=0.5,
                    help="fraction of a bin a presentation must cover for its "
                    "item to land there (default 0.5; a shorter presentation "
                    "keeps its best-covered bin)")
    cp.add_argument("--comparability", metavar="TSV",
                    help="per-model render-falsifier verdicts (columns model, "
                    "comparable[, note]; labels item, item+offset, window, "
                    "none) written into each sidecar's models.<m>.comparable; "
                    "an optional stream column labels a model per stream, "
                    "an empty comparable cell is an explicit null")
    cp.add_argument("--family", metavar="NAME",
                    help="write movies_<NAME>_{frames,audio_frames,"
                    "transcript_chunks} instead of the battery stems, so a "
                    "projection family (e.g. psytwill_space) composes beside "
                    "the battery tables in the same run root")
    cp.add_argument("--dry-run", action="store_true",
                    help="report the plan (runs, streams, models, row "
                    "estimates) without reading values or writing")
    cp.add_argument("--force", action="store_true",
                    help="recompose even when outputs match this input "
                    "signature")
    cp.add_argument("--json", action="store_true",
                    help="print the machine-readable summary instead of prose")
    cp.add_argument("--media", action="store_true",
                    help="also render viewer media per run (frames/, "
                    "audio.m4a, transcript CSVs); needs Pillow, soundfile, "
                    "ffmpeg")
    cp.add_argument("--media-map",
                    help="TSV mapping stimulus_id[, voice] -> media file "
                    "(paths relative to the TSV)")
    cp.add_argument("--stimuli-root",
                    help="root the registry's media columns resolve under "
                    "(<root>/<set>/<file>)")
    cp.add_argument("--screen", default="2048x1280",
                    help="display geometry in pixels WxH (default 2048x1280)")
    cp.add_argument("--image-frac", type=float, default=0.6,
                    help="image height as a fraction of screen height "
                    "(default 0.6)")
    cp.add_argument("--bg-gray", type=int, default=191,
                    help="background gray 0-255 (default 191, PsychoPy "
                    "[.5,.5,.5])")
    cp.add_argument("--media-scale", type=float, default=0.5,
                    help="render scale vs the true screen (default 0.5)")
    cp.set_defaults(func=_run_compose)

    vz = sub.add_parser(
        "viz",
        help="Static feature viewers: self-contained HTML over the features "
        "tables plus the stimuli themselves (works over file://)",
    )
    vzsub = vz.add_subparsers(dest="viz_verb", required=True)
    vm = vzsub.add_parser(
        "movies",
        help="timeline viewer for the movies set: every modality's scalar "
        "features on one time axis, with frames, audio and transcript",
    )
    vm.add_argument(
        "--features-dir",
        required=True,
        help="directory holding the `psytwill features` movie tables "
        "(movies_*_features.parquet)",
    )
    vm.add_argument(
        "--films-dir",
        required=True,
        help="stimuli_features movies/ directory (per-film folders with "
        "frames/, audio, transcript CSVs)",
    )
    vm.add_argument(
        "--registry",
        help="stimuli/stimulus_registry/ directory or movies.tsv, for film "
        "titles and durations",
    )
    vm.add_argument("--slugs", help="comma-separated film subset (default all)")
    vm.add_argument(
        "--projections",
        help="per-modality embedding model for the 2D MDS trajectories, "
        "e.g. 'visual=clip,audio=clap,text=fasttext' (the default); "
        "'none' disables them",
    )
    vm.add_argument(
        "--no-families",
        action="store_true",
        help="skip the projection-family tables (movies_<family>_<stream>, "
        "e.g. psytwill_space), whose models otherwise each get a trajectory",
    )
    vm.add_argument(
        "-o",
        "--output",
        help="bundle directory (default <films-dir>/viz/timeline/)",
    )
    vm.set_defaults(func=_run_viz_movies)

    sp = sub.add_parser(
        "space",
        help="Private-block fits, relations and releases (psytwill-space): fit | project | check | relate | release",
    )
    spsub = sp.add_subparsers(dest="space_verb", required=True)

    def _space_common(q: argparse.ArgumentParser, features_required: bool = True) -> None:
        q.add_argument("--features", nargs="+", required=features_required, help="`psytwill features` table(s)")
        q.add_argument("--key", default="stimulus_id", help="row grain, comma-separated (default stimulus_id)")
        q.add_argument("--window", type=float, help="bin `time` at this width (needs time in --key)")
        q.add_argument("--exclude-ids", help="file of stimulus_ids to drop (one per line)")
        q.add_argument("--exclude-rows",
                       help="CSV or parquet with one column per --key column; matching rows are dropped "
                            "(numeric key values compare as numbers, so 3 matches 3.0). With --window, "
                            "give the binned key")
        q.add_argument("--groups-from-label", action="store_true",
                       help="grouped folds / block nulls keyed on the first key column (clip id)")

    def _space_criterion(q: argparse.ArgumentParser) -> None:
        q.add_argument("--r2-min", type=float, default=0.5)
        q.add_argument("--alpha", type=float, default=0.01)
        q.add_argument("--k-nn", type=int, default=20)
        q.add_argument("--n-perm", type=int, default=250)
        q.add_argument("--eval-n", type=int, default=5000, help="rows per fold for the kNN overlap (0 = all); with --block-size, drawn as whole contiguous blocks")
        q.add_argument("--block-size", type=int, help="block-permutation width for temporal grids")
        q.add_argument("--seed", type=int, default=0)

    f = spsub.add_parser("fit", help="fit one private block and freeze it")
    _space_common(f, features_required=False)
    f.add_argument("--block", help="V | A | L (battery modality) or a custom name with --members (required "
                                   "unless --like)")
    f.add_argument("--members", help="comma-separated member spaces (default: the block's battery members present)")
    f.add_argument("--k-schedule", help="comma-separated k candidates (default 8,16,...,256 below the PR bound)")
    f.add_argument("--n-splits", type=int, default=5)
    f.add_argument("--targets",
                   help="comma-separated OUTSIDE targets: spaces scored by the block like members at "
                        "every k it walks, but never part of it and never deciding k (needs --members "
                        "or --like); the admission screen's question, `per_target` in the manifest")
    f.add_argument("--target-features", nargs="+", metavar="TABLE",
                   help="`psytwill features` table(s) read for the --targets only (never for members), "
                        "e.g. a candidate extracted outside the battery; allowed with --like. "
                        "Recorded as `target_inputs`")
    f.add_argument("--target-rows", choices=("narrow", "block"), default="narrow",
                   help="narrow (default): the fit's rows are those every member AND target hold, so "
                        "a target can narrow them. block: the members' rows alone; a target is absent "
                        "(masked, never filled) from a row its tables lack or on which it is "
                        "undefinable -- a candidate on part of a mix scored against the unchanged block")
    f.add_argument("--leave-out",
                   help="with --like: drop these members from the block and score each as an outside "
                        "target at the source's k (leave-one-member-out coverage)")
    f.add_argument("--defer-members",
                   help="comma-separated members that are scored but do not decide k; the manifest "
                        "records them as not subsumed when they fail (a future version's work)")
    f.add_argument("--corpora-from-label", action="store_true",
                   help="read the corpus from each row's ext-<corpus>-* stimulus_id "
                        "(non-external ids group as 'internal') for the gated-`missing` "
                        "WARNING only: nulls are handled as the producers declare them "
                        "(Contract B 1.1 `nulls`), never inferred from corpus contrast")
    f.add_argument("--per-corpus", action="store_true",
                   help="score the criterion within each corpus (read as --corpora-from-label "
                        "does), once on all its rows placed out of fold, and choose k only when "
                        "every corpus passes; the pooled folds are still reported. For a "
                        "multi-register mix, where pooled R^2 counts between-corpus differences "
                        "as explained")
    f.add_argument("--corpus-weights", choices=["equal"],
                   help="weight rows so every corpus (read as --corpora-from-label does) counts "
                        "equally in the member whiteners and the block covariance; the criterion "
                        "is not weighted. For a mix whose corpora differ in size")
    _space_criterion(f)
    f.add_argument("--member-scaling", choices=["pr", "pcs", "zscore"], default="pr",
                   help="pr: whiten each member to its participation ratio (psytwill-space); pcs: the "
                        "same ceil(PR) directions kept at their own variance (variance-weighted block); "
                        "zscore: z-score only, so members weigh by width (the structure-free baseline)")
    f.add_argument("--fixed-k", type=int,
                   help="freeze at this k instead of walking the schedule; the criterion, when scored, "
                        "is a diagnostic at k")
    f.add_argument("--no-criterion", action="store_true",
                   help="skip the criterion (needs --fixed-k or --like): fit the map on all rows only")
    f.add_argument("--like", metavar="MANIFEST",
                   help="copy inputs, rows, members, folds, corpus handling, criterion and k from this "
                        "fit's manifest; only --member-scaling / --fixed-k / --k-schedule / --no-criterion "
                        "may differ (--k-schedule walks the variant to its own k); --leave-out and "
                        "--targets add outside targets, and --per-corpus adds per-corpus scoring to a "
                        "pooled source at its fixed k")
    f.add_argument("--select", action="append", metavar="EXPR",
                   help="keep only member columns whose Contract B 1.2 labels match, e.g. "
                        "'interpretability=nameable;derivation!=probe,zero-shot'. Clauses (';'-separated, "
                        "or repeat --select) are ANDed; KEY=a,b keeps a value in the set, KEY!=a,b drops "
                        "it; keys interpretability | derivation. Column-level: each member is whitened "
                        "on its selected columns alone, and only their `nulls` act. A member with no "
                        "matching column is dropped; an unlabelled member is refused. Recorded as "
                        "`selection` in the manifest (copied by --like unless given). Bookkeeping "
                        "columns are always left out, with or without --select")
    f.add_argument("-o", "--output", required=True, help="output directory")
    f.add_argument("--stem", help="file stem (default <block>_v1)")
    f.set_defaults(func=_run_space_fit)

    sb = spsub.add_parser("subspaces", help="shared and unique subspaces of a frozen block, gated by split-half "
                                            "reproducibility (psytwill.shared_unique)")
    sb.add_argument("--like", required=True, metavar="MANIFEST",
                    help="the frozen block fit (.json); its inputs and rows are re-read as `space fit --like` does, "
                         "and the refit's rows must be the frozen fit's")
    sb.add_argument("--shared-c", type=float, default=0.5,
                    help="S = the leading block directions with eigenvalue >= 1 + c (default 0.5)")
    sb.add_argument("--unique", type=float, default=0.5,
                    help="a direction of a member is unique when >= this share of its variance is residual "
                         "on every other member (default 0.5)")
    sb.add_argument("--max-absent", type=float, default=0.5,
                    help="a member absent from >= this share of the rows gets no unique subspace (default 0.5)")
    sb.add_argument("--half-seed", type=int, default=0,
                    help="halves = sha256 parity of '<seed>:<stimulus id>' (default 0)")
    sb.add_argument("--n-random", type=int, default=20,
                    help="random-span pairs per object for the gate's reference (default 20)")
    sb.add_argument("--gate-msc", type=float, default=0.5, help="gate: halves' msc >= this (default 0.5)")
    sb.add_argument("--gate-margin", type=float, default=0.25,
                    help="gate: and >= the random pairs' mean msc + this (default 0.25)")
    sb.add_argument("--no-gate", action="store_true", help="the frozen fit's objects only, no half fits")
    sb.add_argument("--draws", type=int, default=20,
                    help="random spans per kind for the brain read, saved with the objects: S@dNNN / U@dNNN at "
                         "d = min(dim S, dim U), ref@dNNN, and S@<member>.dNNN at each unique subspace's dim "
                         "(default 20; 0 = none)")
    sb.add_argument("--draw-seed", type=int, default=0, help="seed of the draws (default 0)")
    sb.add_argument("-o", "--output", required=True, help="output directory")
    sb.add_argument("--stem", help="file stem (default <block>_subspaces)")
    sb.set_defaults(func=_run_space_subspaces)

    pr = spsub.add_parser("project", help="apply a frozen block to a features table")
    _space_common(pr)
    pr.add_argument("--space", required=True, help="the fit's .json manifest")
    pr.add_argument("--view", choices=["covariance", "whitened"], default="covariance",
                    help="covariance: scores as stored, each component at its own variance (default); "
                         "whitened: every component at unit variance on the fit rows")
    pr.add_argument("--chunk-size", type=int,
                    help="project this many stimulus_ids at a time and concatenate (same rows and "
                         "scores, bounded memory; a whole-table load of ~500 M long-form rows "
                         "passes 96 G)")
    pr.add_argument("--k", type=int,
                    help="project through the first K block components (1..the fit's k; default the fit's k). "
                         "A row with a masked member is placed when its present members span K directions, "
                         "so a smaller K places rows the fit's k cannot")
    pr.add_argument("--causal", action="store_true",
                    help="read every centred windowed column as trailing: shift it later by half its window "
                         "(Contract B 1.4 labels in the fit manifest), so a row's value reads only audio up to "
                         "the row's end. The first half-window of each clip has no value: the member is absent "
                         "there, never filled")
    pr.add_argument("--subspaces", metavar="JSON",
                    help="write these subspace objects' coordinates instead of block scores (`space subspaces` "
                         "output for this --space): <object>__<jjj> columns, NaN where an object cannot place a "
                         "row; rows are aligned and masked as for block scores")
    pr.add_argument("--objects", help="with --subspaces: only these objects (comma-separated; default all)")
    pr.add_argument("-o", "--output", required=True, help="scores table (.parquet or .csv)")
    pr.set_defaults(func=_run_space_project)

    lb = spsub.add_parser("labels", help="per-column Contract B 1.2 labels of fitted blocks or a release: "
                                         "a stimulus-free CSV plus a BIDS-style column dictionary")
    lbsrc = lb.add_mutually_exclusive_group(required=True)
    lbsrc.add_argument("--space", nargs="+", help="block .json manifest(s)")
    lbsrc.add_argument("--release", help="release .json (every block it pins; loaded tamper-evident)")
    lb.add_argument("-o", "--output", required=True,
                    help="labels table (.csv); the column dictionary is written beside it as <stem>.columns.json")
    lb.set_defaults(func=_run_space_labels)

    ck = spsub.add_parser("check", help="the subsumption criterion on any table, no refit")
    _space_common(ck)
    ck.add_argument("--space", required=True, help="the fit's .json manifest")
    _space_criterion(ck)
    ck.add_argument("-o", "--output", help="per-member CSV")
    ck.set_defaults(func=_run_space_check)

    rel = spsub.add_parser("relate", help="relations between fitted blocks: fit | check | project")
    relsub = rel.add_subparsers(dest="relate_verb", required=True)

    def _relate_pair_args(q: argparse.ArgumentParser) -> None:
        q.add_argument("--a", required=True, help="side-a block score table (`space project` output)")
        q.add_argument("--b", required=True, help="side-b block score table")
        q.add_argument("--exclude-ids", help="file of stimulus_ids to drop (one per line), matched on the first join column")

    rf = relsub.add_parser("fit", help="count the shared prefix by CV-CCA and freeze the top-k map")
    _relate_pair_args(rf)
    rf.add_argument("--name", required=True, help="relation name, e.g. VL")
    rf.add_argument("--fixed-k", type=int,
                    help="freeze at this k (a baseline matched to another relation's k); the prefix count "
                         "is still measured and recorded")
    rf.add_argument("--join", default="stimulus_id", help="comma-separated key columns the two tables pair on")
    rf.add_argument("--pool-a", choices=["none", "mean"], default="none",
                    help="how side a's rows pool when several share a join key (refused if needed and unset)")
    rf.add_argument("--pool-b", choices=["none", "mean"], default="none", help="as --pool-a, for side b")
    rf.add_argument("--groups-from-label", action="store_true",
                    help="grouped folds keyed on the first join column (clip id)")
    rf.add_argument("--registers-from-label", action="store_true",
                    help="keep each register's coupling (joint moments of the top-k variates), the register "
                         "read as the corpus of the first join column's ext-<corpus>-* id; k and the map are "
                         "still fitted on all rows")
    rf.add_argument("--n-splits", type=int, default=5)
    rf.add_argument("--n-perm", type=int, default=250)
    rf.add_argument("--block-size", type=int, help="block-permutation width for temporal grids")
    rf.add_argument("--r-min", type=float, default=0.5 ** 0.5, help="held-out r a component must reach (default sqrt(.5))")
    rf.add_argument("--rank-cap", type=int, help="PCA rank per side before CCA (default: the wider side + 16)")
    rf.add_argument("--scope", help="free-text caveat recorded in the manifest (what the pairing does and does not cover)")
    rf.add_argument("--seed", type=int, default=0)
    rf.add_argument("-o", "--output", required=True, help="output directory")
    rf.add_argument("--stem", help="file stem (default <name>_v1)")
    rf.set_defaults(func=_run_relate_fit)

    rc = relsub.add_parser("check", help="per-component r of a frozen relation on any paired tables")
    _relate_pair_args(rc)
    rc.add_argument("--relation", required=True, help="the relation's .json manifest")
    rc.add_argument("--n-perm", type=int, default=200)
    rc.add_argument("--alpha", type=float, default=0.01,
                    help="null quantile is 1 - alpha per component (default .01, i.e. q99)")
    rc.add_argument("--block-size", type=int)
    rc.add_argument("--seed", type=int, default=0)
    rc.add_argument("-o", "--output", help="per-component CSV")
    rc.set_defaults(func=_run_relate_check)

    rp = relsub.add_parser("project", help="place one side's rows in the relation's shared coordinates")
    rp.add_argument("--relation", required=True, help="the relation's .json manifest")
    rp.add_argument("--side", required=True, choices=["a", "b"], help="which side the score table belongs to")
    rp.add_argument("--scores", required=True, help="block score table (`space project` output)")
    rp.add_argument("-o", "--output", required=True, help="variates table (.parquet or .csv)")
    rp.set_defaults(func=_run_relate_project)

    rls = spsub.add_parser("release", help="name a set of blocks + relations as one pinned version: write | verify")
    rlsub = rls.add_subparsers(dest="release_verb", required=True)
    rw = rlsub.add_parser("write", help="write an immutable release manifest")
    rw.add_argument("--name", required=True, help="e.g. psytwill-space")
    rw.add_argument("--version", required=True, help="release version, e.g. 0.1.0")
    rw.add_argument("--blocks", nargs="+", required=True, help="block .json manifests")
    rw.add_argument("--relations", nargs="+", help="relation .json manifests (fitted on blocks given here)")
    rw.add_argument("--absent", nargs="+", metavar="'X,Y: evidence'",
                    help="pairs measured to share nothing (k = 0), each with where the measurement is recorded")
    rw.add_argument("--note", help="free-text note recorded in the release")
    rw.add_argument("--model-prefix", default="pspace",
                    help="store model names a projection writes, <prefix>_<block> (default pspace); a "
                         "baseline release takes its own, e.g. b0")
    rw.add_argument("--data-root", help="the fit-corpora directory: every exclusion file a pinned fit names must "
                                        "lie under it (required when any does); each is pinned by sha256")
    rw.add_argument("--attached", metavar="SPEC.json",
                    help="attached members (opt-in producer features in no block): a JSON file "
                         "{\"root\": <stimulus feature store>, \"attached\": [{model, basis: bar|decided, gap, "
                         "evidence, columns?, name?, package?, root?, note?}, ...]}; every sidecar of each model "
                         "under the root, and its tables, is pinned by sha256")
    rw.add_argument("-o", "--output", required=True, help="output directory (file: <name>_<version>.json)")
    rw.set_defaults(func=_run_release_write)
    rpj = rlsub.add_parser("project", help="place a stimulus set in a release, written as a Contract B family")
    rpj.add_argument("--release", required=True, help="release .json (loaded tamper-evident)")
    rpj.add_argument("--set", required=True, help="stimulus set name, e.g. shared1000 (recorded in the sidecar)")
    rpj.add_argument("--registry", required=True, help="the set's registry (.tsv/.csv with stimulus_id); the family must cover it")
    rpj.add_argument("--registry-join", help="registry column external source ids map onto (e.g. nsdId)")
    rpj.add_argument("--id-pattern", help="regex with one capture group read from each source id and compared to "
                                          "--registry-join (e.g. 'ext-nsd-(\\d+)'; numbers compare as numbers)")
    rpj.add_argument("--grain", nargs="+", action="append", required=True, metavar="ARG",
                     help="TABLE BLOCK KEY FEATURES...: one table of the family (TABLE '-' = <stem>.csv, else a "
                          "§4.1 suffix such as chunks), the release block placing it, its comma-separated key, "
                          "and the `psytwill features` table(s) to read. Repeat per table. Every relation side "
                          "on BLOCK is written into the same table")
    rpj.add_argument("--key-map", action="append", metavar="TABLE=FILE",
                     help="for a table whose sub-stimulus key the source numbers differently: a table of "
                          "src_<key> and <key> columns, one to one")
    rpj.add_argument("--skip-member", action="append", metavar="TABLE=MEMBER[,MEMBER]",
                     help="a block member this set does not carry at all: masked in every row (as an "
                          "`undefined` member is), never filled, and listed in the sidecar")
    rpj.add_argument("--partial-coverage", action="store_true",
                     help="allow registry stimuli with no row in the input (a film without dialogue has no "
                          "transcript); they are listed in the sidecar instead of refused")
    rpj.add_argument("-o", "--output", required=True, help="family stem, e.g. <store>/shared1000/psytwill_space")
    rpj.set_defaults(func=_run_release_project)
    rv = rlsub.add_parser("verify", help="check every file a release pins is unchanged")
    rv.add_argument("release", help="release .json")
    rv.set_defaults(func=_run_release_verify)

    from psytwill.bench.cli import register as _register_bench

    _register_bench(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        args.func(args)
    except PsytwillError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
