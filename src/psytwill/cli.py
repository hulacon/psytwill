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
    from psytwill.features import build_features, refresh_group_nulls

    if args.refresh_nulls:
        for table in args.inputs:
            got = refresh_group_nulls(table)
            n_decl = sum(v is not None for v in got.values())
            print(f"psytwill features --refresh-nulls {table}: {n_decl}/{len(got)} models declare nulls")
        return
    if not args.output:
        raise SystemExit("psytwill features: -o/--output is required (unless --refresh-nulls)")
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

    modality_map = None
    if args.modality_map:
        modality_map = dict(
            item.split("=", 1) for item in args.modality_map.split(",") if item
        )
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
    if args.members:
        return [m.strip() for m in args.members.split(",") if m.strip()]
    from psytwill.battery import BATTERY

    modality = {"V": "visual", "A": "audio", "L": "text"}.get(args.block, args.block)
    want = {name for name, m in BATTERY.items()
            if m.modality == modality and m.kind in ("embedding", "profile")}
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


def _space_load(args: argparse.Namespace, members: list[str] | None = None):
    """Load member spaces one model at a time (a 300 M-row group table does
    not fit in pandas whole), dropping --exclude-ids rows. Returns
    (spaces, members, n_excluded, report).

    A member is read from EVERY table that carries it and the rows are
    stacked, so a fit over several corpora (one group table per corpus) sees
    all of them. Before 0.18.1 the first table holding a model won and the
    rest were silently ignored, which would have fit the A block on one
    corpus while reporting seven. Two tables contributing the same row label
    is refused rather than pooled: on a multi-corpus fit a duplicate key
    means the same stimulus was extracted twice, not a replicate.
    """
    import numpy as np

    from psytwill.features import group_nulls
    from psytwill.store import LoadReport, SpaceMatrix, distinct_values, load_spaces

    key = tuple(args.key.split(","))
    rep = LoadReport()
    where: dict[str, list[str]] = {}
    table_nulls = {str(path): group_nulls(path) for path in args.features}
    for path in args.features:
        for m in sorted(distinct_values(path, "model")):
            where.setdefault(m, []).append(str(path))
    if members is None:
        members = _space_members(args, list(where))
    from psytwill.space import member_source, select_features

    source = {m: member_source(m) for m in members}
    missing = [m for m in members if source[m][0] not in where]
    if missing:
        raise SpaceError(f"member(s) {missing} not in the given tables; available {sorted(where)}")
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
                                  stimulus_ids=getattr(args, "include_ids", None))
                if model not in got:
                    raise SpaceError(f"model {model!r} in {path} loaded as none of {sorted(got)} "
                                     "(string-valued or empty?)")
                parts.append(got[model])
            loaded[model] = parts
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
        src = f" from {len(parts)} tables" if len(parts) > 1 else ""
        of = f" (split from {model})" if cols is not None else ""
        print(f"  loaded {m}: {sm.n} rows x {sm.dim} features{src}{of}", flush=True)
    return spaces, members, len(ids), rep


def _run_space_fit(args: argparse.Namespace) -> None:
    from psytwill.space import DEFAULT_K_SCHEDULE, SPACE_SCHEMA_VERSION, fit_block, save_fit

    spaces, members, n_excl, rep = _space_load(args)
    groups = None
    corpora = None
    schedule = tuple(int(k) for k in args.k_schedule.split(",")) if args.k_schedule else DEFAULT_K_SCHEDULE
    if args.groups_from_label or args.corpora_from_label or args.per_corpus or args.corpus_weights:
        from psytwill.store import align_spaces

        _, labels = align_spaces({m: spaces[m] for m in members})
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

    fit = fit_block(spaces, members, block=args.block, k_schedule=schedule, n_splits=args.n_splits,
                    groups=groups, corpora=corpora, nulls=rep.nulls, r2_min=args.r2_min, alpha=args.alpha, k_nn=args.k_nn,
                    n_perm=args.n_perm, eval_n=args.eval_n or None, block_size=args.block_size,
                    random_state=args.seed, progress=progress, per_corpus=args.per_corpus,
                    corpus_weights=args.corpus_weights,
                    on_row=on_row,
                    defer=[m.strip() for m in (args.defer_members or "").split(",") if m.strip()])
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
    fit.manifest["inputs"] = [str(p) for p in args.features]
    fit.manifest["key"] = args.key
    fit.manifest["window"] = args.window
    fit.manifest["n_excluded_ids"] = n_excl
    fit.manifest["exclude_ids_file"] = args.exclude_ids
    fit.manifest["exclude_rows_file"] = args.exclude_rows
    fit.manifest["n_excluded_rows"] = dict(rep.excluded_rows)
    npz, manifest, curve = save_fit(fit, args.output, stem=args.stem)
    partial.unlink(missing_ok=True)
    verdict = "SUBSUMES all members" if fit.manifest["subsumes_all_members"] else "does NOT subsume every member"
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
    print(f"  weights {npz}\n  curve {curve}")


def _verdict(passed: bool | None) -> str:
    return "UNSCOREABLE" if passed is None else "pass" if passed else "FAIL"


def _run_space_project(args: argparse.Namespace) -> None:
    import pandas as pd

    from psytwill.space import load_fit

    fit = load_fit(args.space)
    spaces, _, _, _ = _space_load(args, members=fit.members)
    S, labels = fit.project(spaces)
    df = pd.DataFrame(S, columns=[f"{fit.block}_{j:03d}" for j in range(S.shape[1])])
    keys = args.key.split(",")
    parts = [lab.split("|") for lab in labels]
    for j, kname in enumerate(keys):
        df.insert(j, kname, [p[j] if j < len(p) else None for p in parts])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    meta = {"space": str(args.space), "block": fit.block, "k": fit.k, "inputs": [str(p) for p in args.features],
            "key": args.key, "window": args.window, "rows": int(len(df))}
    (out.parent / (out.name.removesuffix(out.suffix) + ".meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"psytwill space project [{fit.block}, k={fit.k}] -> {out}  ({len(df)} rows)")


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
                        groups_from_label=args.groups_from_label, scope=args.scope, n_splits=args.n_splits,
                        n_perm=args.n_perm, block_size=args.block_size, r_min=args.r_min,
                        rank_cap=args.rank_cap, random_state=args.seed)
    npz, manifest = save_relation(fit, args.output, stem=args.stem)
    m = fit.manifest
    st = m["subspace_stability"]
    print(f"psytwill space relate fit [{m['name']}: {a.block} <-> {b.block}] on {m['n_rows']} paired rows: "
          f"k = {fit.k} (prefix at r >= {m['r_min']:.3f}; {m['count_r_ge_min']} components reach it)")
    print(f"  held-out r: first {m['r_cv'][0]:.3f}, k-th {m['r_cv'][fit.k - 1]:.3f}, next {m['r_cv'][fit.k]:.3f}"
          if fit.k < len(m['r_cv']) else f"  held-out r: first {m['r_cv'][0]:.3f}")
    print(f"  subspace stability (fold vs full, mean sq. canonical r): a {st['a']['mean']:.3f} (min {st['a']['min']:.3f}), "
          f"b {st['b']['mean']:.3f} (min {st['b']['min']:.3f})")
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
    res = check_relation(fit, X, Y, n_perm=args.n_perm, block_size=args.block_size, random_state=args.seed)
    rm = fit.manifest["r_min"]
    print(f"psytwill space relate check [{fit.name}, k={fit.k}] on {res.n} paired rows: "
          f"{res.count} of {fit.k} components at r >= {rm:.3f}, prefix {res.prefix} "
          f"(fit: k = {fit.k}); first r {res.r[0]:.3f}, k-th {res.r[-1]:.3f}, null q99 max {res.null_q.max():.3f}")
    if args.output:
        cv = fit.manifest["r_cv"][: fit.k]
        pd.DataFrame({"component": range(1, fit.k + 1), "r": res.r, "null_q": res.null_q,
                      "r_fit_cv": cv}).to_csv(args.output, index=False)
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
            "scores": str(t.path), "key": ",".join(t.key), "rows": int(len(df)), "n_dropped_nan": int((~ok).sum())}
    (out.parent / (out.name.removesuffix(out.suffix) + ".meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"psytwill space relate project [{fit.name} side {args.side} ({t.block}), k={fit.k}] -> {out}  ({len(df)} rows)")


def _run_release_write(args: argparse.Namespace) -> None:
    from psytwill.release import build_release, parse_absent, write_release

    rel = build_release(name=args.name, version=args.version, blocks=args.blocks, relations=args.relations or [],
                        absent=[parse_absent(a) for a in (args.absent or [])], note=args.note)
    path = write_release(rel, args.output)
    b = ", ".join(f"{n} k={e['k']}" for n, e in rel["blocks"].items())
    r = ", ".join(f"{n} ({e['a']}<->{e['b']}) k={e['k']}" for n, e in rel["relations"].items()) or "none"
    a = ", ".join("<->".join(x["pair"]) for x in rel["absent"]) or "none"
    print(f"psytwill space release {rel['name']} {rel['version']}: blocks {b}; relations {r}; absent {a}\n  {path}")


def _run_release_verify(args: argparse.Namespace) -> None:
    from psytwill.release import load_release

    rel = load_release(args.release)
    m = rel.meta
    print(f"psytwill space release {rel.label}: {len(m['blocks'])} blocks, {len(m['relations'])} relations, "
          f"{len(m['absent'])} absences; every pinned file matches")


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
                    "none) written into each sidecar's models.<m>.comparable")
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

    def _space_common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--features", nargs="+", required=True, help="`psytwill features` table(s)")
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
    _space_common(f)
    f.add_argument("--block", required=True, help="V | A | L (battery modality) or a custom name with --members")
    f.add_argument("--members", help="comma-separated member spaces (default: the block's battery members present)")
    f.add_argument("--k-schedule", help="comma-separated k candidates (default 8,16,...,256 below the PR bound)")
    f.add_argument("--n-splits", type=int, default=5)
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
    f.add_argument("-o", "--output", required=True, help="output directory")
    f.add_argument("--stem", help="file stem (default <block>_v1)")
    f.set_defaults(func=_run_space_fit)

    pr = spsub.add_parser("project", help="apply a frozen block to a features table")
    _space_common(pr)
    pr.add_argument("--space", required=True, help="the fit's .json manifest")
    pr.add_argument("-o", "--output", required=True, help="scores table (.parquet or .csv)")
    pr.set_defaults(func=_run_space_project)

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
    rf.add_argument("--join", default="stimulus_id", help="comma-separated key columns the two tables pair on")
    rf.add_argument("--pool-a", choices=["none", "mean"], default="none",
                    help="how side a's rows pool when several share a join key (refused if needed and unset)")
    rf.add_argument("--pool-b", choices=["none", "mean"], default="none", help="as --pool-a, for side b")
    rf.add_argument("--groups-from-label", action="store_true",
                    help="grouped folds keyed on the first join column (clip id)")
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
