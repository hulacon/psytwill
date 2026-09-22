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
    from psytwill.features import build_features

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
    return members


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

    from psytwill.store import LoadReport, SpaceMatrix, load_spaces, model_inventory

    key = tuple(args.key.split(","))
    rep = LoadReport()
    where: dict[str, list[str]] = {}
    for path in args.features:
        for m in model_inventory(path)["model"].dropna().unique():
            where.setdefault(str(m), []).append(str(path))
    if members is None:
        members = _space_members(args, list(where))
    missing = [m for m in members if m not in where]
    if missing:
        raise SpaceError(f"member(s) {missing} not in the given tables; available {sorted(where)}")
    ids: set[str] = set()
    if args.exclude_ids:
        ids = {line.strip() for line in Path(args.exclude_ids).read_text().splitlines() if line.strip()}
    spaces: dict = {}
    for m in members:
        parts: list[SpaceMatrix] = []
        for path in where[m]:
            got = load_spaces(path, key=key, models=[m], window=args.window, report=rep)
            if m not in got:
                raise SpaceError(f"model {m!r} in {path} loaded as none of {sorted(got)} "
                                 "(string-valued or empty?)")
            parts.append(got[m])
        sm = parts[0]
        if len(parts) > 1:
            feats = parts[0].features
            for part, path in zip(parts[1:], where[m][1:]):
                if part.features != feats:
                    raise SpaceError(
                        f"model {m!r} has {len(part.features)} feature columns in {path} but "
                        f"{len(feats)} in {where[m][0]}; a member must carry the same columns "
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
        spaces[m] = sm
        src = f" from {len(parts)} tables" if len(parts) > 1 else ""
        print(f"  loaded {m}: {sm.n} rows x {sm.dim} features{src}", flush=True)
    return spaces, members, len(ids), rep


def _run_space_fit(args: argparse.Namespace) -> None:
    from psytwill.space import DEFAULT_K_SCHEDULE, fit_block, save_fit

    spaces, members, n_excl, rep = _space_load(args)
    groups = None
    corpora = None
    schedule = tuple(int(k) for k in args.k_schedule.split(",")) if args.k_schedule else DEFAULT_K_SCHEDULE
    if args.groups_from_label or args.corpora_from_label:
        from psytwill.store import align_spaces

        _, labels = align_spaces({m: spaces[m] for m in members})
        if args.groups_from_label:
            groups = [lab.split("|")[0] for lab in labels]
        if args.corpora_from_label:
            from psytwill.fitcorpus import is_external, parse_ext_id

            ids = [lab.split("|")[0] for lab in labels]
            corpora = [parse_ext_id(i)[0] if is_external(i) else "internal" for i in ids]
            print(f"  structural rule on: {len(set(corpora))} corpora "
                  f"({', '.join(sorted(set(corpora)))})")

    def progress(i, n, what):
        if i == 1 or i % 25 == 0 or i == n:
            print(f"  [{i}/{n}] {what}", flush=True)

    fit = fit_block(spaces, members, block=args.block, k_schedule=schedule, n_splits=args.n_splits,
                    groups=groups, corpora=corpora, r2_min=args.r2_min, alpha=args.alpha, k_nn=args.k_nn,
                    n_perm=args.n_perm, eval_n=args.eval_n or None, block_size=args.block_size,
                    random_state=args.seed, progress=progress)
    if corpora is not None:
        for m in members:
            sc = fit.manifest["per_member"][m]["structural_columns"]
            if sc:
                print(f"  {m}: {len(sc)} structural column(s) filled: {', '.join(sc)}")
    fit.manifest["inputs"] = [str(p) for p in args.features]
    fit.manifest["key"] = args.key
    fit.manifest["window"] = args.window
    fit.manifest["n_excluded_ids"] = n_excl
    fit.manifest["exclude_ids_file"] = args.exclude_ids
    npz, manifest, curve = save_fit(fit, args.output, stem=args.stem)
    verdict = "SUBSUMES all members" if fit.manifest["subsumes_all_members"] else "does NOT subsume every member"
    print(f"psytwill space fit [{args.block}] -> {manifest}")
    print(f"  k={fit.k} ({verdict}); PR bound {fit.manifest['pr_sum_bound']:.1f}; "
          f"n={fit.manifest['n_rows']} rows, {len(members)} members, excluded {n_excl} ids")
    for m in members:
        pm = fit.manifest["per_member"][m]
        print(f"  {m:<18} PR {pm['participation_ratio']:6.1f} rank {pm['whitened_rank']:>4}  "
              f"R2 {min(pm['r2_per_fold']) if pm['r2_per_fold'] else float('nan'):.3f}  "
              f"overlap {min(pm['overlap_per_fold']) if pm['overlap_per_fold'] else float('nan'):.3f}  "
              f"{'pass' if pm['passed_all_folds'] else 'FAIL'}")
    print(f"  weights {npz}\n  curve {curve}")


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
    n_pass = sum(r["passed"] for r in rows)
    print(f"psytwill space check [{fit.block}, k={fit.k}] on {rows[0]['n_rows']} rows: {n_pass}/{len(rows)} members pass")
    for r in rows:
        print(f"  {r['member']:<18} R2 {r['r2']:.3f}  overlap {r['overlap']:.3f} (null {r['null_mean']:.3f}, p={r['overlap_p']:.3f})  "
              f"{'pass' if r['passed'] else 'FAIL'}")
    if args.output:
        import pandas as pd

        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"  {args.output}")
    if n_pass < len(rows):
        raise SpaceError(f"{len(rows) - n_pass} member(s) not subsumed")


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
    f.add_argument("inputs", nargs="+", help="Extractor scores CSV/TSV files")
    f.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output table (.parquet preferred, or .csv); "
        "<stem>.meta.json is written alongside",
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
        help="Private-block fits (psytwill-space v0.1): fit | project | check",
    )
    spsub = sp.add_subparsers(dest="space_verb", required=True)

    def _space_common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--features", nargs="+", required=True, help="`psytwill features` table(s)")
        q.add_argument("--key", default="stimulus_id", help="row grain, comma-separated (default stimulus_id)")
        q.add_argument("--window", type=float, help="bin `time` at this width (needs time in --key)")
        q.add_argument("--exclude-ids", help="file of stimulus_ids to drop (one per line)")
        q.add_argument("--groups-from-label", action="store_true",
                       help="grouped folds / block nulls keyed on the first key column (clip id)")

    def _space_criterion(q: argparse.ArgumentParser) -> None:
        q.add_argument("--r2-min", type=float, default=0.5)
        q.add_argument("--alpha", type=float, default=0.01)
        q.add_argument("--k-nn", type=int, default=20)
        q.add_argument("--n-perm", type=int, default=250)
        q.add_argument("--eval-n", type=int, default=5000, help="rows per fold for the kNN overlap (0 = all)")
        q.add_argument("--block-size", type=int, help="block-permutation width for temporal grids")
        q.add_argument("--seed", type=int, default=0)

    f = spsub.add_parser("fit", help="fit one private block and freeze it")
    _space_common(f)
    f.add_argument("--block", required=True, help="V | A | L (battery modality) or a custom name with --members")
    f.add_argument("--members", help="comma-separated member spaces (default: the block's battery members present)")
    f.add_argument("--k-schedule", help="comma-separated k candidates (default 8,16,...,256 below the PR bound)")
    f.add_argument("--n-splits", type=int, default=5)
    f.add_argument("--corpora-from-label", action="store_true",
                   help="structural-missingness rule: read the corpus from each row's "
                        "ext-<corpus>-* stimulus_id (non-external ids group as 'internal') "
                        "and fill columns null under a per-corpus gate with a frozen "
                        "sentinel instead of mean-imputing them")
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
