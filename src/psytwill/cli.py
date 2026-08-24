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

    # How the spaces in N feature tables relate to each other
    psytwill compare image.parquet:image caption.parquet:cap -o geometry/
"""

import argparse
import sys

from psytwill import __version__
from psytwill.compare import DEFAULT_K
from psytwill.exceptions import InputError, PsytwillError


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


def _parse_table_arg(arg: str) -> tuple[str, str | None]:
    """``path.parquet:prefix`` -> (path, prefix). Prefix is optional."""
    path, sep, prefix = arg.rpartition(":")
    if not sep or len(prefix) > 40 or "/" in prefix or "." in prefix:
        return arg, None
    return path, prefix


def _run_compare(args: argparse.Namespace) -> None:
    from psytwill.geometry import compare_spaces, write_geometry
    from psytwill.store import LoadReport, align_spaces, dedupe_spaces, load_spaces

    key = tuple(args.key.split(","))
    models = args.models.split(",") if args.models else None
    measures = args.measures.split(",") if args.measures else None
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

    total_seen = [0]

    def progress(done: int, total: int, label: str) -> None:
        if done == 1 or done == total or done % max(1, total // 20) == 0:
            print(f"  [{done:>6d}/{total}] {label}", flush=True)
        total_seen[0] = total

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
        extra={
            "stride": args.stride,
            "load_report": {
                "deduped": report.deduped,
                "skipped_string": report.skipped_string,
                "skipped_empty": report.skipped_empty,
                "pooled": report.pooled,
                "dropped_provenance": report.dropped_provenance,
            }
        },
    )
    print(f"wrote {len(result.pairs)} pair rows, {len(result.manifest)} spaces")
    for kind in ("pairs", "manifest", "meta_path"):
        print(f"  {paths[kind]}")


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

    c = sub.add_parser(
        "compare",
        help="How the spaces in N feature tables relate (Contract B geometry)",
    )
    c.add_argument(
        "inputs",
        nargs="+",
        help="Long-form feature tables, optionally 'path.parquet:prefix'",
    )
    c.add_argument("-o", "--output", required=True, help="Output directory")
    c.add_argument("--name", default="space_geometry", help="Output file stem")
    c.add_argument(
        "--key",
        default="stimulus_id",
        help="Row grain, comma-separated (e.g. 'stimulus_id,time')",
    )
    c.add_argument("--models", help="Comma-separated model subset")
    c.add_argument("--measures", help="Comma-separated measure subset")
    c.add_argument(
        "--pool",
        choices=("mean", "none"),
        default="mean",
        help="Pool replicate rows sharing a key, or refuse them",
    )
    c.add_argument(
        "--window",
        type=float,
        help="Bin 'time' into windows of this many seconds before pooling; "
        "required on a temporal grid (it is the timescale axis, and it "
        "reconciles bin-start vs bin-center time stamps across groups)",
    )
    c.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth aligned row within each clip (label prefix "
        "before --group-sep) before comparing — for running the n^2 "
        "measures at a grain whose full n cannot afford them",
    )
    c.add_argument("--k", type=int, default=DEFAULT_K, help="Neighbours for overlap")
    c.add_argument("--n-splits", type=int, default=5, help="Ridge CV folds")
    c.add_argument(
        "--permutations",
        type=int,
        default=1000,
        help="Neighbour-overlap null draws (0 skips the null)",
    )
    c.add_argument(
        "--block-size",
        type=int,
        help="Permute contiguous blocks of this size; required on a temporal grid",
    )
    c.add_argument(
        "--group-by",
        action="store_true",
        help="Group ridge folds by the label prefix before --group-sep "
        "(one fold per clip on a movie grid)",
    )
    c.add_argument("--group-sep", default="|", help="Separator for --group-by")
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=_run_compare)

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
