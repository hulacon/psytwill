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
"""

import argparse
import sys

from psytwill import __version__
from psytwill.exceptions import PsytwillError


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
