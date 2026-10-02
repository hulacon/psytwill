"""``psytwill bench <task>``: argument wiring for the benchmark harness.

Kept out of ``psytwill/cli.py`` so the harness can grow without touching the
main parser; ``register`` is the only entry point it uses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from psytwill.exceptions import BenchError


def _common(q: argparse.ArgumentParser) -> None:
    q.add_argument("-o", "--output", required=True, help="output directory")
    q.add_argument("--tag", required=True, help="run name in the output filenames, e.g. the model")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--n-boot", type=int, default=2000)
    q.add_argument("--fit-ids", nargs="+",
                   help="id file(s) (one per line) a fit has seen; any benchmark item found in one stops the run")
    q.add_argument("--drop-ids", help="id file of benchmark items to leave out before scoring")


def _ids(path) -> set[str]:
    return {ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()} if path else set()


def _keep(sm, drop: set[str]):
    from psytwill.store import SpaceMatrix

    if not drop:
        return sm
    k = [i for i, lab in enumerate(sm.labels) if lab.split("|")[0] not in drop]
    return SpaceMatrix(name=sm.name, labels=[sm.labels[i] for i in k], X=sm.X[k], features=sm.features,
                       modality=sm.modality, extractor=sm.extractor, n_replicates=sm.n_replicates)


def _load(features, model, key, window=None, drop=frozenset()):
    from .core import drop_nan_rows, load_model

    sm = load_model(features, model, key=tuple(key.split(",")), window=window)
    sm, n_nan = drop_nan_rows(_keep(sm, set(drop)))
    print(f"  loaded {model}: {sm.n} rows x {sm.dim}" + (f" ({n_nan} NaN rows dropped)" if n_nan else ""),
          flush=True)
    return sm, n_nan


def _read_table(path) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd

    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, sep="\t" if p.suffix in (".tsv", ".tab", ".txt") else ",")


def _report(side: Path, summary: dict, keys) -> None:
    print(f"  {side}")
    for k in keys:
        v = summary.get(k)
        if isinstance(v, dict) and "mean" in v:
            print(f"  {k}: {v['mean']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}] (n={v['n']}, units={v['n_units']})")


def _run_retrieval(args) -> None:
    from .core import refuse_overlap, write_run
    from .retrieval import retrieval

    drop = _ids(args.drop_ids)
    q, qn = _load(args.query_features, args.query_model, args.query_key, drop=drop)
    t, tn = _load(args.target_features, args.target_model, args.target_key, drop=drop)
    guard = refuse_overlap({lab.split("|")[0] for lab in q.labels}, args.fit_ids, what="retrieval")
    items, summary = retrieval(q, t, mode=args.mode, metric=args.metric, n_splits=args.n_splits, seed=args.seed,
                               n_boot=args.n_boot)
    summary["leak_guard"] = guard
    params = {k: v for k, v in vars(args).items() if k != "func"} | {"nan_rows_dropped": {"query": qn, "target": tn}}
    side = write_run(args.output, "retrieval", args.tag, items, summary, params=params,
                     inputs=[*args.query_features, *args.target_features])
    print(f"psytwill bench retrieval [{args.tag}, {args.mode}]: {summary['n_items']} items")
    print(f"  {side}")
    for d in ("query_to_target", "target_to_query"):
        s = summary[d]["pct_beaten"]
        print(f"  {d}: pct_beaten {s['mean']:.4f} [{s['lo']:.4f}, {s['hi']:.4f}]; "
              f"top1 {summary[d]['top1']['mean']:.4f} of ~{summary[d]['n_candidates_mean']:.0f}")


def _run_oddoneout(args) -> None:
    from .core import refuse_overlap, write_run
    from .oddoneout import oddoneout

    emb, n_nan = _load(args.features, args.model, args.key)
    tri = _read_table(args.triplets)
    if args.split:
        col, _, val = args.split.partition("=")
        tri = tri[tri[col].astype(str) == val]
        if tri.empty:
            raise BenchError(f"--split {args.split} keeps no triplets")
    n_all = len(tri)
    if args.only_ids:
        keep = _ids(args.only_ids)
        tri = tri[tri["item1"].astype(str).isin(keep) & tri["item2"].astype(str).isin(keep)
                  & tri["item3"].astype(str).isin(keep)]
        if tri.empty:
            raise BenchError(f"--only-ids {args.only_ids} keeps no triplets")
    ids = set(tri["item1"].astype(str)) | set(tri["item2"].astype(str)) | set(tri["item3"].astype(str))
    guard = refuse_overlap(ids, args.fit_ids, what="triplet")
    items, summary = oddoneout(emb, tri, metric=args.metric, n_boot=args.n_boot, seed=args.seed)
    summary["leak_guard"] = guard
    summary["n_triplets_before_only_ids"] = n_all
    params = {k: v for k, v in vars(args).items() if k != "func"} | {"nan_rows_dropped": n_nan}
    side = write_run(args.output, "oddoneout", args.tag, items, summary, params=params,
                     inputs=[*args.features, args.triplets])
    print(f"psytwill bench oddoneout [{args.tag}]: {summary['n_triplets']} triplets over {summary['n_items']} items")
    _report(side, summary, ["accuracy"])


def _run_nextwindow(args) -> None:
    from .core import refuse_overlap, write_run
    from .nextwindow import next_window

    drop = _ids(args.drop_ids)
    emb, n_nan = _load(args.features, args.model, "stimulus_id,time", window=args.window, drop=drop)
    guard = refuse_overlap({lab.split("|")[0] for lab in emb.labels}, args.fit_ids, what="film")
    items, summary = next_window(emb, horizon=args.horizon, gap=args.gap, predictors=args.predictors.split(","),
                                 metric=args.metric, n_splits=args.n_splits, n_boot=args.n_boot, seed=args.seed)
    summary["leak_guard"] = guard
    summary["window_sec"] = args.window
    params = {k: v for k, v in vars(args).items() if k != "func"} | {"nan_rows_dropped": n_nan}
    side = write_run(args.output, "nextwindow", args.tag, items, summary, params=params, inputs=args.features)
    print(f"psytwill bench nextwindow [{args.tag}, {args.window}s, h={args.horizon}]: {summary['n_films']} films")
    print(f"  {side}")
    for p in args.predictors.split(","):
        s = summary[p]["pct_beaten"]
        print(f"  {p}: pct_beaten {s['mean']:.4f} [{s['lo']:.4f}, {s['hi']:.4f}]")
    for p in ("ridge", "delta"):
        if f"{p}_minus_persistence" in summary:
            s = summary[f"{p}_minus_persistence"]
            print(f"  {p} - persistence: {s['mean']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}]")


def _run_congruence(args) -> None:
    from .congruence import congruence
    from .core import refuse_overlap, write_run

    pairs = _read_table(args.pairs)
    img, n1 = _load(args.image_features, args.image_model, "stimulus_id")
    wrd, n2 = _load(args.word_features, args.word_model, "stimulus_id")
    mi = mt = None
    if args.mode == "mapped":
        if not (args.map_image_features and args.map_text_features):
            raise BenchError("--mode mapped needs --map-image-features/--map-text-features (a paired training set)")
        mi, _ = _load(args.map_image_features, args.map_image_model or args.image_model, "stimulus_id")
        mt, _ = _load(args.map_text_features, args.map_text_model or args.word_model, args.map_text_key)
    ids = set(pairs["image_id"].astype(str)) | set(pairs["word_id"].astype(str))
    guard = refuse_overlap(ids, args.fit_ids, what="pair")
    outcomes = args.outcomes.split(",")
    items, summary = congruence(img, wrd, pairs, outcomes, mode=args.mode, map_image=mi, map_text=mt,
                                metric=args.metric, n_splits=args.n_splits, n_boot=args.n_boot, seed=args.seed)
    summary["leak_guard"] = guard
    params = {k: v for k, v in vars(args).items() if k != "func"} | {"nan_rows_dropped": {"image": n1, "word": n2}}
    inputs = [*args.image_features, *args.word_features, args.pairs, *(args.map_image_features or []),
              *(args.map_text_features or [])]
    side = write_run(args.output, "congruence", args.tag, items, summary, params=params, inputs=inputs)
    print(f"psytwill bench congruence [{args.tag}, {args.mode}]: {summary['n_pairs']} pairs, "
          f"{summary['n_subjects']} subjects")
    print(f"  {side}")
    for oc in outcomes:
        s = summary[oc]
        print(f"  {oc}: mean within-subject rho {s['mean_rho']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}]")


def register(sub) -> None:
    b = sub.add_parser("bench", help="Score embeddings on benchmark tasks: retrieval | oddoneout | nextwindow | "
                                     "congruence")
    bsub = b.add_subparsers(dest="bench_task", required=True)

    r = bsub.add_parser("retrieval", help="cross-modal identification between two keyed sets")
    r.add_argument("--query-features", nargs="+", required=True)
    r.add_argument("--query-model", required=True)
    r.add_argument("--query-key", default="stimulus_id")
    r.add_argument("--target-features", nargs="+", required=True)
    r.add_argument("--target-model", required=True)
    r.add_argument("--target-key", default="stimulus_id,chunk_idx",
                   help="replicate rows (several captions per image) collapse to one prototype per id")
    r.add_argument("--mode", choices=["mapped", "zero-shot"], default="mapped")
    r.add_argument("--metric", choices=["correlation", "cosine"], default="correlation",
                   help="similarity after the map (zero-shot always uses cosine)")
    r.add_argument("--n-splits", type=int, default=5)
    _common(r)
    r.set_defaults(func=_run_retrieval)

    o = bsub.add_parser("oddoneout", help="triplet odd-one-out against human choices")
    o.add_argument("--features", nargs="+", required=True)
    o.add_argument("--model", required=True)
    o.add_argument("--key", default="stimulus_id")
    o.add_argument("--triplets", required=True, help="table with item1, item2, item3, odd")
    o.add_argument("--split", help="COL=VALUE: score only the triplets in that split (e.g. split=test)")
    o.add_argument("--only-ids", help="id file: score only triplets whose three items are all in it")
    o.add_argument("--metric", choices=["cosine", "correlation"], default="cosine")
    _common(o)
    o.set_defaults(func=_run_oddoneout)

    n = bsub.add_parser("nextwindow", help="identify a film's next window among its other windows")
    n.add_argument("--features", nargs="+", required=True)
    n.add_argument("--model", required=True)
    n.add_argument("--window", type=float, required=True, help="bin width in seconds")
    n.add_argument("--horizon", type=int, default=1, help="target is this many bins ahead")
    n.add_argument("--gap", type=int, help="exclude candidates within this many bins of the target "
                                           "(default = horizon, which also excludes the source)")
    n.add_argument("--predictors", default="persistence,ridge,delta")
    n.add_argument("--metric", choices=["correlation", "cosine"], default="correlation")
    n.add_argument("--n-splits", type=int, default=5, help="film folds for the ridge predictor")
    _common(n)
    n.set_defaults(func=_run_nextwindow)

    c = bsub.add_parser("congruence", help="per-pair image-word congruence vs per-pair human outcomes")
    c.add_argument("--image-features", nargs="+", required=True)
    c.add_argument("--image-model", required=True)
    c.add_argument("--word-features", nargs="+", required=True)
    c.add_argument("--word-model", required=True)
    c.add_argument("--pairs", required=True, help="table with subject, image_id, word_id and the outcome columns")
    c.add_argument("--outcomes", required=True, help="comma-separated outcome columns")
    c.add_argument("--mode", choices=["mapped", "zero-shot"], default="mapped")
    c.add_argument("--map-image-features", nargs="+")
    c.add_argument("--map-image-model", help="default: --image-model")
    c.add_argument("--map-text-features", nargs="+")
    c.add_argument("--map-text-model", help="default: --word-model")
    c.add_argument("--map-text-key", default="stimulus_id,chunk_idx")
    c.add_argument("--metric", choices=["correlation", "cosine"], default="correlation")
    c.add_argument("--n-splits", type=int, default=5)
    _common(c)
    c.set_defaults(func=_run_congruence)
