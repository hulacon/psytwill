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


#: arms built by `_load` from a `a+b+c` model spec, keyed by spec; read back into each run's sidecar
CONCAT_INFO: dict = {}


def _load(features, model, key, window=None, drop=frozenset()):
    from .core import drop_nan_rows, load_concat, load_model

    if "+" in model:
        sm, info = load_concat(features, model.split("+"), key=tuple(key.split(",")), window=window)
        CONCAT_INFO[model] = info
        print(f"  concatenated {len(info['members'])} members: {info['n_columns']} columns "
              f"({sum(info['nan_columns_dropped'].values())} with NaN dropped)", flush=True)
    else:
        sm = load_model(features, model, key=tuple(key.split(",")), window=window)
    sm, n_nan = drop_nan_rows(_keep(sm, set(drop)))
    print(f"  loaded {model}: {sm.n} rows x {sm.dim}" + (f" ({n_nan} NaN rows dropped)" if n_nan else ""),
          flush=True)
    return sm, n_nan


def _concat_record(*models) -> dict:
    return {m: CONCAT_INFO[m] for m in models if m in CONCAT_INFO}


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


def _load_side(features, model, key, segments, window, drop, side):
    """One retrieval side: a keyed table, or a time grid mean-pooled into segments."""
    if not segments:
        sm, n_nan = _load(features, model, key, drop=drop)
        return sm, n_nan, None
    from .core import pool_segments

    if window is None:
        raise BenchError(f"--{side}-segments pools a time grid; give its --{side}-window (seconds)")
    sm, n_nan = _load(features, model, "stimulus_id,time", window=window, drop=drop)
    sm, pooling = pool_segments(sm, _read_table(segments))
    print(f"  pooled {model} into {pooling['n_segments']} segments "
          f"(min {pooling['grid_rows_per_segment_min']} grid rows each)", flush=True)
    return sm, n_nan, pooling


def _run_retrieval(args) -> None:
    from .core import in_subset, parse_subsets, refuse_overlap, write_run
    from .retrieval import retrieval, summarize

    drop = _ids(args.drop_ids)
    subsets = parse_subsets(args.subset)
    q, qn, q_pool = _load_side(args.query_features, args.query_model, args.query_key, args.query_segments,
                               args.query_window, drop, "query")
    t, tn, t_pool = _load_side(args.target_features, args.target_model, args.target_key, args.target_segments,
                               args.target_window, drop, "target")
    guard = refuse_overlap({lab.split("|")[0] for lab in q.labels}, args.fit_ids, what="retrieval")
    items, summary = retrieval(q, t, mode=args.mode, metric=args.metric, n_splits=args.n_splits, seed=args.seed,
                               n_boot=args.n_boot, within_stimulus=args.within_stimulus)
    summary["leak_guard"] = guard
    summary["concatenated"] = _concat_record(args.query_model, args.target_model)
    if q_pool:
        summary["query_pooling"] = q_pool
    if t_pool:
        summary["target_pooling"] = t_pool
    if subsets:
        summary["subsets"] = {}
        for name, ids in subsets.items():
            keep = in_subset(items["stimulus_id"], ids)
            if not keep.any():
                raise BenchError(f"subset {name!r} matches no scored item")
            summary["subsets"][name] = summarize(items[keep], within_stimulus=args.within_stimulus,
                                                 n_boot=args.n_boot, seed=args.seed)
    params = {k: v for k, v in vars(args).items() if k != "func"} | {"nan_rows_dropped": {"query": qn, "target": tn}}
    inputs = [*args.query_features, *args.target_features,
              *[x for x in (args.query_segments, args.target_segments) if x],
              *[sp.partition("=")[2] for sp in (args.subset or [])]]
    side = write_run(args.output, "retrieval", args.tag, items, summary, params=params, inputs=inputs)
    print(f"psytwill bench retrieval [{args.tag}, {args.mode}]: {summary['n_items']} items")
    print(f"  {side}")
    for d in ("query_to_target", "target_to_query"):
        s = summary[d]["pct_beaten"]
        print(f"  {d}: pct_beaten {s['mean']:.4f} [{s['lo']:.4f}, {s['hi']:.4f}]; "
              f"top1 {summary[d]['top1']['mean']:.4f} of ~{summary[d]['n_candidates_mean']:.0f}")
    s = summary["both_directions"]["pct_beaten"]
    print(f"  both directions: pct_beaten {s['mean']:.4f} [{s['lo']:.4f}, {s['hi']:.4f}] "
          f"(CI unit: {summary['ci_unit']}, n={s['n_units']})")
    for name, sub in summary.get("subsets", {}).items():
        s = sub["both_directions"]["pct_beaten"]
        print(f"  subset {name}: pct_beaten {s['mean']:.4f} [{s['lo']:.4f}, {s['hi']:.4f}] "
              f"({sub['n_items_scored']} items, n={s['n_units']})")


def _run_oddoneout(args) -> None:
    from .core import refuse_overlap, write_run
    from .oddoneout import oddoneout

    drop = _ids(args.drop_ids)
    emb, n_nan = _load(args.features, args.model, args.key, drop=drop)
    tri = _read_table(args.triplets)
    if drop:
        tri = tri[~(tri["item1"].astype(str).isin(drop) | tri["item2"].astype(str).isin(drop)
                    | tri["item3"].astype(str).isin(drop))]
        if tri.empty:
            raise BenchError(f"--drop-ids {args.drop_ids} leaves no triplets")
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
    summary["concatenated"] = _concat_record(args.model)
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
                                 metric=args.metric, n_splits=args.n_splits, n_boot=args.n_boot, seed=args.seed,
                                 window=args.window)
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

    drop = _ids(args.drop_ids)
    pairs = _read_table(args.pairs)
    if drop:
        pairs = pairs[~(pairs["image_id"].astype(str).isin(drop) | pairs["word_id"].astype(str).isin(drop))]
        if pairs.empty:
            raise BenchError(f"--drop-ids {args.drop_ids} leaves no pairs")
    img, n1 = _load(args.image_features, args.image_model, "stimulus_id", drop=drop)
    wrd, n2 = _load(args.word_features, args.word_model, "stimulus_id", drop=drop)
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


def _run_rdm(args) -> None:
    from .core import refuse_overlap, write_run
    from .rdm import rdm

    drop = _ids(args.drop_ids)
    emb, n_nan = _load(args.features, args.model, args.key, drop=drop)
    pairs = _read_table(args.pairs)
    n_all = len(pairs)
    if drop:
        pairs = pairs[~(pairs["image_a"].astype(str).isin(drop) | pairs["image_b"].astype(str).isin(drop))]
        if pairs.empty:
            raise BenchError(f"--drop-ids {args.drop_ids} leaves no pairs")
    ids = set(pairs["image_a"].astype(str)) | set(pairs["image_b"].astype(str))
    guard = refuse_overlap(ids, args.fit_ids, what="judged")
    items, summary = rdm(emb, pairs, metric=args.metric, n_boot=args.n_boot, seed=args.seed)
    summary["leak_guard"] = guard
    summary["concatenated"] = _concat_record(args.model)
    summary["n_pairs_before_drop_ids"] = n_all
    params = {k: v for k, v in vars(args).items() if k != "func"} | {"nan_rows_dropped": n_nan}
    side = write_run(args.output, "rdm", args.tag, items, summary, params=params, inputs=[*args.features, args.pairs])
    print(f"psytwill bench rdm [{args.tag}, {args.metric}]: {summary['n_items']} items, "
          f"{summary['n_participants']} participants")
    print(f"  {side}")
    for k in ("rho", "ceiling_lower", "ceiling_upper", "share_of_ceiling_lower"):
        v = summary[k]
        print(f"  {k}: {v['mean']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]")


def _run_rating(args) -> None:
    from .core import refuse_overlap, write_run
    from .rating import rating

    drop = _ids(args.drop_ids)
    emb, n_nan = _load(args.features, args.model, args.key, drop=drop)
    ratings = _read_table(args.ratings)
    if drop:
        ratings = ratings[~ratings["stimulus_id"].astype(str).isin(drop)]
        if ratings.empty:
            raise BenchError(f"--drop-ids {args.drop_ids} leaves no ratings")
    guard = refuse_overlap(set(ratings["stimulus_id"].astype(str)), args.fit_ids, what="rated")
    dims = args.outcomes.split(",")
    items, summary = rating(emb, ratings, dims, n_splits=args.n_splits, n_boot=args.n_boot, seed=args.seed,
                            n_reliability_splits=args.n_reliability_splits)
    summary["leak_guard"] = guard
    summary["concatenated"] = _concat_record(args.model)
    params = {k: v for k, v in vars(args).items() if k != "func"} | {"nan_rows_dropped": n_nan}
    side = write_run(args.output, "rating", args.tag, items, summary, params=params,
                     inputs=[*args.features, args.ratings])
    print(f"psytwill bench rating [{args.tag}, mapped]: {', '.join(dims)}")
    print(f"  {side}")
    for d in dims:
        v, c = summary[d]["r"], summary[d]["ceiling"]
        print(f"  {d}: r {v['mean']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}] over {v['n_units']} items; "
              f"split-half reliability {c['reliability']:.3f} (r ceiling {c['r_ceiling']:.3f})")


def _run_boundary(args) -> None:
    import numpy as np

    from psytwill.store import SpaceMatrix

    from .boundary import boundary
    from .core import refuse_overlap, write_run

    drop = _ids(args.drop_ids)
    parts, n_nan = [], {}
    for m in args.model.split(","):
        sm, n_nan[m] = _load(args.features, m, "stimulus_id,time", window=args.window, drop=drop)
        parts.append(sm)
    if len(parts) == 1:
        sm = parts[0]
    else:
        # the blocks side by side in their own units (no rescaling), on the bins every block has
        common = sorted(set.intersection(*(set(p.labels) for p in parts)))
        if not common:
            raise BenchError(f"models {args.model} share no grid row")
        cols, X = [], []
        for p in parts:
            pos = {lab: i for i, lab in enumerate(p.labels)}
            X.append(p.X[[pos[lab] for lab in common]])
            cols += [f"{p.name}:{f}" for f in p.features]
        sm = SpaceMatrix(name="+".join(p.name for p in parts), labels=common, X=np.hstack(X), features=cols)
        print(f"  joined {len(parts)} blocks: {sm.n} common rows x {sm.dim}", flush=True)
    kernel = None
    if args.sfa_kernel:
        z = np.load(args.sfa_kernel)
        need = [f"{args.sfa_prefix}W", f"{args.sfa_prefix}lam"]
        miss = [k for k in need if k not in z.files]
        if miss:
            raise BenchError(f"{args.sfa_kernel} lacks {miss} (has {z.files[:6]}...)")
        kernel = (z[need[0]], z[need[1]])
    guard = refuse_overlap({lab.split("|")[0] for lab in sm.labels}, args.fit_ids, what="film")
    items, summary = boundary(sm, _read_table(args.boundaries), _read_table(args.cuts), _read_table(args.durations),
                              level=args.level, window=args.window, kernel=kernel, tol=args.tol, exclude=args.exclude,
                              n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed)
    summary["leak_guard"] = guard
    params = {k: v for k, v in vars(args).items() if k != "func"} | {"nan_rows_dropped": n_nan}
    inputs = [*args.features, args.boundaries, args.cuts, args.durations, *([args.sfa_kernel] if args.sfa_kernel else [])]
    side = write_run(args.output, "boundary", args.tag, items, summary, params=params, inputs=inputs)
    print(f"psytwill bench boundary [{args.tag}, level {args.level}]: {summary['n_films']} films")
    print(f"  {side}")
    for st in ("change", "pe_sfa"):
        if st not in summary:
            continue
        c = summary[st]
        f, r, far = c["film_sd"], c["raw"], c["far_from_cut_film_sd"]
        print(f"  {st}: {f['mean']:+.4f} film-SD [{f['lo']:+.4f}, {f['hi']:+.4f}], p {f['p_sign_flip']:.4g} "
              f"({f['n_annotators_positive']}/{f['n_annotators']} annotators > 0); raw {r['mean']:+.5f} "
              f"p {r['p_sign_flip']:.4g}; {c['n_boundaries']} boundaries, {c['n_dropped']} dropped; "
              f"> {args.tol:g} s from a cut {far['mean']:+.4f} film-SD")


def _run_ceiling(args) -> None:
    import pandas as pd

    from .core import write_run
    from .oddoneout import noise_ceiling

    tri = pd.concat([_read_table(p) for p in args.triplets], ignore_index=True)
    n_all = len(tri)
    if args.only_ids:
        keep = _ids(args.only_ids)
        tri = tri[tri["item1"].astype(str).isin(keep) & tri["item2"].astype(str).isin(keep)
                  & tri["item3"].astype(str).isin(keep)]
    s = noise_ceiling(tri, n_boot=args.n_boot, seed=args.seed)
    s["n_answers_before_only_ids"] = n_all
    params = {k: v for k, v in vars(args).items() if k != "func"}
    side = write_run(args.output, "ceiling", args.tag, tri, s, params=params, inputs=args.triplets)
    c = s["ceiling"]
    print(f"psytwill bench ceiling [{args.tag}]: {c['mean']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}] over "
          f"{s['n_triplets_repeated']} repeated triplets (median {s['answers_per_triplet_median']:.0f} answers each; "
          f"{s['n_triplets_single']} answered once, left out)")
    print(f"  {side}")


def _run_select(args) -> None:
    import json

    import pandas as pd

    from . import compare as C
    from .core import write_run
    from .select import select_best

    sides = {p: json.loads(Path(p).read_text()) for p in args.runs}
    first = next(iter(sides.values()))
    task = first["task"]
    for p, sd in sides.items():
        if sd["task"] != task:
            raise BenchError(f"{p} is a {sd['task']} run; every member run must be {task}")
        C.check_params(task, first["params"], sd["params"])
    runs = {}
    for p, sd in sides.items():
        pr = sd["params"]
        # a cross-modal run's member is the pair it read, query side -> target side
        name = (f"{pr['query_model']}->{pr['target_model']}" if pr.get("query_model") and pr.get("target_model")
                else pr.get("model") or sd["tag"])
        if name in runs:
            raise BenchError(f"two runs score member {name!r}")
        runs[name] = pd.read_csv(Path(p).parent / sd["items_file"])
    within = bool(first["params"].get("within_stimulus"))
    items, summary = select_best(runs, task, statistic=args.statistic, within_stimulus=within,
                                 n_splits=args.n_splits, seed=args.seed, n_boot=args.n_boot)
    summary["member_runs"] = {n: str(Path(p).resolve()) for n, p in zip(runs, sides)}
    params = dict(first["params"]) | {"arm": "B2", "select_statistic": args.statistic,
                                       "select_n_splits": args.n_splits, "select_seed": args.seed}
    for k in ("query_model", "model", "tag"):
        if k in params:
            params[k] = "B2"
    side = write_run(args.output, task, args.tag, items, summary, params=params, inputs=args.runs)
    print(f"psytwill bench select [{task}, B2 by {args.statistic}]: {len(summary['candidates'])} candidates; chose "
          + ", ".join(f"fold {p['fold']} {p['member']}" for p in summary["folds"]))
    if summary["not_candidates_incomplete"]:
        print(f"  not candidates (some items unscored): {', '.join(summary['not_candidates_incomplete'])}")
    print(f"  {side}")


def _run_reconstruct(args) -> None:
    from .core import load_model
    from .reconstruct import fit_reconstruction, scores_matrix, write_reconstruction

    scores = scores_matrix(args.scores, args.scores_key.split(","), _ids(args.exclude_ids) or None)
    member = load_model(args.member_features, args.member, key=tuple(args.member_key.split(",")))
    print(f"  fit rows: block scores {scores.n} x {scores.dim}; {args.member} {member.n} x {member.dim}", flush=True)
    lm, fit = fit_reconstruction(scores, member, seed=args.seed)
    apply = load_model(args.apply_features, args.apply_model, key=tuple(args.apply_key.split(",")),
                       window=args.apply_window)
    if apply.dim != scores.dim:
        raise BenchError(f"{args.apply_model} has {apply.dim} columns but the block scores {scores.dim}; "
                         "apply the map to the same block it was fit on")
    prov = {"fit": fit, "scores": str(Path(args.scores).resolve()), "exclude_ids": args.exclude_ids,
            "member_features": [str(p) for p in args.member_features], "apply_model": args.apply_model,
            "apply_features": [str(p) for p in args.apply_features], "apply_window": args.apply_window}
    side = write_reconstruction(lm, apply, name=args.name, member=member, key=args.apply_key.split(","),
                                out=args.output, provenance=prov)
    print(f"psytwill bench reconstruct [{args.name} = {args.member} from {args.apply_model}]: {apply.n} items; "
          f"fit rows {fit['n_fit_rows']}, alpha {fit['alpha']:g}, out-of-fold R2 {fit['cv_r2_variance_weighted']:.3f}")
    print(f"  {side}")


def _run_compare(args) -> None:
    import json

    import pandas as pd

    from . import compare as C
    from .core import write_run

    sides = [json.loads(Path(p).read_text()) for p in (args.a, args.b)]
    task = sides[0]["task"]
    if sides[1]["task"] != task:
        raise BenchError(f"cannot compare a {task} run with a {sides[1]['task']} run")
    C.check_params(task, sides[0]["params"], sides[1]["params"])
    items = [pd.read_csv(Path(p).parent / s["items_file"]) for p, s in zip((args.a, args.b), sides)]
    if task == "retrieval":
        within = bool(sides[0]["params"].get("within_stimulus"))
        from .core import parse_subsets

        joined, summary = C.compare_retrieval(*items, within_stimulus=within, n_boot=args.n_boot, seed=args.seed,
                                              subsets=parse_subsets(args.subset))
        stats = ["pct_beaten", "top1"]
    elif task == "oddoneout":
        joined, summary = C.compare_oddoneout(*items, n_boot=args.n_boot, seed=args.seed)
        stats = ["accuracy"]
    elif task == "rdm":
        from .rdm import compare_rdm

        joined, summary = compare_rdm(*items, n_boot=args.n_boot, seed=args.seed)
        stats = ["rho"]
    elif task == "boundary":
        from .boundary import compare_boundary

        joined, summary = compare_boundary(*items, n_boot=args.n_boot, seed=args.seed)
        stats = [k for k in ("change", "pe_sfa") if k in summary]
    elif task == "rating":
        from .rating import compare_rating

        joined, summary = compare_rating(*items, n_boot=args.n_boot, seed=args.seed)
        stats = [d for d in sides[0]["params"]["outcomes"].split(",")]
    else:
        outcomes = (args.outcomes or sides[0]["params"]["outcomes"]).split(",")
        joined, summary = C.compare_congruence(*items, outcomes, n_boot=args.n_boot, seed=args.seed)
        stats = outcomes
    summary |= {"task": task, "a": {"tag": sides[0]["tag"], "sidecar": str(Path(args.a).resolve())},
                "b": {"tag": sides[1]["tag"], "sidecar": str(Path(args.b).resolve())}}
    params = {k: v for k, v in vars(args).items() if k != "func"}
    side = write_run(args.output, f"compare_{task}", args.tag, joined, summary, params=params, inputs=[args.a, args.b])
    print(f"psytwill bench compare [{task}]: {sides[0]['tag']} - {sides[1]['tag']} on {summary['n_items']} items "
          f"(CI unit: {summary['ci_unit']})")
    print(f"  {side}")
    for st in stats:
        d = summary[st]["diff"]
        print(f"  {st}: a {summary[st]['a']:.4f}, b {summary[st]['b']:.4f}, a - b {d['mean']:+.4f} "
              f"[{d['lo']:+.4f}, {d['hi']:+.4f}] -> {summary[st]['verdict']}")
        if "retention" in summary[st]:
            q = summary[st]["retention"]
            print(f"    retention (a - {summary[st]['retention_chance']}) / (b - {summary[st]['retention_chance']}) "
                  f"{q['mean']:.4f} [{q['lo']:.4f}, {q['hi']:.4f}]"
                  + ("" if summary[st]["retention_defined"] else " (undefined: b's interval reaches chance)"))
        if "ratio" in summary[st]:
            q = summary[st]["ratio"]
            print(f"    a / b {q['mean']:.4f} [{q['lo']:.4f}, {q['hi']:.4f}]"
                  + ("" if summary[st]["ratio_defined"] else " (undefined: b's interval reaches 0)"))
    for name, sub in summary.get("subsets", {}).items():
        d = sub["pct_beaten"]["diff"]
        line = (f"  subset {name} ({sub['n_items']} items): pct_beaten a {sub['pct_beaten']['a']:.4f}, "
                f"b {sub['pct_beaten']['b']:.4f}, a - b {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")
        q = sub["pct_beaten"]["retention"]
        print(line + f"; retention {q['mean']:.4f} [{q['lo']:.4f}, {q['hi']:.4f}]"
              + ("" if sub["pct_beaten"]["retention_defined"] else " (undefined)"))


def register(sub) -> None:
    b = sub.add_parser("bench", help="Score embeddings on benchmark tasks: retrieval | oddoneout | nextwindow | "
                                     "congruence | rdm | rating | boundary; compare two runs")
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
    r.add_argument("--within-stimulus", action="store_true",
                   help="items are parts of a stimulus (labels stimulus|chunk_idx): candidates are the other parts "
                        "of the same stimulus, folds and CIs are grouped by stimulus")
    r.add_argument("--target-segments",
                   help="table of stimulus_id, chunk_idx, onset, offset: the target is a time grid, mean-pooled "
                        "into these segments (needs --target-window)")
    r.add_argument("--target-window", type=float, help="the target grid's bin width in seconds")
    r.add_argument("--query-segments",
                   help="as --target-segments, for the query side (both sides may be pooled grids, e.g. a film's "
                        "sound and picture over the same segments; needs --query-window)")
    r.add_argument("--query-window", type=float, help="the query grid's bin width in seconds")
    r.add_argument("--subset", action="append", metavar="NAME=FILE",
                   help="also report the statistics over the items whose stimulus or label is listed in FILE "
                        "(maps and candidates stay those of the full run); repeatable")
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

    s = bsub.add_parser("rdm", help="model item distances vs human dissimilarity judgements (RSA, item bootstrap)")
    s.add_argument("--features", nargs="+", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--key", default="stimulus_id")
    s.add_argument("--pairs", required=True, help="table with participant, image_a, image_b, dissimilarity")
    s.add_argument("--metric", choices=["cosine", "correlation"], default="cosine",
                   help="model dissimilarity = 1 - this similarity")
    _common(s)
    s.set_defaults(func=_run_rdm)

    g = bsub.add_parser("rating", help="mapped ridge from coordinates to each item's mean human rating")
    g.add_argument("--features", nargs="+", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--key", default="stimulus_id")
    g.add_argument("--ratings", required=True,
                   help="table with participant, stimulus_id and one column per rated dimension")
    g.add_argument("--outcomes", required=True, help="comma-separated rated dimensions, e.g. valence,arousal")
    g.add_argument("--n-splits", type=int, default=5, help="item folds for the ridge")
    g.add_argument("--n-reliability-splits", type=int, default=100, help="random participant halves")
    _common(g)
    g.set_defaults(func=_run_rating)

    bd = bsub.add_parser("boundary", help="film change / prediction error at annotated event boundaries vs "
                                           "cut-matched controls")
    bd.add_argument("--features", nargs="+", required=True)
    bd.add_argument("--model", required=True, help="a grid model; comma-separated models are joined side by side "
                                                   "(raw units) on their common bins")
    bd.add_argument("--window", type=float, required=True, help="grid bin width in seconds")
    bd.add_argument("--boundaries", required=True, help="table with stimulus_id, level, onset, annotator")
    bd.add_argument("--level", required=True, help="the level scored; every level's onsets are kept from the controls")
    bd.add_argument("--cuts", required=True, help="table with stimulus_id, time (the shot cuts matched on)")
    bd.add_argument("--durations", required=True,
                    help="table with stimulus_id, duration: each film's scored span, from 0")
    bd.add_argument("--sfa-kernel", help="npz holding <prefix>W (d, k) and <prefix>lam (k,): adds pe_sfa")
    bd.add_argument("--sfa-prefix", default="", help="array-name prefix in --sfa-kernel")
    bd.add_argument("--tol", type=float, default=1.0, help="half-width of a boundary's window, seconds")
    bd.add_argument("--exclude", type=float, default=2.0, help="controls lie further than this from any boundary")
    bd.add_argument("--n-perm", type=int, default=10_000, help="sign-flip draws")
    _common(bd)
    bd.set_defaults(func=_run_boundary)

    rc = bsub.add_parser("reconstruct", help="rebuild a member from a space's block scores (arm R)")
    rc.add_argument("--scores", required=True, help="block scores on the block's fit rows (`space project` parquet)")
    rc.add_argument("--scores-key", default="stimulus_id")
    rc.add_argument("--exclude-ids", help="id file of fit-row stimuli to leave out of the map")
    rc.add_argument("--member-features", nargs="+", required=True, help="the member's features on those fit rows")
    rc.add_argument("--member", required=True, help="member model to rebuild, e.g. ebind")
    rc.add_argument("--member-key", default="stimulus_id")
    rc.add_argument("--apply-features", nargs="+", required=True, help="the space's coordinates for the items")
    rc.add_argument("--apply-model", required=True, help="e.g. pspace_v")
    rc.add_argument("--apply-key", default="stimulus_id")
    rc.add_argument("--apply-window", type=float, help="bin width when the items are a time grid")
    rc.add_argument("--name", required=True, help="model name to write, e.g. r_ebind")
    rc.add_argument("-o", "--output", required=True, help="output features table (.parquet)")
    rc.add_argument("--seed", type=int, default=0)
    rc.set_defaults(func=_run_reconstruct)

    sel = bsub.add_parser("select", help="best single member (B2), chosen per outer fold on the other folds")
    sel.add_argument("runs", nargs="+", help="sidecar JSON of each member's run of one task")
    sel.add_argument("--statistic", required=True,
                     help="what 'best' means: retrieval pct_beaten | top1 (both directions averaged); "
                          "oddoneout correct")
    sel.add_argument("--n-splits", type=int, default=5, help="outer folds over the task's unit")
    sel.add_argument("-o", "--output", required=True, help="output directory")
    sel.add_argument("--tag", required=True)
    sel.add_argument("--seed", type=int, default=0)
    sel.add_argument("--n-boot", type=int, default=2000)
    sel.set_defaults(func=_run_select)

    e = bsub.add_parser("ceiling", help="odd-one-out noise ceiling from repeated triplets")
    e.add_argument("--triplets", nargs="+", required=True,
                   help="table(s) of item1, item2, item3, odd; tables are pooled, so a triplet's repeats may sit "
                        "in different files (a test set and its repeat)")
    e.add_argument("--only-ids", help="id file: use only triplets whose three items are all in it")
    e.add_argument("-o", "--output", required=True, help="output directory")
    e.add_argument("--tag", required=True)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--n-boot", type=int, default=2000)
    e.set_defaults(func=_run_ceiling)

    k = bsub.add_parser("compare", help="paired difference between two runs of one task, with a win/loss/tie verdict")
    k.add_argument("a", help="sidecar JSON of run a")
    k.add_argument("b", help="sidecar JSON of run b; the difference is a - b")
    k.add_argument("--outcomes", help="congruence: outcome columns to compare (default: run a's)")
    k.add_argument("--subset", action="append", metavar="NAME=FILE",
                   help="retrieval: also compare over the items whose stimulus or label is listed in FILE; repeatable")
    k.add_argument("-o", "--output", required=True, help="output directory")
    k.add_argument("--tag", required=True, help="run name in the output filenames, e.g. pspace_v-vs-pca")
    k.add_argument("--seed", type=int, default=0)
    k.add_argument("--n-boot", type=int, default=2000)
    k.set_defaults(func=_run_compare)
