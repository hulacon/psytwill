"""`psytwill bench`: one protocol for every embedding, on four task shapes.

What these pin: identification counts ties half and honours the candidate
mask; grouped bootstrap resamples groups; each task scores a planted
structure well above chance and a destroyed one at chance; the guards that
would otherwise produce a quietly wrong number refuse (unequal dims in
zero-shot, a source window left among next-window candidates, an odd item
not in its triplet, a benchmark item a fit has seen); and the CLI writes an
items file whose rows recompute the sidecar's summary.
"""

import json

import numpy as np
import pandas as pd
import pytest

from psytwill.bench import compare as C
from psytwill.bench.congruence import congruence
from psytwill.bench.core import bootstrap_mean, identify, pool_segments, refuse_overlap
from psytwill.bench.nextwindow import next_window
from psytwill.bench.core import load_concat
from psytwill.bench.oddoneout import noise_ceiling, oddoneout
from psytwill.bench.select import select_best
from psytwill.bench.retrieval import retrieval
from psytwill.cli import main
from psytwill.exceptions import BenchError
from psytwill.store import SpaceMatrix


def _sm(name, labels, X):
    return SpaceMatrix(name=name, labels=list(labels), X=np.asarray(X, float), features=[f"f{i}" for i in
                                                                                        range(np.shape(X)[1])])


def _pair_sets(rng, n=120, latent=4, reps=3, noise=0.1):
    Z = rng.normal(size=(n, latent))
    A, B = rng.normal(size=(latent, 8)), rng.normal(size=(latent, 6))
    ids = [f"s{i:03d}" for i in range(n)]
    q = _sm("q", ids, Z @ A + noise * rng.normal(size=(n, 8)))
    tl, tx = [], []
    for i in range(n):
        for r in range(reps):
            tl.append(f"{ids[i]}|{r}")
            tx.append(Z[i] @ B + noise * rng.normal(size=6))
    return q, _sm("t", tl, tx), Z


# --------------------------------------------------------------------- core


def test_identify_counts_ties_half_and_respects_mask():
    S = np.array([[1.0, 1.0, 0.0, 5.0]])
    allowed = np.array([[True, True, True, False]])
    r = identify(S, np.array([0]), allowed).iloc[0]
    assert r["n_candidates"] == 3
    assert r["pct_beaten"] == pytest.approx((1 + 0.5) / 2)
    assert not r["top1"]


def test_grouped_bootstrap_resamples_groups():
    v = np.r_[np.ones(100), np.zeros(2)]
    g = np.r_[np.zeros(100), [1, 2]]
    flat = bootstrap_mean(v, n_boot=500)
    grp = bootstrap_mean(v, g, n_boot=500)
    assert grp["n_units"] == 3 and flat["n_units"] == 102
    assert (grp["hi"] - grp["lo"]) > (flat["hi"] - flat["lo"])


def test_refuse_overlap(tmp_path):
    f = tmp_path / "fit.txt"
    f.write_text("a\nb\n")
    with pytest.raises(BenchError, match="appear in"):
        refuse_overlap(["b", "c"], [f], what="x")
    assert refuse_overlap(["c"], [f], what="x")["checked"]
    assert refuse_overlap(["c"], None, what="x") == {"checked": False}


# ----------------------------------------------------------------- retrieval


def test_retrieval_mapped_finds_planted_pairs_and_shuffle_is_chance():
    rng = np.random.default_rng(0)
    q, t, _ = _pair_sets(rng)
    items, s = retrieval(q, t, mode="mapped", n_boot=200)
    assert s["query_to_target"]["pct_beaten"]["mean"] > 0.95
    assert s["target_to_query"]["pct_beaten"]["mean"] > 0.95
    assert set(items["fold"]) == set(range(5))
    perm = rng.permutation(len(q.labels))
    shuf = _sm("q", q.labels, q.X[perm])
    _, s0 = retrieval(shuf, t, mode="mapped", n_boot=200)
    assert abs(s0["query_to_target"]["pct_beaten"]["mean"] - 0.5) < 0.08


def test_retrieval_zero_shot_needs_one_space():
    rng = np.random.default_rng(1)
    q, t, _ = _pair_sets(rng)
    with pytest.raises(BenchError, match="shared space"):
        retrieval(q, t, mode="zero-shot")
    same = _sm("t", [lab.split("|")[0] for lab in q.labels], q.X + 0.01 * rng.normal(size=q.X.shape))
    _, s = retrieval(q, same, mode="zero-shot", n_boot=200)
    assert s["query_to_target"]["top1"]["mean"] > 0.95


def _segment_sets(rng, n_films=12, per=8, film_scale=5.0, content=1.0):
    """Films whose segments share a strong film component plus per-segment content.

    The film component makes "which film" trivial, so only the per-segment
    content can score above chance when candidates are the film's own segments.
    """
    lat = 4
    A, B = rng.normal(size=(lat, 9)), rng.normal(size=(lat, 7))
    Fq, Ft = rng.normal(size=(n_films, 9)) * film_scale, rng.normal(size=(n_films, 7)) * film_scale
    ql, qx, tl, tx = [], [], [], []
    for f in range(n_films):
        for c in range(per):
            z = rng.normal(size=lat) * content
            lab = f"film{f:02d}|{c}"
            ql.append(lab)
            qx.append(Fq[f] + z @ A + 0.1 * rng.normal(size=9))
            tl.append(lab)
            tx.append(Ft[f] + z @ B + 0.1 * rng.normal(size=7))
    return _sm("q", ql, qx), _sm("t", tl, tx)


def test_retrieval_within_stimulus_scores_content_not_film_identity():
    rng = np.random.default_rng(7)
    q, t = _segment_sets(rng)
    items, s = retrieval(q, t, mode="mapped", within_stimulus=True, n_boot=200)
    assert s["center_within"]
    assert s["both_directions"]["pct_beaten"]["mean"] > 0.9
    assert s["both_directions"]["pct_beaten"]["n_units"] == 12  # films, not segments
    # uncentred, the films' own offsets swamp the map: why centring is the default
    _, s_raw = retrieval(q, t, mode="mapped", within_stimulus=True, center_within=False, n_boot=200)
    assert s_raw["both_directions"]["pct_beaten"]["mean"] < s["both_directions"]["pct_beaten"]["mean"] - 0.2
    assert (items["n_candidates"] == 8).all()  # only the film's own segments
    film_folds = items.assign(film=items["stimulus_id"].str.split("|").str[0]).groupby("film")["fold"].nunique()
    assert (film_folds == 1).all()  # a film is never on both sides of a map
    # film identity alone: easy across films, chance within
    q0, t0 = _segment_sets(rng, content=0.0)
    _, s_within = retrieval(q0, t0, mode="mapped", within_stimulus=True, n_boot=200)
    assert abs(s_within["both_directions"]["pct_beaten"]["mean"] - 0.5) < 0.08
    _, s_zero = retrieval(q0, _sm("t", q0.labels, q0.X + 0.01 * rng.normal(size=q0.X.shape)), mode="zero-shot",
                          within_stimulus=True, n_boot=200)
    assert s_zero["ci_unit"] == "stimulus"


def test_retrieval_within_stimulus_refuses_single_segment_film():
    rng = np.random.default_rng(8)
    q, t = _segment_sets(rng, per=3)
    keep = [i for i, lab in enumerate(q.labels) if not (lab.startswith("film00|") and not lab.endswith("|0"))]
    q1 = _sm("q", [q.labels[i] for i in keep], q.X[keep])
    with pytest.raises(BenchError, match="single item"):
        retrieval(q1, t, within_stimulus=True)


def test_pool_segments_means_grid_rows_and_refuses_empty_segment():
    labels = [f"a|{x * 0.5}" for x in range(8)] + [f"b|{x * 0.5}" for x in range(4)]
    X = np.arange(12, dtype=float)[:, None] * np.ones((1, 2))
    grid = _sm("g", labels, X)
    seg = pd.DataFrame({"stimulus_id": ["a", "a", "b"], "chunk_idx": [0, 1, 0], "onset": [0.0, 2.0, 0.0],
                        "offset": [2.0, 4.0, 1.0]})
    pooled, info = pool_segments(grid, seg)
    assert pooled.labels == ["a|0", "a|1", "b|0"]
    assert pooled.X[:, 0].tolist() == [1.5, 5.5, 8.5]  # [onset, offset): a rows 0-3, 4-7; b rows 8-9
    assert info["grid_rows_per_segment_min"] == 2
    with pytest.raises(BenchError, match="no grid row"):
        pool_segments(grid, pd.concat([seg, pd.DataFrame({"stimulus_id": ["b"], "chunk_idx": [1],
                                                          "onset": [9.0], "offset": [10.0]})]))
    with pytest.raises(BenchError, match="end at or before"):
        pool_segments(grid, seg.assign(offset=seg["onset"]))


# ---------------------------------------------------------------- oddoneout


def test_oddoneout_planted_clusters():
    rng = np.random.default_rng(2)
    centers = rng.normal(size=(2, 10)) * 5
    ids = [f"c{i}" for i in range(20)]
    X = np.vstack([centers[i % 2] + rng.normal(size=10) for i in range(20)])
    emb = _sm("e", ids, X)
    tri = pd.DataFrame([{"item1": f"c{a}", "item2": f"c{a + 2}", "item3": f"c{b}", "odd": f"c{b}"}
                        for a in range(0, 16, 2) for b in range(1, 20, 4)])
    items, s = oddoneout(emb, tri, n_boot=200)
    assert s["accuracy"]["mean"] == 1.0
    assert (items["model_odd"] == items["odd"]).all()
    bad = tri.assign(odd="c19x")
    with pytest.raises(BenchError, match="not one of their three"):
        oddoneout(emb, bad)


# --------------------------------------------------------------- nextwindow


def _films(rng, n_films=8, T=80, d=12):
    A = 0.9 * np.linalg.qr(rng.normal(size=(d, d)))[0]  # a shared rotation-like dynamic
    labels, X = [], []
    for f in range(n_films):
        x = rng.normal(size=d)
        for t in range(T):
            labels.append(f"film{f}|{t * 0.5}")
            X.append(x)
            x = A @ x + 0.3 * rng.normal(size=d)
    return _sm("m", labels, np.array(X))


def test_nextwindow_ridge_learns_dynamics_and_gap_guard():
    rng = np.random.default_rng(3)
    emb = _films(rng)
    items, s = next_window(emb, horizon=1, n_boot=200)
    assert s["n_films"] == 8
    assert s["ridge"]["pct_beaten"]["mean"] > 0.85
    assert s["ridge_minus_persistence"]["mean"] > 0
    assert s["delta_minus_persistence"]["mean"] > 0
    assert s["ridge"]["pct_beaten"]["n_units"] == 8  # films, not windows
    with pytest.raises(BenchError, match="source window"):
        next_window(emb, horizon=2, gap=1)


# --------------------------------------------------------------- congruence


def _pairs(rng, n_img=150, subjects=3, per=100):
    Z = rng.normal(size=(n_img, 5))
    Bi, Bt = rng.normal(size=(5, 10)), rng.normal(size=(5, 7))
    img_ids = [f"img{i:03d}" for i in range(n_img)]
    image = _sm("img", img_ids, Z @ Bi + 0.05 * rng.normal(size=(n_img, 10)))
    caps = _sm("cap", img_ids, Z @ Bt + 0.05 * rng.normal(size=(n_img, 7)))
    Wz = rng.normal(size=(n_img, 5))  # a word per image-sized pool, latent in the same space
    word = _sm("w", [f"w{i:03d}" for i in range(n_img)], Wz @ Bt)
    rows = []
    for s in range(subjects):
        for k in range(per):
            i, j = rng.integers(n_img), rng.integers(n_img)
            c = Z[i] @ Wz[j] / (np.linalg.norm(Z[i]) * np.linalg.norm(Wz[j]))
            rows.append({"subject": f"s{s}", "image_id": img_ids[i], "word_id": f"w{j:03d}",
                         "rating": c + 0.2 * rng.normal(), "noise": rng.normal()})
    return image, caps, word, pd.DataFrame(rows)


def test_congruence_mapped_tracks_outcome_not_noise():
    rng = np.random.default_rng(4)
    image, caps, word, pairs = _pairs(rng)
    items, s = congruence(image, word, pairs, ["rating", "noise"], mode="mapped", map_image=image, map_text=caps,
                          n_boot=100)
    assert s["rating"]["mean_rho"] > 0.6
    assert abs(s["noise"]["mean_rho"]) < 0.2
    assert (items["fold"] >= 0).all()  # every pair image held out of its own map
    with pytest.raises(BenchError, match="shared space"):
        congruence(image, word, pairs, ["rating"], mode="zero-shot")


# --------------------------------------------------------------------- CLI


def _long(sm, extractor, key="chunk_idx"):
    rows = []
    for lab, x in zip(sm.labels, sm.X):
        sid, _, rest = lab.partition("|")
        for j, v in enumerate(x):
            rows.append((sid, int(rest) if rest else None, sm.name, f"{sm.name}_{j:03d}", float(v)))
    df = pd.DataFrame(rows, columns=["stimulus_id", key, "model", "feature", "value"])
    df["value_str"] = pd.Series([None] * len(df), dtype="string")
    df["modality"], df["extractor"], df["extractor_version"] = "x", extractor, "0.0"
    return df


def test_cli_retrieval_items_recompute_summary(tmp_path):
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(5)
    q, t, _ = _pair_sets(rng, n=60)
    qp, tp = tmp_path / "q_features.parquet", tmp_path / "t_chunks_features.parquet"
    _long(q, "viz2psy").drop(columns="chunk_idx").to_parquet(qp, index=False)
    _long(t, "word2psy").to_parquet(tp, index=False)
    out = tmp_path / "out"
    assert main(["bench", "retrieval", "--query-features", str(qp), "--query-model", "q", "--target-features",
                 str(tp), "--target-model", "t", "-o", str(out), "--tag", "toy", "--n-boot", "100"]) == 0
    side = json.loads((out / "retrieval__toy.json").read_text())
    items = pd.read_csv(out / side["items_file"])
    d = items[items["direction"] == "query_to_target"]
    assert side["summary"]["query_to_target"]["pct_beaten"]["mean"] == pytest.approx(d["pct_beaten"].mean())
    assert side["summary"]["leak_guard"] == {"checked": False}
    fit = tmp_path / "fit.txt"
    fit.write_text("s000\n")
    assert main(["bench", "retrieval", "--query-features", str(qp), "--query-model", "q", "--target-features",
                 str(tp), "--target-model", "t", "-o", str(out), "--tag", "toy2", "--fit-ids", str(fit)]) == 1


def test_cli_oddoneout_only_ids_restricts_triplets(tmp_path):
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(6)
    ids = [f"c{i}" for i in range(12)]
    centers = rng.normal(size=(2, 6)) * 5
    emb = _sm("e", ids, np.vstack([centers[i % 2] + rng.normal(size=6) for i in range(12)]))
    fp = tmp_path / "things_features.parquet"
    _long(emb, "viz2psy").drop(columns="chunk_idx").to_parquet(fp, index=False)
    tri = pd.DataFrame([{"item1": f"c{a}", "item2": f"c{a + 2}", "item3": f"c{b}", "odd": f"c{b}"}
                        for a in range(0, 8, 2) for b in (1, 3, 5)])
    tp = tmp_path / "tri.csv"
    tri.to_csv(tp, index=False)
    keep = tmp_path / "keep.txt"
    keep.write_text("\n".join(f"c{i}" for i in range(10) if i != 5) + "\n")
    out = tmp_path / "out"
    assert main(["bench", "oddoneout", "--features", str(fp), "--model", "e", "--triplets", str(tp), "--only-ids",
                 str(keep), "-o", str(out), "--tag", "t", "--n-boot", "50"]) == 0
    side = json.loads((out / "oddoneout__t.json").read_text())
    assert side["summary"]["n_triplets_before_only_ids"] == len(tri)
    assert side["n_items"] == len(tri) - 4  # every triplet holding c5 is dropped


def test_cli_retrieval_pools_target_segments_within_stimulus(tmp_path):
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(9)
    q, t = _segment_sets(rng, n_films=10, per=6)
    # target as a 0.5 s grid: each segment c spans [2c, 2c + 2) seconds, four grid rows of its vector + noise
    gl, gx, segs = [], [], []
    for lab, x in zip(t.labels, t.X):
        film, c = lab.split("|")
        c = int(c)
        segs.append({"stimulus_id": film, "chunk_idx": c, "onset": 2.0 * c, "offset": 2.0 * c + 2})
        for k in range(4):
            gl.append(f"{film}|{2.0 * c + 0.5 * k}")
            gx.append(x + 0.05 * rng.normal(size=x.size))
    grid = _sm("t", gl, np.array(gx))
    gp, qp, sp = tmp_path / "frames_features.parquet", tmp_path / "annot_chunks_features.parquet", tmp_path / "s.csv"
    g = pd.DataFrame([(lab.split("|")[0], float(lab.split("|")[1]), "t", f"t_{j:03d}", float(v))
                      for lab, x in zip(grid.labels, grid.X) for j, v in enumerate(x)],
                     columns=["stimulus_id", "time", "model", "feature", "value"])
    g["value_str"] = pd.Series([None] * len(g), dtype="string")
    g["modality"], g["extractor"], g["extractor_version"] = "x", "viz2psy", "0.0"
    g.to_parquet(gp, index=False)
    _long(q, "word2psy").to_parquet(qp, index=False)
    pd.DataFrame(segs).to_csv(sp, index=False)
    out = tmp_path / "out"
    assert main(["bench", "retrieval", "--query-features", str(qp), "--query-model", "q", "--query-key",
                 "stimulus_id,chunk_idx", "--target-features", str(gp), "--target-model", "t", "--target-segments",
                 str(sp), "--target-window", "0.5", "--within-stimulus", "-o", str(out), "--tag", "seg",
                 "--n-boot", "100"]) == 0
    side = json.loads((out / "retrieval__seg.json").read_text())
    assert side["summary"]["target_pooling"]["n_segments"] == 60
    assert side["summary"]["target_pooling"]["grid_rows_per_segment_min"] == 4
    assert side["summary"]["both_directions"]["pct_beaten"]["mean"] > 0.85
    assert main(["bench", "retrieval", "--query-features", str(qp), "--query-model", "q", "--target-features",
                 str(gp), "--target-model", "t", "--target-segments", str(sp), "-o", str(out), "--tag", "x"]) == 1


# ------------------------------------------------------------------ compare


def test_compare_retrieval_pairs_items_and_calls_verdicts():
    rng = np.random.default_rng(10)
    q, t = _segment_sets(rng)
    a, _ = retrieval(q, t, within_stimulus=True, n_boot=50)
    noisy = _sm("q", q.labels, q.X + 3.0 * rng.normal(size=q.X.shape))
    b, _ = retrieval(noisy, t, within_stimulus=True, n_boot=50)
    joined, s = C.compare_retrieval(a, b, within_stimulus=True, n_boot=300)
    assert s["pct_beaten"]["verdict"] == "win" and s["ci_unit"] == "stimulus"
    assert s["pct_beaten"]["diff"]["n_units"] == 12
    _, s_rev = C.compare_retrieval(b, a, within_stimulus=True, n_boot=300)
    assert s_rev["pct_beaten"]["verdict"] == "loss"
    _, s_self = C.compare_retrieval(a, a, within_stimulus=True, n_boot=300)
    assert s_self["pct_beaten"]["verdict"] == "tie"
    with pytest.raises(BenchError, match="same items"):
        C.compare_retrieval(a, b[b["stimulus_id"] != b["stimulus_id"].iloc[0]], within_stimulus=True)


def test_compare_refuses_runs_with_different_folds():
    C.check_params("retrieval", {"n_splits": 5, "seed": 0}, {"n_splits": 5, "seed": 0})
    with pytest.raises(BenchError, match="differ in"):
        C.check_params("retrieval", {"n_splits": 5, "seed": 0}, {"n_splits": 10, "seed": 0})
    with pytest.raises(BenchError, match="no paired comparison"):
        C.check_params("nextwindow", {}, {})


def test_compare_congruence_differences_spearman_per_subject():
    rng = np.random.default_rng(11)
    image, caps, word, pairs = _pairs(rng)
    a, _ = congruence(image, word, pairs, ["rating"], mode="mapped", map_image=image, map_text=caps, n_boot=20)
    b = a.assign(congruence=rng.normal(size=len(a)))
    _, s = C.compare_congruence(a, b, ["rating"], n_boot=200)
    assert s["rating"]["verdict"] == "win" and s["rating"]["diff"]["n_units"] == 3
    assert s["rating"]["a"] > 0.6 and abs(s["rating"]["b"]) < 0.2


def test_cli_compare_oddoneout(tmp_path):
    pytest.importorskip("pyarrow")
    tri = pd.DataFrame({"item1": ["a", "b"] * 50, "item2": ["c", "d"] * 50, "item3": ["e", "f"] * 50,
                        "odd": ["e", "f"] * 50})
    out = tmp_path / "out"
    for tag, correct in (("good", [True] * 90 + [False] * 10), ("bad", [True] * 40 + [False] * 60)):
        items = tri.assign(model_odd="x", correct=correct, tied=False)
        from psytwill.bench.core import write_run
        write_run(out, "oddoneout", tag, items, {}, params={"triplets": "t.csv", "split": None, "only_ids": None},
                  inputs=[])
    assert main(["bench", "compare", str(out / "oddoneout__good.json"), str(out / "oddoneout__bad.json"),
                 "-o", str(out), "--tag", "g-b", "--n-boot", "200"]) == 0
    side = json.loads((out / "compare_oddoneout__g-b.json").read_text())
    assert side["summary"]["accuracy"]["verdict"] == "win"
    assert side["summary"]["accuracy"]["diff"]["mean"] == pytest.approx(0.5)


def test_noise_ceiling_is_modal_share_over_repeated_triplets():
    rows = [("a", "b", "c", "c")] * 3 + [("b", "a", "c", "a")]  # one triplet, order varies: modal share 3/4
    rows += [("d", "e", "f", "d"), ("f", "e", "d", "e")]  # second triplet: 1/2
    rows += [("x", "y", "z", "z")]  # answered once: left out
    s = noise_ceiling(pd.DataFrame(rows, columns=["item1", "item2", "item3", "odd"]), n_boot=100)
    assert s["ceiling"]["mean"] == pytest.approx((0.75 + 0.5) / 2)
    assert s["n_triplets_repeated"] == 2 and s["n_triplets_single"] == 1
    with pytest.raises(BenchError, match="more than once"):
        noise_ceiling(pd.DataFrame([("x", "y", "z", "z")], columns=["item1", "item2", "item3", "odd"]))


# ------------------------------------------------------- arms B1 and B2


def test_load_concat_zscores_drops_nan_columns_and_keeps_shared_rows(tmp_path):
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(12)
    a = _sm("a", [f"s{i}" for i in range(20)], rng.normal(size=(20, 3)) * 100)
    Xb = rng.normal(size=(19, 2))
    Xb[4, 1] = np.nan  # a declared null in one column
    b = _sm("b", [f"s{i}" for i in range(19)], Xb)
    fp = tmp_path / "x_features.parquet"
    pd.concat([_long(a, "v").drop(columns="chunk_idx"), _long(b, "v").drop(columns="chunk_idx")]).to_parquet(fp)
    sm, info = load_concat([fp], ["a", "b"])
    assert sm.n == 19 and info["rows_lost"] == {"a": 1, "b": 0}
    assert info["nan_columns_dropped"] == {"a": 0, "b": 1} and sm.dim == 4
    np.testing.assert_allclose(sm.X.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(sm.X.std(0), 1, atol=1e-12)


def test_select_best_chooses_out_of_fold_and_covers_every_item():
    rng = np.random.default_rng(13)
    q, t = _segment_sets(rng)
    good, _ = retrieval(q, t, within_stimulus=True, n_boot=20)
    bad, _ = retrieval(_sm("q", q.labels, rng.normal(size=q.X.shape)), t, within_stimulus=True, n_boot=20)
    partial = good[good["stimulus_id"] != good["stimulus_id"].iloc[0]]
    items, s = select_best({"good": good, "bad": bad, "partial": partial}, "retrieval", statistic="pct_beaten",
                           within_stimulus=True, n_splits=4)
    assert s["members_chosen"] == ["good"] and s["not_candidates_incomplete"] == ["partial"]
    assert set(items["stimulus_id"]) == set(good["stimulus_id"]) and len(items) == len(good)
    # each film's items sit in one outer fold
    film_folds = items.assign(f=items["stimulus_id"].str.split("|").str[0]).groupby("f")["b2_fold"].nunique()
    assert (film_folds == 1).all()
    _, cmp = C.compare_retrieval(items, good, within_stimulus=True, n_boot=50)
    assert cmp["pct_beaten"]["diff"]["mean"] == pytest.approx(0.0)
    with pytest.raises(BenchError, match="no best-member selection"):
        select_best({"x": good}, "congruence", statistic="rho")



def test_reconstruct_rebuilds_a_member_from_block_scores(tmp_path):
    pytest.importorskip("pyarrow")
    from psytwill.bench.reconstruct import fit_reconstruction, scores_matrix
    rng = np.random.default_rng(14)
    n, k = 400, 6
    S = rng.normal(size=(n, k))
    W = rng.normal(size=(k, 10))
    ids = [f"img{i:04d}" for i in range(n)]
    sp = tmp_path / "V_scores.parquet"
    pd.DataFrame(S, columns=[f"V_{j:03d}" for j in range(k)]).assign(stimulus_id=ids).to_parquet(sp)
    member = _sm("ebind", ids, S @ W + 0.01 * rng.normal(size=(n, 10)))
    lm, fit = fit_reconstruction(scores_matrix(sp, ["stimulus_id"]), member, cv_rows=300)
    assert fit["cv_r2_variance_weighted"] > 0.99 and fit["n_fit_rows"] == n
    # through the CLI: items carry the block's coordinates as a store model, output is a store model
    fp = tmp_path / "m_features.parquet"
    _long(member, "viz2psy").drop(columns="chunk_idx").to_parquet(fp)
    items = _sm("pspace_v", [f"item{i}" for i in range(30)], rng.normal(size=(30, k)))
    ap = tmp_path / "items_features.parquet"
    _long(items, "psytwill").drop(columns="chunk_idx").to_parquet(ap)
    out = tmp_path / "r" / "r_ebind_features.parquet"
    assert main(["bench", "reconstruct", "--scores", str(sp), "--member-features", str(fp), "--member", "ebind",
                 "--apply-features", str(ap), "--apply-model", "pspace_v", "--name", "r_ebind", "-o", str(out)]) == 0
    from psytwill.bench.core import load_model
    r = load_model([out], "r_ebind")
    assert r.n == 30 and r.dim == 10
    pos = {lab: i for i, lab in enumerate(r.labels)}
    want = lm.predict(items.X)
    np.testing.assert_allclose(r.X[[pos[f"item{i}"] for i in range(30)]], want, rtol=1e-6, atol=1e-6)


def test_cli_select_names_cross_modal_members_by_pair(tmp_path):
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(15)
    q, t, _ = _pair_sets(rng, n=60)
    noisy = _sm("qn", q.labels, rng.normal(size=q.X.shape))
    qp, tp = tmp_path / "q_features.parquet", tmp_path / "t_chunks_features.parquet"
    pd.concat([_long(q, "viz2psy"), _long(noisy, "viz2psy")]).drop(columns="chunk_idx").to_parquet(qp, index=False)
    _long(t, "word2psy").to_parquet(tp, index=False)
    out = tmp_path / "out"
    for m in ("q", "qn"):
        assert main(["bench", "retrieval", "--query-features", str(qp), "--query-model", m, "--target-features",
                     str(tp), "--target-model", "t", "-o", str(out), "--tag", f"m_{m}", "--n-boot", "20"]) == 0
    assert main(["bench", "select", str(out / "retrieval__m_q.json"), str(out / "retrieval__m_qn.json"),
                 "--statistic", "pct_beaten", "-o", str(out), "--tag", "b2"]) == 0
    side = json.loads((out / "retrieval__b2.json").read_text())
    assert side["summary"]["members_chosen"] == ["q->t"]
    assert side["params"]["query_model"] == "B2" and side["params"]["arm"] == "B2"
