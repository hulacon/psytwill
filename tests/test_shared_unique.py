"""Shared and unique subspaces of a block, and their split-half gate, on planted structure."""

import json

import numpy as np
import pandas as pd
import pytest

from psytwill import shared_unique as su
from psytwill.cli import main
from psytwill.exceptions import SpaceError
from psytwill.space import fit_block, fit_block_map, load_fit
from psytwill.store import SpaceMatrix

N = 1200


def _space(name, X, labels=None):
    return SpaceMatrix(name=name, labels=labels or [f"s{i:04d}" for i in range(X.shape[0])], X=X,
                       features=[f"{name}_{j:02d}" for j in range(X.shape[1])])


def _embed(rng, Z, dim):
    """Noiseless orthonormal embedding: the member's whitened rank is Z's column count."""
    Q, _ = np.linalg.qr(rng.normal(size=(dim, Z.shape[1])))
    return Z @ Q.T


def _planted(seed=0, n=N):
    """a, b, c share two latents; a holds two more alone, b one; c holds nothing of its own."""
    rng = np.random.default_rng(seed)
    zs, za, zb = rng.normal(size=(n, 2)), rng.normal(size=(n, 2)), rng.normal(size=(n, 1))
    sp = {"a": _space("a", _embed(rng, np.hstack([zs, za]), 7)),
          "b": _space("b", _embed(rng, np.hstack([zs, zb]), 5)),
          "c": _space("c", _embed(rng, zs, 4))}
    return sp, {"s": zs, "a": za, "b": zb}


def _objects(sp, **kw):
    bm = fit_block_map(sp, list(sp), np.arange(len(next(iter(sp.values())).labels)))
    W = bm.concat(sp)
    objs, info = su.block_subspaces(bm, W, **kw)
    return bm, W, objs, info


def test_planted_shared_and_unique_subspaces_are_recovered():
    sp, z = _planted()
    bm, W, objs, info = _objects(sp)
    assert [bm.whiteners[m].rank for m in sp] == [4, 3, 2]
    # three members share z_s at r = 1: eigenvalue 3; everything else is private: eigenvalue 1
    np.testing.assert_allclose(bm.block_eigenvalues[:2], 3.0, atol=0.05)
    assert list(objs) == ["S", "U:a", "U:b", "U:c", "U"]
    assert {n: o.dim for n, o in objs.items()} == {"S": 2, "U:a": 2, "U:b": 1, "U:c": 0, "U": 3}
    assert info["shared_only"] == ["c"] and info["skipped"] == {} and info["read"] == ["a", "b", "c"]
    for name, latent in (("S", z["s"]), ("U:a", z["a"]), ("U:b", z["b"])):
        assert su.span_overlap(objs[name].scores(bm, W), latent)["msc"] > 0.999, name
    union = np.hstack([z["a"], z["b"]])
    assert su.span_overlap(objs["U"].scores(bm, W), union)["msc"] > 0.999
    assert su.span_overlap(objs["S"].scores(bm, W), objs["U"].scores(bm, W))["msc"] < 1e-3
    assert objs["S"].detail["next_eigenvalue"] == pytest.approx(1.0, abs=0.1)


def test_scores_are_uncorrelated_unit_variance_on_the_fit_rows():
    sp, _ = _planted(1)
    bm, W, objs, _ = _objects(sp)
    for name, o in objs.items():
        if o.dim:
            S = o.scores(bm, W)
            np.testing.assert_allclose(S.mean(axis=0), 0, atol=1e-10)
            np.testing.assert_allclose(np.cov(S, rowvar=False, bias=True).reshape(o.dim, o.dim), np.eye(o.dim),
                                       atol=1e-8, err_msg=name)


def test_the_unique_share_is_the_residual_share_of_variance():
    # d's one column is 0.6 z_s + 0.8 z_d: 0.64 of its variance lies outside the other members
    rng = np.random.default_rng(2)
    zs, zd = rng.normal(size=(N, 1)), rng.normal(size=(N, 1))
    zs = (zs - zs.mean()) / zs.std()
    zd = zd - zd.mean()
    zd = zd - zs * (zs.T @ zd) / (zs.T @ zs)  # exactly uncorrelated on these rows
    zd /= zd.std()
    sp = {"e": _space("e", _embed(rng, zs, 3)), "d": _space("d", 0.6 * zs + 0.8 * zd)}
    for cut, dim in ((0.6, 1), (0.7, 0)):
        _, _, objs, _ = _objects(sp, unique=cut)
        assert objs["U:d"].dim == dim, cut
    _, _, objs, _ = _objects(sp)
    assert objs["U:d"].detail["unique_shares"][0] == pytest.approx(0.64, abs=1e-6)


def test_a_member_absent_from_most_rows_gets_no_unique_subspace_and_s_still_places_its_rows():
    sp, _ = _planted(3)
    rng = np.random.default_rng(3)
    X = sp["b"].X.copy()
    gone = rng.random(N) < 0.7
    X[gone] = np.nan
    sp["b"] = _space("b", X)
    bm = fit_block_map(sp, list(sp), np.arange(N))
    W = bm.concat(sp)
    objs, info = su.block_subspaces(bm, W)
    assert "U:b" not in objs and set(info["skipped"]) == {"b"} and info["skipped"]["b"] == pytest.approx(gone.mean())
    S = objs["S"].scores(bm, W)
    assert np.isfinite(S[gone]).all()  # a and c determine the shared directions without b
    assert np.isfinite(objs["U:a"].scores(bm, W)).all()  # b is not read by the unique objects


def test_a_non_pr_block_is_refused():
    sp, _ = _planted(4)
    bm = fit_block_map(sp, list(sp), np.arange(N), member_scaling="pcs")
    with pytest.raises(SpaceError, match="pr"):
        su.shared_subspace(bm)


def test_span_overlap_and_the_random_reference():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(N, 3))
    assert su.span_overlap(A, A @ rng.normal(size=(3, 3)))["msc"] == pytest.approx(1.0)
    assert su.span_overlap(A, rng.normal(size=(N, 3)))["msc"] < 0.02
    assert su.span_overlap(A, np.zeros((N, 0)))["msc"] is None
    X = rng.normal(size=(N, 40))
    lo, hi = su.random_span_msc(X, 4, 20), su.random_span_msc(X, 36, 20)
    assert 0.03 < lo < 0.25 and hi > 0.8  # random spans overlap by dimensionality alone
    assert su.random_span_msc(X, 0) is None
    assert su.random_span_msc(X, 4, 20, seed=1) == su.random_span_msc(X, 4, 20, seed=1)


def test_unit_halves_keep_a_unit_together_and_are_seeded():
    labels = [f"ext-x-{u:03d}|{t}" for u in range(400) for t in (0.25, 0.75)]
    h = su.unit_halves(labels, 0)
    assert (h[::2] == h[1::2]).all()
    assert 0.4 < h.mean() < 0.6
    assert (su.unit_halves(labels, 0) == h).all() and (su.unit_halves(labels, 1) != h).any()


def _halves(sp, seed=0):
    labels = sp["a"].labels
    half = su.unit_halves(labels, seed)
    out = []
    for h in (0, 1):
        bm = fit_block_map(sp, list(sp), np.flatnonzero(half == h))
        W = bm.concat(sp)
        objs, _ = su.block_subspaces(bm, W, rows=half == h, basis=False)
        out.append((objs, bm, W))
    return out


def test_the_gate_passes_planted_objects_and_marks_dimensionless_ones():
    sp, _ = _planted(6)
    bm, W, objs, _ = _objects(sp)
    g = su.reproducibility(_halves(sp), W, n_random=10)
    g = g.set_index("object")
    for name in ("S", "U:a", "U:b", "U"):
        assert g.loc[name, "passed"] is True and g.loc[name, "msc"] > 0.98, name
        assert g.loc[name, "random_msc"] < 0.75
    assert g.loc["U:c", "passed"] is None and g.loc["U:c", "reason"] == "no dimensions in either half"


def test_the_gate_fails_a_span_that_does_not_reproduce_and_one_that_overlaps_by_dimension_alone():
    sp, _ = _planted(7)
    halves = _halves(sp)
    W = halves[0][2]
    V = np.linalg.svd(W - W.mean(0), full_matrices=False)[2].T  # the block's 5 identified directions first

    def read(name, cols):
        return su.Subspace(name=name, kind="unique", reads=["a", "b", "c"], weights=V[:, cols],
                           center=np.zeros(W.shape[1]))

    for (objs, _, _), a, b in zip(halves, ([0, 1], [2, 3]), ([0, 1, 2, 3], [1, 2, 3, 4])):
        objs["apart"], objs["crowded"] = read("apart", a), read("crowded", b)
    # the halves' maps differ a little, so read each half's object on half 0's coordinates
    halves = [(halves[0][0], halves[0][1], W), (halves[1][0], halves[0][1], W)]
    g = su.reproducibility(halves, W, n_random=20).set_index("object")
    assert g.loc["apart", "passed"] is False and g.loc["apart", "msc"] < 0.01
    assert g.loc["apart", "reason"] == "msc < 0.5"
    # two 4-dim spans of a 5-dim block share 3 dimensions: msc .75, but random 4-dim pairs do as well
    assert g.loc["crowded", "msc"] == pytest.approx(0.75, abs=1e-6)
    assert g.loc["crowded", "passed"] is False and g.loc["crowded", "reason"] == "msc - random < 0.25"
    halves[1][0]["noise"] = read("noise", [0])
    # an object present in one half only cannot pass
    g = su.reproducibility(halves, W, n_random=5).set_index("object")
    assert g.loc["noise", "passed"] is False and g.loc["noise", "reason"] == "dimensions in one half only"


def test_draws_are_orthonormal_inside_their_parent_and_paired_by_index():
    sp, _ = _planted(10)
    bm, W, objs, _ = _objects(sp)
    draws, info = su.draw_subspaces(objs, bm, W, n=6, seed=0)
    assert info["d"] == 2 and info["per_member_dim"] == {"a": 2, "b": 1}  # min(dim S 2, dim U 3)
    names = set(draws)
    assert {f"S@d{r:03d}" for r in range(6)} <= names and {f"ref@d{r:03d}" for r in range(6)} <= names
    assert {f"S@b.d{r:03d}" for r in range(6)} <= names and not any(n.startswith("S@c.") for n in names)
    Sx, Ux = objs["S"].scores(bm, W), objs["U"].scores(bm, W)
    for name, o in draws.items():
        X = o.scores(bm, W)
        np.testing.assert_allclose(np.cov(X, rowvar=False, bias=True).reshape(o.dim, o.dim), np.eye(o.dim),
                                   atol=1e-8, err_msg=name)
        parent = {"S": Sx, "U": Ux}.get(name[0]) if not name.startswith("ref") else None
        if parent is not None:
            assert su.span_overlap(X, parent)["msc"] == pytest.approx(1.0), name  # inside its parent
    # S has exactly d dims, so every S draw is S itself; U (3 dims) draws differ
    assert su.span_overlap(draws["S@d000"].scores(bm, W), Sx)["msc"] == pytest.approx(1.0)
    assert su.span_overlap(draws["U@d000"].scores(bm, W), draws["U@d001"].scores(bm, W))["msc"] < 0.999
    again, _ = su.draw_subspaces(objs, bm, W, n=6, seed=0)
    np.testing.assert_array_equal(again["U@d003"].basis, draws["U@d003"].basis)


def test_save_and_load_round_trip(tmp_path):
    sp, _ = _planted(8)
    bm, W, objs, _ = _objects(sp)
    npz, js = su.save_subspaces(objs, {"block": "T"}, tmp_path, stem="T_sub")
    back, doc = su.load_subspaces(js)
    assert doc["schema_version"] == su.SUBSPACES_SCHEMA_VERSION and doc["block"] == "T"
    assert list(back) == list(objs)
    for n in objs:
        np.testing.assert_allclose(back[n].scores(bm, W), objs[n].scores(bm, W), atol=1e-12, err_msg=n)
    with pytest.raises(SpaceError, match="orthonormalise"):
        su.save_subspaces({"S": su.shared_subspace(bm)}, {}, tmp_path, stem="x")


def test_fit_block_keeps_its_rows_and_projection_places_them_as_the_fit_saw_them():
    sp, _ = _planted(9)
    fit = fit_block(sp, list(sp), block="T", fixed_k=5, score=False, keep_rows=True)
    assert fit.rows is not None and list(fit.rows) == list(fit.members) and fit.row_weights is None
    assert all(fit.rows[m].labels == fit.labels for m in fit.members)
    W = fit.map.concat(fit.rows)
    objs, _ = su.block_subspaces(fit.map, W)
    placed, labels = su.project_subspaces(objs, fit, sp)
    assert labels == fit.labels
    for n, o in objs.items():
        np.testing.assert_allclose(placed[n], o.scores(fit.map, W), atol=1e-10, err_msg=n)
    assert fit_block(sp, list(sp), block="T", fixed_k=5, score=False).rows is None


# --- `psytwill space subspaces` ---------------------------------------------

N_STIM, N_BINS = 60, 5


def _frames_table(path, corpus, seed):
    """Long-form frames table: a, b, c share two latents; a holds one alone, c none (as in _planted)."""
    rng = np.random.default_rng(seed)
    n = N_STIM * N_BINS
    zs, za = rng.normal(size=(n, 2)), rng.normal(size=(n, 1))
    X = {"a": _embed(rng, np.hstack([zs, za]), 5), "b": _embed(rng, zs, 3), "c": _embed(rng, zs, 4)}
    rows = []
    for i in range(n):
        sid, b = f"ext-{corpus}-{i // N_BINS:03d}", i % N_BINS
        for model, M in X.items():
            for j, v in enumerate(M[i]):
                rows.append((sid, 0.25 + 0.5 * b, model, f"{model}_{j:02d}", float(v)))
    df = pd.DataFrame(rows, columns=["stimulus_id", "time", "model", "feature", "value"])
    df["value_str"] = pd.Series([None] * len(df), dtype="string")
    df["modality"], df["extractor"], df["extractor_version"] = "audio", "testx", "0.0"
    df.to_parquet(path, index=False)
    return path


def test_space_subspaces_writes_the_frozen_fits_objects_and_their_gate(tmp_path, capsys):
    table = _frames_table(tmp_path / "librispeech_audio_frames_features.parquet", "librispeech", 0)
    out = tmp_path / "space"
    assert main(["space", "fit", "--features", str(table), "--key", "stimulus_id,time", "--window", "0.5",
                 "--groups-from-label", "--members", "a,b,c", "--block", "T", "--fixed-k", "3",
                 "--n-splits", "3", "--n-perm", "150", "--k-nn", "10", "--eval-n", "0", "--seed", "0",
                 "-o", str(out), "--stem", "T"]) == 0
    sub = tmp_path / "sub"
    capsys.readouterr()
    assert main(["space", "subspaces", "--like", str(out / "T.json"), "--n-random", "5", "-o", str(sub)]) == 0
    printed = capsys.readouterr().out
    assert "T_subspaces.json" in printed and "PASS" in printed
    doc = json.loads((sub / "T_subspaces.json").read_text())
    assert doc["refit_vs_frozen"]["msc"] == pytest.approx(1.0)
    dims = {o["name"]: o["dim"] for o in doc["objects"] if "@" not in o["name"]}
    assert dims == {"S": 2, "U:a": 1, "U:b": 0, "U:c": 0, "U": 1}
    assert doc["draws"]["d"] == 1 and sum("@" in o["name"] for o in doc["objects"]) == 4 * 20  # S, U, ref + S@a
    assert doc["unique_info"]["shared_only"] == ["b", "c"]
    assert [h["half"] for h in doc["gate_settings"]["halves"]] == [0, 1]
    gate = pd.read_csv(sub / "T_subspaces_gate.csv").set_index("object")
    assert bool(gate.loc["S", "passed"]) and bool(gate.loc["U:a", "passed"])
    objs, _ = su.load_subspaces(sub / "T_subspaces.json")
    assert objs["S"].n_block == 2 and objs["S"].dim == 2
    assert load_fit(out / "T.json").map.k_max >= objs["S"].n_block


def test_space_subspaces_refuses_a_frozen_fit_whose_inputs_changed(tmp_path, capsys):
    table = _frames_table(tmp_path / "librispeech_audio_frames_features.parquet", "librispeech", 0)
    out = tmp_path / "space"
    assert main(["space", "fit", "--features", str(table), "--key", "stimulus_id,time", "--window", "0.5",
                 "--members", "a,b,c", "--block", "T", "--fixed-k", "3", "--no-criterion",
                 "-o", str(out), "--stem", "T"]) == 0
    df = pd.read_parquet(table)
    df[~df["stimulus_id"].str.endswith("000")].to_parquet(table, index=False)  # one unit gone
    capsys.readouterr()
    assert main(["space", "subspaces", "--like", str(out / "T.json"), "--no-gate", "-o", str(tmp_path / "s")]) == 1
    assert "inputs changed" in capsys.readouterr().err
