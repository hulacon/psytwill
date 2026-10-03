"""Relations between fitted blocks: a frozen CV-CCA map, pinned to its inputs.

What these pin: the shared prefix is recovered and nothing is invented (k = 0
is refused, not saved); the frozen map reproduces itself on its own rows;
pooling is declared, never inferred; and a score table projected through a
different block cannot be read through the relation. Plus a regression on
`cv_cca`, whose fold loop now runs through the shared `fit_cca`.
"""

import json

import numpy as np
import pandas as pd
import pytest

from psytwill.cli import main
from psytwill.decompose import cv_cca, fit_cca
from psytwill.exceptions import SpaceError
from psytwill.relate import (
    check_relation,
    fit_relation,
    load_relation,
    read_scores,
    relate_tables,
    save_relation,
)

N_ITEMS, K_SHARED, D_A, D_B, CAPS = 400, 3, 12, 10, 4


def _latents(seed=0, n=N_ITEMS, k=K_SHARED, d_a=D_A, d_b=D_B, noise=0.15):
    rng = np.random.RandomState(seed)
    z = rng.randn(n, k)
    A = np.hstack([z, rng.randn(n, d_a - k)]) @ rng.randn(d_a, d_a) + noise * rng.randn(n, d_a)
    B = np.hstack([z, rng.randn(n, d_b - k)]) @ rng.randn(d_b, d_b)
    return A, B, rng


def _write_block(tmp_path, block, X, key_cols: dict, tag="v1"):
    """A score table + sidecar + the (stand-in) block manifest it names."""
    manifest = tmp_path / f"{block}_{tag}.json"
    manifest.write_text(json.dumps({"block": block, "tag": tag}))
    df = pd.DataFrame(X, columns=[f"{block}_{j:03d}" for j in range(X.shape[1])])
    for j, (name, vals) in enumerate(key_cols.items()):
        df.insert(j, name, vals)
    path = tmp_path / f"{block}_{tag}_scores.parquet"
    df.to_parquet(path, index=False)
    (tmp_path / f"{block}_{tag}_scores.meta.json").write_text(json.dumps(
        {"space": str(manifest), "block": block, "k": X.shape[1], "key": ",".join(key_cols), "rows": len(df)}))
    return path


@pytest.fixture
def tables(tmp_path):
    """Side a: one row per item. Side b: CAPS noisy 'captions' per item."""
    A, B, rng = _latents()
    ids = [f"ext-test-{i:04d}" for i in range(N_ITEMS)]
    a = _write_block(tmp_path, "P", A, {"stimulus_id": ids})
    Bc = np.repeat(B, CAPS, axis=0) + 0.1 * rng.randn(N_ITEMS * CAPS, D_B)
    b = _write_block(tmp_path, "Q", Bc, {"stimulus_id": np.repeat(ids, CAPS), "chunk_idx": np.tile(np.arange(CAPS), N_ITEMS)})
    return a, b, A, Bc


# --- the count and the refusal ---------------------------------------------


def test_planted_prefix_is_recovered():
    A, B, _ = _latents()
    fit = fit_relation(A, B, name="PQ", n_perm=50)
    assert fit.k == K_SHARED
    assert fit.map.U.shape[1] == K_SHARED and fit.map.Vt.shape[0] == K_SHARED
    r = fit.manifest["r_cv"]
    assert min(r[:K_SHARED]) >= fit.manifest["r_min"] > r[K_SHARED]


def test_independent_blocks_are_refused_not_saved():
    rng = np.random.RandomState(1)
    with pytest.raises(SpaceError, match="measured shared k is 0"):
        fit_relation(rng.randn(300, 8), rng.randn(300, 6), name="PQ", n_perm=20)


def test_prefix_not_count():
    """A component past the first failure is never kept, even if it passes."""
    A, B, _ = _latents()
    hi = fit_relation(A, B, name="PQ", n_perm=0)
    r2 = hi.manifest["r_cv"][1]
    fit = fit_relation(A, B, name="PQ", n_perm=0, r_min=r2 + 1e-9)  # component 2 now fails
    assert fit.k == 1


def test_subspace_stability_is_reported_and_high_when_planted():
    A, B, _ = _latents()
    st = fit_relation(A, B, name="PQ", n_perm=0).manifest["subspace_stability"]
    for side in "ab":
        assert len(st[side]["per_fold"]) == 5
        assert 0.95 < st[side]["min"] <= st[side]["mean"] <= 1.0


# --- the frozen map ---------------------------------------------------------


def test_round_trip_reproduces_the_fit(tmp_path):
    A, B, _ = _latents()
    fit = fit_relation(A, B, name="PQ", n_perm=0)
    _, manifest = save_relation(fit, tmp_path, stem="PQ_v0.1")
    back = load_relation(manifest)
    assert back.k == fit.k
    np.testing.assert_allclose(back.project(A, "a"), fit.project(A, "a"))
    np.testing.assert_allclose(back.project(B, "b"), fit.project(B, "b"))
    res = check_relation(back, A, B, n_perm=20)
    np.testing.assert_allclose(res.r, back.manifest["r_insample"], atol=1e-10)
    assert res.prefix == res.count == fit.k


def test_check_scores_held_out_rows_without_refit():
    A, B, _ = _latents(n=600)
    fit = fit_relation(A[:500], B[:500], name="PQ", n_perm=0)
    res = check_relation(fit, A[500:], B[500:], n_perm=50)
    assert res.n == 100 and res.prefix == K_SHARED
    assert (res.r > res.null_q).all()


# --- pairing and provenance -------------------------------------------------


def test_repeated_keys_need_a_declared_pool(tables):
    a, b, _, _ = tables
    with pytest.raises(SpaceError, match="Declare how they pool"):
        relate_tables(read_scores(a), read_scores(b), name="PQ", join=["stimulus_id"], n_perm=0)


def test_mean_pool_matches_hand_pooling(tables):
    a, b, A, Bc = tables
    fit = relate_tables(read_scores(a), read_scores(b), name="PQ", join=["stimulus_id"], pool_b="mean", n_perm=0)
    hand = fit_relation(A, Bc.reshape(N_ITEMS, CAPS, D_B).mean(axis=1), name="PQ", n_perm=0)
    assert fit.k == hand.k
    np.testing.assert_allclose(fit.manifest["r_cv"], hand.manifest["r_cv"], atol=1e-10)
    assert fit.manifest["pairing"]["rows_per_key_b"] == CAPS
    assert fit.manifest["sides"]["b"]["pool"] == "mean"


def test_cli_fit_check_project_and_provenance(tables, tmp_path, capsys):
    a, b, A, _ = tables
    out = tmp_path / "rel"
    assert main(["space", "relate", "fit", "--a", str(a), "--b", str(b), "--name", "PQ", "--pool-b", "mean",
                 "--n-perm", "20", "--scope", "test pairing", "-o", str(out), "--stem", "PQ_v0.1"]) == 0
    m = json.loads((out / "PQ_v0.1.json").read_text())
    assert m["k"] == K_SHARED and m["scope"] == "test pairing"
    assert m["sides"]["a"]["block"] == "P" and len(m["sides"]["a"]["manifest_sha256"]) == 64

    assert main(["space", "relate", "check", "--relation", str(out / "PQ_v0.1.json"), "--a", str(a), "--b", str(b),
                 "--n-perm", "20", "-o", str(out / "check.csv")]) == 0
    assert len(pd.read_csv(out / "check.csv")) == K_SHARED

    proj = out / "P_in_PQ.parquet"
    assert main(["space", "relate", "project", "--relation", str(out / "PQ_v0.1.json"), "--side", "a",
                 "--scores", str(a), "-o", str(proj)]) == 0
    P = pd.read_parquet(proj)
    assert list(P.columns) == ["stimulus_id"] + [f"PQ_{j:03d}" for j in range(K_SHARED)]
    assert len(P) == N_ITEMS

    # a table projected through a different block manifest is refused
    other = _write_block(tmp_path, "P", A, {"stimulus_id": P.stimulus_id.to_numpy()}, tag="v2")
    capsys.readouterr()
    assert main(["space", "relate", "project", "--relation", str(out / "PQ_v0.1.json"), "--side", "a",
                 "--scores", str(other), "-o", str(out / "x.parquet")]) == 1
    assert "Re-project the table" in capsys.readouterr().err
    # and so is the right table on the wrong side
    assert main(["space", "relate", "project", "--relation", str(out / "PQ_v0.1.json"), "--side", "b",
                 "--scores", str(a), "-o", str(out / "y.parquet")]) == 1


# --- cv_cca through the shared fit -------------------------------------------


def test_cv_cca_unchanged_by_fit_cca_refactor():
    """Values pinned from psytwill 0.30.1, before the fold loop called fit_cca."""
    rng = np.random.RandomState(0)
    z = rng.randn(500, 3)
    A = np.hstack([z, rng.randn(500, 17)]) @ rng.randn(20, 20) + 0.3 * rng.randn(500, 20)
    B = np.hstack([z, rng.randn(500, 27)]) @ rng.randn(30, 30) + 0.3 * rng.randn(500, 30)
    g = np.repeat(np.arange(25), 20)
    ung = cv_cca(A, B, n_perm=50, rank_cap=16, random_state=0)
    grp = cv_cca(A, B, groups=g, n_perm=50, rank_cap=16, random_state=0)
    np.testing.assert_allclose(ung.r_cv[:4], [0.48141377792295853, 0.49675668675239154,
                                              0.2793962651393823, 0.02345702065099453], rtol=1e-9)
    np.testing.assert_allclose(grp.r_cv[:4], [0.4845029201444371, 0.48697230332125796,
                                              0.3097067814164539, 0.08176737766990819], rtol=1e-9)
    np.testing.assert_allclose(ung.r_train[:2], [0.6164819487849129, 0.5746333577106509], rtol=1e-9)
    assert ung.shared_dims == grp.shared_dims == 3


def test_fit_cca_variates_are_unit_variance_and_paired():
    A, B, _ = _latents()
    cm = fit_cca(A, B, rank_cap=64)
    Pa, Pb = cm.transform_a(A, 3), cm.transform_b(B, 3)
    np.testing.assert_allclose(Pa.std(axis=0, ddof=1), 1.0, atol=1e-8)
    r = [np.corrcoef(Pa[:, j], Pb[:, j])[0, 1] for j in range(3)]
    n = len(A)
    np.testing.assert_allclose(r, cm.r[:3] * n / (n - 1), atol=1e-8)  # see CcaMap.r


def test_fixed_k_freezes_the_map_and_still_records_the_prefix():
    A, B, _ = _latents()
    measured = fit_relation(A, B, name="PQ", n_perm=0)
    k = measured.k
    fixed = fit_relation(A, B, name="PQ", n_perm=0, fixed_k=k + 2)
    assert fixed.k == k + 2 and fixed.manifest["fixed_k"] == k + 2
    assert fixed.manifest["prefix_k"] == k and measured.manifest["prefix_k"] == k
    assert fixed.manifest["k_rule"].startswith("fixed")
    with pytest.raises(SpaceError, match="outside"):
        fit_relation(A, B, name="PQ", n_perm=0, fixed_k=10_000)
    # a fixed k is honoured even where nothing is shared, so a baseline is never refused for it
    rng = np.random.RandomState(1)
    none = fit_relation(rng.randn(300, 8), rng.randn(300, 6), name="PQ", n_perm=0, fixed_k=2)
    assert none.k == 2 and none.manifest["prefix_k"] == 0
