"""Per-direction member energy, universality and split-half reliability of a block, on planted structure."""

from types import SimpleNamespace

import numpy as np
import pytest

from psytwill.directions import direction_reliability, direction_table, member_energy, universality
from psytwill.exceptions import SpaceError
from psytwill.space import fit_block_map
from psytwill.store import SpaceMatrix

N = 800


def _stub(V, widths, eig=None):
    members = [f"m{i}" for i in range(len(widths))]
    return SimpleNamespace(
        members=members,
        whiteners={m: SimpleNamespace(rank=w) for m, w in zip(members, widths)},
        block_components=np.asarray(V, float),
        block_eigenvalues=np.arange(len(V), 0, -1, dtype=float) if eig is None else np.asarray(eig, float),
    )


def test_energy_and_u_worked_by_hand():
    s = 1 / np.sqrt(2)
    V = np.array([
        [1, 0, 0, 0],      # all on m0 (width 2)
        [0, 0, s, s],      # all on m1 (width 2): spread inside one member is still one member
        [s, 0, s, 0],      # half each
        [0.6, 0.0, 0.8, 0.0],  # 0.36 / 0.64
    ])
    b = _stub(V, [2, 2])
    E = member_energy(b)
    assert list(E.columns) == ["component", "member", "width", "energy"]
    mat = E["energy"].to_numpy().reshape(4, 2)
    np.testing.assert_allclose(mat, [[1, 0], [0, 1], [0.5, 0.5], [0.36, 0.64]], atol=1e-12)
    u = universality(mat)
    np.testing.assert_allclose(u[:3], [0.5, 0.5, 1.0], atol=1e-12)
    assert u[3] == pytest.approx(1 / (0.36 ** 2 + 0.64 ** 2) / 2)
    t = direction_table(b)
    assert list(t.columns) == ["component", "eigenvalue", "u"]
    assert t["component"].tolist() == [0, 1, 2, 3]
    assert direction_table(b, k=2)["u"].tolist() == pytest.approx([0.5, 0.5])


def test_u_runs_from_one_over_m_to_one():
    M = 5
    assert universality(np.eye(M)) == pytest.approx(np.full(M, 1 / M))
    assert universality(np.full((1, M), 0.2)) == pytest.approx([1.0])
    rng = np.random.default_rng(0)
    u = universality(rng.dirichlet(np.ones(M), size=200))
    assert (u >= 1 / M - 1e-12).all() and (u <= 1 + 1e-12).all()


def _space(name, X):
    return SpaceMatrix(name=name, labels=[f"s{i:04d}" for i in range(X.shape[0])], X=X,
                       features=[f"{name}_{j:02d}" for j in range(X.shape[1])])


def test_on_a_fitted_block_a_shared_latent_is_universal_and_a_private_one_is_not():
    # One latent a and b share, one c holds alone. Their block eigenvalues differ (~2 vs ~1), so the PCA
    # identifies each direction. Two private latents of equal strength would form a degenerate
    # eigenspace whose basis, and so whose u, the PCA leaves arbitrary (see the next test).
    rng = np.random.default_rng(1)
    zs, zc = rng.normal(size=(N, 1)), rng.normal(size=(N, 1))

    def emb(Z, dim):  # noiseless, so each member whitens to rank 1: a noise direction whitened to unit
        return Z @ rng.normal(size=(Z.shape[1], dim))  # variance would sit at c's eigenvalue (~1) and mix with it

    sp = {"a": _space("a", emb(zs, 6)), "b": _space("b", emb(zs, 5)), "c": _space("c", zc)}
    assert [fit_block_map(sp, [m], np.arange(N)).whiteners[m].rank for m in sp] == [1, 1, 1]
    bm = fit_block_map(sp, ["a", "b", "c"], np.arange(N))
    E = member_energy(bm).pivot(index="component", columns="member", values="energy")
    t = direction_table(bm)
    assert np.allclose(E.sum(axis=1), 1.0)
    # the leading direction is the shared latent: split between a and b, none on c
    assert E.loc[0, "c"] < 0.01 and min(E.loc[0, "a"], E.loc[0, "b"]) > 0.45
    assert t.loc[0, "u"] == pytest.approx(2 / 3, abs=0.01)
    # c's private latent: one member carries it
    private = E[E["c"] > 0.99].index
    assert len(private) == 1
    assert t.loc[private[0], "u"] == pytest.approx(1 / 3, abs=0.01)
    assert t.loc[private[0], "eigenvalue"] < t.loc[0, "eigenvalue"]


def test_u_inside_a_degenerate_eigenspace_depends_on_the_basis():
    # Disclosed property, not a defect of the formula: two private directions of equal variance span an
    # eigenspace the PCA may return in any rotation, and u changes with it.
    s = 1 / np.sqrt(2)
    axes = _stub(np.eye(2), [1, 1])
    rotated = _stub([[s, s], [s, -s]], [1, 1])
    assert direction_table(axes)["u"].tolist() == pytest.approx([0.5, 0.5])
    assert direction_table(rotated)["u"].tolist() == pytest.approx([1.0, 1.0])


def test_reliability_is_high_for_a_stable_direction_and_low_for_noise():
    rng = np.random.default_rng(2)
    stable = rng.normal(size=(N, 2)) * [3.0, 2.0]
    ref = np.hstack([stable, rng.normal(size=(N, 1))])
    # each half reproduces the two stable directions (sign-flipped, with a little noise) but not the third
    half_a = np.hstack([-stable[:, :1], stable[:, 1:]]) + 0.1 * rng.normal(size=(N, 2))
    half_a = np.hstack([half_a, rng.normal(size=(N, 1))])
    half_b = np.hstack([stable + 0.1 * rng.normal(size=(N, 2)), rng.normal(size=(N, 1))])
    for how in ("procrustes", "sign"):
        rho = direction_reliability(ref, half_a, half_b, align=how)
        assert (rho[:2] > 0.99).all(), how
        assert abs(rho[2]) < 0.15, how


def test_sign_alignment_is_strict_where_procrustes_absorbs_a_swap():
    rng = np.random.default_rng(3)
    S = rng.normal(size=(N, 2))
    ref = S.copy()
    half_a = S + 0.05 * rng.normal(size=(N, 2))
    half_b = S[:, ::-1] + 0.05 * rng.normal(size=(N, 2))  # two near-equal eigenvalues swapped order
    assert (direction_reliability(ref, half_a, half_b) > 0.99).all()
    assert (np.abs(direction_reliability(ref, half_a, half_b, align="sign")) < 0.15).all()


def test_table_takes_rho_and_rows_with_nan_are_dropped():
    b = _stub(np.eye(3), [1, 1, 1])
    t = direction_table(b, rho=[0.9, 0.5, 0.1])
    assert t["rho"].tolist() == [0.9, 0.5, 0.1]
    rng = np.random.default_rng(4)
    S = rng.normal(size=(50, 2))
    A = S.copy()
    A[3, 1] = np.nan
    assert np.isfinite(direction_reliability(S, A, S)).all()


@pytest.mark.parametrize("call, msg", [
    (lambda: direction_table(_stub(np.eye(3), [1, 1, 1]), rho=[0.1, 0.2]), "rho has shape"),
    (lambda: member_energy(_stub(np.eye(3), [1, 1])), "ranks sum to 2"),
    (lambda: universality(np.zeros((2, 3))), "zero energy"),
    (lambda: universality(-np.eye(2)), "non-negative"),
    (lambda: direction_reliability(np.zeros((5, 2)), np.zeros((5, 3)), np.zeros((5, 2))), "one shape"),
    (lambda: direction_reliability(np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2)), align="x"), "align"),
])
def test_refusals_name_the_problem(call, msg):
    with pytest.raises(SpaceError, match=msg):
        call()

