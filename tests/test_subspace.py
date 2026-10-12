"""Variance-matched random subspaces (the bottleneck-share control)."""

import numpy as np
import pytest

from psytwill.exceptions import SpaceError
from psytwill.subspace import captured_share, matched_subspaces, route_share

D, K, N = 24, 4, 600


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    scales = np.geomspace(5.0, 0.2, D)  # a steep spectrum: uniform k-subspaces sit near k/d
    return rng.normal(size=(N, D)) * scales @ np.linalg.qr(rng.normal(size=(D, D)))[0]


def _top_pcs(X, k):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T


def test_route_share_of_the_top_pcs_is_their_eigenvalue_share(X):
    Xc = X - X.mean(0)
    ev = np.linalg.eigvalsh(np.cov(Xc, rowvar=False))[::-1]
    Q = _top_pcs(X, K)
    want = ev[:K].sum() / ev.sum()
    assert route_share(X, Xc @ Q) == pytest.approx(want, abs=1e-10)
    # coordinates given in any invertible mix of the route measure the same span
    mix = np.random.default_rng(1).normal(size=(K, K))
    assert route_share(X, Xc @ Q @ mix + 3.0) == pytest.approx(want, abs=1e-10)
    assert captured_share(X, Q) == pytest.approx(want, abs=1e-10)


@pytest.mark.parametrize("which", ["high", "low"])
def test_matches_shares_uniform_draws_cannot_reach(X, which):
    uniform = [captured_share(X, np.linalg.qr(np.random.default_rng(s).normal(size=(D, K)))[0]) for s in range(200)]
    # halfway between the uniform draws' reach and the extreme (top or bottom k PCs, which no draw attains)
    if which == "high":
        target = (max(uniform) + captured_share(X, _top_pcs(X, K))) / 2
    else:
        target = (min(uniform) + captured_share(X, _top_pcs(X, D)[:, -K:])) / 2
    assert not (min(uniform) <= target <= max(uniform)), "fixture should put the target outside uniform reach"
    m = matched_subspaces(X, K, target, n=50, n_pool=2000, tol=0.02, random_state=0)
    assert len(m.bases) == 50 and np.all(np.abs(m.shares - target) <= 0.02)
    for B in m.bases[:5]:
        assert B.shape == (D, K)
        np.testing.assert_allclose(B.T @ B, np.eye(K), atol=1e-10)


def test_same_seed_same_draws_and_a_shared_rng_is_consumed(X):
    a = matched_subspaces(X, K, 0.3, n=10, n_pool=500, tol=0.05, random_state=7)
    b = matched_subspaces(X, K, 0.3, n=10, n_pool=500, tol=0.05, random_state=np.random.RandomState(7))
    np.testing.assert_array_equal(a.shares, b.shares)
    rng = np.random.RandomState(7)
    matched_subspaces(X, K, 0.3, n=10, n_pool=500, tol=0.05, random_state=rng)
    assert rng.standard_normal() != np.random.RandomState(7).standard_normal()


def test_route_carrying_the_signal_beats_matched_random(X):
    """The use it exists for: a route that holds the target's direction stands out from matched draws."""
    rng = np.random.default_rng(2)
    Xc = X - X.mean(0)
    Q = _top_pcs(X, D)[:, 8:8 + K]  # mid-spectrum route
    y = Xc @ Q[:, 0] + 0.1 * rng.normal(size=N)

    def r2_through(B):
        Z = Xc @ B
        beta, *_ = np.linalg.lstsq(Z, y - y.mean(), rcond=None)
        return 1 - ((y - y.mean() - Z @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    m = matched_subspaces(X, K, captured_share(X, Q), n=100, n_pool=3000, tol=0.02, random_state=0)
    rand = np.array([r2_through(B) for B in m.bases])
    assert r2_through(Q) > np.quantile(rand, 0.99)


def test_refuses_what_it_cannot_match(X):
    with pytest.raises(SpaceError, match="brackets"):
        matched_subspaces(X, K, 0.999, n=5, n_pool=50, random_state=0)
    with pytest.raises(SpaceError, match="matching failed"):
        matched_subspaces(X, K, 0.3, n=50, n_pool=50, tol=1e-9, random_state=0)
    with pytest.raises(SpaceError, match="strictly between"):
        matched_subspaces(X, D, 0.5, random_state=0)
