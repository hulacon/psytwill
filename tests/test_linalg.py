"""psytwill.linalg.thin_svd: the gesvd fallback for whitener SVDs."""

import numpy as np
import pytest

from psytwill import decompose, linalg


def _fail_gesdd(monkeypatch):
    real = np.linalg.svd

    def svd(*a, **kw):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(np.linalg, "svd", svd)
    return real


def test_thin_svd_is_numpy_when_gesdd_converges():
    M = np.random.default_rng(0).normal(size=(20, 6))
    for a, b in zip(linalg.thin_svd(M), np.linalg.svd(M, full_matrices=False)):
        assert np.array_equal(a, b)


def test_thin_svd_falls_back_to_gesvd_and_warns(monkeypatch):
    rng = np.random.default_rng(1)
    M = rng.normal(size=(40, 3)) @ rng.normal(size=(3, 12))  # exactly rank 3, the 0.3.0 case
    real = _fail_gesdd(monkeypatch)
    with pytest.warns(RuntimeWarning, match="retried with gesvd"):
        U, s, Vt = linalg.thin_svd(M)
    _, s0, _ = real(M, full_matrices=False)
    assert U.shape == (40, 12) and Vt.shape == (12, 12)
    assert np.allclose(s, s0) and np.allclose((U * s) @ Vt, M)
    with pytest.warns(RuntimeWarning):
        assert np.allclose(linalg.thin_svd(M, compute_uv=False), s0)


def test_the_cca_whitener_uses_the_fallback(monkeypatch):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 2)) @ rng.normal(size=(2, 8))
    want = decompose._fit_whitener(X, rank_cap=8)
    _fail_gesdd(monkeypatch)
    with pytest.warns(RuntimeWarning, match="gesvd"):
        got = decompose._fit_whitener(X, rank_cap=8)
    assert got.rank == want.rank == 2
    assert np.allclose(np.abs(got.components), np.abs(want.components)) and np.allclose(got.scale, want.scale)
