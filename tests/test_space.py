"""Private-block fit tests (psytwill.space).

Synthetic members are linear read-outs of one low-rank latent plus noise, so
the fitted block must subsume every member at a small k, and a member built
from an independent latent must fail the criterion.
"""

import numpy as np
import pytest

from psytwill.space import (
    BlockFit,
    check_fit,
    fit_block,
    fit_whitener,
    load_fit,
    save_fit,
)
from psytwill.store import SpaceMatrix

N, LATENT = 600, 4


def _member(name, rng, Z, dim, noise=0.05):
    W = rng.normal(size=(Z.shape[1], dim))
    X = Z @ W + noise * rng.normal(size=(Z.shape[0], dim))
    return SpaceMatrix(name=name, labels=[f"s{i:04d}" for i in range(Z.shape[0])],
                       X=X, features=[f"{name}_{j:03d}" for j in range(dim)])


@pytest.fixture
def members():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(N, LATENT))
    return {
        "a": _member("a", rng, Z, 64),
        "b": _member("b", rng, Z, 12),
        "c": _member("c", rng, Z, 6),
    }, Z


def _fit_kwargs():
    # small null / subset so the suite stays fast; the criterion is unchanged
    return dict(n_splits=3, n_perm=150, eval_n=None, k_nn=10, k_schedule=(2, 4, 8, 16))


class TestWhitener:
    def test_whitened_rank_follows_participation_ratio(self, members):
        sp, _ = members
        w = fit_whitener(sp["a"], sp["a"].X)
        assert 1 <= w.rank <= 8  # 4 latent dims + noise: PR is a little above 4
        Z = w.transform(sp["a"].X)
        assert Z.shape == (N, w.rank)
        cov = np.cov(Z, rowvar=False)
        assert np.allclose(np.diag(cov), 1.0, atol=1e-6)

    def test_constant_columns_do_not_blow_up(self, members):
        sp, _ = members
        X = sp["b"].X.copy()
        X[:, 0] = 3.0
        w = fit_whitener(sp["b"], X)
        assert np.isfinite(w.transform(X)).all()


class TestFit:
    def test_shared_latent_is_subsumed_at_small_k(self, members):
        sp, _ = members
        fit = fit_block(sp, ["a", "b", "c"], block="T", **_fit_kwargs())
        assert fit.manifest["subsumes_all_members"]
        assert fit.k <= 8
        assert fit.manifest["k_below_pr_bound"]
        for m in fit.members:
            assert fit.manifest["per_member"][m]["passed_all_folds"]
        assert all(r["k"] <= fit.k for r in fit.curve)  # walk stopped at the first pass

    def test_independent_member_raises_k(self, members):
        # a block is the union of its members: an unrelated member is still
        # subsumed, but only once k grows to cover its own latent
        sp, _ = members
        rng = np.random.default_rng(1)
        Z2 = rng.normal(size=(N, LATENT))
        sp = dict(sp)
        sp["d"] = _member("d", rng, Z2, 10)
        alone = fit_block(sp, ["a"], block="T", r2_min=0.9, **_fit_kwargs())
        both = fit_block(sp, ["a", "d"], block="T", r2_min=0.9, **_fit_kwargs())
        assert both.manifest["subsumes_all_members"]
        assert both.k > alone.k
        assert both.k >= 2 * LATENT

    def test_lenient_r2_retains_fewer_dims(self, members):
        # the pre-registered R^2 >= .5 accepts half the latent; a stricter
        # threshold pushes k up to the true latent rank — the threshold is
        # part of the retention rule, not just the verdict
        sp, _ = members
        kw = _fit_kwargs()
        lenient = fit_block(sp, ["a", "b", "c"], block="T", r2_min=0.5, **kw)
        strict = fit_block(sp, ["a", "b", "c"], block="T", r2_min=0.9, **kw)
        assert lenient.k <= strict.k
        assert strict.k >= LATENT

    def test_project_and_roundtrip(self, members, tmp_path):
        sp, Z = members
        fit = fit_block(sp, ["a", "b", "c"], block="T", r2_min=0.9, **_fit_kwargs())
        S, labels = fit.project(sp)
        assert S.shape == (N, fit.k)
        assert labels == sp["a"].labels
        # block scores span the latent: ridge from scores recovers Z
        from sklearn.linear_model import Ridge
        r2 = Ridge(alpha=1e-3).fit(S, Z).score(S, Z)
        assert r2 > 0.95
        npz, manifest, curve = save_fit(fit, tmp_path)
        assert npz.exists() and manifest.exists() and curve.exists()
        back = load_fit(manifest)
        assert isinstance(back, BlockFit)
        S2, _ = back.project(sp)
        assert np.allclose(S, S2)

    def test_check_on_a_fresh_table(self, members, tmp_path):
        sp, _ = members
        fit = fit_block(sp, ["a", "b"], block="T", **_fit_kwargs())
        rng = np.random.default_rng(7)
        Z = rng.normal(size=(200, LATENT))
        # fresh rows from the same generative model, same feature columns
        fresh = {}
        for m in ("a", "b"):
            W = np.linalg.lstsq(np.random.default_rng(0).normal(size=(N, LATENT)), sp[m].X, rcond=None)[0]
            fresh[m] = SpaceMatrix(name=m, labels=[f"f{i}" for i in range(200)], X=Z @ W,
                                   features=sp[m].features)
        rows = check_fit(fit, fresh, n_perm=150, eval_n=None, k_nn=10)
        assert {r["member"] for r in rows} == {"a", "b"}
        assert all(r["passed"] for r in rows)

    def test_feature_mismatch_is_refused(self, members):
        sp, _ = members
        fit = fit_block(sp, ["a"], block="T", **_fit_kwargs())
        bad = {"a": SpaceMatrix(name="a", labels=sp["a"].labels, X=sp["a"].X[:, :10],
                                features=sp["a"].features[:10])}
        with pytest.raises(Exception):
            check_fit(fit, bad, n_perm=150, eval_n=None)


    def test_permutation_floor_is_refused(self, members):
        sp, _ = members
        with pytest.raises(Exception, match="cannot beat alpha"):
            fit_block(sp, ["a"], n_perm=50, alpha=0.01, eval_n=None, n_splits=3)
