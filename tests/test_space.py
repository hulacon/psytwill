"""Private-block fit tests (psytwill.space).

Synthetic members are linear read-outs of one low-rank latent plus noise, so
the fitted block must subsume every member at a small k, and a member built
from an independent latent must fail the criterion.
"""

import numpy as np
import pytest

from psytwill.space import (
    BlockFit,
    apply_structural_fill,
    detect_structural_columns,
    prepare_member,
    check_fit,
    check_member,
    eval_subsample,
    fit_block,
    fit_whitener,
    load_fit,
    save_fit,
    structural_fill_values,
)
from psytwill.compare import neighbor_overlap_null
from psytwill.exceptions import SpaceError
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


class TestNaNPolicy:
    def test_mostly_undefined_column_is_dropped_and_recorded(self, members):
        sp, _ = members
        X = sp["b"].X.copy()
        X[:, 0] = np.nan
        X[::50, 0] = 1.0  # defined for 2 % of rows
        X[::7, 1] = np.nan  # sparse gaps stay, mean-imputed for the fit
        sp = dict(sp)
        sp["b"] = SpaceMatrix(name="b", labels=sp["b"].labels, X=X, features=sp["b"].features)
        kept, dropped = prepare_member(sp["b"])
        assert dropped == ["b_000"] and kept.dim == 11
        fit = fit_block(sp, ["a", "b"], block="T", r2_min=0.9, **_fit_kwargs())
        assert fit.manifest["per_member"]["b"]["dropped_columns"] == ["b_000"]
        assert fit.manifest["subsumes_all_members"]
        # projecting the original (13-column) table selects the kept columns by name
        S, _ = fit.project(sp)
        assert np.isfinite(S).all() and S.shape[1] == fit.k
        rows = check_fit(fit, sp, n_perm=150, eval_n=None, k_nn=10)
        assert all(r["passed"] for r in rows)


class TestRankMetricRule:
    """Cosine neighbour graphs are degenerate below whitened rank 3.

    In one dimension cosine similarity takes only +/-1, so every pair ties and
    the stable argsort hands back the same index list for every row. The graph
    is then a constant and its overlap with anything is chance. DECIDED
    2026-09-12: score those members with Euclidean neighbours instead.
    """

    def test_metric_selected_by_rank(self):
        from psytwill.space import EUCLIDEAN_RANK_BELOW, metric_for_rank

        assert EUCLIDEAN_RANK_BELOW == 3
        assert [metric_for_rank(r) for r in (1, 2)] == ["euclidean", "euclidean"]
        assert [metric_for_rank(r) for r in (3, 4, 91)] == ["cosine"] * 3

    def test_cosine_graph_is_constant_on_a_one_column_member(self):
        """The defect itself, pinned: an all-positive 1-d member yields one
        neighbour set for (almost) every row, so overlap cannot beat its null."""
        from collections import Counter

        from psytwill.compare import knn_indices

        rng = np.random.default_rng(0)
        y = 5.0 + 2.0 * rng.normal(size=(N, 1))  # an all-positive rating scale
        cos = [tuple(r) for r in knn_indices(y, k=10, metric="cosine")]
        euc = [tuple(r) for r in knn_indices(y, k=10, metric="euclidean")]
        assert Counter(cos).most_common(1)[0][1] / len(cos) > 0.95  # ~one set
        assert len(set(euc)) == len(euc)  # a genuine ordering, all distinct

    def test_rank_one_member_subsumes_under_the_rule(self, members):
        """A 1-d member that is a clean read-out of the shared latent must pass.

        Under cosine it cannot, at any k -- which is what stalled the 2026-09-10
        V fit -- so this is the regression test for the rule.
        """
        sp, Z = members
        rng = np.random.default_rng(7)
        y = Z @ rng.normal(size=Z.shape[1]) + 0.02 * rng.normal(size=Z.shape[0])
        sp = dict(sp)
        sp["d"] = SpaceMatrix(name="d", labels=sp["a"].labels,
                              X=(10.0 + y)[:, None], features=["d_000"])
        fit = fit_block(sp, ["a", "b", "c", "d"], block="V", **_fit_kwargs())
        assert fit.map.whiteners["d"].rank == 1
        assert fit.manifest["per_member"]["d"]["metric"] == "euclidean"
        assert fit.manifest["per_member"]["a"]["metric"] == "cosine"
        assert fit.manifest["per_member"]["d"]["passed_all_folds"]
        assert fit.manifest["subsumes_all_members"]


class TestPRBasisAndReporting:
    """The bound is summed on the same basis as the whitening (DECIDED
    2026-09-12), and the numbers that carry the compression claim are
    reported alongside it."""

    def test_bound_matches_the_whitener_participation_ratios(self, members, tmp_path):
        sp, _ = members
        fit = fit_block(sp, list(sp), block="V", **_fit_kwargs())
        assert fit.manifest["pr_basis"] == "correlation"
        pr_sum = sum(fit.map.whiteners[m].participation_ratio for m in fit.members)
        assert fit.manifest["pr_sum_bound"] == pytest.approx(pr_sum)
        # and the persisted member_pr must agree, or a reader can derive two bounds
        _, manifest, _ = save_fit(fit, tmp_path, stem="V_v0.1")
        import json

        meta = json.loads(manifest.read_text())
        assert sum(meta["member_pr"].values()) == pytest.approx(meta["pr_sum_bound"])
        assert meta["space_schema_version"] == "1.2"

    def test_compression_numbers_are_reported(self, members):
        sp, _ = members
        fit = fit_block(sp, list(sp), block="V", **_fit_kwargs())
        man = fit.manifest
        assert man["n_raw_columns"] == 64 + 12 + 6
        assert man["concat_rank"] >= man["k"]
        # the block's own effective dimensionality sits at or below its rank
        assert 0 < man["block_pr"] <= man["concat_rank"]

    def test_eval_rows_recorded_per_member(self, members):
        sp, _ = members
        fit = fit_block(sp, list(sp), block="V", **_fit_kwargs())
        for m in fit.members:
            rows = fit.manifest["per_member"][m]["eval_rows"]
            assert rows and all(r > 0 for r in rows)


class TestStructuralRule:
    """Structurally-conditional columns (DECIDED 2026-09-13).

    A column that is null whenever its condition is absent (a formant on a
    frame with no voice) has *no value* there, not a missing one. The rule
    detects such columns by per-corpus null contrast and fills them with a
    frozen sentinel, so neither branch of the old policy fires: the column
    is not dropped when the gated corpus dominates the mix, and no mean is
    fabricated when it does not.
    """

    N_ON, N_OFF = 250, 350  # gate-on rows, gate-off rows (off majority)

    @pytest.fixture
    def gated(self):
        rng = np.random.default_rng(7)
        Z = rng.normal(size=(self.N_ON + self.N_OFF, LATENT))
        sp = {
            "a": _member("a", rng, Z, 64),
            "b": _member("b", rng, Z, 12),
        }
        corpora = ["on"] * self.N_ON + ["off"] * self.N_OFF
        X = sp["b"].X.copy()
        # b_000 is gated: null on every gate-off row and 4 % of gate-on rows
        X[self.N_ON:, 0] = np.nan
        X[: self.N_ON : 25, 0] = np.nan
        # b_001 has sparse incidental gaps everywhere -- not structural
        X[::20, 1] = np.nan
        sp["b"] = SpaceMatrix(name="b", labels=sp["b"].labels, X=X, features=sp["b"].features)
        return sp, corpora

    def test_detection_needs_corpus_contrast(self, gated):
        sp, corpora = gated
        assert detect_structural_columns(sp["b"], corpora) == ["b_000"]
        assert detect_structural_columns(sp["a"], corpora) == []
        # one corpus: no contrast to read, nothing detected
        assert detect_structural_columns(sp["b"], ["one"] * sp["b"].n) == []

    def test_fill_sits_below_the_defined_range(self, gated):
        sp, _ = gated
        fills = structural_fill_values(sp["b"], ["b_000"])
        col = sp["b"].X[:, 0]
        assert fills["b_000"] == pytest.approx(np.nanmean(col) - 3 * np.nanstd(col))
        assert fills["b_000"] < np.nanpercentile(col, 1)
        filled = apply_structural_fill(sp["b"], fills)
        assert not np.isnan(filled.X[:, 0]).any()
        assert np.isnan(filled.X[:, 1]).any()  # sparse gaps untouched

    def test_gated_column_is_kept_not_dropped_or_mean_imputed(self, gated):
        sp, corpora = gated
        # without corpora the old policy drops it: pooled null 0.60 > 0.5
        fit_old = fit_block(sp, ["a", "b"], block="T", **_fit_kwargs())
        assert "b_000" in fit_old.manifest["per_member"]["b"]["dropped_columns"]
        # with corpora it is detected, filled and kept
        fit = fit_block(sp, ["a", "b"], block="T", corpora=corpora, **_fit_kwargs())
        pm = fit.manifest["per_member"]["b"]
        assert pm["structural_columns"] == ["b_000"]
        assert "b_000" not in pm["dropped_columns"]
        assert pm["dim"] == 12
        rule = fit.manifest["structural_rule"]
        assert rule["n_corpora"] == 2 and rule["n_structural_columns"] == 1
        assert fit.manifest["subsumes_all_members"]

    def test_undefined_is_distinct_from_the_mean_in_the_whitened_space(self, gated):
        sp, corpora = gated
        fit = fit_block(sp, ["a", "b"], block="T", corpora=corpora, **_fit_kwargs())
        w = fit.map.whiteners["b"]
        base = np.nanmean(np.asarray(sp["b"].X, dtype=float), axis=0)[None, :]
        undefined = base.copy()
        undefined[0, 0] = np.nan  # gate off: no value
        at_mean = base.copy()  # gate on, value happens to sit at the mean
        d = np.linalg.norm(w.transform(undefined) - w.transform(at_mean))
        assert d > 1.0  # mean imputation would give d == 0

    def test_roundtrip_project_and_check_on_a_null_bearing_table(self, gated, tmp_path):
        sp, corpora = gated
        fit = fit_block(sp, ["a", "b"], block="T", corpora=corpora, **_fit_kwargs())
        _, manifest, _ = save_fit(fit, tmp_path)
        loaded = load_fit(manifest)
        assert loaded.map.whiteners["b"].structural_fill == fit.map.whiteners["b"].structural_fill
        # projecting the raw table (nulls and all) matches the fitting-time fill
        S_fit, _ = fit.project(sp)
        S_loaded, _ = loaded.project(sp)
        assert np.isfinite(S_loaded).all()
        np.testing.assert_allclose(S_loaded, S_fit, atol=1e-10)
        # the check covers the gate-off rows instead of dropping them
        rows = check_fit(loaded, sp, n_perm=150, eval_n=None, k_nn=10)
        assert all(r["passed"] for r in rows)
        assert all(r["n_rows"] == self.N_ON + self.N_OFF for r in rows)

    def test_no_corpora_means_no_behavior_change(self, members, tmp_path):
        sp, _ = members
        fit = fit_block(sp, list(sp), block="V", **_fit_kwargs())
        assert fit.manifest["structural_rule"] is None
        assert all(fit.manifest["per_member"][m]["structural_columns"] == []
                   for m in fit.members)
        _, manifest, _ = save_fit(fit, tmp_path)
        assert load_fit(manifest).map.whiteners["a"].structural_fill == {}


class TestContiguousEvalBlocks:
    """With ``block_size`` set, the overlap subsample is whole grid blocks.

    Before 0.18.2 the subsample was drawn row-wise and then cut into blocks,
    so a "block" held rows from unrelated clips and the block null was a row
    null in disguise (MEASURED on the A v0.1 fit: overlap p at the permutation
    floor in all 680 criterion rows).
    """

    @staticmethod
    def _smooth(rng, n, dim, width):
        kernel = np.ones(width) / width
        noise = rng.normal(size=(n + width - 1, dim))
        return np.stack([np.convolve(noise[:, j], kernel, mode="valid") for j in range(dim)], axis=1)

    def test_draws_whole_blocks_on_the_grid(self):
        sub, how = eval_subsample(1000, 250, 20, np.random.default_rng(0))
        assert how == "blocks"
        blocks = sub.reshape(-1, 20)
        assert blocks.shape[0] == 250 // 20
        assert (blocks[:, 0] % 20 == 0).all()
        np.testing.assert_array_equal(blocks - blocks[:, :1], np.tile(np.arange(20), (len(blocks), 1)))
        assert len(np.unique(blocks[:, 0])) == len(blocks)

    def test_trailing_partial_block_is_never_drawn(self):
        sub, _ = eval_subsample(1010, 1000, 20, np.random.default_rng(0))
        assert sub.size == 1000 and sub.max() < 1000

    def test_row_path_is_unchanged(self):
        sub, how = eval_subsample(1000, 250, None, np.random.default_rng(7))
        expected = np.sort(np.random.default_rng(7).choice(1000, size=250, replace=False))
        assert how == "rows"
        np.testing.assert_array_equal(sub, expected)
        assert eval_subsample(100, 250, 20, np.random.default_rng(0))[1] == "all"

    def test_too_few_blocks_is_refused_before_fitting(self, members, tmp_path):
        sp, _ = members
        kw = {**_fit_kwargs(), "eval_n": 100, "block_size": 20}
        with pytest.raises(SpaceError, match="only 5 blocks"):
            fit_block(sp, list(sp), **kw)
        fit = fit_block(sp, list(sp), **_fit_kwargs())
        with pytest.raises(SpaceError, match="only 5 blocks"):
            check_fit(fit, sp, n_perm=150, eval_n=100, block_size=20)

    def test_independent_smooth_spaces_fail_the_block_null(self):
        rng = np.random.default_rng(1)
        X, Y = self._smooth(rng, 6000, 8, 20), self._smooth(rng, 6000, 8, 20)
        mc = check_member(X, Y, member="y", k=8, fold=0, k_nn=10, n_perm=200,
                          eval_n=1200, block_size=40, random_state=0)
        assert mc.eval_sampling == "blocks"
        assert not mc.overlap_p < 0.01
        # the same contiguous rows against a ROW null: temporal adjacency alone "passes"
        sub, _ = eval_subsample(6000, 1200, 40, np.random.default_rng(0))
        row = neighbor_overlap_null(X[sub], Y[sub], k=10, n_perm=200, block_size=None)
        assert row.p_value < 0.01

    def test_shared_signal_still_passes_the_block_null(self):
        rng = np.random.default_rng(2)
        # a latent that revisits similar states in unrelated blocks
        Z = np.repeat(rng.normal(size=(6000 // 5, 3)), 5, axis=0)
        X = Z @ rng.normal(size=(3, 8)) + 0.05 * rng.normal(size=(6000, 8))
        Y = Z @ rng.normal(size=(3, 6)) + 0.05 * rng.normal(size=(6000, 6))
        mc = check_member(X, Y, member="y", k=8, fold=0, k_nn=10, n_perm=200,
                          eval_n=1200, block_size=40, random_state=0)
        assert mc.eval_sampling == "blocks"
        assert mc.overlap_p < 0.01

    def test_null_spread_is_carried_through(self, members):
        rng = np.random.default_rng(2)
        Z = np.repeat(rng.normal(size=(6000 // 5, 3)), 5, axis=0)
        X = Z @ rng.normal(size=(3, 8)) + 0.05 * rng.normal(size=(6000, 8))
        Y = Z @ rng.normal(size=(3, 6)) + 0.05 * rng.normal(size=(6000, 6))
        mc = check_member(X, Y, member="y", k=8, fold=0, k_nn=10, n_perm=200,
                          eval_n=1200, block_size=40, random_state=0)
        # the same subsample and seed through the null directly
        sub, _ = eval_subsample(6000, 1200, 40, np.random.default_rng(0))
        direct = neighbor_overlap_null(X[sub], Y[sub], k=10, n_perm=200, block_size=40, random_state=0)
        assert mc.null_sd == pytest.approx(direct.null_sd)
        assert mc.null_q99 == pytest.approx(np.quantile(direct.null, 0.99))
        assert mc.null_mean <= mc.null_q99 <= max(direct.null)
        sp, _ = members
        fit = fit_block(sp, list(sp), **_fit_kwargs())
        assert all(np.isfinite(r["null_sd"]) and np.isfinite(r["null_q99"]) for r in fit.curve)

    def test_sampling_is_recorded_in_manifest_and_curve(self, members):
        sp, _ = members
        fit = fit_block(sp, list(sp), **{**_fit_kwargs(), "eval_n": 150, "block_size": 10})
        assert fit.manifest["criterion"]["eval_sampling"] == "blocks"
        assert {r["eval_sampling"] for r in fit.curve} == {"blocks"}
        assert fit_block(sp, list(sp), **_fit_kwargs()).manifest["criterion"]["eval_sampling"] == "all"
