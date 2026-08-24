"""Second-order measures: known values, invariances, and the two silent traps.

The traps are the reason this file exists. Random folds on autocorrelated rows
and a row-shuffle null on temporally ordered data both produce confident,
plausible, wrong numbers — no exception, no warning. Each has a test here that
fails if the guard is removed.
"""

import numpy as np
import pytest

from psytwill.compare import (
    MEASURE_REGISTRY,
    applicability,
    block_permutation,
    cka,
    get_measure,
    knn_indices,
    linear_cka,
    neighbor_overlap,
    neighbor_overlap_null,
    participation_ratio,
    permutation_null,
    rbf_cka,
    ridge_predictivity,
    second_order_rsa,
)
from psytwill.exceptions import SpaceError


@pytest.fixture
def rng():
    return np.random.RandomState(0)


def smooth_walk(n, d, rng, step=1.0):
    """Temporally autocorrelated rows — a stand-in for a 0.5 s movie grid."""
    return np.cumsum(rng.randn(n, d) * step, axis=0)


# --- effective dimensionality ---------------------------------------------


def test_participation_ratio_isotropic_is_near_d(rng):
    X = rng.randn(2000, 8)
    assert participation_ratio(X) == pytest.approx(8, rel=0.15)


def test_participation_ratio_rank_one_is_one(rng):
    X = rng.randn(200, 1) @ rng.randn(1, 10)
    assert participation_ratio(X) == pytest.approx(1.0, abs=1e-6)


def test_participation_ratio_needs_rows():
    with pytest.raises(SpaceError):
        participation_ratio(np.zeros((1, 4)))


# --- ridge predictivity ----------------------------------------------------


def test_ridge_recovers_an_exact_linear_map(rng):
    X = rng.randn(200, 10)
    Y = X @ rng.randn(10, 4)
    assert ridge_predictivity(X, Y).r2 == pytest.approx(1.0, abs=1e-3)


def test_ridge_on_independent_spaces_is_about_zero(rng):
    X, Y = rng.randn(200, 10), rng.randn(200, 4)
    assert ridge_predictivity(X, Y).r2 < 0.1


def test_ridge_is_asymmetric(rng):
    """The whole point of including it: A->B is not B->A."""
    X = rng.randn(300, 10)
    Y = X[:, :3]
    assert ridge_predictivity(X, Y).r2 > 0.99  # subspace is fully contained
    assert ridge_predictivity(Y, X).r2 < 0.6   # 3 dims cannot rebuild 10


def test_ridge_selects_alpha_inside_each_fold(rng):
    X, Y = rng.randn(120, 6), rng.randn(120, 2)
    assert ridge_predictivity(X, Y).alphas_selected  # recorded, not assumed


def test_ridge_drops_nan_rows_and_reports_them(rng):
    X, Y = rng.randn(100, 5), rng.randn(100, 2)
    X[3, 0] = np.nan
    Y[7, 1] = np.nan
    r = ridge_predictivity(X, Y)
    assert (r.n_used, r.n_dropped) == (98, 2)


def test_ridge_grouped_folds_prevent_leakage(rng):
    """CONSTRAINT: fold by clip, never by frame.

    Rows carry a group identity that X can read off. With random folds the
    model memorizes each group's target from its other members and R^2 goes
    to ~1 with no generalizable signal present; with grouped folds the held-out
    group has never been seen and R^2 collapses. On a 0.5 s grid, adjacent
    frames play the role of group members.
    """
    n_groups, per_group = 20, 10
    groups = np.repeat(np.arange(n_groups), per_group)
    means = rng.randn(n_groups)
    X = np.eye(n_groups)[groups] + rng.randn(n_groups * per_group, n_groups) * 0.01
    Y = (means[groups] + rng.randn(n_groups * per_group) * 0.01)[:, None]

    leaky = ridge_predictivity(X, Y).r2
    honest = ridge_predictivity(X, Y, groups=groups).r2
    assert leaky > 0.8, "random folds should look great here — that is the trap"
    assert honest < 0.2, "grouped folds must not see the held-out group"
    assert ridge_predictivity(X, Y, groups=groups).grouped is True


def test_ridge_refuses_a_single_group(rng):
    X, Y = rng.randn(40, 4), rng.randn(40, 2)
    with pytest.raises(SpaceError, match="single group"):
        ridge_predictivity(X, Y, groups=np.zeros(40))


# --- CKA -------------------------------------------------------------------


def test_linear_cka_self_is_one(rng):
    X = rng.randn(100, 8)
    assert linear_cka(X, X) == pytest.approx(1.0)


def test_linear_cka_is_rotation_invariant(rng):
    X, Y = rng.randn(100, 8), rng.randn(100, 5)
    Q, _ = np.linalg.qr(rng.randn(8, 8))
    assert linear_cka(X @ Q, Y) == pytest.approx(linear_cka(X, Y), abs=1e-10)


def test_linear_cka_is_isotropic_scale_invariant(rng):
    X, Y = rng.randn(100, 8), rng.randn(100, 5)
    assert linear_cka(X * 17.0, Y) == pytest.approx(linear_cka(X, Y), abs=1e-10)


def test_linear_cka_independent_is_small(rng):
    assert linear_cka(rng.randn(500, 10), rng.randn(500, 10)) < 0.15


def test_rbf_cka_self_is_one(rng):
    X = rng.randn(80, 6)
    assert rbf_cka(X, X) == pytest.approx(1.0)


def test_cka_rejects_unknown_kernel(rng):
    with pytest.raises(SpaceError, match="linear, rbf"):
        cka(rng.randn(10, 3), rng.randn(10, 3), kernel="laplacian")


# --- second-order RSA ------------------------------------------------------


def test_rsa_identical_spaces_is_one(rng):
    X = rng.randn(60, 7)
    assert second_order_rsa(X, X) == pytest.approx(1.0)


def test_rsa_independent_is_about_zero(rng):
    assert abs(second_order_rsa(rng.randn(100, 7), rng.randn(100, 7))) < 0.2


# --- neighbour overlap -----------------------------------------------------


def test_neighbor_overlap_identical_is_one(rng):
    X = rng.randn(80, 6)
    assert neighbor_overlap(X, X, k=10) == pytest.approx(1.0)


def test_neighbor_overlap_independent_is_near_chance(rng):
    X, Y = rng.randn(200, 6), rng.randn(200, 6)
    assert neighbor_overlap(X, Y, k=10) < 0.2


def test_neighbor_overlap_clamps_k_to_available_rows(rng):
    X = rng.randn(5, 3)
    assert neighbor_overlap(X, X, k=100) == pytest.approx(1.0)


# --- permutation nulls -----------------------------------------------------


def test_block_permutation_keeps_blocks_contiguous():
    perm = block_permutation(20, np.random.RandomState(1), block_size=5)
    assert sorted(perm) == list(range(20))
    diffs = np.diff(perm)
    # 4 steps of +1 inside each of 4 blocks = 16 contiguous steps
    assert (diffs == 1).sum() >= 16


def test_block_permutation_none_is_a_full_shuffle():
    perm = block_permutation(50, np.random.RandomState(1), block_size=None)
    assert sorted(perm) == list(range(50))
    assert (np.diff(perm) == 1).sum() < 10


def test_block_null_is_the_conservative_one_on_temporal_data(rng):
    """CONSTRAINT: temporal-block permutation, not row shuffle.

    Two *independent* smooth trajectories agree strongly on nearest neighbours
    for a reason that has nothing to do with their content: in either space a
    row's closest neighbours are the rows just before and after it. A
    row-shuffle null destroys that shared autocorrelation, so this
    uninformative agreement clears it easily. A block null keeps it, and
    correctly declines to call it a relationship.
    """
    X = smooth_walk(200, 3, rng)
    Y = smooth_walk(200, 3, rng)  # independent content, same smoothness

    def measure(a, b):
        return neighbor_overlap(a, b, k=5)

    shuffled = permutation_null(measure, X, Y, n_perm=60, block_size=None)
    blocked = permutation_null(measure, X, Y, n_perm=60, block_size=20)

    z_shuffled = (shuffled.observed - shuffled.null_mean) / shuffled.null_sd
    z_blocked = (blocked.observed - blocked.null_mean) / blocked.null_sd

    assert blocked.null_mean > shuffled.null_mean + 0.05
    assert z_shuffled > 10, (
        "shared autocorrelation alone puts the observed value many SDs above "
        "the row-shuffle null — no content agreement required"
    )
    assert z_shuffled > 5 * z_blocked, (
        "the row-shuffle null is the permissive one: it inflates the same "
        "uninformative agreement by an order of magnitude"
    )
    # Asserted on effect size, not on p: with n_perm this small the block
    # null's p is seed-dependent (measured 0.016-0.98 across six seeds) while
    # the z-ratio holds at >=9x everywhere. The block null is conservative but
    # not perfectly calibrated either — blocks land at new offsets, so a small
    # positive z survives.


def test_permutation_p_is_bounded_below(rng):
    X = rng.randn(50, 4)
    null = permutation_null(lambda a, b: neighbor_overlap(a, b, k=5), X, X, n_perm=19)
    assert null.p_value == pytest.approx(1 / 20)


# --- alignment, registry, applicability ------------------------------------


def test_mismatched_rows_name_the_fix(rng):
    with pytest.raises(SpaceError, match="common grain"):
        linear_cka(rng.randn(10, 3), rng.randn(11, 3))


def test_registry_lists_alternatives_on_a_typo():
    with pytest.raises(SpaceError, match="cka_linear"):
        get_measure("cka")


def test_registry_records_which_measures_are_symmetric():
    assert MEASURE_REGISTRY["ridge"].symmetric is False
    assert all(
        MEASURE_REGISTRY[n].symmetric
        for n in ("cka_linear", "cka_rbf", "rsa", "neighbor_overlap")
    )


def test_applicability_flags_scalar_heads_without_refusing_them():
    assert applicability("ridge", 1024, 1) is None       # scalar target is fine
    assert "1-d source" in applicability("ridge", 1, 1024)
    assert "single ordering" in applicability("neighbor_overlap", 1024, 1)
    assert applicability("rsa", 512, 20) is None


# --- the fast neighbour-overlap null ---------------------------------------


def test_knn_indices_excludes_self_and_clamps_k(rng):
    X = rng.randn(6, 4)
    idx = knn_indices(X, k=99)
    assert idx.shape == (6, 5)                       # k clamped to n - 1
    assert all(i not in idx[i] for i in range(6))    # never its own neighbour


@pytest.mark.parametrize("block_size", [None, 20])
def test_fast_null_reproduces_the_generic_one_exactly(rng, block_size):
    """The relabeling shortcut must be an optimization, not a second measure.

    Permuting Y's rows only relabels its neighbour graph, so both graphs are
    built once instead of once per permutation. If that identity is ever
    broken, this test catches it as a numeric divergence rather than as a
    plausible-looking null.
    """
    X = rng.randn(120, 8)
    Y = X @ rng.randn(8, 5) + rng.randn(120, 5) * 2

    generic = permutation_null(
        lambda a, b: neighbor_overlap(a, b, k=10),
        X, Y, n_perm=30, block_size=block_size, random_state=3,
    )
    fast = neighbor_overlap_null(
        X, Y, k=10, n_perm=30, block_size=block_size, random_state=3
    )
    assert fast.observed == pytest.approx(generic.observed)
    assert np.allclose(fast.null, generic.null)
    assert fast.p_value == pytest.approx(generic.p_value)
