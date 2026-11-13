"""Readout of an opaque target through a battery's nameable members, on planted ground truth."""

import json

import numpy as np
import pytest

from psytwill.exceptions import SpaceError
from psytwill.readout import readout

N, N_GROUPS = 600, 20
FAST = dict(n_control=40, n_pool=2000, n_perm=200, k_nn=10)


def _battery(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(N, 3))
    b = rng.normal(size=(N, 2))
    # a wide member with a steep spectrum, so a uniform k-subspace would not match the named share
    embed = rng.normal(size=(N, 20)) * np.geomspace(3.0, 0.3, 20)
    return {"named_a": a, "named_b": b, "embed": embed}


def _groups():
    return np.repeat(np.arange(N_GROUPS), N // N_GROUPS)


def _target(*parts, noise=0.3, dims=8, seed=1):
    rng = np.random.default_rng(seed)
    T = sum(p @ rng.normal(size=(p.shape[1], dims)) for p in parts)
    return T + noise * T.std() * rng.normal(size=T.shape)


def test_a_target_built_from_the_named_members_beats_the_control():
    bat = _battery()
    T = _target(bat["named_a"], bat["named_b"])
    r = readout(bat, T, nameable=["named_a", "named_b"], groups=_groups(), **FAST).read("nameable")
    assert r.k == 5
    assert r.r2 > 0.8
    assert r.r2_percentile == 1.0
    assert max(r.control_r2) < r.r2
    assert r.knn_p <= 0.01 and r.knn_percentile == 1.0


def test_a_target_the_named_members_do_not_carry_sits_below_the_control():
    bat = _battery()
    T = _target(bat["embed"][:, :5])
    r = readout(bat, T, nameable=["named_a", "named_b"], groups=_groups(), **FAST).read("nameable")
    assert r.r2 < 0.05
    assert r.r2_percentile <= 0.05
    assert np.median(r.control_r2) > r.r2 + 0.1


def test_control_subspaces_match_the_named_span_in_dimension_and_variance():
    bat = _battery()
    T = _target(bat["named_a"])
    r = readout(bat, T, nameable=["named_a", "named_b"], groups=_groups(), knn=False, **{
        k: v for k, v in FAST.items() if k not in ("n_perm", "k_nn")}).read("nameable")
    X = np.hstack(list(bat.values()))
    Xc = X - X.mean(0)
    want = (Xc[:, :5] ** 2).sum() / (Xc ** 2).sum()
    assert r.share == pytest.approx(want, abs=1e-12)
    lo, hi = r.control_share_range
    assert abs(lo - r.share) <= 0.01 and abs(hi - r.share) <= 0.01
    # the steep spectrum puts uniform draws well away from the named share, so the match is doing work
    assert abs(5 / 25 - r.share) > 0.03
    assert len(r.control_r2) == FAST["n_control"]


def test_unique_share_lands_on_the_member_that_carries_the_target():
    bat = _battery()
    T = _target(bat["named_a"], noise=0.2)
    out = readout(bat, T, nameable=["named_a", "named_b"], groups=_groups(), n_control=0, knn=False)
    rows = {row["member"]: row for row in out.per_member}
    assert rows["named_a"]["r2_unique"] > 0.5
    assert abs(rows["named_b"]["r2_unique"]) < 0.02
    assert rows["named_a"]["r2_alone"] > 0.9 and rows["named_b"]["r2_alone"] < 0.02
    assert out.read("nameable").control_r2 == [] and out.read("nameable").r2_percentile is None


def test_strict_then_full_form_a_nested_curve_each_with_its_own_control():
    bat = _battery()
    T = _target(bat["named_a"], bat["named_b"])
    out = readout(bat, T, nameable=["named_a", "named_b"], strict=["named_a"], groups=_groups(), **FAST)
    assert [r.label for r in out.reads] == ["strict", "nameable"]
    strict, full = out.reads
    assert (strict.k, full.k) == (3, 5)
    assert full.r2 > strict.r2
    assert len(strict.control_r2) == len(full.control_r2) == FAST["n_control"]
    assert abs(strict.control_share_range[0] - strict.share) <= 0.01


def test_a_one_dimensional_member_beside_a_wide_embedding_gets_a_one_dimensional_control():
    rng = np.random.default_rng(3)
    one = rng.normal(size=(N, 1))
    embed = rng.normal(size=(N, 30)) * np.geomspace(3.0, 0.3, 30)
    T = _target(one, embed[:, :3], noise=0.2)
    out = readout({"one": one, "embed": embed}, T, nameable=["one"], groups=_groups(), **FAST)
    r = out.read("nameable")
    assert r.k == 1 and out.per_member[0]["r2_unique"] == pytest.approx(r.r2)
    assert abs(r.control_share_range[1] - r.share) <= 0.01


def test_non_finite_rows_are_dropped_not_filled():
    bat = _battery()
    T = _target(bat["named_a"])
    bat["embed"][5, 2] = np.nan
    T[9, 0] = np.inf
    out = readout(bat, T, nameable=["named_a"], groups=_groups(), n_control=0, knn=False, per_member=False)
    assert (out.n_rows, out.n_dropped, out.n_groups) == (N - 2, 2, N_GROUPS)


def test_same_seed_same_result_and_the_record_is_json():
    bat = _battery()
    T = _target(bat["named_a"])
    kw = dict(nameable=["named_a", "named_b"], groups=_groups(), **FAST)
    a, b = readout(bat, T, **kw), readout(bat, T, **kw)
    assert a.to_dict() == b.to_dict()
    rec = json.loads(json.dumps(a.to_dict()))
    assert rec["schema"] == "1.1" and rec["battery"] == ["named_a", "named_b", "embed"]


@pytest.mark.parametrize("kw, msg", [
    (dict(nameable=["nope"]), "not in the battery"),
    (dict(nameable=[]), "nameable is empty"),
    (dict(nameable=["named_a"], strict=["named_b"]), "not among the nameable"),
    (dict(nameable=["named_a", "named_b", "embed"]), "no random subspace"),
])
def test_refusals_name_the_problem(kw, msg):
    bat = _battery()
    T = _target(bat["named_a"])
    with pytest.raises(SpaceError, match=msg):
        readout(bat, T, groups=_groups(), knn=False, n_control=5, n_pool=100, **kw)


def test_rows_must_agree():
    bat = _battery()
    with pytest.raises(SpaceError, match="do not have the target's"):
        readout(bat, np.zeros((N - 1, 2)), nameable=["named_a"])
    with pytest.raises(SpaceError, match="groups has"):
        readout(bat, _target(bat["named_a"]), nameable=["named_a"], groups=np.arange(5))


def test_every_read_sees_a_sets_span_in_the_callers_metric_not_its_axes():
    # An anisotropic member, then the same member with its columns rotated: the same span and the same
    # metric, on different coordinate axes. The share and the Euclidean k-NN read are blind to the axes,
    # so the R^2 read must be too; z-scoring each column would re-weight the rotated directions. The
    # control's draws are an arbitrary orthonormal basis of their span, so this is also what makes a
    # control draw's R^2 independent of which basis it came in.
    rng = np.random.default_rng(5)
    bat = _battery()
    bat["named_a"] = bat["named_a"] * np.array([3.0, 1.0, 0.1])
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rot = dict(bat, named_a=bat["named_a"] @ R)
    T = _target(bat["named_a"], bat["embed"][:, :3], noise=0.3)
    kw = dict(nameable=["named_a", "named_b"], groups=_groups(), n_control=0, n_perm=50, k_nn=10)
    a, b = readout(bat, T, **kw), readout(rot, T, **kw)
    ra, rb = a.read("nameable"), b.read("nameable")
    assert rb.share == pytest.approx(ra.share, abs=1e-12)
    assert rb.knn_overlap == pytest.approx(ra.knn_overlap, abs=1e-12)
    assert rb.r2 == pytest.approx(ra.r2, abs=1e-8)
    for pa, pb in zip(a.per_member, b.per_member):
        for key in ("r2_alone", "r2_without", "r2_unique"):
            assert pb[key] == pytest.approx(pa[key], abs=1e-8)
    assert a.params["ridge_scaling"] == "set"


def test_a_control_draws_r2_does_not_depend_on_the_basis_it_came_in():
    from psytwill.compare import ridge_predictivity
    rng = np.random.default_rng(6)
    bat = _battery()
    X = np.hstack(list(bat.values()))
    Xc = X - X.mean(0)
    B, _ = np.linalg.qr(rng.normal(size=(X.shape[1], 4)))
    R, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    T = _target(bat["embed"][:, :4])
    one = ridge_predictivity(Xc @ B, T, groups=_groups(), scaling="set").r2
    other = ridge_predictivity(Xc @ B @ R, T, groups=_groups(), scaling="set").r2
    assert other == pytest.approx(one, abs=1e-8)
    # the per-column z-score that space/geometry keep is not basis-free, which is the point of the option
    zs = ridge_predictivity(Xc @ B, T, groups=_groups()).r2
    zr = ridge_predictivity(Xc @ B @ R, T, groups=_groups()).r2
    assert abs(zs - zr) > 1e-6
