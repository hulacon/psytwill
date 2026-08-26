"""CV-CCA decomposition: recovers a planted shared subspace, and nothing else.

The measure exists because plain CCA is catastrophically optimistic in-sample
— at modest n it finds near-perfect correlations between independent spaces.
The tests here plant a known number of shared dimensions and require the
cross-validated count to find exactly that many, and to find zero where zero
were planted. If either fails, the count is reading fit, not structure.
"""

import json

import numpy as np
import pytest

from psytwill.decompose import (
    COMPONENT_COLUMNS,
    SUMMARY_COLUMNS,
    CvCcaResult,
    cv_cca,
    decompose_spaces,
    shared_prefix,
    write_decomposition,
)
from psytwill.exceptions import SpaceError
from psytwill.store import SpaceMatrix


@pytest.fixture
def rng():
    return np.random.RandomState(0)


def planted_pair(rng, n=600, k=3, d_a=20, d_b=30, noise=0.3):
    """Two spaces sharing exactly ``k`` latent dimensions."""
    z = rng.randn(n, k)
    Xa = np.hstack([z, rng.randn(n, d_a - k)]) @ rng.randn(d_a, d_a)
    Xb = np.hstack([z, rng.randn(n, d_b - k)]) @ rng.randn(d_b, d_b)
    Xa += noise * rng.randn(*Xa.shape)
    Xb += noise * rng.randn(*Xb.shape)
    return Xa, Xb


# --- the count -------------------------------------------------------------


def test_recovers_planted_shared_dimensions(rng):
    Xa, Xb = planted_pair(rng, k=3)
    res = cv_cca(Xa, Xb, n_perm=100)
    assert res.shared_dims == 3


def test_independent_spaces_share_nothing(rng):
    Xa = rng.randn(400, 15)
    Xb = rng.randn(400, 25)
    res = cv_cca(Xa, Xb, n_perm=100)
    assert res.shared_dims == 0


def test_correlated_scalars_share_one_dimension(rng):
    a = rng.randn(300, 1)
    b = a + 0.2 * rng.randn(300, 1)
    res = cv_cca(a, b, n_perm=100)
    assert res.n_components == 1
    assert res.shared_dims == 1


def test_independent_scalars_share_zero(rng):
    res = cv_cca(rng.randn(300, 1), rng.randn(300, 1), n_perm=100)
    assert res.shared_dims == 0


def test_train_correlations_are_optimistic_and_cv_is_not(rng):
    """The reason this measure is cross-validated at all."""
    Xa = rng.randn(120, 40)
    Xb = rng.randn(120, 40)
    res = cv_cca(Xa, Xb, n_perm=0)
    assert res.r_train[0] > 0.7  # in-sample CCA "finds" a strong correlation
    assert abs(res.r_cv[0]) < 0.5  # held-out, it is noise
    assert res.shared_dims == -1  # no null, no count


def test_prefix_rule_stops_at_first_failure():
    r_cv = np.array([0.9, 0.1, 0.8])
    null_q = np.array([0.2, 0.2, 0.2])
    assert shared_prefix(r_cv, null_q) == 1


# --- structure handling ----------------------------------------------------


def test_nan_rows_are_dropped_and_counted(rng):
    Xa, Xb = planted_pair(rng, k=2)
    Xb[5, 0] = np.nan
    res = cv_cca(Xa, Xb, n_perm=50)
    assert res.n_used == Xa.shape[0] - 1
    assert res.n_dropped == 1


def test_grouped_folds_hold_whole_groups_out(rng):
    Xa, Xb = planted_pair(rng, n=600, k=2)
    groups = np.repeat(np.arange(10), 60)
    res = cv_cca(Xa, Xb, groups=groups, n_perm=50, block_size=20)
    assert res.grouped
    assert res.shared_dims == 2


def test_single_group_is_refused(rng):
    Xa, Xb = planted_pair(rng)
    with pytest.raises(SpaceError):
        cv_cca(Xa, Xb, groups=np.zeros(Xa.shape[0]))


def test_rank_cap_bounds_the_component_count(rng):
    Xa, Xb = planted_pair(rng, d_a=30, d_b=30)
    res = cv_cca(Xa, Xb, rank_cap=5, n_perm=0)
    assert res.n_components == 5
    assert res.rank_x == 5 and res.rank_y == 5


def test_deterministic_under_a_seed(rng):
    Xa, Xb = planted_pair(rng, k=2)
    a = cv_cca(Xa, Xb, n_perm=50, random_state=7)
    b = cv_cca(Xa, Xb, n_perm=50, random_state=7)
    assert a.r_cv == b.r_cv and a.null_q == b.null_q


def test_constant_space_is_refused(rng):
    with pytest.raises(SpaceError):
        cv_cca(np.ones((100, 3)), rng.randn(100, 3), n_perm=0)


# --- driver ----------------------------------------------------------------


def space(name, X):
    return SpaceMatrix(
        name=name,
        labels=[str(i) for i in range(X.shape[0])],
        X=X,
        features=[f"f{j}" for j in range(X.shape[1])],
        modality="visual",
        extractor="test",
    )


@pytest.fixture
def inventory(rng):
    z = rng.randn(300, 2)
    return {
        "ebind": space("ebind", np.hstack([z, rng.randn(300, 8)])),
        "twin": space("twin", np.hstack([z, rng.randn(300, 4)])),
        "alien": space("alien", rng.randn(300, 6)),
    }


def test_driver_runs_source_against_every_other_space(inventory):
    res = decompose_spaces(inventory, "ebind", n_perm=50)
    assert sorted(res.summary["target"]) == ["alien", "twin"]
    assert set(res.summary["source"]) == {"ebind"}
    by_target = res.summary.set_index("target")["shared_dims"]
    assert by_target["twin"] == 2
    assert by_target["alien"] == 0


def test_driver_component_rows_match_summary_counts(inventory):
    res = decompose_spaces(inventory, "ebind", n_perm=50)
    for row in res.summary.itertuples():
        comp = res.components[res.components["target"] == row.target]
        assert len(comp) == row.n_components
        assert int(comp["shared"].sum()) == row.shared_dims


def test_driver_refuses_an_unloaded_source(inventory):
    with pytest.raises(SpaceError):
        decompose_spaces(inventory, "nope")


def test_columns_are_the_declared_schema(inventory):
    res = decompose_spaces(inventory, "ebind", n_perm=10)
    assert tuple(res.components.columns) == COMPONENT_COLUMNS
    assert tuple(res.summary.columns) == SUMMARY_COLUMNS


def test_cli_decompose_writes_all_three_outputs(tmp_path):
    import pandas as pd

    from psytwill.cli import main

    rng = np.random.RandomState(0)
    stimuli = [f"img{i:03d}" for i in range(30)]
    rows = [
        {
            "stimulus_id": s,
            "modality": "visual",
            "extractor": "viz2psy",
            "model": model,
            "feature": f"{model}_{d:03d}",
            "value": float(rng.randn()),
            "value_str": None,
        }
        for model, dim in (("ebind", 5), ("gist", 4))
        for s in stimuli
        for d in range(dim)
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "image.parquet", index=False)

    out = tmp_path / "decomp"
    rc = main(
        [
            "decompose",
            f"{tmp_path/'image.parquet'}:image",
            "-o",
            str(out),
            "--source",
            "image:ebind",
            "--permutations",
            "10",
            "--n-splits",
            "3",
        ]
    )
    assert rc == 0
    summ = pd.read_parquet(out / "space_decomposition.summary.parquet")
    assert list(summ["target"]) == ["image:gist"]
    meta = json.loads((out / "space_decomposition.meta.json").read_text())
    assert meta["config"]["source"] == "image:ebind"
    assert "load_report" in meta


def test_write_round_trips_with_a_versioned_sidecar(inventory, tmp_path):
    import pandas as pd

    res = decompose_spaces(inventory, "ebind", n_perm=10)
    paths = write_decomposition(
        res, tmp_path, name="t", inputs=["a.parquet"], labels=["0", "1"]
    )
    meta = json.loads((tmp_path / "t.meta.json").read_text())
    assert meta["schema_version"] == "1.0"
    assert meta["table"] == "space_decomposition"
    assert meta["config"]["source"] == "ebind"
    comp = pd.read_parquet(paths["components"])
    summ = pd.read_parquet(paths["summary"])
    assert tuple(comp.columns) == COMPONENT_COLUMNS
    assert tuple(summ.columns) == SUMMARY_COLUMNS
