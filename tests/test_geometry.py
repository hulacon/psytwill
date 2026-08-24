"""The driver: what a matrix over many spaces must not get wrong.

The measures themselves are tested in test_compare.py. What is testable only
here is the bookkeeping — that a symmetric measure is stored once and read
back mirrored, that a cached neighbour graph is the *same* measure and not a
faster approximation of it, and that adding a measure to the registry without
teaching the driver about it fails loudly instead of silently vanishing from
the output.
"""

import json

import numpy as np
import pandas as pd
import pytest

from psytwill.compare import MEASURE_REGISTRY, get_measure, neighbor_overlap_null
from psytwill.exceptions import SpaceError
from psytwill.geometry import (
    GEOMETRY_SCHEMA_VERSION,
    MANIFEST_COLUMNS,
    PAIR_COLUMNS,
    compare_spaces,
    write_geometry,
)
from psytwill.store import SpaceMatrix


@pytest.fixture
def rng():
    return np.random.RandomState(0)


def space(name, X):
    return SpaceMatrix(
        name=name,
        labels=[f"s{i:04d}" for i in range(X.shape[0])],
        X=np.asarray(X, dtype=float),
        features=[f"{name}_{j}" for j in range(X.shape[1])],
    )


@pytest.fixture
def inventory(rng):
    """Three spaces on 60 shared rows: two related, one independent."""
    A = rng.randn(60, 6)
    B = A @ rng.randn(6, 4) + 0.3 * rng.randn(60, 4)  # predictable from A
    C = rng.randn(60, 5)  # unrelated
    return {"a": space("a", A), "b": space("b", B), "c": space("c", C)}


# --- shape and bookkeeping -------------------------------------------------


def test_symmetric_measures_run_once_per_unordered_pair(inventory):
    res = compare_spaces(inventory, n_permutations=0)
    counts = res.pairs.groupby("measure").size().to_dict()
    for name, cfg in MEASURE_REGISTRY.items():
        # 3 spaces: 6 ordered pairs, 3 unordered
        assert counts[name] == (3 if cfg.symmetric else 6), name


def test_every_registered_measure_has_a_driver_branch(inventory):
    """The contract that keeps the registry and the driver from drifting."""
    res = compare_spaces(inventory, measures=list(MEASURE_REGISTRY), n_permutations=0)
    assert set(res.pairs["measure"]) == set(MEASURE_REGISTRY)


def test_columns_are_the_declared_schema(inventory):
    res = compare_spaces(inventory, n_permutations=0)
    assert tuple(res.pairs.columns) == PAIR_COLUMNS
    assert tuple(res.manifest.columns) == MANIFEST_COLUMNS


def test_manifest_reports_effective_not_nominal_dimensionality(rng):
    """A rank-1 space is 5 columns wide and one degree of freedom."""
    v = rng.randn(40, 1)
    flat = space("flat", v @ np.ones((1, 5)))
    iso = space("iso", rng.randn(40, 5))
    res = compare_spaces({"flat": flat, "iso": iso}, n_permutations=0)
    m = res.manifest.set_index("space")
    assert m.loc["flat", "dim"] == 5
    assert m.loc["flat", "participation_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert m.loc["iso", "participation_ratio"] > 3.0
    assert m.loc["flat", "pr_fraction"] == pytest.approx(0.2, abs=1e-6)


# --- the cached neighbour graph is the same measure ------------------------


@pytest.mark.parametrize("block_size", [None, 7])
def test_cached_knn_reproduces_the_uncached_null_exactly(inventory, block_size):
    res = compare_spaces(
        inventory,
        measures=["neighbor_overlap"],
        n_permutations=50,
        block_size=block_size,
        random_state=3,
        k=5,
    )
    for r in res.pairs.itertuples():
        direct = neighbor_overlap_null(
            inventory[r.source].X,
            inventory[r.target].X,
            k=5,
            n_perm=50,
            block_size=block_size,
            random_state=3,
        )
        assert r.value == direct.observed
        assert r.p_value == direct.p_value
        assert r.null_mean == direct.null_mean


def test_nan_rows_bypass_the_cache_and_are_counted(inventory, rng):
    """A cached graph built before NaN removal would index rows that moved."""
    holey = inventory["c"].X.copy()
    holey[[2, 9], 0] = np.nan
    inv = dict(inventory, c=space("c", holey))
    res = compare_spaces(inv, measures=["neighbor_overlap"], n_permutations=20, k=5)
    involving_c = res.pairs[(res.pairs.source == "c") | (res.pairs.target == "c")]
    assert (involving_c["n_dropped"] == 2).all()
    assert (involving_c["n_used"] == 58).all()
    for r in involving_c.itertuples():
        direct = neighbor_overlap_null(
            inv[r.source].X, inv[r.target].X, k=5, n_perm=20, random_state=0
        )
        assert r.value == direct.observed
    ab = res.pairs[(res.pairs.source == "a") & (res.pairs.target == "b")].iloc[0]
    assert ab["n_dropped"] == 0 and ab["n_used"] == 60


# --- the measures land where they should -----------------------------------


def test_ridge_is_asymmetric_and_finds_the_predictable_direction(inventory):
    res = compare_spaces(inventory, measures=["ridge"], n_splits=4)
    r = res.pairs.set_index(["source", "target"])["value"]
    assert r[("a", "b")] > 0.8  # b was built from a
    assert r[("a", "c")] < 0.2  # c is independent
    assert r[("a", "b")] != r[("b", "a")]


def test_ridge_records_the_alphas_it_selected(inventory):
    res = compare_spaces(inventory, measures=["ridge"], n_splits=4)
    assert res.pairs["alpha_min"].notna().all()
    assert (res.pairs["alpha_min"] <= res.pairs["alpha_max"]).all()
    assert (res.pairs["n_splits"] == 4).all()
    assert not res.pairs["grouped"].any()


def test_groups_are_passed_through_to_the_folds(inventory):
    groups = np.repeat(np.arange(6), 10)
    res = compare_spaces(inventory, measures=["ridge"], groups=groups, n_splits=3)
    assert res.pairs["grouped"].all()
    assert (res.pairs["n_splits"] == 3).all()


def test_null_z_is_recorded_beside_the_p(inventory):
    res = compare_spaces(
        inventory, measures=["neighbor_overlap"], n_permutations=50, k=5
    )
    assert res.pairs["null_z"].notna().all()
    assert (res.pairs["n_perm"] == 50).all()
    assert (res.pairs["k"] == 5).all()


def test_scalar_head_carries_its_note(rng):
    inv = {"scalar": space("scalar", rng.randn(40, 1)), "wide": space("wide", rng.randn(40, 6))}
    res = compare_spaces(inv, n_permutations=0)
    ridge = res.pairs[(res.pairs.measure == "ridge") & (res.pairs.source == "scalar")]
    assert "1-d source" in ridge["note"].iloc[0]
    cka = res.pairs[res.pairs.measure == "cka_linear"]
    assert "single ordering" in cka["note"].iloc[0]


# --- refusals --------------------------------------------------------------


def test_unaligned_spaces_name_the_fix(rng):
    inv = {"a": space("a", rng.randn(60, 3)), "b": space("b", rng.randn(50, 3))}
    with pytest.raises(SpaceError, match="align_spaces"):
        compare_spaces(inv, n_permutations=0)


def test_one_space_is_not_a_comparison(rng):
    with pytest.raises(SpaceError, match="at least 2 spaces"):
        compare_spaces({"a": space("a", rng.randn(10, 3))}, n_permutations=0)


def test_unknown_measure_lists_the_alternatives(inventory):
    with pytest.raises(SpaceError, match="Available"):
        compare_spaces(inventory, measures=["ridgeregression"])


def test_groups_length_is_checked(inventory):
    with pytest.raises(SpaceError, match="entries for 60 rows"):
        compare_spaces(inventory, measures=["ridge"], groups=np.arange(7))


# --- reading it back -------------------------------------------------------


def test_matrix_mirrors_symmetric_and_not_asymmetric(inventory):
    res = compare_spaces(inventory, measures=["ridge", "cka_linear"], n_permutations=0)
    sym = res.matrix("cka_linear")
    assert sym.loc["a", "b"] == sym.loc["b", "a"]
    assert sym.loc["a", "a"] != sym.loc["a", "a"]  # diagonal stays NaN
    asym = res.matrix("ridge")
    assert asym.loc["a", "b"] != asym.loc["b", "a"]


def test_matrix_names_the_measures_it_has(inventory):
    res = compare_spaces(inventory, measures=["ridge"], n_permutations=0)
    with pytest.raises(SpaceError, match="Present: ridge"):
        res.matrix("rsa")


def test_write_geometry_round_trips_with_a_versioned_sidecar(inventory, tmp_path):
    res = compare_spaces(inventory, n_permutations=20, k=5)
    paths = write_geometry(res, tmp_path, inputs=["fixture.parquet"], labels=["x"] * 60)
    back = pd.read_parquet(paths["pairs"])
    assert tuple(back.columns) == PAIR_COLUMNS
    assert len(back) == len(res.pairs)
    manifest = pd.read_parquet(paths["manifest"])
    assert list(manifest["space"]) == ["a", "b", "c"]
    meta = json.loads(open(paths["meta_path"]).read())
    assert meta["schema_version"] == GEOMETRY_SCHEMA_VERSION
    assert meta["config"]["n_permutations"] == 20
    assert meta["config"]["n_spaces"] == 3
    assert meta["inputs"] == ["fixture.parquet"]
    assert meta["n_labels"] == 60


# --- the CLI verb ----------------------------------------------------------


def _long_rows(stimuli, model, dim, rng):
    return [
        {
            "stimulus_id": s,
            "chunk_idx": i,
            "modality": "visual",
            "extractor": "viz2psy",
            "model": model,
            "feature": f"{model}_{d:03d}",
            "value": float(rng.randn()),
            "value_str": None,
        }
        for i, s in enumerate(stimuli)
        for d in range(dim)
    ]


@pytest.fixture
def two_tables(tmp_path):
    rng = np.random.RandomState(0)
    stimuli = [f"img{i:03d}" for i in range(30)]
    a = _long_rows(stimuli, "clip", 5, rng) + _long_rows(stimuli, "gist", 4, rng)
    b = _long_rows(stimuli, "clip", 3, rng)  # same model name, different table
    pd.DataFrame(a).to_parquet(tmp_path / "image.parquet", index=False)
    pd.DataFrame(b).to_parquet(tmp_path / "caption.parquet", index=False)
    return tmp_path


def test_cli_compare_writes_all_three_outputs(two_tables, tmp_path):
    from psytwill.cli import main

    out = tmp_path / "geo"
    rc = main(
        [
            "compare",
            f"{two_tables/'image.parquet'}:image",
            f"{two_tables/'caption.parquet'}:cap",
            "-o",
            str(out),
            "--permutations",
            "10",
            "--k",
            "5",
            "--n-splits",
            "3",
        ]
    )
    assert rc == 0
    pairs = pd.read_parquet(out / "space_geometry.pairs.parquet")
    assert set(pairs["source"]) | set(pairs["target"]) == {
        "image:clip",
        "image:gist",
        "cap:clip",
    }
    meta = json.loads((out / "space_geometry.meta.json").read_text())
    assert meta["config"]["n_spaces"] == 3
    assert meta["n_labels"] == 30
    assert "load_report" in meta


def test_cli_compare_stride_keeps_every_nth_row_per_clip(tmp_path):
    from psytwill.cli import main

    rng = np.random.RandomState(0)
    rows = [
        {
            "stimulus_id": clip,
            "time": 0.5 * t,
            "modality": "visual",
            "extractor": "viz2psy",
            "model": model,
            "feature": f"{model}_{d:03d}",
            "value": float(rng.randn()),
            "value_str": None,
        }
        for clip, n_t in (("clipA", 7), ("clipB", 5))  # uneven clip lengths
        for t in range(n_t)
        for model, dim in (("m1", 3), ("m2", 2))
        for d in range(dim)
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "grid.parquet", index=False)
    out = tmp_path / "geo"
    rc = main(
        [
            "compare",
            str(tmp_path / "grid.parquet"),
            "-o",
            str(out),
            "--key",
            "stimulus_id,time",
            "--stride",
            "2",
            "--measures",
            "cka_linear,rsa",
            "--permutations",
            "0",
        ]
    )
    assert rc == 0
    meta = json.loads((out / "space_geometry.meta.json").read_text())
    # ceil(7/2) + ceil(5/2) = 4 + 3, restarting the count at each clip
    assert meta["n_labels"] == 7
    assert meta["stride"] == 2
    pairs = pd.read_parquet(out / "space_geometry.pairs.parquet")
    assert set(pairs["n_used"]) == {7}


def test_cli_compare_refuses_colliding_names_without_a_prefix(two_tables, tmp_path):
    """Two tables both offering 'clip' would silently overwrite one another."""
    from psytwill.cli import main

    rc = main(
        [
            "compare",
            str(two_tables / "image.parquet"),
            str(two_tables / "caption.parquet"),
            "-o",
            str(tmp_path / "geo"),
            "--permutations",
            "0",
        ]
    )
    assert rc == 1  # InputError naming the prefix fix


@pytest.mark.parametrize(
    "arg,expected",
    [
        ("/a/b/x.parquet:image", ("/a/b/x.parquet", "image")),
        ("/a/b/x.parquet", ("/a/b/x.parquet", None)),
        ("x.parquet", ("x.parquet", None)),
    ],
)
def test_table_arg_parses_optional_prefix(arg, expected):
    from psytwill.cli import _parse_table_arg

    assert _parse_table_arg(arg) == expected
