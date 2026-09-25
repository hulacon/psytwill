"""`psytwill space fit` over several group tables (one per corpus).

A member must be stacked from every table that carries it. The silent
failure this pins: the first table holding a model won and the rest were
ignored, so a seven-corpus fit would have run on one corpus while its
manifest listed seven inputs.
"""

import json

import numpy as np
import pandas as pd
import pytest

from psytwill.cli import main

N_STIM, N_BINS, LATENT = 24, 6, 2
FIT_ARGS = ["--members", "a,b", "--block", "T", "--k-schedule", "2,4", "--n-splits", "3",
            "--n-perm", "150", "--k-nn", "10", "--eval-n", "0", "--seed", "0"]


def _table(path, corpus, rng, W):
    """Long-form frames table for one corpus on a 0.5 s bin-centre grid."""
    rows = []
    for s in range(N_STIM):
        sid = f"ext-{corpus}-{s:03d}"
        for b in range(N_BINS):
            z = rng.normal(size=LATENT)
            for model, w in W.items():
                x = z @ w + 0.05 * rng.normal(size=w.shape[1])
                for j, v in enumerate(x):
                    rows.append((sid, 0.25 + 0.5 * b, model, f"{model}_{j:02d}", float(v)))
    df = pd.DataFrame(rows, columns=["stimulus_id", "time", "model", "feature", "value"])
    df["value_str"] = pd.Series([None] * len(df), dtype="string")
    df["modality"] = "audio"
    df["extractor"] = "testx"
    df["extractor_version"] = "0.0"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def two_corpora(tmp_path):
    rng = np.random.default_rng(0)
    W = {"a": rng.normal(size=(LATENT, 6)), "b": rng.normal(size=(LATENT, 3))}
    return [_table(tmp_path / f"{c}_audio_frames_features.parquet", c, rng, W)
            for c in ("librispeech", "musopen")]


def test_member_is_stacked_across_tables(two_corpora, tmp_path, capsys):
    out = tmp_path / "space"
    argv = ["space", "fit", "--features", *map(str, two_corpora), "--key", "stimulus_id,time",
            "--window", "0.5", "--groups-from-label", "--corpora-from-label",
            "-o", str(out), "--stem", "T_test", *FIT_ARGS]
    assert main(argv) == 0
    manifest = json.loads((out / "T_test.json").read_text())
    assert manifest["n_rows"] == 2 * N_STIM * N_BINS
    assert manifest["null_policy"]["gated_missing_detector"]["n_corpora"] == 2
    assert [p.split("/")[-1] for p in manifest["inputs"]] == [p.name for p in two_corpora]
    text = capsys.readouterr().out
    assert f"loaded a: {2 * N_STIM * N_BINS} rows x 6 features from 2 tables" in text


def test_single_table_output_is_unchanged(two_corpora, tmp_path, capsys):
    out = tmp_path / "space"
    argv = ["space", "fit", "--features", str(two_corpora[0]), "--key", "stimulus_id,time",
            "--window", "0.5", "--groups-from-label", "-o", str(out), "--stem", "T_one", *FIT_ARGS]
    assert main(argv) == 0
    manifest = json.loads((out / "T_one.json").read_text())
    assert manifest["n_rows"] == N_STIM * N_BINS
    assert "from 1 tables" not in capsys.readouterr().out


def test_same_stimulus_in_two_tables_is_refused(two_corpora, tmp_path, capsys):
    dup = tmp_path / "dup.parquet"
    pd.read_parquet(two_corpora[0]).to_parquet(dup, index=False)
    out = tmp_path / "space"
    argv = ["space", "fit", "--features", str(two_corpora[0]), str(dup), "--key", "stimulus_id,time",
            "--window", "0.5", "-o", str(out), "--stem", "T_dup", *FIT_ARGS]
    assert main(argv) == 1
    assert "more than one table" in capsys.readouterr().err


def _with_trailing_nulls(path, nulls):
    """Blank member b's last bin in every stimulus; optionally declare it."""
    df = pd.read_parquet(path)
    last = (df["model"] == "b") & (df["time"] == 0.25 + 0.5 * (N_BINS - 1))
    df.loc[last, "value"] = np.nan
    df.to_parquet(path, index=False)
    if nulls is not None:
        meta = {"schema_version": "1.1", "table": "features", "models": ["a", "b"],
                "model_nulls": {"a": {}, "b": nulls}}
        path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return path


def test_undeclared_nan_refuses_the_fit(two_corpora, tmp_path, capsys):
    table = _with_trailing_nulls(two_corpora[0], None)
    argv = ["space", "fit", "--features", str(table), "--key", "stimulus_id,time",
            "--window", "0.5", "-o", str(tmp_path / "space"), "--stem", "T_ref", *FIT_ARGS]
    with pytest.warns(UserWarning, match="no sidecar"):
        assert main(argv) == 1
    assert "no Contract B `nulls` entry" in capsys.readouterr().err


def test_declared_undefinable_rows_are_dropped(two_corpora, tmp_path, capsys):
    pos = {"means": "undefinable", "when": "trailing window"}
    tables = [_with_trailing_nulls(p, {"b_00": pos, "b_01": pos, "b_02": pos}) for p in two_corpora]
    out = tmp_path / "space"
    argv = ["space", "fit", "--features", *map(str, tables), "--key", "stimulus_id,time",
            "--window", "0.5", "--groups-from-label", "-o", str(out), "--stem", "T_def", *FIT_ARGS]
    assert main(argv) == 0
    manifest = json.loads((out / "T_def.json").read_text())
    assert manifest["n_rows"] == 2 * N_STIM * (N_BINS - 1)
    assert manifest["per_member"]["b"]["undefinable_rows_dropped"] == 2 * N_STIM
    assert "b: 48 undefinable row(s) dropped" in capsys.readouterr().out


def test_tables_that_disagree_on_nulls_are_refused(two_corpora, tmp_path, capsys):
    pos = {"means": "undefinable", "when": "trailing window"}
    _with_trailing_nulls(two_corpora[0], {"b_00": pos, "b_01": pos, "b_02": pos})
    _with_trailing_nulls(two_corpora[1], {"b_00": pos})
    argv = ["space", "fit", "--features", *map(str, two_corpora), "--key", "stimulus_id,time",
            "--window", "0.5", "-o", str(tmp_path / "space"), "--stem", "T_dis", *FIT_ARGS]
    assert main(argv) == 1
    assert "different `nulls` declarations" in capsys.readouterr().err


# --------------------------------------------------------------------------
# member splits (DECIDED 2026-09-24): faces -> faces_extent + faces_layout
# --------------------------------------------------------------------------

FACE_COLS = ["faces_center_dist", "faces_count", "faces_max_area", "faces_mutual_dist",
             "faces_total_area"]


@pytest.fixture
def faces_table(tmp_path):
    """An image table holding `faces` and `llstat`, with faces' configuration
    columns undefined (declared) where the image has too few faces."""
    rng = np.random.default_rng(1)
    n = 240
    count = rng.choice([0, 1, 2, 3], size=n, p=[0.3, 0.3, 0.25, 0.15])
    z = rng.normal(size=(n, 2))
    vals = {
        "faces_count": count.astype(float),
        "faces_max_area": z[:, 0] + 0.1 * rng.normal(size=n),
        "faces_total_area": z[:, 0] + 0.3 * count + 0.1 * rng.normal(size=n),
        "faces_center_dist": np.where(count >= 1, z[:, 1] + 0.1 * rng.normal(size=n), np.nan),
        "faces_mutual_dist": np.where(count >= 2, z[:, 1] + 0.2 * rng.normal(size=n), np.nan),
    }
    ll = z @ rng.normal(size=(2, 4)) + 0.05 * rng.normal(size=(n, 4))
    rows = []
    for i in range(n):
        sid = f"img{i:04d}"
        for c in FACE_COLS:
            rows.append((sid, "faces", c, float(vals[c][i])))
        for j in range(4):
            rows.append((sid, "llstat", f"llstat_{j:02d}", float(ll[i, j])))
    df = pd.DataFrame(rows, columns=["stimulus_id", "model", "feature", "value"])
    df["value_str"] = pd.Series([None] * len(df), dtype="string")
    df["modality"] = "visual"
    df["extractor"] = "viz2psy"
    df["extractor_version"] = "0.0"
    path = tmp_path / "img_features.parquet"
    df.to_parquet(path, index=False)
    undef = {"means": "undefined", "when": "too few faces"}
    meta = {"schema_version": "1.1", "table": "features", "models": ["faces", "llstat"],
            "model_nulls": {"faces": {"faces_center_dist": undef, "faces_mutual_dist": undef},
                            "llstat": {}}}
    path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return path, int((count >= 2).sum()), n


SPLIT_ARGS = ["--block", "V", "--k-schedule", "2,4", "--n-splits", "3", "--n-perm", "150",
              "--k-nn", "10", "--eval-n", "0", "--seed", "0"]


def test_block_default_splits_faces(faces_table, tmp_path, capsys):
    path, n_layout, n = faces_table
    out = tmp_path / "space"
    argv = ["space", "fit", "--features", str(path), "-o", str(out), "--stem", "V_split", *SPLIT_ARGS]
    main(argv)
    manifest = json.loads((out / "V_split.json").read_text())
    assert manifest["members"] == ["faces_extent", "faces_layout", "llstat"]
    assert manifest["member_sources"] == {"faces_extent": "faces", "faces_layout": "faces"}
    assert manifest["member_features"]["faces_extent"] == ["faces_count", "faces_max_area",
                                                           "faces_total_area"]
    assert manifest["member_features"]["faces_layout"] == ["faces_center_dist", "faces_mutual_dist"]
    pm = manifest["per_member"]
    # "absent unless complete" now costs only the gated columns
    assert pm["faces_extent"]["rows_present"] == n
    assert pm["faces_layout"]["rows_present"] == n_layout
    assert pm["faces_extent"]["masked_columns"] == []
    assert pm["faces_layout"]["masked_columns"] == ["faces_center_dist", "faces_mutual_dist"]
    assert "loaded faces_layout: 240 rows x 2 features (split from faces)" in capsys.readouterr().out


def test_explicit_model_member_loads_whole(faces_table, tmp_path):
    path, n_layout, _ = faces_table
    out = tmp_path / "space"
    argv = ["space", "fit", "--features", str(path), "--members", "faces,llstat",
            "-o", str(out), "--stem", "V_whole", *SPLIT_ARGS]
    main(argv)
    manifest = json.loads((out / "V_whole.json").read_text())
    assert manifest["members"] == ["faces", "llstat"]
    assert manifest["member_sources"] == {}
    assert len(manifest["member_features"]["faces"]) == 5
    assert manifest["per_member"]["faces"]["rows_present"] == n_layout


def test_split_fit_projects(faces_table, tmp_path):
    path, _, n = faces_table
    out = tmp_path / "space"
    main(["space", "fit", "--features", str(path), "-o", str(out), "--stem", "V_split", *SPLIT_ARGS])
    proj = tmp_path / "proj.parquet"
    assert main(["space", "project", "--space", str(out / "V_split.json"), "--features", str(path),
                 "-o", str(proj)]) == 0
    assert len(pd.read_parquet(proj)) == n


def test_manifest_reports_walk_top_and_captured_share(faces_table, tmp_path):
    path, _, _ = faces_table
    out = tmp_path / "space"
    main(["space", "fit", "--features", str(path), "-o", str(out), "--stem", "V_split", *SPLIT_ARGS])
    manifest = json.loads((out / "V_split.json").read_text())
    # concat_rank is the fold maps' full rank, not the chosen k (bug before 1.5)
    assert manifest["concat_rank"] == max(manifest["k_schedule"])
    assert manifest["space_schema_version"] == "1.5"
    for m, pm in manifest["per_member"].items():
        assert len(pm["block_captured_at_k_max"]) == 3
        assert all(0.0 <= c <= 1.0 + 1e-9 for c in pm["block_captured_at_k_max"])
        assert all(a <= b + 1e-9 for a, b in zip(pm["block_captured_at_k"], pm["block_captured_at_k_max"]))


def test_defer_members_flag_reaches_the_manifest(faces_table, tmp_path, capsys):
    path, _, _ = faces_table
    out = tmp_path / "space"
    main(["space", "fit", "--features", str(path), "-o", str(out), "--stem", "V_def",
          "--defer-members", "faces_layout", *SPLIT_ARGS])
    manifest = json.loads((out / "V_def.json").read_text())
    assert manifest["deferred_members"] == ["faces_layout"]
