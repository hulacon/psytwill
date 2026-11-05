"""Contract B 1.2 per-column labels through psytwill.

Reader validation, carry-through into the features sidecar (with the
stacking rule), ``features --refresh-labels``, bookkeeping exclusion,
``space fit --select``, the ``circular_with`` flag, and the labels export.
Fixtures are synthetic (``tests.test_contract.labelled_fixture``).
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from psytwill.cli import main
from psytwill.exceptions import InputError, SpaceError
from psytwill.features import build_features, group_labels
from psytwill.labels import parse_select, select_text
from tests.test_contract import CLIP_CKPT, labelled_fixture, labelled_models

FAST = ["--n-splits", "3", "--no-criterion", "--fixed-k", "2"]


def _features(tmp_path, csvs, name="features.parquet"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_features(csvs, tmp_path / name)
    return tmp_path / name


def _fit(tmp_path, table, stem, *extra, members="clip,aesthetics,llstat"):
    out = tmp_path / "space"
    argv = ["space", "fit", "--features", str(table), "--block", "V", "--members", members,
            "-o", str(out), "--stem", stem, *FAST, *extra]
    assert main(argv) == 0
    return json.loads((out / f"{stem}.json").read_text()), out


# --- reader -----------------------------------------------------------------


@pytest.mark.parametrize("mutate, match", [
    (lambda m: m["llstat"]["labels"]["default"].update(interpretability="obvious"), "interpretability='obvious'"),
    (lambda m: m["llstat"]["labels"]["default"].update(derivation="magic"), "derivation='magic'"),
    (lambda m: m["aesthetics"]["labels"]["default"].pop("derived_from"), "requires `derived_from`"),
    (lambda m: m["llstat"]["labels"]["default"].update(derived_from=CLIP_CKPT), "only allowed"),
    # pattern-aware: clip's inventory is `clip_{NNN}`, so a non-numeric suffix is not a column
    (lambda m: m["clip"]["labels"].setdefault("columns", {}).update(clip_extra={"Units": "x"}),
     "'clip_extra' is not a column"),
    (lambda m: m["llstat"]["labels"]["columns"].update(llstat_nope={"Units": "x"}), "'llstat_nope' is not"),
    (lambda m: m["clip"]["labels"]["default"].pop("derivation"), "no `derivation`"),
])
def test_malformed_producer_labels_fail_loudly(tmp_path, mutate, match):
    models = labelled_models()
    mutate(models)
    csv = labelled_fixture(tmp_path, models=models)
    with pytest.raises(InputError, match=match):
        build_features([csv], tmp_path / "f.parquet")


def test_labels_are_carried_into_the_features_sidecar(tmp_path):
    table = _features(tmp_path, [labelled_fixture(tmp_path)])
    meta = json.loads(table.with_suffix(".meta.json").read_text())
    assert meta["schema_version"] == "1.2"
    assert meta["model_labels"]["clip"]["default"]["derivation"] == "embedding"
    assert meta["model_labels_from"]["clip"]["created_at"].startswith("2026-01-01")
    digest = meta["inputs"][0]["models"]["clip"]["labels_sha256"]
    assert isinstance(digest, str) and len(digest) == 16
    labels, _ = group_labels(table)
    assert set(labels) == {"clip", "aesthetics", "llstat"}


def test_1_1_input_is_unlabelled_not_an_error(tmp_path):
    table = _features(tmp_path, [labelled_fixture(tmp_path, schema="1.1")])
    labels, origin = group_labels(table)
    assert labels == {"aesthetics": None, "clip": None, "llstat": None} and origin == {}
    meta = json.loads(table.with_suffix(".meta.json").read_text())
    assert meta["inputs"][0]["models"]["clip"]["labels_sha256"] is None
    # and fits normally without a selection
    manifest, _ = _fit(tmp_path, table, "V_unl")
    assert manifest["per_member"]["clip"]["labelled"] is False
    assert manifest["per_member"]["clip"]["labels"] is None and manifest["selection"] is None


def test_a_table_built_before_1_2_reads_as_unlabelled(tmp_path):
    table = _features(tmp_path, [labelled_fixture(tmp_path)])
    side = table.with_suffix(".meta.json")
    meta = json.loads(side.read_text())
    for key in ("model_labels", "model_labels_from"):
        meta.pop(key)
    meta["schema_version"] = "1.1"
    side.write_text(json.dumps(meta))
    assert group_labels(table)[0] == {"aesthetics": None, "clip": None, "llstat": None}


# --- stacking -----------------------------------------------------------------


def test_structural_label_difference_refuses_the_stack(tmp_path):
    a = labelled_fixture(tmp_path, name="a.csv")
    models = labelled_models()
    models["llstat"]["labels"]["columns"]["llstat_contrast"] = {"interpretability": "opaque"}
    b = labelled_fixture(tmp_path, name="b.csv", models=models, ids=[f"other{i:03d}" for i in range(60)])
    with pytest.raises(SpaceError, match=r"`labels` mismatch for model 'llstat'.*llstat_contrast\.interpretability"):
        build_features([a, b], tmp_path / "f.parquet")


def test_labelled_and_unlabelled_inputs_of_one_model_refuse(tmp_path):
    a = labelled_fixture(tmp_path, name="a.csv")
    b = labelled_fixture(tmp_path, name="b.csv", schema="1.1", ids=[f"other{i:03d}" for i in range(60)])
    with pytest.raises(SpaceError, match="declares labels and .* declares none"):
        build_features([a, b], tmp_path / "f.parquet")


def test_descriptive_difference_warns_and_the_newer_sidecar_wins(tmp_path):
    old = labelled_fixture(tmp_path, name="a.csv", created_at="2026-01-01T00:00:00+00:00")
    models = labelled_models()
    models["clip"]["labels"]["default"]["Description"] = "CLIP ViT-B/32 image embedding"
    new = labelled_fixture(tmp_path, name="b.csv", models=models, created_at="2026-03-01T00:00:00+00:00",
                           ids=[f"other{i:03d}" for i in range(60)])
    # newer input first: order does not decide, created_at does
    with pytest.warns(UserWarning, match="descriptive labels differ.*keeping the newer"):
        build_features([new, old], tmp_path / "f.parquet")
    labels, origin = group_labels(tmp_path / "f.parquet")
    assert labels["clip"]["default"]["Description"] == "CLIP ViT-B/32 image embedding"
    assert origin["clip"]["path"].endswith("b.meta.json")


def test_tables_with_structurally_different_labels_refuse_a_fit(tmp_path, capsys):
    a = _features(tmp_path, [labelled_fixture(tmp_path, name="a.csv")], name="a_features.parquet")
    models = labelled_models()
    models["clip"]["labels"]["default"]["interpretability"] = "nameable"
    b = _features(tmp_path, [labelled_fixture(tmp_path, name="b.csv", models=models,
                                              ids=[f"other{i:03d}" for i in range(60)])],
                  name="b_features.parquet")
    argv = ["space", "fit", "--features", str(a), str(b), "--block", "V", "--members", "clip,llstat",
            "-o", str(tmp_path / "space"), *FAST]
    assert main(argv) == 1
    assert "`labels` mismatch for model 'clip'" in capsys.readouterr().err


# --- refresh ------------------------------------------------------------------


def test_refresh_labels_rewrites_only_the_sidecar(tmp_path, capsys):
    csv = labelled_fixture(tmp_path, schema="1.1")
    table = _features(tmp_path, [csv])
    before = table.read_bytes()
    assert group_labels(table)[0]["clip"] is None
    # the producer refresh: same values, a 1.2 sidecar
    side = csv.with_suffix(".meta.json")
    meta = json.loads(side.read_text())
    meta["schema_version"] = "1.2"
    meta["models"] = labelled_models()
    side.write_text(json.dumps(meta))
    assert main(["features", "--refresh-labels", str(table)]) == 0
    labels, _ = group_labels(table)
    assert labels["llstat"]["default"]["derivation"] == "formula"
    out = json.loads(table.with_suffix(".meta.json").read_text())
    assert out["schema_version"] == "1.2"
    assert out["refreshed"][-1]["fields"][-2:] == ["model_labels", "model_labels_from"]
    assert out["inputs"][0]["models"]["clip"]["labels_sha256"]
    assert table.read_bytes() == before
    assert "3/3 models declare labels" in capsys.readouterr().out


# --- selection grammar --------------------------------------------------------


def test_select_grammar():
    c = parse_select(["interpretability=nameable", "derivation!=zero-shot,probe"])
    assert select_text(c) == "interpretability=nameable;derivation!=probe,zero-shot"
    assert select_text(parse_select(select_text(c))) == select_text(c)
    assert parse_select("interpretability=nameable; derivation != probe") == parse_select(
        "interpretability=nameable;derivation!=probe")
    for bad, match in (("colour=red", "not one of"), ("derivation=magic", "vocabulary"),
                       ("interpretability", "does not parse"), ("derivation=bookkeeping", "never swept")):
        with pytest.raises(SpaceError, match=match):
            parse_select(bad)


# --- fits ---------------------------------------------------------------------


@pytest.fixture
def labelled_table(tmp_path):
    return _features(tmp_path, [labelled_fixture(tmp_path)])


def test_bookkeeping_is_never_swept_in_and_probes_are_flagged_circular(labelled_table, tmp_path, capsys):
    manifest, _ = _fit(tmp_path, labelled_table, "V_all")
    assert manifest["space_schema_version"] == "1.16"
    aes = manifest["per_member"]["aesthetics"]
    assert aes["bookkeeping_columns"] == ["aesthetics_n_crops"]
    assert "aesthetics_n_crops" not in manifest["member_features"]["aesthetics"]
    assert "aesthetics_n_crops" not in aes["labels"]
    # probe and zero-shot columns derived from clip's checkpoint, with clip in the block
    assert aes["circular_with"] == {c: ["clip"] for c in
                                    ("aesthetics_artistic", "aesthetics_quality", "aesthetics_score")}
    assert manifest["per_member"]["clip"]["circular_with"] == {}
    assert manifest["per_member"]["clip"]["checkpoint"] == CLIP_CKPT
    assert manifest["per_member"]["llstat"]["labels"]["llstat_luminance_mean"]["Units"] == "arbitrary"
    assert manifest["selection"] is None
    text = capsys.readouterr().out
    assert "bookkeeping column(s) left out: aesthetics_n_crops" in text and "circular_with" in text


def test_no_circular_flag_without_the_parent_member(labelled_table, tmp_path):
    manifest, _ = _fit(tmp_path, labelled_table, "V_nc", members="aesthetics,llstat")
    assert manifest["per_member"]["aesthetics"]["circular_with"] == {}


def test_select_nameable_is_column_level(labelled_table, tmp_path):
    manifest, _ = _fit(tmp_path, labelled_table, "V_name", "--select", "interpretability=nameable")
    sel = manifest["selection"]
    assert sel["expression"] == "interpretability=nameable"
    assert sel["dropped_members"] == ["clip"]  # every clip column is opaque
    assert manifest["members"] == ["aesthetics", "llstat"]
    assert manifest["member_features"]["llstat"] == ["llstat_colorfulness", "llstat_contrast",
                                                    "llstat_luminance_mean"]
    assert manifest["per_member"]["llstat"]["deselected_columns"] == ["llstat_hist_00", "llstat_hist_01"]
    assert all(e["interpretability"] == "nameable"
               for m in manifest["members"] for e in manifest["per_member"][m]["labels"].values())
    # the member is whitened on its selected columns alone
    assert manifest["per_member"]["llstat"]["dim"] == 3


def test_select_excluding_a_derivation_set_and_anding(labelled_table, tmp_path):
    manifest, _ = _fit(tmp_path, labelled_table, "V_np", "--select", "derivation!=probe,zero-shot")
    assert manifest["selection"]["dropped_members"] == ["aesthetics"]
    assert manifest["members"] == ["clip", "llstat"]
    both, _ = _fit(tmp_path, labelled_table, "V_and", "--select", "interpretability=nameable",
                   "--select", "derivation!=probe")
    assert both["selection"]["expression"] == "interpretability=nameable;derivation!=probe"
    assert both["member_features"]["aesthetics"] == ["aesthetics_quality"]  # the zero-shot one


def test_like_rebuilds_the_selection_from_the_manifest(labelled_table, tmp_path):
    _, out = _fit(tmp_path, labelled_table, "V_src", "--select", "interpretability=nameable")
    argv = ["space", "fit", "--like", str(out / "V_src.json"), "--no-criterion", "-o", str(out),
            "--stem", "V_copy"]
    assert main(argv) == 0
    copy = json.loads((out / "V_copy.json").read_text())
    assert copy["selection"]["expression"] == "interpretability=nameable"
    assert copy["like"]["selection_from"] == "manifest"
    assert copy["member_features"] == json.loads((out / "V_src.json").read_text())["member_features"]


def test_select_refuses_an_unlabelled_member_naming_its_table(tmp_path, capsys):
    table = _features(tmp_path, [labelled_fixture(tmp_path, schema="1.1")])
    argv = ["space", "fit", "--features", str(table), "--block", "V", "--members", "clip,llstat",
            "-o", str(tmp_path / "space"), "--select", "interpretability=nameable", *FAST]
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "have none" in err and str(table) in err and "--refresh-labels" in err


def test_select_refusal_message(tmp_path):
    from psytwill.space import apply_labels
    from psytwill.store import SpaceMatrix

    sp = {"a": SpaceMatrix(name="a", labels=["x"], X=np.zeros((1, 1)), features=["a_0"])}
    with pytest.raises(SpaceError, match=r"member\(s\) \['a'\] have none.*a: t\.parquet.*--refresh-labels"):
        apply_labels(sp, ["a"], column_labels={"a": None}, select="interpretability=nameable",
                     label_sources={"a": ["t.parquet"]})


def test_without_select_the_fit_matches_an_unlabelled_one(tmp_path):
    """No bookkeeping column, no --select: labels change nothing in the weights."""
    lab = _features(tmp_path, [labelled_fixture(tmp_path, name="lab.csv")], name="lab.parquet")
    unl = _features(tmp_path, [labelled_fixture(tmp_path, name="unl.csv", schema="1.1")], name="unl.parquet")
    a, out = _fit(tmp_path, lab, "V_lab", members="clip,llstat")
    b, _ = _fit(tmp_path, unl, "V_unl", members="clip,llstat")
    wa, wb = np.load(out / a["weights"]), np.load(out / b["weights"])
    assert sorted(wa.files) == sorted(wb.files)
    for key in wa.files:
        np.testing.assert_array_equal(wa[key], wb[key])


def test_projection_ignores_labels_a_table_gains_later(tmp_path):
    """A frozen fit projects by its recorded columns, whatever the table now declares."""
    unl = _features(tmp_path, [labelled_fixture(tmp_path, name="unl.csv", schema="1.1")], name="unl.parquet")
    manifest, out = _fit(tmp_path, unl, "V_old", members="aesthetics,llstat")
    assert "aesthetics_n_crops" in manifest["member_features"]["aesthetics"]
    lab = _features(tmp_path, [labelled_fixture(tmp_path, name="lab.csv")], name="lab.parquet")
    argv = ["space", "project", "--features", str(lab), "--space", str(out / "V_old.json"),
            "-o", str(tmp_path / "scores.csv")]
    assert main(argv) == 0
    assert len(pd.read_csv(tmp_path / "scores.csv")) == 60


# --- export -------------------------------------------------------------------


def test_space_labels_export(labelled_table, tmp_path):
    _, out = _fit(tmp_path, labelled_table, "V_all")
    assert main(["space", "labels", "--space", str(out / "V_all.json"), "-o", str(tmp_path / "lab.csv")]) == 0
    table = pd.read_csv(tmp_path / "lab.csv")
    assert "stimulus_id" not in table.columns
    assert len(table) == 6 + 3 + 5  # bookkeeping column excluded
    row = table.set_index("column").loc["aesthetics_score"]
    assert row["circular_with"] == "clip" and row["derivation"] == "probe"
    assert json.loads(row["source_validation"])[0]["metric"] == "spearman"
    cols = json.loads((tmp_path / "lab.columns.json").read_text())
    assert set(cols) == set(table["column"])
    lum = cols["llstat_luminance_mean"]
    assert lum["LongName"] == "Mean luminance" and lum["TermURL"].startswith("https://")
    assert lum["interpretability"] == "nameable" and lum["derivation"] == "formula"
    assert "derived_from" not in lum and "circular_with" not in lum
    assert cols["aesthetics_score"]["derived_from"] == CLIP_CKPT


def test_release_write_emits_labels_beside_it(labelled_table, tmp_path):
    _, out = _fit(tmp_path, labelled_table, "V_rel")
    rel_dir = tmp_path / "rel"
    argv = ["space", "release", "write", "--name", "demo", "--version", "0.1.0",
            "--blocks", str(out / "V_rel.json"), "-o", str(rel_dir)]
    assert main(argv) == 0
    rel = json.loads((rel_dir / "demo_0.1.0.json").read_text())
    assert rel["release_schema_version"] == "1.3"
    assert rel["labels"] == {"table": "demo_0.1.0_labels.csv", "columns": "demo_0.1.0_labels.columns.json"}
    table = pd.read_csv(rel_dir / rel["labels"]["table"])
    assert set(table["block"]) == {"V"}
    cols = json.loads((rel_dir / rel["labels"]["columns"]).read_text())
    assert cols["clip_000"]["interpretability"] == "opaque"
    # the release export also runs from `space labels --release`
    assert main(["space", "labels", "--release", str(rel_dir / "demo_0.1.0.json"),
                 "-o", str(tmp_path / "again.csv")]) == 0
    assert pd.read_csv(tmp_path / "again.csv").equals(table)


def test_old_manifests_still_load_and_export_refuses_cleanly(labelled_table, tmp_path, capsys):
    from psytwill.space import load_fit

    _, out = _fit(tmp_path, labelled_table, "V_new")
    path = out / "V_new.json"
    meta = json.loads(path.read_text())
    meta["space_schema_version"] = "1.15"
    meta.pop("selection")
    meta.pop("labels_source")
    for pm in meta["per_member"].values():
        for key in ("labelled", "checkpoint", "labels", "bookkeeping_columns", "deselected_columns",
                    "circular_with"):
            pm.pop(key)
    path.write_text(json.dumps(meta))
    assert load_fit(path).k == 2
    assert main(["space", "labels", "--space", str(path), "-o", str(tmp_path / "x.csv")]) == 1
    assert "no labels to export" in capsys.readouterr().err
    rel_dir = tmp_path / "rel"
    assert main(["space", "release", "write", "--name", "old", "--version", "0.1.0", "--blocks", str(path),
                 "-o", str(rel_dir)]) == 0
    assert json.loads((rel_dir / "old_0.1.0.json").read_text())["labels"] is None


def test_inventory_matcher_covers_word2psy_chunk_inventories():
    """word2psy declares chunk columns outside ``features``: the
    ``chunk_aggregates`` list and ``chunk_pooling``'s count column and pooled
    pattern. Labels keyed on those columns must validate."""
    from psytwill.labels import inventory_matcher, validate_labels

    pooled = {
        "features": {"pattern": "emb_{NNN}", "range": [0, 2], "count": 3, "level": "word"},
        "chunk_pooling": {"stat": "mean", "count_column": "emb_n_pooled",
                          "features": {"pattern": "emb_{NNN}", "range": [0, 2], "count": 3,
                                       "level": "chunk"}},
    }
    m = inventory_matcher(pooled)
    assert m("emb_000") and m("emb_n_pooled") and not m("emb_count")
    validate_labels(
        {"default": {"interpretability": "opaque", "derivation": "embedding"},
         "columns": {"emb_n_pooled": {"derivation": "bookkeeping"}}},
        model="emb", where="t", is_column=m)

    aggregated = {
        "features": {"columns": ["wf_length"], "count": 1, "level": "word"},
        "chunk_aggregates": {"stats": ["mean", "sd"], "columns": ["wf_length_mean", "wf_length_sd"]},
    }
    m = inventory_matcher(aggregated)
    assert m("wf_length") and m("wf_length_mean") and m("wf_length_sd") and not m("wf_length_max")


def test_inventory_matcher_none_without_inventory():
    from psytwill.labels import inventory_matcher

    assert inventory_matcher({"checkpoint": "x"}) is None
