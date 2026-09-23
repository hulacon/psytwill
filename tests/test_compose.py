"""compose: sparse runs onto the dense movie grid, grain-driven and generic.

All synthetic and offline: a generic events file that resolves by its own
stimulus_id column (no registry), plus three little item stores — one per
grain — written in the features-table schema.
"""

import json

import numpy as np
import pandas as pd
import pytest

from psytwill.compose import (
    COMPOSE_SCHEMA_VERSION,
    build_composed,
    classify_store,
    read_runs,
    slug_for,
    stream_for,
)
from psytwill.exceptions import InputError

W = 0.5  # test grid width


def _store(path, rows, checkpoints=None):
    """Write a features-schema parquet + minimal sidecar."""
    cols = ["stimulus_id", "voice", "time", "onset", "offset", "chunk_idx",
            "word_idx", "modality", "extractor", "extractor_version",
            "model", "feature", "value", "value_str"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df[cols].to_parquet(path, index=False)
    if checkpoints:
        meta = {"schema_version": "1.0",
                "inputs": [{"models": {m: {"checkpoint": c}
                                       for m, c in checkpoints.items()}}]}
        path.with_name(path.name.replace(".parquet", ".meta.json")).write_text(
            json.dumps(meta))
    return path


@pytest.fixture
def visual_store(tmp_path):
    rows = []
    for sid, a, b in (("imgA", 1.0, 0.0), ("imgB", 0.0, 1.0)):
        for j, v in enumerate((a, b)):
            rows.append({"stimulus_id": sid, "model": "clip",
                         "feature": f"clip_{j:03d}", "value": v,
                         "modality": "visual", "extractor": "viz2psy"})
        rows.append({"stimulus_id": sid, "model": "caption",
                     "feature": "caption_text", "value": None,
                     "value_str": f"a picture of {sid}",
                     "modality": "visual", "extractor": "viz2psy"})
    return _store(tmp_path / "img_features.parquet", rows,
                  checkpoints={"clip": "ViT-test/ckpt"})


@pytest.fixture
def audio_store(tmp_path):
    rows = []
    # cabin exists in two voices; river is voice-blind (voice null).
    for voice, val in (("nova", 0.5), ("echo", 9.9)):
        for t in (0.25, 0.75):
            rows.append({"stimulus_id": "cabin", "voice": voice, "time": t,
                         "model": "loudness", "feature": "loudness_rms",
                         "value": val, "modality": "audio",
                         "extractor": "aud2psy"})
    for t in (0.25, 0.75):
        rows.append({"stimulus_id": "river", "voice": None, "time": t,
                     "model": "loudness", "feature": "loudness_rms",
                     "value": 0.7, "modality": "audio", "extractor": "aud2psy"})
    return _store(tmp_path / "aud_features.parquet", rows)


@pytest.fixture
def text_store(tmp_path):
    rows = []
    for sid, v in (("cabin", 0.1), ("river", 0.2)):
        for j in range(2):
            rows.append({"stimulus_id": sid, "chunk_idx": 0, "word_idx": 0,
                         "model": "fasttext", "feature": f"fasttext_{j:03d}",
                         "value": v + j, "modality": "text",
                         "extractor": "word2psy"})
    return _store(tmp_path / "txt_features.parquet", rows)


@pytest.fixture
def events(tmp_path):
    f = tmp_path / "sub-01_ses-04_task-TBencoding_run-01_events.tsv"
    pd.DataFrame([
        {"onset": 9.0, "duration": 3.0, "stimulus_id": "imgA", "voice": "n/a"},
        {"onset": 9.0, "duration": 0.6, "stimulus_id": "cabin", "voice": "nova"},
        {"onset": 11.0, "duration": 1.0, "stimulus_id": "n/a", "voice": "n/a",
         "trial_type": "rest"},
        {"onset": 13.0, "duration": 3.0, "stimulus_id": "imgB", "voice": "n/a"},
        {"onset": 13.0, "duration": 0.6, "stimulus_id": "river", "voice": "n/a"},
    ]).to_csv(f, sep="\t", index=False)
    return f


SLUG = "sub-01_ses-04_task-TBencoding_run-01"


@pytest.fixture
def img_events(tmp_path):
    """Image-only run, for tests that give only the visual store."""
    f = tmp_path / "sub-01_ses-05_task-TBencoding_run-01_events.tsv"
    pd.DataFrame([
        {"onset": 9.0, "duration": 3.0, "stimulus_id": "imgA"},
        {"onset": 13.0, "duration": 3.0, "stimulus_id": "imgB"},
    ]).to_csv(f, sep="\t", index=False)
    return f


# --------------------------------------------------------------------------
# classification and routing
# --------------------------------------------------------------------------

def test_classify_grains(visual_store, audio_store, text_store):
    assert classify_store(visual_store).set_index("model")["grain"].to_dict() == {
        "clip": "untimed", "caption": "untimed"}
    assert classify_store(audio_store)["grain"].tolist() == ["gridded"]
    assert classify_store(text_store)["grain"].tolist() == ["chunked"]


def test_classify_refuses_mixed_grain(tmp_path):
    p = _store(tmp_path / "mixed_features.parquet", [
        {"stimulus_id": "x", "model": "m", "feature": "f", "value": 1.0,
         "time": 0.0, "modality": "visual"},
        {"stimulus_id": "x", "model": "m", "feature": "f", "value": 1.0,
         "chunk_idx": 0, "modality": "visual"},
    ])
    with pytest.raises(InputError, match="both a time grid and chunk"):
        classify_store(p)


def test_classify_needs_modality_and_takes_overrides(tmp_path):
    p = _store(tmp_path / "nomod_features.parquet", [
        {"stimulus_id": "x", "model": "m", "feature": "f", "value": 1.0},
    ])
    with pytest.raises(InputError, match="--modality-map m="):
        classify_store(p)
    cls = classify_store(p, {"m": "audio"})
    assert cls["modality"].tolist() == ["audio"]


def test_stream_routing_and_refusals():
    assert stream_for("visual", "untimed") == "movies_frames"
    assert stream_for("visual", "gridded") == "movies_frames"
    assert stream_for("audio", "gridded") == "movies_audio_frames"
    assert stream_for("text", "chunked") == "movies_transcript_words"
    assert stream_for("text", "untimed") == "movies_transcript_words"
    with pytest.raises(InputError, match="gridded text"):
        stream_for("text", "gridded")
    with pytest.raises(InputError, match="no composition rule"):
        stream_for("visual", "chunked")


def test_slug_strips_events_suffix():
    assert slug_for("a/b/sub-01_ses-04_task-X_run-01_events.tsv") == \
        "sub-01_ses-04_task-X_run-01"
    assert slug_for("story_listening.tsv") == "story_listening"


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------

@pytest.fixture
def composed(events, visual_store, audio_store, text_store, tmp_path):
    out = tmp_path / "run_root"
    summary = build_composed([events], [visual_store, audio_store, text_store],
                             out, window=W)
    return out, summary


def test_untimed_expands_onto_the_grid(composed):
    out, _ = composed
    df = pd.read_parquet(out / "features" / "movies_frames_features.parquet")
    a = df[(df["source_stimulus_id"] == "imgA") & (df["feature"] == "clip_000")]
    assert a["time"].tolist() == [9.0, 9.5, 10.0, 10.5, 11.0, 11.5]
    assert (a["value"] == 1.0).all()
    assert (df["stimulus_id"] == SLUG).all()
    cap = df[(df["model"] == "caption") & df["value_str"].notna()]
    assert set(cap["value_str"]) == {"a picture of imgA", "a picture of imgB"}


def test_gridded_shifts_and_respects_voice(composed):
    out, _ = composed
    df = pd.read_parquet(out / "features" / "movies_audio_frames_features.parquet")
    # 0.6 s words: the store's second frame (0.75) lands on a bin the word
    # covers only 20 % of, so it is left empty (majority-coverage rule)
    cabin = df[df["source_stimulus_id"] == "cabin"]
    assert sorted(cabin["time"]) == [9.25]
    assert (cabin["value"] == 0.5).all()  # nova, never the echo 9.9
    river = df[df["source_stimulus_id"] == "river"]
    assert sorted(river["time"]) == [13.25]  # voice-blind rows joined by id
    assert df.loc[df["time"] == 9.75, "value"].isna().all()


def test_chunked_renumbers_and_takes_the_trial_window(composed):
    out, _ = composed
    df = pd.read_parquet(out / "features" / "movies_transcript_words_features.parquet")
    one = df[df["feature"] == "fasttext_000"].sort_values("onset")
    assert one["chunk_idx"].tolist() == [0, 1]
    assert one["word_idx"].tolist() == [0, 1]
    assert one["onset"].tolist() == [9.0, 13.0]
    assert one["offset"].tolist() == [9.6, 13.6]


def test_empty_bins_are_explicit_nan_rows(composed):
    out, _ = composed
    df = pd.read_parquet(out / "features" / "movies_frames_features.parquet")
    # run_end = 16.0 -> 32 bins; every (model, feature) covers every bin.
    per_feat = df.groupby("feature")["time"].nunique()
    assert (per_feat == 32).all()
    gap = df[(df["time"] == 0.0) & (df["feature"] == "clip_000")]
    assert len(gap) == 1 and np.isnan(gap["value"].iloc[0])
    # audio fill stamps bin centers
    aud = pd.read_parquet(out / "features" / "movies_audio_frames_features.parquet")
    assert aud["time"].nunique() == 32
    assert set(np.round((aud["time"] % W).unique(), 6)) == {0.25}


def test_sparse_skips_the_fill(img_events, visual_store, tmp_path):
    out = tmp_path / "sparse_root"
    build_composed([img_events], [visual_store], out, window=W, sparse=True)
    df = pd.read_parquet(out / "features" / "movies_frames_features.parquet")
    assert df["value"].notna().sum() + df["value_str"].notna().sum() == len(df)
    assert df["time"].min() == 9.0


def test_sidecar_carries_contract(composed, visual_store):
    out, summary = composed
    meta = json.loads(
        (out / "features" / "movies_frames_features.meta.json").read_text())
    assert meta["schema_version"] == COMPOSE_SCHEMA_VERSION
    assert meta["table"] == "composed"
    assert meta["grid"] == {"window": W, "stamp": "start", "fill": "full",
                            "min_coverage": 0.5}
    assert meta["models"]["clip"]["checkpoint"] == "ViT-test/ckpt"
    assert meta["models"]["clip"]["comparable"] is None
    assert meta["runs"][SLUG]["n_presentations"] == 4
    assert meta["runs"][SLUG]["entities"]["subject"] == "01"
    assert meta["inputs_signature"]["params"]["window"] == W
    assert "experimental time" in meta["time_semantics"]


def test_idempotent_until_inputs_change(img_events, visual_store, tmp_path):
    out = tmp_path / "idem_root"
    s1 = build_composed([img_events], [visual_store], out, window=W)
    assert "up_to_date" not in s1
    s2 = build_composed([img_events], [visual_store], out, window=W)
    assert s2["up_to_date"] is True
    # a changed input (different size) recomposes
    df = pd.read_parquet(visual_store)
    pd.concat([df, df.iloc[[0]].assign(value=5.0, stimulus_id="imgC")]
              ).to_parquet(visual_store, index=False)
    s3 = build_composed([img_events], [visual_store], out, window=W)
    assert "up_to_date" not in s3
    s4 = build_composed([img_events], [visual_store], out, window=W, force=True)
    assert "up_to_date" not in s4


def test_dry_run_writes_nothing(img_events, visual_store, tmp_path):
    out = tmp_path / "dry_root"
    summary = build_composed([img_events], [visual_store], out, window=W,
                             dry_run=True)
    assert summary["dry_run"] is True
    assert not out.exists()
    est = summary["streams"]["movies_frames"]
    assert est["estimated_rows_lower_bound"] > 0


def test_unmatched_presentation_is_an_error(events, visual_store, tmp_path):
    # visual store alone: cabin and river have nowhere to compose
    with pytest.raises(InputError, match="no rows in any given store.*cabin"):
        build_composed([events], [visual_store], tmp_path / "x", window=W)


def test_model_collision_across_stores_is_refused(events, visual_store,
                                                  tmp_path):
    dup = _store(tmp_path / "dup_features.parquet", [
        {"stimulus_id": "cabin", "model": "clip", "feature": "clip_000",
         "value": 1.0, "modality": "visual"},
        {"stimulus_id": "river", "model": "clip", "feature": "clip_000",
         "value": 1.0, "modality": "visual"},
    ])
    with pytest.raises(InputError, match="appears in both"):
        build_composed([events], [visual_store, dup], tmp_path / "x", window=W)


def test_models_subset_must_exist(img_events, visual_store, tmp_path):
    with pytest.raises(InputError, match="not found in any given store"):
        build_composed([img_events], [visual_store], tmp_path / "x", window=W,
                       models=["nope"])


# --------------------------------------------------------------------------
# the no-special-case claim
# --------------------------------------------------------------------------

def test_composed_loads_beside_a_real_movie_table(composed, tmp_path):
    from psytwill.store import align_spaces, load_spaces

    out, _ = composed
    movie_rows = []
    for t in np.arange(0.0, 3.0, W):
        for j in range(2):
            movie_rows.append({"stimulus_id": "some-film", "time": t,
                               "model": "clip", "feature": f"clip_{j:03d}",
                               "value": float(j) + t, "modality": "visual",
                               "extractor": "viz2psy"})
    movie = _store(tmp_path / "movie_features.parquet", movie_rows)

    composed_df = pd.read_parquet(out / "features" / "movies_frames_features.parquet")
    both = tmp_path / "both_features.parquet"
    pd.concat([pd.read_parquet(movie), composed_df], ignore_index=True
              ).to_parquet(both, index=False)

    spaces = load_spaces(both, key=("stimulus_id", "time"), window=W,
                         models=["clip"])
    labels = spaces["clip"].labels
    assert any(lab.startswith("some-film|") for lab in labels)
    assert any(lab.startswith(SLUG + "|") for lab in labels)
    aligned, common = align_spaces(spaces)
    assert len(common) == len(labels)


# --------------------------------------------------------------------------
# runs / CLI
# --------------------------------------------------------------------------

def test_two_files_one_slug_is_refused(events, tmp_path):
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    twin = other_dir / events.name
    twin.write_text(events.read_text())
    with pytest.raises(InputError, match="distinct name"):
        read_runs([events, twin], None)


def test_onset_column_override_and_missing(events):
    with pytest.raises(InputError, match="--onset-column"):
        read_runs([events], None, onset_column="onset_actual")


def test_cli_compose_json(events, visual_store, audio_store, text_store,
                          tmp_path, capsys):
    from psytwill.cli import main

    out = tmp_path / "cli_root"
    rc = main(["compose", str(events), "--stores", str(visual_store),
               str(audio_store), str(text_store), "-o", str(out), "--window",
               str(W), "--json"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert set(summary["streams"]) == {"movies_frames", "movies_audio_frames",
                                       "movies_transcript_words"}
    assert (out / "features" / "movies_frames_features.parquet").exists()


# --------------------------------------------------------------------------
# bin coverage
# --------------------------------------------------------------------------

def _one_run(tmp_path, rows, name="sub-01_ses-06_task-TBencoding_run-01"):
    f = tmp_path / f"{name}_events.tsv"
    pd.DataFrame(rows).to_csv(f, sep="\t", index=False)
    return f


def test_untimed_takes_bins_the_item_mostly_covers(visual_store, tmp_path):
    ev = _one_run(tmp_path, [
        {"onset": 9.1, "duration": 3.0, "stimulus_id": "imgA"},   # off-grid
        {"onset": 14.0, "duration": 0.6, "stimulus_id": "imgB"},  # 1.2 bins
    ])
    out = tmp_path / "cov_root"
    build_composed([ev], [visual_store], out, window=W, sparse=True)
    df = pd.read_parquet(out / "features" / "movies_frames_features.parquet")
    a = df[(df["source_stimulus_id"] == "imgA") & (df["feature"] == "clip_000")]
    # 9.0 is 80 % covered (kept), 12.0 only 20 % (dropped)
    assert a["time"].tolist() == [9.0, 9.5, 10.0, 10.5, 11.0, 11.5]
    b = df[(df["source_stimulus_id"] == "imgB") & (df["feature"] == "clip_000")]
    assert b["time"].tolist() == [14.0]


def test_short_presentation_keeps_its_best_bin(audio_store, tmp_path):
    ev = _one_run(tmp_path, [
        {"onset": 9.1, "duration": 0.2, "stimulus_id": "river"},  # 40 % of one bin
    ])
    out = tmp_path / "short_root"
    build_composed([ev], [audio_store], out, window=W, sparse=True)
    df = pd.read_parquet(out / "features" / "movies_audio_frames_features.parquet")
    assert sorted(df["time"]) == [9.25]


def test_min_coverage_is_a_parameter_and_is_validated(events, visual_store,
                                                      audio_store, text_store,
                                                      tmp_path):
    out = tmp_path / "loose_root"
    build_composed([events], [visual_store, audio_store, text_store], out,
                   window=W, min_coverage=0.1, sparse=True)
    df = pd.read_parquet(out / "features" / "movies_audio_frames_features.parquet")
    assert sorted(df.loc[df["source_stimulus_id"] == "cabin", "time"]) == [9.25, 9.75]
    with pytest.raises(InputError, match="min-coverage"):
        build_composed([events], [visual_store], tmp_path / "x", window=W,
                       min_coverage=0.0)


# --------------------------------------------------------------------------
# comparability
# --------------------------------------------------------------------------

def _comp(tmp_path, rows):
    p = tmp_path / "comparability.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return p


def test_comparability_labels_land_in_the_sidecar(img_events, visual_store,
                                                   tmp_path):
    comp = _comp(tmp_path, [
        {"model": "clip", "comparable": "item", "note": "id 1.000"},
        {"model": "notcomposed", "comparable": "none", "note": ""},
    ])
    out = tmp_path / "comp_root"
    s = build_composed([img_events], [visual_store], out, window=W,
                       comparability=comp)
    meta = json.loads(
        (out / "features" / "movies_frames_features.meta.json").read_text())
    assert meta["models"]["clip"]["comparable"] == "item"
    assert meta["models"]["clip"]["comparability_note"] == "id 1.000"
    assert meta["models"]["caption"]["comparable"] is None
    assert meta["comparability"]["table"] == str(comp.resolve())
    assert set(meta["comparability"]["labels"]) == {"item", "item+offset",
                                                    "window", "none"}
    assert s["comparability"]["unlabelled"] == ["caption"]


def test_relabel_recomposes(img_events, visual_store, tmp_path):
    comp = _comp(tmp_path, [{"model": "clip", "comparable": "item"}])
    out = tmp_path / "relabel_root"
    build_composed([img_events], [visual_store], out, window=W, comparability=comp)
    s2 = build_composed([img_events], [visual_store], out, window=W,
                        comparability=comp)
    assert s2["up_to_date"] is True
    _comp(tmp_path, [{"model": "clip", "comparable": "none"}])  # same byte count
    s3 = build_composed([img_events], [visual_store], out, window=W,
                        comparability=comp)
    assert "up_to_date" not in s3
    meta = json.loads(
        (out / "features" / "movies_frames_features.meta.json").read_text())
    assert meta["models"]["clip"]["comparable"] == "none"


@pytest.mark.parametrize("rows, match", [
    ([{"model": "clip", "comparable": "maybe"}], "unknown comparable label"),
    ([{"model": "clip"}], "lacks column"),
    ([{"model": "clip", "comparable": "item"},
      {"model": "clip", "comparable": "none"}], "listed twice"),
])
def test_comparability_table_is_validated(img_events, visual_store, tmp_path,
                                          rows, match):
    comp = _comp(tmp_path, rows)
    with pytest.raises(InputError, match=match):
        build_composed([img_events], [visual_store], tmp_path / "x", window=W,
                       comparability=comp)


def test_missing_comparability_table_is_loud(img_events, visual_store, tmp_path):
    with pytest.raises(InputError, match="comparability table not found"):
        build_composed([img_events], [visual_store], tmp_path / "x", window=W,
                       comparability=tmp_path / "nope.tsv")
