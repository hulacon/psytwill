"""timelines: events -> registry ids -> ordered presentations with lags.

All synthetic and offline: a three-table registry, a subject's events across
two TB sessions plus one NAT run, and a small psytwill features table.
"""

import json

import numpy as np
import pandas as pd
import pytest

from psytwill.exceptions import InputError
from psytwill.timelines import (
    KEY_COLUMNS,
    TIMELINE_COLUMNS,
    TIMELINES_SCHEMA_VERSION,
    Registry,
    add_context_distance,
    add_lags,
    attach_features,
    build_timeline,
    load_item_features,
    parse_entities,
    read_events,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def registry_dir(tmp_path):
    d = tmp_path / "stimulus_registry"
    d.mkdir()
    pd.DataFrame(
        {"stimulus_id": ["shared0001_nsd00001", "shared0002_nsd00002", "shared0003_nsd00003"],
         "image_file": ["images/a.png", "images/b.png", "images/c.png"],
         "mmmId": [1, 2, 3], "nsdId": [1, 2, 3], "cocoId": [11, 12, 13], "cocoSplit": ["val"] * 3}
    ).to_csv(d / "shared1000.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"stimulus_id": ["cabin", "river", "spoon"], "itmno": [1, 2, 3],
         "presented_voice": ["nova", "echo", "onyx"]}
    ).to_csv(d / "twp1000.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"stimulus_id": ["snack-attack", "table-7"],
         "movie_name": ["Snack Attack", "Table 7"],
         "movie_name_variants": ["snack attack|SNACK ATTACK", ""],
         "video_file": ["movie_files/Snack_Attack.mov", "movie_files/Table_7.mov"]}
    ).to_csv(d / "movies.tsv", sep="\t", index=False)
    return d


def _tb_events(path, trials):
    """TB-encoding-shaped events: image + word rows share an onset; rest rows."""
    rows = []
    for i, t in enumerate(trials):
        onset = 9.0 + 4.5 * i
        if t is None:
            rows.append({"onset": onset, "duration": 3.0, "trial_type": "rest", "modality": "n/a",
                         "word": "n/a", "mmmId": "n/a", "voice": "n/a", "pairId": "n/a", "enCon": "n/a"})
            continue
        mmm, word, voice, pair = t
        shared = {"word": word, "mmmId": mmm, "voice": voice, "pairId": pair, "enCon": 1}
        rows.append({"onset": onset, "duration": 3.0, "trial_type": "image", "modality": "visual", **shared})
        rows.append({"onset": onset, "duration": 0.6, "trial_type": "word", "modality": "auditory", **shared})
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _nat_events(path, movies):
    rows = []
    t = 5.0
    for name in movies:
        rows.append({"onset": t, "duration": 2.0, "trial_type": "title", "movie_name": ""})
        rows.append({"onset": t + 2.0, "duration": 240.0, "trial_type": "movie", "movie_name": name})
        rows.append({"onset": t + 242.0, "duration": 10.0, "trial_type": "fixation", "movie_name": ""})
        t += 260.0
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


@pytest.fixture
def events(tmp_path):
    f = tmp_path / "func"
    f.mkdir()
    e1 = f / "sub-01_ses-04_task-TBencoding_run-01_events.tsv"
    e2 = f / "sub-01_ses-04_task-TBencoding_run-02_events.tsv"
    e3 = f / "sub-01_ses-05_task-TBencoding_run-01_events.tsv"
    e4 = f / "sub-01_ses-19_task-NATencoding_run-01_events.tsv"
    # ses-04 run-01: A, rest, B, A again (repeat within run)
    _tb_events(e1, [(1, "cabin", "nova", 1), None, (2, "river", "echo", 2), (1, "cabin", "nova", 1)])
    # ses-04 run-02: C
    _tb_events(e2, [(3, "spoon", "onyx", 3)])
    # ses-05 run-01: B again (repeat across sessions)
    _tb_events(e3, [(2, "river", "echo", 2)])
    _nat_events(e4, ["SNACK ATTACK", "Table 7"])
    return [e1, e2, e3, e4]


@pytest.fixture
def features_table(tmp_path):
    """Item-level CLIP for images (3-d), voice-specific 'egemaps' for words, a timed row to ignore."""
    rows = []
    emb = {"shared0001_nsd00001": [1, 0, 0], "shared0002_nsd00002": [0, 1, 0], "shared0003_nsd00003": [1, 0, 0]}
    for sid, vec in emb.items():
        for j, v in enumerate(vec):
            rows.append({"stimulus_id": sid, "voice": None, "time": None, "onset": None, "offset": None,
                         "model": "clip", "feature": f"clip_{j:03d}", "value": float(v)})
    for word, voice, val in (("cabin", "nova", 0.5), ("river", "echo", 0.7), ("spoon", "onyx", 0.9)):
        rows.append({"stimulus_id": word, "voice": voice, "time": None, "onset": None, "offset": None,
                     "model": "egemaps", "feature": "egemaps_loudness", "value": val})
    rows.append({"stimulus_id": "snack-attack", "voice": None, "time": 0.0, "onset": None, "offset": None,
                 "model": "clip", "feature": "clip_000", "value": 99.0})  # timed: must be ignored
    p = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------
# registry + resolution
# --------------------------------------------------------------------------

def test_registry_loads_all_three_tables(registry_dir):
    reg = Registry.from_dir(registry_dir)
    assert reg.rows == {"shared1000": 3, "twp1000": 3, "movies": 2}
    assert reg.by_mmm_id[2] == "shared0002_nsd00002"
    assert reg.by_movie_name["snack attack"] == "snack-attack"


def test_registry_missing_table_names_the_fix(tmp_path):
    (tmp_path / "shared1000.tsv").write_text("stimulus_id\tmmmId\n")
    with pytest.raises(InputError, match="lacks.*twp1000.tsv.*build_stimulus_registry"):
        Registry.from_dir(tmp_path)


def test_parse_entities_with_and_without_run():
    e = parse_entities("sub-03_ses-04_task-TBencoding_run-01_events.tsv")
    assert e == {"subject": "03", "session": "04", "task": "TBencoding", "run": "01"}
    assert parse_entities("sub-03_ses-30_task-motor_events.tsv")["run"] is None
    with pytest.raises(InputError, match="not a BIDS events file name"):
        parse_entities("motor_timing.csv")


def test_resolution_follows_the_three_rules(events, registry_dir):
    tl = read_events(events, Registry.from_dir(registry_dir))
    img = tl[tl["trial_type"] == "image"]
    assert set(img["stimulus_set"]) == {"shared1000"}
    assert img["stimulus_id"].iloc[0] == "shared0001_nsd00001"  # mmmId 1
    wrd = tl[tl["trial_type"] == "word"]
    assert set(wrd["stimulus_set"]) == {"twp1000"}
    assert list(zip(wrd["stimulus_id"], wrd["voice"]))[0] == ("cabin", "nova")
    mov = tl[tl["trial_type"] == "movie"]
    assert list(mov["stimulus_id"]) == ["snack-attack", "table-7"]  # case-insensitive variant
    non = tl[~tl["is_stimulus"]]
    assert set(non["trial_type"]) == {"rest", "title", "fixation"}
    assert non["stimulus_id"].isna().all()


def test_unresolved_reference_is_a_named_error(tmp_path, registry_dir):
    f = tmp_path / "sub-01_ses-04_task-TBencoding_run-01_events.tsv"
    _tb_events(f, [(1, "cabin", "nova", 1), (42, "zebra", "nova", 2)])
    with pytest.raises(InputError, match=r"(?s)2 stimulus reference.*mmmId 42.*zebra"):
        read_events([f], Registry.from_dir(registry_dir))


def test_word_without_voice_is_refused(tmp_path, registry_dir):
    f = tmp_path / "sub-01_ses-04_task-TBencoding_run-01_events.tsv"
    pd.DataFrame([{"onset": 9.0, "duration": 0.6, "trial_type": "word", "word": "cabin", "voice": "n/a"}]).to_csv(
        f, sep="\t", index=False)
    with pytest.raises(InputError, match="without a voice"):
        read_events([f], Registry.from_dir(registry_dir))


def test_order_is_session_run_onset(events, registry_dir):
    tl = read_events(events, Registry.from_dir(registry_dir))
    key = list(zip(tl["session"], tl["run"]))
    assert key == sorted(key, key=lambda k: (int(k[0]), int(k[1])))
    assert tl["row_idx"].tolist() == list(range(len(tl)))


# --------------------------------------------------------------------------
# lags
# --------------------------------------------------------------------------

def test_lags_in_trials_and_seconds(events, registry_dir):
    tl = add_lags(read_events(events, Registry.from_dir(registry_dir)))
    a = tl[tl["stimulus_id"] == "shared0001_nsd00001"]
    assert a["n_prior"].tolist() == [0, 1]
    first, second = a.iloc[0], a.iloc[1]
    assert np.isnan(first["lag_trials"]) and np.isnan(first["lag_seconds"])
    # Presentations between A and A within ses-04 run-01: A(word), B(image), B(word) -> A shown 4 later
    assert second["lag_trials"] == 4
    assert second["lag_seconds"] == pytest.approx((9.0 + 4.5 * 3) - 9.0)
    assert second["prev_session"] == "04" and second["prev_run"] == "01"


def test_lag_across_sessions_has_trials_but_no_seconds(events, registry_dir):
    tl = add_lags(read_events(events, Registry.from_dir(registry_dir)))
    b = tl[tl["stimulus_id"] == "shared0002_nsd00002"]
    assert len(b) == 2
    again = b.iloc[1]
    assert again["session"] == "05"
    assert again["lag_trials"] > 0
    assert np.isnan(again["lag_seconds"])  # different run: no shared clock
    assert again["prev_session"] == "04" and again["prev_run"] == "01"


def test_presentation_idx_skips_non_stimulus_rows(events, registry_dir):
    tl = add_lags(read_events(events, Registry.from_dir(registry_dir)))
    assert tl.loc[~tl["is_stimulus"], "presentation_idx"].isna().all()
    stim = tl.loc[tl["is_stimulus"], "presentation_idx"]
    assert stim.tolist() == list(range(len(stim)))


# --------------------------------------------------------------------------
# features + context
# --------------------------------------------------------------------------

def test_item_features_pivot_ignores_timed_rows(features_table):
    wide, model_cols = load_item_features(features_table)
    assert set(model_cols) == {"clip", "egemaps"}
    assert model_cols["clip"] == ["clip_000", "clip_001", "clip_002"]
    snack = wide[wide["stimulus_id"] == "snack-attack"]
    assert snack.empty  # its only row was timed


def test_attach_features_voice_specific_and_not(events, registry_dir, features_table):
    tl = add_lags(read_events(events, Registry.from_dir(registry_dir)))
    wide, _ = load_item_features(features_table)
    out = attach_features(tl, wide)
    img = out[(out["stimulus_id"] == "shared0002_nsd00002")].iloc[0]
    assert [img["clip_000"], img["clip_001"], img["clip_002"]] == [0.0, 1.0, 0.0]
    word = out[(out["stimulus_id"] == "river")].iloc[0]
    assert word["egemaps_loudness"] == 0.7
    assert np.isnan(word["clip_000"])  # images-only space stays NaN on words
    movie = out[out["stimulus_id"] == "table-7"].iloc[0]
    assert np.isnan(movie["clip_000"])  # no item-level features in the table


def test_context_distance_to_preceding_items(events, registry_dir, features_table):
    tl = add_lags(read_events(events, Registry.from_dir(registry_dir)))
    wide, cols = load_item_features(features_table)
    out = add_context_distance(attach_features(tl, wide), "clip", cols["clip"], k=5)
    col = "ctx_clip_k5_cosdist"
    run1 = out[(out["session"] == "04") & (out["run"] == "01") & out["trial_type"].eq("image")]
    # A first: no context. B: context = {A}, orthogonal -> 1.0. A again: context mean of (A, B) -> 1 - 1/sqrt(2)
    assert np.isnan(run1[col].iloc[0])
    assert run1[col].iloc[1] == pytest.approx(1.0)
    assert run1[col].iloc[2] == pytest.approx(1 - 1 / np.sqrt(2))
    # Words carry no clip features: they neither get a value nor enter the context.
    assert out.loc[out["trial_type"] == "word", col].isna().all()
    # A new run starts with no context.
    run2 = out[(out["session"] == "04") & (out["run"] == "02") & out["trial_type"].eq("image")]
    assert np.isnan(run2[col].iloc[0])


def test_context_needs_features_and_a_known_model(events, registry_dir, features_table, tmp_path):
    with pytest.raises(InputError, match="needs --features"):
        build_timeline(events, registry_dir, tmp_path / "t.parquet", context_model="clip")
    with pytest.raises(InputError, match="not among the attached models"):
        build_timeline(events, registry_dir, tmp_path / "t.parquet", features=features_table,
                       models=["clip"], context_model="egemaps")


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def test_build_timeline_writes_table_and_sidecar(events, registry_dir, features_table, tmp_path):
    out = tmp_path / "sub-01_timeline.parquet"
    summary = build_timeline(events, registry_dir, out, features=features_table,
                             context_model="clip", context_k=3)
    tl = pd.read_parquet(out)
    assert list(tl.columns[: len(TIMELINE_COLUMNS)]) == TIMELINE_COLUMNS
    assert "pairId" in tl.columns and "enCon" in tl.columns  # passthrough survives
    assert "mmmId" not in tl.columns  # consumed by the join
    assert "clip_000" in tl.columns and "ctx_clip_k3_cosdist" in tl.columns
    assert not tl.duplicated(KEY_COLUMNS).any()
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["schema_version"] == TIMELINES_SCHEMA_VERSION
    assert meta["table"] == "timelines"
    assert meta["output"]["key_columns"] == KEY_COLUMNS
    assert meta["sets"] == {"shared1000": 5, "twp1000": 5, "movies": 2}
    assert meta["context"] == {"model": "clip", "k": 3}
    assert summary["presentations"] == 12 and summary["models"] == ["clip", "egemaps"]


def test_build_timeline_without_features_and_as_tsv(events, registry_dir, tmp_path):
    out = tmp_path / "t.tsv"
    build_timeline(events, registry_dir, out)
    tl = pd.read_csv(out, sep="\t")
    assert len(tl) > 0 and "lag_trials" in tl.columns
    with pytest.raises(InputError, match="output must be"):
        build_timeline(events, registry_dir, tmp_path / "t.xlsx")


def test_cli_timelines(events, registry_dir, features_table, tmp_path, capsys):
    from psytwill.cli import main

    out = tmp_path / "cli.parquet"
    rc = main(["timelines", *map(str, events), "--registry", str(registry_dir),
               "--features", str(features_table), "--context-model", "clip", "-o", str(out)])
    assert rc == 0
    assert out.exists() and out.with_suffix(".meta.json").exists()
    assert "presentations" in capsys.readouterr().out
