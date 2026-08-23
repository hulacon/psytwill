"""`psytwill features` — the long-form feature table (Contract B §4.2.4).

Fixtures reuse the contract-test shapes (CSV + §4.1 sidecar per
extractor). CSV outputs are exercised throughout so the suite stays free
of a pyarrow dependency; one test covers parquet and skips without it.
"""

import json

import numpy as np
import pandas as pd
import pytest

from psytwill.exceptions import InputError, SpaceError
from psytwill.features import (
    FEATURES_SCHEMA_VERSION,
    KEY_COLUMNS,
    OUTPUT_COLUMNS,
    build_features,
)

CLIP_CKPT = "ViT-B-32/laion2b_s34b_b79k"


def _write(path, df, sidecar=None):
    df.to_csv(path, index=False)
    if sidecar is not None:
        path.with_suffix(".meta.json").write_text(json.dumps(sidecar))
    return path


def image_fixture(tmp_path, checkpoint=CLIP_CKPT, name="img_scores.csv"):
    """viz2psy-shaped: stimulus rows, clip embedding + scalar + string."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "stimulus_id": [f"shared{i:04d}_nsd{i:05d}" for i in range(4)],
        "filename": [f"shared{i:04d}_nsd{i:05d}.png" for i in range(4)],
        **{f"clip_{i:03d}": rng.normal(size=4) for i in range(4)},
        "resmem_memorability": rng.uniform(size=4),
        "caption_text": [f"a photo of thing {i}" for i in range(4)],
    })
    sidecar = {
        "schema_version": "1.0",
        "extractor": "viz2psy",
        "extractor_version": "0.7.0",
        "models": {
            "clip": {"checkpoint": checkpoint},
            "resmem": {"checkpoint": "resmem-pretrained"},
            "caption": {"checkpoint": "Salesforce/blip-image-captioning-large",
                        "dtype": "string"},
        },
    }
    return _write(tmp_path / name, df, sidecar)


def frames_fixture(tmp_path):
    """aud2psy-shaped: one movie, time-resolved rows, voice absent."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "stimulus_id": ["adventure-time"] * 3,
        "time": [0.25, 0.75, 1.25],
        **{f"ebind_audio_{i:04d}": rng.normal(size=3) for i in range(4)},
    })
    sidecar = {
        "schema_version": "1.0",
        "extractor": "aud2psy",
        "extractor_version": "0.13.1",
        "models": {"ebind_audio": {"checkpoint": "encord-team/ebind-full"}},
    }
    return _write(tmp_path / "movie_frames.csv", df, sidecar)


def test_long_table_schema_and_keys(tmp_path):
    out = tmp_path / "features.csv"
    summary = build_features([image_fixture(tmp_path)], out)
    table = pd.read_csv(out)
    assert list(table.columns) == OUTPUT_COLUMNS
    # 4 stimuli x (4 clip + 1 resmem + 1 caption) = 24 rows
    assert len(table) == 24 == summary["rows"]
    assert set(table["model"]) == {"clip", "resmem", "caption"}
    assert (table["modality"] == "visual").all()
    assert (table["extractor"] == "viz2psy").all()
    # String feature lands in value_str, not dropped and not in value.
    captions = table[table["feature"] == "caption_text"]
    assert captions["value"].isna().all()
    assert captions["value_str"].str.startswith("a photo").all()
    # Sidecar carries the features-table schema + provenance.
    meta = json.loads((tmp_path / "features.meta.json").read_text())
    assert meta["schema_version"] == FEATURES_SCHEMA_VERSION
    assert meta["table"] == "features"
    assert meta["output"]["key_columns"] == KEY_COLUMNS
    assert meta["inputs"][0]["models"]["clip"]["checkpoint"] == CLIP_CKPT


def test_multi_input_and_time_key(tmp_path):
    out = tmp_path / "features.csv"
    build_features([image_fixture(tmp_path), frames_fixture(tmp_path)], out)
    table = pd.read_csv(out)
    frames = table[table["model"] == "ebind_audio"]
    assert (frames["modality"] == "audio").all()
    assert sorted(frames["time"].unique()) == [0.25, 0.75, 1.25]
    # Image rows keep a null time; key uniqueness holds across inputs.
    assert table[table["model"] == "clip"]["time"].isna().all()
    assert not table.duplicated(subset=KEY_COLUMNS).any()


def test_duplicate_keys_refused(tmp_path):
    a = image_fixture(tmp_path, name="a.csv")
    b = image_fixture(tmp_path, name="b.csv")
    with pytest.raises(InputError, match="duplicate feature key"):
        build_features([a, b], tmp_path / "features.csv")


def test_checkpoint_mismatch_refused(tmp_path):
    a = image_fixture(tmp_path, name="a.csv")
    b = image_fixture(tmp_path, checkpoint="ViT-L-14/openai", name="b.csv")
    with pytest.raises(SpaceError, match="Checkpoint mismatch"):
        build_features([a, b], tmp_path / "features.csv")


def test_legacy_input_null_provenance(tmp_path):
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "chunk_label": [f"c{i}" for i in range(3)],
        **{f"minilm_{i:03d}": rng.normal(size=3) for i in range(3)},
    })
    csv = _write(tmp_path / "plain.csv", df)
    with pytest.warns(UserWarning):
        summary = build_features([csv], tmp_path / "features.csv")
    table = pd.read_csv(tmp_path / "features.csv")
    # No sidecar: embedding detection still attributes the model; the
    # provenance columns stay null and the label chain fills stimulus_id.
    assert (table["model"] == "minilm").all()
    assert table["extractor"].isna().all() and table["modality"].isna().all()
    assert set(table["stimulus_id"]) == {"c0", "c1", "c2"}
    assert summary["inputs"][0]["label_column"] == "chunk_label"


def test_unattributed_column_kept_with_null_model(tmp_path):
    csv = image_fixture(tmp_path)
    df = pd.read_csv(csv)
    df["mystery_score"] = 0.5
    df.to_csv(csv, index=False)
    with pytest.warns(UserWarning, match="mystery_score"):
        build_features([csv], tmp_path / "features.csv")
    table = pd.read_csv(tmp_path / "features.csv")
    mystery = table[table["feature"] == "mystery_score"]
    assert len(mystery) == 4 and mystery["model"].isna().all()


def test_bad_output_suffix_states_fix(tmp_path):
    with pytest.raises(InputError, match=r"\.parquet"):
        build_features([image_fixture(tmp_path)], tmp_path / "features.xlsx")


def test_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    out = tmp_path / "features.parquet"
    build_features([image_fixture(tmp_path), frames_fixture(tmp_path)], out)
    table = pd.read_parquet(out)
    assert list(table.columns) == OUTPUT_COLUMNS
    assert str(table["value"].dtype) == "float64"
    assert str(table["time"].dtype) == "float64"


def test_cli_features_verb(tmp_path, capsys):
    from psytwill.cli import main

    csv = image_fixture(tmp_path)
    out = tmp_path / "features.csv"
    assert main(["features", str(csv), "-o", str(out)]) == 0
    assert out.exists() and out.with_suffix(".meta.json").exists()
    stdout = capsys.readouterr().out
    assert "24 rows" in stdout and "3 models" in stdout


# --------------------------------------------------------------------------
# Streaming writes (psytwill 0.5.0)
#
# build_features used to melt every input, concat, then write. On the
# MMMData store that is ~302 M rows at ~450 B/row resident -- ~135 GB, which
# no node has. Inputs are now melted and written one at a time, so peak
# memory tracks the largest single input instead of the total. These cover
# the guarantees that rewrite had to preserve.
# --------------------------------------------------------------------------

def test_one_row_group_per_input(tmp_path):
    """The streaming write is observable: N inputs -> N parquet row groups."""
    pq = pytest.importorskip("pyarrow.parquet")
    inputs = [image_fixture(tmp_path, name=f"s{i}.csv") for i in range(3)]
    # Disambiguate so the three inputs do not collide on the key.
    for i, path in enumerate(inputs):
        df = pd.read_csv(path)
        df["stimulus_id"] = df["stimulus_id"] + f"_r{i}"
        df.to_csv(path, index=False)

    out = tmp_path / "features.parquet"
    build_features(inputs, out)
    assert pq.ParquetFile(out).num_row_groups == 3


def test_refusal_leaves_no_output_and_no_partial(tmp_path):
    """A mid-stream refusal must not leave a truncated table behind."""
    a = image_fixture(tmp_path, name="a.csv")
    b = image_fixture(tmp_path, checkpoint="ViT-L-14/openai", name="b.csv")
    out = tmp_path / "features.parquet"
    with pytest.raises(SpaceError):
        build_features([a, b], out)
    assert not out.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_refusal_does_not_clobber_an_existing_table(tmp_path):
    """The previous good table survives a failed rebuild."""
    good = image_fixture(tmp_path, name="good.csv")
    out = tmp_path / "features.parquet"
    build_features([good], out)
    before = out.read_bytes()

    a = image_fixture(tmp_path, name="a.csv")
    b = image_fixture(tmp_path, checkpoint="ViT-L-14/openai", name="b.csv")
    with pytest.raises(SpaceError):
        build_features([a, b], out)
    assert out.read_bytes() == before


def test_streaming_matches_a_single_concat(tmp_path):
    """Row-for-row equality with the pre-streaming semantics."""
    inputs = [image_fixture(tmp_path, name=f"s{i}.csv") for i in range(3)]
    for i, path in enumerate(inputs):
        df = pd.read_csv(path)
        df["stimulus_id"] = df["stimulus_id"] + f"_r{i}"
        df.to_csv(path, index=False)

    out = tmp_path / "features.csv"
    summary = build_features(inputs, out)
    table = pd.read_csv(out)

    # The reference: melt each input and concat, the way it used to work.
    from psytwill.features import MODALITY_MAP, _coerce, _melt_input
    parts = [_melt_input(p, MODALITY_MAP)[0] for p in inputs]
    expected = _coerce(pd.concat(parts, ignore_index=True))

    assert len(table) == len(expected) == summary["rows"]
    assert list(table.columns) == OUTPUT_COLUMNS
    assert table["feature"].tolist() == expected["feature"].tolist()
    assert table["stimulus_id"].tolist() == expected["stimulus_id"].tolist()


def test_meta_totals_are_accumulated_not_recomputed(tmp_path):
    """rows / n_stimuli / models come from the stream, not a resident table."""
    inputs = [image_fixture(tmp_path, name=f"s{i}.csv") for i in range(2)]
    for i, path in enumerate(inputs):
        df = pd.read_csv(path)
        df["stimulus_id"] = df["stimulus_id"] + f"_r{i}"
        df.to_csv(path, index=False)

    out = tmp_path / "features.csv"
    summary = build_features(inputs, out)
    table = pd.read_csv(out)
    meta = json.loads((tmp_path / "features.meta.json").read_text())

    assert meta["output"]["rows"] == len(table) == summary["rows"]
    assert meta["n_stimuli"] == table["stimulus_id"].nunique() == 8
    assert meta["models"] == sorted(table["model"].dropna().unique())


def test_duplicate_detection_survives_a_hash_collision(tmp_path, monkeypatch):
    """Candidate collisions are confirmed against real keys, not trusted.

    _key_hash is 64-bit, so distinct keys can collide. Forcing every row to
    one hash must not manufacture a duplicate that is not there.
    """
    import psytwill.features as F

    inputs = [image_fixture(tmp_path, name=f"s{i}.csv") for i in range(2)]
    for i, path in enumerate(inputs):
        df = pd.read_csv(path)
        df["stimulus_id"] = df["stimulus_id"] + f"_r{i}"
        df.to_csv(path, index=False)

    monkeypatch.setattr(
        F, "_key_hash", lambda table: np.zeros(len(table), dtype="uint64")
    )
    out = tmp_path / "features.csv"
    build_features(inputs, out)          # must NOT raise
    assert out.exists()


def test_real_duplicates_still_refused_across_many_inputs(tmp_path):
    """The exact check runs only on suspects, but still catches them."""
    inputs = [image_fixture(tmp_path, name=f"s{i}.csv") for i in range(4)]
    for i, path in enumerate(inputs[1:3], start=1):
        df = pd.read_csv(path)
        df["stimulus_id"] = df["stimulus_id"] + f"_r{i}"
        df.to_csv(path, index=False)
    # inputs[0] and inputs[3] keep the same ids -> one real collision, found
    # only after two unrelated inputs have already streamed out.
    with pytest.raises(InputError, match="duplicate feature key"):
        build_features(inputs, tmp_path / "features.csv")


def test_duplicate_confirmation_is_bounded(tmp_path, monkeypatch):
    """Confirming duplicates must not materialise the whole suspect set.

    When a column is duplicated across every input the suspect set is the
    table -- tens of millions of rows in the MMMData store. An unbounded
    confirmation pass OOMed a 32 GB job; only a sample is inspected now.
    """
    import psytwill.features as F

    monkeypatch.setattr(F, "DUP_CONFIRM_HASHES", 2)
    seen = []
    real_melt = F._melt_input

    def counting_melt(path, mm):
        seen.append(str(path))
        return real_melt(path, mm)

    monkeypatch.setattr(F, "_melt_input", counting_melt)

    # Four identical inputs: every key collides, four ways.
    inputs = [image_fixture(tmp_path, name=f"s{i}.csv") for i in range(4)]
    with pytest.raises(InputError, match="duplicate feature key"):
        build_features(inputs, tmp_path / "features.csv")

    # 4 melts to stream + at most 4 to confirm; never a re-melt per suspect.
    assert len(seen) <= 8


def test_duplicate_count_is_reported_from_the_hash_pass(tmp_path):
    """The count covers all duplicates, not just the confirmed sample."""
    inputs = [image_fixture(tmp_path, name=f"s{i}.csv") for i in range(3)]
    with pytest.raises(InputError) as exc:
        build_features(inputs, tmp_path / "features.csv")
    # 3 identical inputs -> 2 excess rows per key across all keys.
    one = build_features([inputs[0]], tmp_path / "one.csv")
    assert f"~{one['rows'] * 2} duplicate" in str(exc.value)


# --------------------------------------------------------------------------
# Structural ordinals in the key, and empty inputs (psytwill 0.6.0)
# --------------------------------------------------------------------------

def text_fixture(tmp_path, name, n_chunks=2, words_per_chunk=3, timed=False):
    """word2psy-shaped word-level table: untimed unless asked otherwise.

    word2psy times nothing -- only aud2psy transcripts carry real onsets --
    so the default here is the shape that actually dominates the store.
    """
    rows = []
    for c in range(n_chunks):
        for w in range(words_per_chunk):
            rows.append({
                "stimulus_id": "shared0001",
                "chunk_idx": c,
                "word_idx": w,
                "onset": float(c * 10 + w) if timed else None,
                "offset": float(c * 10 + w + 1) if timed else None,
                "wordform_length": float(w + 3),
            })
    df = pd.DataFrame(rows)
    sidecar = {
        "schema_version": "1.0",
        "extractor": "word2psy",
        "extractor_version": "0.5.1",
        "models": {"wordform": {"checkpoint": "builtin",
                                "features": {"columns": ["wordform_length"]}}},
    }
    return _write(tmp_path / name, df, sidecar)


def test_untimed_word_rows_are_keyed_by_their_ordinals(tmp_path):
    """The P2 case: all-null onset, so word_idx must carry the identity."""
    path = text_fixture(tmp_path, "w.csv")
    out = tmp_path / "features.csv"
    summary = build_features([path], out)
    table = pd.read_csv(out)

    assert summary["rows"] == 6                      # 2 chunks x 3 words
    assert not table.duplicated(subset=KEY_COLUMNS).any()
    assert set(table["chunk_idx"]) == {0, 1}
    assert set(table["word_idx"]) == {0, 1, 2}
    assert table["onset"].isna().all()


def test_multiple_untimed_chunks_per_stimulus_survive(tmp_path):
    """The P1 case: five captions of one image, no timing to separate them."""
    rows = [{"stimulus_id": "shared0001", "chunk_idx": i,
             "readability_flesch": float(i)} for i in range(5)]
    sidecar = {
        "schema_version": "1.0", "extractor": "word2psy",
        "extractor_version": "0.5.1",
        "models": {"readability": {"checkpoint": "builtin",
                   "features": {"columns": ["readability_flesch"]}}},
    }
    path = _write(tmp_path / "c.csv", pd.DataFrame(rows), sidecar)
    summary = build_features([path], tmp_path / "features.csv")
    assert summary["rows"] == 5
    assert summary["n_stimuli"] == 1


def test_ordinals_are_in_the_fixed_schema(tmp_path):
    path = text_fixture(tmp_path, "w.csv")
    build_features([path], tmp_path / "features.csv")
    table = pd.read_csv(tmp_path / "features.csv")
    assert list(table.columns) == OUTPUT_COLUMNS
    assert "chunk_idx" in KEY_COLUMNS and "word_idx" in KEY_COLUMNS


def test_ordinals_are_nullable_for_inputs_that_lack_them(tmp_path):
    """Image tables have no chunks; the columns stay null, not absent."""
    path = image_fixture(tmp_path)
    build_features([path], tmp_path / "features.csv")
    table = pd.read_csv(tmp_path / "features.csv")
    assert table["chunk_idx"].isna().all()
    assert table["word_idx"].isna().all()


def test_empty_input_is_skipped_not_fatal(tmp_path):
    """A movie with no speech must not take down its 60-movie group."""
    good = image_fixture(tmp_path, name="good.csv")
    empty = _write(
        tmp_path / "empty.csv",
        pd.DataFrame(columns=["stimulus_id", "clip_000"]),
        {"schema_version": "1.0", "extractor": "viz2psy",
         "extractor_version": "0.7.1",
         "models": {"clip": {"checkpoint": CLIP_CKPT}}},
    )
    out = tmp_path / "features.csv"
    with pytest.warns(UserWarning, match="no rows"):
        summary = build_features([good, empty], out)
    assert summary["rows"] > 0
    assert summary["skipped_empty"] == [str(empty)]
    meta = json.loads((tmp_path / "features.meta.json").read_text())
    assert meta["skipped_empty"] == [str(empty)]


def test_all_inputs_empty_still_writes_a_schema_correct_table(tmp_path):
    empty = _write(
        tmp_path / "empty.csv",
        pd.DataFrame(columns=["stimulus_id", "clip_000"]),
        {"schema_version": "1.0", "extractor": "viz2psy",
         "extractor_version": "0.7.1",
         "models": {"clip": {"checkpoint": CLIP_CKPT}}},
    )
    out = tmp_path / "features.csv"
    with pytest.warns(UserWarning, match="no rows"):
        summary = build_features([empty], out)
    assert summary["rows"] == 0
    assert list(pd.read_csv(out).columns) == OUTPUT_COLUMNS
