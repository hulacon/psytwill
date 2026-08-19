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
