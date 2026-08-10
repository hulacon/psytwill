"""Contract tests against the extractor-output convention (§4.1).

Tiny synthetic fixtures shaped like each sibling's output (CSV +
.meta.json sidecar) validate that psytwill's reader honors the contract:
schema_version gating, family sidecar resolution, checkpoint equality on
cross-modal pairs, stimulus_id as reserved label column, and provenance
passthrough into matrices.meta.json. Fully offline.
"""

import json

import numpy as np
import pandas as pd
import pytest

from psytwill.exceptions import InputError, SpaceError
from psytwill.pipeline import build_quilt
from psytwill.sidecar import find_sidecar, load_sidecar, model_checkpoint
from psytwill.spaces import detect_spaces

CLIP_CKPT = "ViT-B-32/laion2b_s34b_b79k"


def _write(path, df, sidecar=None, sidecar_path=None):
    df.to_csv(path, index=False)
    if sidecar is not None:
        target = sidecar_path or path.with_suffix(".meta.json")
        target.write_text(json.dumps(sidecar))
    return path


def viz2psy_fixture(tmp_path, checkpoint=CLIP_CKPT, schema="1.0"):
    """viz2psy-shaped scores.csv + sidecar (image rows, clip embedding)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "filename": [f"shared{i:04d}_nsd{i:05d}.png" for i in range(4)],
        "stimulus_id": [f"shared{i:04d}_nsd{i:05d}" for i in range(4)],
        **{f"clip_{i:03d}": rng.normal(size=4) for i in range(4)},
        "resmem_memorability": rng.uniform(size=4),
    })
    sidecar = {
        "schema_version": schema,
        "extractor": "viz2psy",
        "extractor_version": "0.6.0",
        "models": {
            "clip": {"checkpoint": checkpoint, "package_version": "3.2.0"},
            "resmem": {"checkpoint": "resmem-pretrained"},
        },
    }
    return _write(tmp_path / "img_scores.csv", df, sidecar)


def word2psy_fixture(tmp_path, checkpoint=CLIP_CKPT):
    """word2psy-shaped features_chunks.csv + family features.meta.json."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "chunk_idx": range(4),
        "chunk_label": [f"word{i}" for i in range(4)],
        **{f"clip_text_{i:03d}": rng.normal(size=4) for i in range(4)},
    })
    sidecar = {
        "schema_version": "1.0",
        "extractor": "word2psy",
        "extractor_version": "0.2.0",
        "models": {"clip_text": {"checkpoint": checkpoint}},
    }
    return _write(
        tmp_path / "features_chunks.csv", df, sidecar,
        sidecar_path=tmp_path / "features.meta.json",
    )


def test_family_sidecar_resolution(tmp_path):
    csv = word2psy_fixture(tmp_path)
    assert find_sidecar(csv) == tmp_path / "features.meta.json"
    meta = load_sidecar(csv)
    assert meta["extractor"] == "word2psy"
    assert model_checkpoint(meta, "clip_text") == CLIP_CKPT


def test_missing_sidecar_warns_and_passes(tmp_path):
    df = pd.DataFrame({f"minilm_{i:03d}": np.random.default_rng(2).normal(size=3)
                       for i in range(3)})
    csv = _write(tmp_path / "plain.csv", df)
    with pytest.warns(UserWarning, match="legacy input"):
        result = build_quilt(csv, output_dir=tmp_path / "out")
    assert "minilm__cosine" in result["matrix_files"]


def test_bad_schema_version_rejected(tmp_path):
    csv = viz2psy_fixture(tmp_path, schema="2.0")
    with pytest.raises(InputError, match="schema_version '2.0'"):
        load_sidecar(csv)


def test_cross_modal_checkpoint_match_builds(tmp_path):
    csv_a = word2psy_fixture(tmp_path)
    csv_b = viz2psy_fixture(tmp_path)
    result = build_quilt(csv_a, csv_b, output_dir=tmp_path / "out")
    assert any("clip_text__x__clip" in key for key in result["matrix_files"])
    # Provenance chain lands in psytwill's own sidecar.
    inputs = result["meta"]["inputs"]
    assert inputs[0]["extractor"] == "word2psy"
    assert inputs[1]["checkpoints"]["clip"] == CLIP_CKPT
    assert result["meta"]["schema_version"] == "1.0"
    assert result["meta"]["extractor"] == "psytwill"


def test_cross_modal_checkpoint_mismatch_refused(tmp_path):
    csv_a = word2psy_fixture(tmp_path)
    csv_b = viz2psy_fixture(tmp_path, checkpoint="ViT-L-14/openai")
    with pytest.raises(SpaceError, match="Checkpoint mismatch"):
        build_quilt(csv_a, csv_b, output_dir=tmp_path / "out")


def test_stimulus_id_is_reserved_and_preferred_label(tmp_path):
    csv = viz2psy_fixture(tmp_path)
    df = pd.read_csv(csv)
    spaces = detect_spaces(df.columns)
    swept = {c for s in spaces.values() for c in s.columns}
    assert "stimulus_id" not in swept and "filename" not in swept
    result = build_quilt(csv, output_dir=tmp_path / "out")
    assert result["meta"]["inputs"][0]["label_column"] == "stimulus_id"
