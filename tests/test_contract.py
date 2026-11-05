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


# --- schema_version 1.2: per-column `labels` ------------------------------
#
# A viz2psy-shaped 1.2 fixture: an embedding model (pattern inventory), a
# probe model on that embedding's checkpoint (so its columns are circular
# with it in one block), a formula model, and one bookkeeping column.

AESTHETICS_CKPT = "aesthetic-head/v2"
N_LABELLED = 60


def labelled_models(clip_checkpoint=CLIP_CKPT):
    """The 1.2 sidecar `models` map of :func:`labelled_fixture` (a fresh copy)."""
    return {
        "clip": {
            "checkpoint": clip_checkpoint,
            "features": {"pattern": "clip_{NNN}", "range": [0, 5], "count": 6},
            "nulls": {},
            "labels": {"default": {"interpretability": "opaque", "derivation": "embedding",
                                   "Description": "CLIP image embedding dimension"}},
        },
        "aesthetics": {
            "checkpoint": AESTHETICS_CKPT,
            "features": {"columns": ["aesthetics_score", "aesthetics_artistic", "aesthetics_quality",
                                     "aesthetics_n_crops"]},
            "nulls": {},
            "labels": {
                "default": {"interpretability": "nameable", "derivation": "probe",
                            "derived_from": clip_checkpoint, "Description": "predicted aesthetic rating"},
                "columns": {
                    "aesthetics_score": {
                        "LongName": "Aesthetic score", "Units": "rating (1-10)",
                        "source_validation": [{"citation": "https://example.org/aesthetics",
                                               "eval_dataset": "a public ratings set",
                                               "metric": "spearman", "value": 0.7}]},
                    "aesthetics_quality": {"derivation": "zero-shot",
                                           "Description": "similarity to a quality prompt"},
                    "aesthetics_n_crops": {"derivation": "bookkeeping", "derived_from": None,
                                           "interpretability": "opaque",
                                           "Description": "number of crops pooled"},
                },
            },
        },
        "llstat": {
            "checkpoint": None,
            "features": {"columns": ["llstat_luminance_mean", "llstat_contrast", "llstat_colorfulness",
                                     "llstat_hist_00", "llstat_hist_01"]},
            "nulls": {},
            "labels": {
                "default": {"interpretability": "nameable", "derivation": "formula"},
                "columns": {
                    "llstat_luminance_mean": {"LongName": "Mean luminance", "Units": "arbitrary",
                                              "TermURL": "https://example.org/term/luminance"},
                    "llstat_hist_00": {"interpretability": "opaque"},
                    "llstat_hist_01": {"interpretability": "opaque"},
                },
            },
        },
    }


def labelled_fixture(tmp_path, *, name="img_scores.csv", models=None, schema="1.2",
                     created_at="2026-01-01T00:00:00+00:00", seed=0, ids=None):
    """viz2psy-shaped 1.2 scores CSV + sidecar with per-column labels.

    Columns share a 2-d latent, so any subset of members makes a fittable
    block; ``aesthetics_n_crops`` is a bookkeeping count.
    """
    rng = np.random.default_rng(seed)
    n = N_LABELLED
    z = rng.normal(size=(n, 2))
    cols = {"stimulus_id": ids or [f"img{i:03d}" for i in range(n)]}
    blocks = {"clip": [f"clip_{i:03d}" for i in range(6)],
              "aesthetics": ["aesthetics_score", "aesthetics_artistic", "aesthetics_quality"],
              "llstat": ["llstat_luminance_mean", "llstat_contrast", "llstat_colorfulness",
                         "llstat_hist_00", "llstat_hist_01"]}
    for names in blocks.values():
        W = rng.normal(size=(2, len(names)))
        X = z @ W + 0.3 * rng.normal(size=(n, len(names)))
        cols.update({c: X[:, j] for j, c in enumerate(names)})
    cols["aesthetics_n_crops"] = rng.integers(1, 30, size=n).astype(float)
    df = pd.DataFrame(cols)
    models = labelled_models() if models is None else models
    if schema in ("1.0", "1.1"):
        for entry in models.values():
            entry.pop("labels", None)
            if schema == "1.0":
                entry.pop("nulls", None)
    sidecar = {"schema_version": schema, "extractor": "viz2psy", "extractor_version": "0.12.0",
               "created_at": created_at, "models": models}
    return _write(tmp_path / name, df, sidecar)


def test_labels_reader_accepts_1_2_and_resolves(tmp_path):
    from psytwill.labels import read_model_labels, resolve

    meta = load_sidecar(labelled_fixture(tmp_path))
    assert meta["schema_version"] == "1.2"
    clip = read_model_labels(meta, "clip")
    got = resolve(clip, [f"clip_{i:03d}" for i in range(6)], model="clip")
    assert {e["derivation"] for e in got.values()} == {"embedding"}
    aes = resolve(read_model_labels(meta, "aesthetics"),
                  ["aesthetics_score", "aesthetics_quality", "aesthetics_n_crops"], model="aesthetics")
    # key-wise merge: the override replaces only the keys it names
    assert aes["aesthetics_score"]["derived_from"] == CLIP_CKPT
    assert aes["aesthetics_score"]["Description"] == "predicted aesthetic rating"
    assert aes["aesthetics_quality"]["derivation"] == "zero-shot"
    # a null override removes the default's key (a bookkeeping column has no parent)
    assert "derived_from" not in aes["aesthetics_n_crops"]


def test_1_1_sidecar_reads_as_unlabelled(tmp_path):
    from psytwill.labels import read_model_labels

    meta = load_sidecar(labelled_fixture(tmp_path, schema="1.1"))
    assert read_model_labels(meta, "clip") is None
    assert read_model_labels(None, "clip") is None
