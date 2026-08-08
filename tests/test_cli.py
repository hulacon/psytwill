"""End-to-end CLI runs on synthetic CSVs, including outputs + sidecar."""

import json

import numpy as np
import pandas as pd
import pytest

from psyquilt.cli import main

from tests.conftest import embedding_frame, unit


@pytest.fixture
def chunks_csv(tmp_path, two_topic_frame):
    # Add an emotion profile so two spaces are detected
    df = two_topic_frame.copy()
    rng = np.random.RandomState(1)
    for name in ("joy", "fear", "sadness"):
        df[f"emotion_{name}"] = rng.rand(len(df))
    path = tmp_path / "scores_chunks.csv"
    df.to_csv(path, index=False)
    return path


def test_self_mode_end_to_end(chunks_csv, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["matrices", str(chunks_csv), "-o", str(out)]) == 0

    assert (out / "minilm__cosine.csv").exists()
    assert (out / "emotion__correlation.csv").exists()
    assert (out / "transitions.csv").exists()

    M = pd.read_csv(out / "minilm__cosine.csv", index_col="label")
    assert M.shape == (8, 8)
    assert list(M.index) == list(M.columns)
    assert M.index[0] == "s0/A"

    trans = pd.read_csv(out / "transitions.csv")
    assert set(trans["space"]) == {"minilm", "emotion"}
    assert len(trans) == 7 * 2

    meta = json.loads((out / "matrices.meta.json").read_text())
    assert meta["mode"] == "self"
    assert meta["matrices"]["minilm__cosine"]["space"]["pattern"] == "minilm_{NNN}"
    assert meta["matrices"]["minilm__cosine"]["n_valid"] == 8
    assert meta["matrices"]["emotion__correlation"]["space"]["columns"] == [
        "emotion_joy", "emotion_fear", "emotion_sadness"
    ]
    assert meta["inputs"][0]["label_column"] == "chunk_label"
    assert meta["transitions"]["rows"] == 14


def test_spaces_subset_and_metric_override(chunks_csv, tmp_path):
    out = tmp_path / "out"
    assert main([
        "matrices", str(chunks_csv), "-o", str(out),
        "--spaces", "minilm:euclidean",
    ]) == 0
    assert (out / "minilm__euclidean.csv").exists()
    assert not (out / "emotion__correlation.csv").exists()
    meta = json.loads((out / "matrices.meta.json").read_text())
    assert meta["matrices"]["minilm__euclidean"]["form"] == "distance"


def test_unknown_space_errors(chunks_csv, tmp_path, capsys):
    ret = main([
        "matrices", str(chunks_csv), "-o", str(tmp_path / "out"),
        "--spaces", "nope",
    ])
    assert ret == 1
    assert "not found" in capsys.readouterr().err


def test_cross_mode_clip(tmp_path):
    texts = embedding_frame(
        [unit(8, 0), unit(8, 1)], prefix="clip_text", labels=["cat", "dog"]
    )
    images = embedding_frame([unit(8, 1), unit(8, 0)], prefix="clip")
    images["filename"] = ["dog.png", "cat.png"]
    ta, tb = tmp_path / "text.csv", tmp_path / "img.csv"
    texts.to_csv(ta, index=False)
    images.to_csv(tb, index=False)
    out = tmp_path / "out"

    assert main(["matrices", str(ta), str(tb), "-o", str(out), "--diagonal"]) == 0
    sim = pd.read_csv(out / "clip_text__x__clip__cosine.csv", index_col="label")
    assert sim.loc["cat", "cat.png"] == pytest.approx(1.0)
    assert not (out / "transitions.csv").exists()
    diag = pd.read_csv(out / "diagonal.csv")
    assert len(diag) == 2

    meta = json.loads((out / "matrices.meta.json").read_text())
    assert meta["mode"] == "cross"
    key = "clip_text__x__clip__cosine"
    assert meta["matrices"][key]["space_b"]["name"] == "clip"
    assert meta["matrices"][key]["n_valid"] == {"a": 2, "b": 2}


def test_cross_mode_no_shared_spaces_errors(tmp_path, capsys):
    a = embedding_frame([unit(4, 0)], prefix="minilm")
    b = embedding_frame([unit(4, 0)], prefix="gist")
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
    a.to_csv(pa, index=False)
    b.to_csv(pb, index=False)
    ret = main(["matrices", str(pa), str(pb), "-o", str(tmp_path / "out")])
    assert ret == 1
    assert "No shared or compatible spaces" in capsys.readouterr().err


def test_diagonal_in_self_mode_errors(chunks_csv, tmp_path, capsys):
    ret = main([
        "matrices", str(chunks_csv), "-o", str(tmp_path / "out"), "--diagonal"
    ])
    assert ret == 1
    assert "two inputs" in capsys.readouterr().err


def test_missing_file_errors(tmp_path, capsys):
    ret = main(["matrices", str(tmp_path / "nope.csv"), "-o", str(tmp_path / "o")])
    assert ret == 1
    assert "not found" in capsys.readouterr().err


def test_nan_rows_survive_to_output(tmp_path):
    vecs = np.array([unit(4, 0), [np.nan] * 4, unit(4, 1)])
    df = embedding_frame(vecs, labels=["a", "oov", "b"])
    path = tmp_path / "scores.csv"
    df.to_csv(path, index=False)
    out = tmp_path / "out"
    assert main(["matrices", str(path), "-o", str(out)]) == 0
    M = pd.read_csv(out / "minilm__cosine.csv", index_col="label")
    assert M.loc["oov"].isna().all()
    meta = json.loads((out / "matrices.meta.json").read_text())
    assert meta["matrices"]["minilm__cosine"]["n_valid"] == 2
    assert meta["matrices"]["minilm__cosine"]["nan_labels"] == ["oov"]
    trans = pd.read_csv(out / "transitions.csv")
    assert trans["value"].isna().sum() == 2  # both boundaries touching oov


def test_spaces_subcommand(chunks_csv, capsys):
    assert main(["spaces", str(chunks_csv)]) == 0
    out = capsys.readouterr().out
    assert "minilm" in out and "emotion" in out
    assert "16d" in out and "3d" in out
