"""Tests for the movies timeline viewer (psytwill.viz).

Fully offline: synthetic features parquets + a fake film directory, with
placeholder names throughout (film-a, film-b) per the describe-data-
generically norm.
"""

import json
import re

import pandas as pd
import pytest

from psytwill.exceptions import InputError
from psytwill.viz import movies as M
from psytwill.viz.build import build_movies_bundle

pytest.importorskip("pyarrow")

COLUMNS = ["stimulus_id", "voice", "time", "onset", "offset", "chunk_idx",
           "word_idx", "modality", "extractor", "extractor_version",
           "model", "feature", "value", "value_str"]


def _rows(stimulus_id, model, feature, points, modality="visual",
          value_str=None, chunk_idx=None, word_idx=None):
    """points: [(time, value), ...]"""
    out = []
    for i, (t, v) in enumerate(points):
        out.append({
            "stimulus_id": stimulus_id, "voice": None, "time": t,
            "onset": None, "offset": None,
            "chunk_idx": chunk_idx[i] if chunk_idx else None,
            "word_idx": word_idx[i] if word_idx else None,
            "modality": modality, "extractor": "x2psy",
            "extractor_version": "0.0", "model": model, "feature": feature,
            "value": v, "value_str": value_str[i] if value_str else None,
        })
    return out


@pytest.fixture
def features_dir(tmp_path):
    d = tmp_path / "psytwill"
    d.mkdir()
    grid = [(i * 0.5, float(i)) for i in range(6)]
    frames = (
        _rows("film-a", "motion", "motion_energy", grid)
        + _rows("film-a", "clip", "clip_000", grid)          # embedding: out
        + _rows("film-a", "saliency", "saliency_00_01", grid)  # grid cell: out
        + _rows("film-a", "caption", "caption_text",
                [(0.0, None), (0.5, None), (1.0, None)],
                value_str=["a scene", "a scene", "another scene"])
        + _rows("film-b", "motion", "motion_energy", grid)
    )
    pd.DataFrame(frames, columns=COLUMNS).to_parquet(
        d / "movies_frames_features.parquet")

    audio = _rows("film-a", "loudness", "loudness_db",
                  [(0.25 + i * 0.5, -20.0 + i) for i in range(6)],
                  modality="audio")
    pd.DataFrame(audio, columns=COLUMNS).to_parquet(
        d / "movies_audio_frames_features.parquet")

    # word grain: word2psy-style GLOBAL chunk_idx (film-a starts at 100)
    # and global word_idx, one extra token vs the transcript CSV in chunk 0
    words = _rows("film-a", "surprise", "surprise_bits",
                  [(None, 1.0), (None, 2.0), (None, 3.0), (None, 4.0)],
                  modality="text",
                  chunk_idx=[100, 100, 100, 101],
                  word_idx=[40, 41, 42, 43])
    pd.DataFrame(words, columns=COLUMNS).to_parquet(
        d / "movies_transcript_words_features.parquet")
    return d


@pytest.fixture
def films_dir(tmp_path):
    root = tmp_path / "movies"
    for slug in ("film-a", "film-b"):
        film = root / slug
        (film / "frames").mkdir(parents=True)
        for i in range(6):
            (film / "frames" / f"frame_{i * 0.5:.3f}.jpg").write_bytes(b"\xff")
    (root / "film-a" / "audio.m4a").write_bytes(b"\x00")
    (root / "film-a" / "transcribe_transcript_words.csv").write_text(
        "stimulus_id,chunk_idx,word_idx,word,onset,offset,transcribe_probability\n"
        "film-a,0,0,hello,0.10,0.50,0.9\n"
        "film-a,0,1,there,0.50,0.90,0.9\n"
        "film-a,1,0,friend,1.20,1.60,0.9\n")
    (root / "film-a" / "transcribe_transcript.csv").write_text(
        "stimulus_id,chunk_idx,onset,offset,transcribe_text,transcribe_asr_confidence\n"
        "film-a,0,0.10,0.90,hello there,0.9\n"
        "film-a,1,1.20,1.60,friend,0.9\n")
    (root / "film-a" / "diarize_speakers.csv").write_text(
        "stimulus_id,turn_idx,speaker,onset,offset\n"
        "film-a,0,SPEAKER_00,0.0,1.0\n")
    return root


def test_scalar_features_excludes_embeddings_and_grid_cells(features_dir):
    names = M.scalar_features(features_dir / "movies_frames_features.parquet")
    assert "motion_energy" in names and "caption_text" in names
    assert "clip_000" not in names
    assert "saliency_00_01" not in names


def test_load_tables_missing_dir_is_loud(tmp_path):
    with pytest.raises(InputError, match=str(tmp_path)):
        M.load_tables(tmp_path)


def test_film_payload_grids_captions_and_media(features_dir, films_dir):
    tables = M.load_tables(features_dir)
    assert M.film_slugs(tables) == ["film-a", "film-b"]
    p = M.film_payload("film-a", tables, films_dir)

    motion = next(s for s in p["series"] if s["f"] == "motion_energy")
    assert motion["align"] == "grid" and motion["t0"] == 0.0 and motion["dt"] == 0.5
    assert motion["v"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    loud = next(s for s in p["series"] if s["f"] == "loudness_db")
    assert loud["t0"] == 0.25  # the audio grid offset stays visible as data

    # captions collapse to change-points
    assert p["captions"] == [[0.0, "a scene"], [1.0, "another scene"]]

    assert p["media"]["audio"] == "film-a/audio.m4a"
    assert p["media"]["frames_dir"] == "film-a/frames"
    assert p["media"]["frames"][0] == 0.0 and len(p["media"]["frames"]) == 6
    assert [w[4] for w in p["media"]["words"]] == ["hello", "there", "friend"]
    assert p["media"]["speakers"] == [[0.0, 1.0, "SPEAKER_00"]]
    assert p["duration"] >= 2.75


def test_word_alignment_rebases_global_chunk_and_word_indices(features_dir, films_dir):
    tables = M.load_tables(features_dir)
    p = M.film_payload("film-a", tables, films_dir)
    sup = next(s for s in p["series"] if s["f"] == "surprise_bits")
    assert sup["align"] == "word"
    # chunk 100 -> local 0 (words hello, there + one dropped extra token);
    # chunk 101 -> local 1 (word friend)
    assert sup["v"] == [1.0, 2.0, 4.0]


def test_film_without_transcript_or_audio_is_graceful(features_dir, films_dir):
    tables = M.load_tables(features_dir)
    p = M.film_payload("film-b", tables, films_dir)
    assert p["media"]["audio"] is None
    assert p["media"]["words"] == [] and p["media"]["chunks"] == []
    assert any(s["f"] == "motion_energy" for s in p["series"])
    assert not any(s["align"] in ("word", "chunk") for s in p["series"])


def test_payload_js_shape(features_dir, films_dir):
    tables = M.load_tables(features_dir)
    p = M.film_payload("film-a", tables, films_dir)
    js = M.payload_js(p)
    body = json.loads(re.search(r"= (\{.*\});", js, re.S).group(1))
    assert body["slug"] == "film-a"


def test_build_bundle_layout_and_media_root(features_dir, films_dir):
    page = build_movies_bundle(features_dir, films_dir)
    out = films_dir / "viz" / "timeline"
    assert page == out / "index.html"
    html = page.read_text()
    assert 'MEDIA_ROOT = "../.."' in html
    assert "__PSYTWILL_INDEX__" not in html and "__PSYTWILL_VERSION__" not in html
    index = json.loads(re.search(r"INDEX = (\[.*?\]);", html).group(1))
    assert [f["slug"] for f in index] == ["film-a", "film-b"]
    assert (out / "data" / "film-a.js").exists()
    meta = json.loads((out / "viewer.meta.json").read_text())
    assert meta["n_films"] == 2


def test_build_bundle_slug_subset_and_unknown(features_dir, films_dir, tmp_path):
    out = tmp_path / "bundle"
    build_movies_bundle(features_dir, films_dir, out_dir=out, slugs=["film-b"])
    assert not (out / "data" / "film-a.js").exists()
    with pytest.raises(InputError, match="film-zz"):
        build_movies_bundle(features_dir, films_dir, out_dir=out, slugs=["film-zz"])


def test_registry_titles(features_dir, films_dir, tmp_path):
    reg = tmp_path / "movies.tsv"
    reg.write_text("stimulus_id\tmovie_name\tstyle\tduration_s\n"
                   "film-a\tFilm A\tAnimated\t120\n")
    rows = M.load_registry(reg)
    tables = M.load_tables(features_dir)
    p = M.film_payload("film-a", tables, films_dir, rows["film-a"])
    assert p["title"] == "Film A"
    assert p["duration"] == 120.0
