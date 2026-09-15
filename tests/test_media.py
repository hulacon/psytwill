"""media: file resolution and the viewer transcript CSVs (render legs that
need Pillow/ffmpeg are exercised only where the dependency is present)."""

import pandas as pd
import pytest

from psytwill.exceptions import InputError
from psytwill.media import (
    _lookup,
    _write_transcript,
    media_map_from_registry,
    media_map_from_tsv,
    render_run_media,
)


def test_media_map_from_tsv_and_lookup(tmp_path):
    tsv = tmp_path / "map.tsv"
    pd.DataFrame([
        {"stimulus_id": "imgA", "file": "images/a.png", "voice": None},
        {"stimulus_id": "cabin", "file": "audio/cabin_nova.mp3", "voice": "nova"},
    ]).to_csv(tsv, sep="\t", index=False)
    m = media_map_from_tsv(tsv)
    assert _lookup(m, "imgA", None).name == "a.png"
    assert _lookup(m, "imgA", "nova").name == "a.png"  # voiceless fallback
    assert _lookup(m, "cabin", "nova").name == "cabin_nova.mp3"
    assert _lookup(m, "cabin", "echo") is None
    with pytest.raises(InputError, match="not a media map"):
        bad = tmp_path / "bad.tsv"
        pd.DataFrame([{"x": 1}]).to_csv(bad, sep="\t", index=False)
        media_map_from_tsv(bad)


def test_media_map_from_registry(tmp_path):
    reg = tmp_path / "reg"
    reg.mkdir()
    pd.DataFrame({"stimulus_id": ["s1"], "image_file": ["images/s1.png"],
                  "mmmId": [1]}).to_csv(reg / "shared1000.tsv", sep="\t",
                                        index=False)
    pd.DataFrame({"stimulus_id": ["cabin"],
                  "audio_file_nova": ["audio/nova/cabin.mp3"],
                  "audio_file_echo": ["audio/echo/cabin.mp3"]}).to_csv(
        reg / "twp1000.tsv", sep="\t", index=False)
    m = media_map_from_registry(reg, tmp_path / "stimuli")
    assert m[("s1", None)].parts[-3:] == ("shared1000", "images", "s1.png")
    assert m[("cabin", "nova")].parts[-2:] == ("nova", "cabin.mp3")
    assert "twp1000" in m[("cabin", "nova")].parts


def test_write_transcript(tmp_path):
    audio = [{"sid": "river", "onset": 13.0, "dur": 0.6},
             {"sid": "cabin", "onset": 9.0, "dur": 0.6}]
    _write_transcript(tmp_path, audio, "some-run")
    words = pd.read_csv(tmp_path / "transcribe_transcript_words.csv")
    assert words["word"].tolist() == ["cabin", "river"]  # onset order
    assert words["chunk_idx"].tolist() == [0, 1]
    chunks = pd.read_csv(tmp_path / "transcribe_transcript.csv")
    assert chunks["transcribe_text"].tolist() == ["cabin", "river"]


def test_render_frames_with_pillow(tmp_path):
    PIL = pytest.importorskip("PIL")
    from PIL import Image

    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(img)
    pres = pd.DataFrame([{"_sid": "imgA", "_voice": None, "_onset": 0.5,
                          "_dur": 1.0}])
    media = {("imgA", None): img}
    out = render_run_media(tmp_path / "run", pres, media, run_end=2.0,
                           window=0.5, screen=(64, 40), scale=1.0)
    assert out["image_items"] == 1 and out["unmapped"] == 0
    frames = sorted((tmp_path / "run" / "frames").glob("frame_*.jpg"))
    assert len(frames) == out["frames"] == 5  # 0.0..2.0 inclusive
