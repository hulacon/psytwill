"""Optional media render for composed runs — the viewer's eyes and ears.

:mod:`psytwill.compose` writes the feature tables; this module writes what
``psytwill viz movies`` shows beside them: a frames/ directory rendering the
display at each grid bin, an ``audio.m4a`` with each auditory item muxed at
its onset, and the transcript CSVs. None of it affects any measured number —
it exists so a composed run can be *browsed* like a film.

Deliberately outside psytwill's core dependency policy: Pillow and soundfile
are imported lazily and ffmpeg is invoked as a subprocess, all three named in
the error when absent. Display geometry is fully parameterized — the caller
states the screen, image fraction and background their task program used; a
render at the wrong geometry is a different stimulus.

Media files resolve per item from either a generic ``media_map`` TSV
(``stimulus_id``, ``file``, optional ``voice``; paths relative to the TSV's
directory) or the mmmdata-style registry tables (``image_file`` /
``audio_file_<voice>`` columns, relative to ``<stimuli_root>/<set>/``).
Routing is by file suffix — an image renders, an audio file muxes — so the
same map format serves any dataset.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from psytwill.exceptions import InputError

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}

AUDIO_SR = 48000


def _need(what: str, hint: str) -> "Exception":
    return InputError(
        f"--media needs {what} ({hint}); install it, or drop --media — the "
        "composed feature tables do not depend on it."
    )


def media_map_from_tsv(path: str | Path) -> dict[tuple[str, Optional[str]], Path]:
    """(stimulus_id, voice|None) -> media file, from a generic TSV."""
    p = Path(path)
    df = pd.read_csv(p, sep="\t", dtype=str)
    if not {"stimulus_id", "file"} <= set(df.columns):
        raise InputError(
            f"{p.name} is not a media map: needs columns stimulus_id, file "
            "(optional voice); paths resolve relative to the TSV's directory."
        )
    out: dict[tuple[str, Optional[str]], Path] = {}
    for r in df.itertuples(index=False):
        voice = getattr(r, "voice", None)
        voice = None if voice is None or pd.isna(voice) else str(voice)
        out[(str(r.stimulus_id), voice)] = (p.parent / str(r.file)).resolve()
    return out


def media_map_from_registry(
    registry_dir: str | Path, stimuli_root: str | Path
) -> dict[tuple[str, Optional[str]], Path]:
    """(stimulus_id, voice|None) -> media file, from mmmdata-style registry tables.

    ``image_file``-style columns map voice-less; ``audio_file_<voice>``
    columns map per voice. Paths are relative to ``<stimuli_root>/<set>/``,
    the layout the registry documents. Movie sets (``video_file``) are
    skipped: a movie presentation's media is the film itself, which the
    movies pipeline already owns.
    """
    reg = Path(registry_dir)
    root = Path(stimuli_root)
    out: dict[tuple[str, Optional[str]], Path] = {}
    for table in sorted(reg.glob("*.tsv")):
        df = pd.read_csv(table, sep="\t", dtype=str)
        if "stimulus_id" not in df.columns:
            continue
        set_dir = root / table.stem
        for col in df.columns:
            if col == "image_file" or col.endswith(("_image_file",)):
                for sid, f in zip(df["stimulus_id"], df[col]):
                    if isinstance(f, str) and f:
                        out.setdefault((sid, None), set_dir / f)
            elif col.startswith("audio_file_"):
                voice = col[len("audio_file_"):]
                for sid, f in zip(df["stimulus_id"], df[col]):
                    if isinstance(f, str) and f:
                        out[(sid, voice)] = set_dir / f
    return out


def _lookup(media: dict[tuple[str, Optional[str]], Path],
            sid: str, voice: Optional[str]) -> Optional[Path]:
    if voice is not None and (sid, str(voice)) in media:
        return media[(sid, str(voice))]
    return media.get((sid, None))


def render_run_media(
    run_dir: str | Path,
    pres: pd.DataFrame,
    media: dict[tuple[str, Optional[str]], Path],
    run_end: float,
    *,
    window: float = 0.5,
    screen: tuple[int, int] = (2048, 1280),
    image_frac: float = 0.6,
    bg_gray: int = 191,
    fixation_frac: float = 0.01,
    scale: float = 0.5,
) -> dict[str, Any]:
    """Frames, muxed audio and transcript CSVs for one composed run.

    ``pres`` is the run's presentations frame (compose's ``_sid``/``_voice``/
    ``_onset``/``_dur`` columns). Items route by media suffix; an item with
    no media entry is counted, not guessed at — the count comes back in the
    summary so a caller can decide whether silence was expected.
    """
    run_dir = Path(run_dir)
    rows = []
    n_unmapped = 0
    for r in pres[["_sid", "_voice", "_onset", "_dur"]].to_dict("records"):
        f = _lookup(media, r["_sid"], r["_voice"])
        if f is None:
            n_unmapped += 1
            continue
        rows.append({"sid": r["_sid"], "voice": r["_voice"],
                     "onset": float(r["_onset"]), "dur": float(r["_dur"]),
                     "file": f,
                     "kind": ("image" if f.suffix.lower() in IMAGE_SUFFIXES
                              else "audio" if f.suffix.lower() in AUDIO_SUFFIXES
                              else "other")})
    images = [r for r in rows if r["kind"] == "image"]
    audio = [r for r in rows if r["kind"] == "audio"]

    n_frames = _render_frames(run_dir, images, run_end, window=window,
                              screen=screen, image_frac=image_frac,
                              bg_gray=bg_gray, fixation_frac=fixation_frac,
                              scale=scale)
    if audio:
        _mux_audio(run_dir, audio, run_end)
        _write_transcript(run_dir, audio, run_dir.name)
    return {"frames": n_frames, "audio_items": len(audio),
            "image_items": len(images), "unmapped": n_unmapped}


def _render_frames(run_dir: Path, images: list[dict], run_end: float, *,
                   window: float, screen: tuple[int, int], image_frac: float,
                   bg_gray: int, fixation_frac: float, scale: float) -> int:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise _need("Pillow", "pip install pillow") from exc

    W, H = int(screen[0] * scale), int(screen[1] * scale)
    img_px = int(image_frac * H)
    fix_px = max(2, int(fixation_frac * H))
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    fixation = Image.new("RGB", (W, H), (bg_gray,) * 3)
    d = ImageDraw.Draw(fixation)
    d.ellipse([(W - fix_px) // 2, (H - fix_px) // 2,
               (W + fix_px) // 2, (H + fix_px) // 2], fill=(0, 0, 0))

    windows = [(r["onset"], r["onset"] + r["dur"], r["file"]) for r in images]
    cache: dict[Path, Any] = {}
    n = 0
    for t in np.arange(0.0, run_end + 1e-9, window):
        shown = next((f for a, b, f in windows if a - 1e-9 <= t < b - 1e-9), None)
        if shown is None:
            canvas = fixation
        else:
            if shown not in cache:
                stim = Image.open(shown).convert("RGB").resize(
                    (img_px, img_px), Image.LANCZOS)
                c = Image.new("RGB", (W, H), (bg_gray,) * 3)
                c.paste(stim, ((W - img_px) // 2, (H - img_px) // 2))
                cache[shown] = c
            canvas = cache[shown]
        canvas.save(frames_dir / f"frame_{t:.3f}.jpg", quality=85)
        n += 1
    return n


def _mux_audio(run_dir: Path, audio: list[dict], run_end: float) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise _need("soundfile", "pip install soundfile") from exc
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise _need("ffmpeg on PATH", "https://ffmpeg.org")

    mix = np.zeros((int(run_end * AUDIO_SR) + AUDIO_SR, 2), dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        for r in audio:
            wav = Path(td) / (r["file"].stem + ".wav")
            if not wav.exists():
                subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(r["file"]),
                                "-ar", str(AUDIO_SR), "-ac", "2", str(wav)],
                               check=True)
            data, _ = sf.read(wav, dtype="float32", always_2d=True)
            start = int(r["onset"] * AUDIO_SR)
            mix[start:start + len(data)] += data
        mix_wav = Path(td) / "mix.wav"
        sf.write(mix_wav, np.clip(mix, -1, 1), AUDIO_SR)
        run_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(mix_wav),
                        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                        str(run_dir / "audio.m4a")], check=True)


def _write_transcript(run_dir: Path, audio: list[dict], slug: str) -> None:
    """The two transcript CSVs the movies viewer reads, one word per item."""
    run_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(audio, key=lambda r: r["onset"])
    words = [{"stimulus_id": slug, "chunk_idx": i, "word_idx": 0,
              "word": r["sid"], "onset": r["onset"],
              "offset": r["onset"] + r["dur"], "transcribe_probability": 1.0}
             for i, r in enumerate(ordered)]
    pd.DataFrame(words).to_csv(run_dir / "transcribe_transcript_words.csv",
                               index=False)
    chunks = [{"stimulus_id": slug, "chunk_idx": w["chunk_idx"],
               "onset": w["onset"], "offset": w["offset"],
               "transcribe_text": w["word"], "transcribe_asr_confidence": 1.0}
              for w in words]
    pd.DataFrame(chunks).to_csv(run_dir / "transcribe_transcript.csv",
                                index=False)
