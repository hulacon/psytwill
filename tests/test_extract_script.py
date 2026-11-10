"""Offline checks on scripts/fitcorpus/extract.py's aggregate step (no extractors run)."""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "fitcorpus_extract", Path(__file__).parents[1] / "scripts" / "fitcorpus" / "extract.py")
extract = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = extract  # dataclasses resolve annotations through it
_SPEC.loader.exec_module(extract)


def test_aggregate_refuses_a_group_one_of_whose_models_is_missing_units(tmp_path, monkeypatch, capsys):
    durable = tmp_path / "durable"
    units = [extract.Unit(id=f"u{i}", out_dir=durable / "c" / "features" / f"u{i}") for i in range(2)]
    src = extract.Source("c", "image", "viz2psy", "", ["clip", "gist"], "synthetic", lambda: units)
    done = {("u0", "clip"), ("u1", "clip"), ("u0", "gist")}  # gist never finished u1
    for u in units:
        u.out_dir.mkdir(parents=True)
        for m in src.models:
            if (u.id, m) in done:
                (u.out_dir / f"{m}.csv").write_text(f"stimulus_id,{m}_000,{m}_001\nx,1,2\n")
    monkeypatch.setattr(extract, "ROOTS", types.SimpleNamespace(durable=durable, scratch=None, nsd_hdf5=None),
                        raising=False)  # set in main()
    monkeypatch.setattr(extract, "build_sources", lambda: [src])
    monkeypatch.setattr(extract, "is_done", lambda stem, *_: (stem.parent.name, stem.stem) in done)
    monkeypatch.setattr(extract._dir_stems, "cache", {}, raising=False)
    args = argparse.Namespace(model=None, skip_model=None, unit=None, unit_index=None, corpus=None, source=None,
                              allow_partial=False, dry_run=True, redo=False, fail_fast=False)
    assert extract.cmd_aggregate(args) == 1
    assert "INCOMPLETE c/image/main" in capsys.readouterr().out


def _roots(tmp_path, table=None):
    args = argparse.Namespace(durable_root=str(tmp_path / "durable"), scratch_root=None, nsd_hdf5=None,
                              media_root=str(tmp_path / "media"),
                              framing_table=str(table) if table else None)
    return extract.Roots(args)


def _table(tmp_path, sha="f" * 64):
    t = tmp_path / "framing_table.csv"
    t.write_text("path\n")
    (tmp_path / "framing_table.csv.json").write_text(f'{{"sha256": "{sha}"}}')
    return t


def _video_cell(tmp_path):
    unit = extract.Unit(id="bourne01", out_dir=tmp_path / "durable" / "movie10" / "features" / "bourne01",
                        inputs=[str(tmp_path / "media" / "cneuromod" / "bourne01.mkv")])
    src = extract.Source("movie10", "frames", "viz2psy", "", ["clip"], "synthetic", lambda: [unit])
    return src, unit


def test_video_cells_pass_the_framing_table(tmp_path, monkeypatch):
    table = _table(tmp_path)
    monkeypatch.setattr(extract, "ROOTS", _roots(tmp_path, table), raising=False)
    src, unit = _video_cell(tmp_path)
    cmd = extract.command_for(src, unit, "clip")
    assert cmd[cmd.index("--framing-table") + 1] == str(table)
    # staged PNG frames (AVCaps) are stills: no table
    png = extract.Unit(id="u0", out_dir=unit.out_dir, inputs=["a.png", "b.png"])
    assert "--framing-table" not in extract.command_for(src, png, "clip")


def test_video_cell_without_table_refuses(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(extract, "ROOTS", _roots(tmp_path), raising=False)
    src, unit = _video_cell(tmp_path)
    with pytest.raises(SystemExit, match="framing-table"):
        extract.command_for(src, unit, "clip")


def test_video_cell_done_only_under_the_current_table(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(extract, "ROOTS", _roots(tmp_path, _table(tmp_path, sha="new" * 8)), raising=False)
    monkeypatch.setattr(extract, "check_sidecar", lambda sidecar, source: [])
    src, unit = _video_cell(tmp_path)
    stem = extract.stem_for(src, unit, "clip")
    stem.parent.mkdir(parents=True)
    for sha, want in [("old" * 8, False), ("new" * 8, True), (None, False)]:
        framing = {"frame_source": "raw decode"} if sha is None else {"framing_table_sha256": sha}
        extract.sidecar_for(stem).write_text(json.dumps({"input": {"type": "video", "framing": framing}}))
        assert extract.is_done(stem, src, unit) is want
    # non-video cells ignore framing
    assert extract.is_done(stem) is True


def test_cneuromod_media_comes_from_the_media_root(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "ROOTS", _roots(tmp_path), raising=False)
    d = tmp_path / "media" / "cneuromod" / "algonauts_2025.competitors" / "stimuli" / "movies" / "movie10" / "bourne"
    d.mkdir(parents=True)
    (d / "bourne01.mkv").write_bytes(b"x")
    got = extract._cneuromod_media("movie10")
    assert got == [d / "bourne01.mkv"]


def test_blender_films_are_units_from_the_media_root(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "ROOTS", _roots(tmp_path), raising=False)
    d = tmp_path / "media" / "blender"
    d.mkdir(parents=True)
    for name in ("sintel.mkv", "big-buck-bunny.mov", "MANIFEST.sha256", "notes.txt"):
        (d / name).write_bytes(b"x")
    units = extract._video_units("blender")
    assert [u.id for u in units] == ["big-buck-bunny", "sintel"]
    assert units[0].inputs == [str(d / "big-buck-bunny.mov")]
    assert units[0].extra == ["--stimulus-id", "ext-blender-big-buck-bunny"]
    assert units[0].out_dir == tmp_path / "durable" / "blender" / "features" / "big-buck-bunny"
    tr = extract._video_transcript_units("blender")[1]
    assert tr.inputs == [str(tmp_path / "durable" / "blender" / "inputs" / "sintel_transcript.csv")]


def test_blender_runs_like_the_other_video_corpora():
    srcs = {s.key: s for s in extract.build_sources()}
    for source in ("frames", "audio", "transcript", "vlmcap"):
        assert srcs[f"blender/{source}"].models == srcs[f"movie10/{source}"].models
    assert srcs["blender/transcript"].depends == "blender/audio/transcribe"


# ---- caption_vlm wiring (psytwill-space 0.2.0 caption member) ----------------

def _models_by_source():
    return {s.key: s.models for s in extract.build_sources()}


def test_text_kind_viz2psy_models_never_enter_V():
    assert "caption" not in extract.V_MODELS and "caption_vlm" not in extract.V_MODELS


def test_caption_vlm_runs_on_video_corpora_only():
    by = _models_by_source()
    with_vlm = sorted(k for k, models in by.items() if "caption_vlm" in models)
    assert with_vlm == ["blender/frames", "friends/frames", "movie10/frames"]
    assert "caption_vlm" not in by["nsd/image"] and "caption_vlm" not in by["avcaps/frames"]


def test_vlmcap_sources_score_the_caption_vlm_tables_as_text():
    srcs = {s.key: s for s in extract.build_sources()}
    for corpus in extract.VIDEO_CORPORA:
        s = srcs[f"{corpus}/vlmcap"]
        assert (s.package, s.prefix, s.depends) == ("word2psy", "vlmcap_", f"{corpus}/frames/caption_vlm")
        assert s.models == extract.L_MODELS


def test_caption_vlm_batch_size_is_the_reference_runs(tmp_path):
    src = extract.Source("c", "image", "viz2psy", "", ["clip", "caption_vlm"], "synthetic", lambda: [])
    unit = extract.Unit(id="u", out_dir=tmp_path, inputs=[str(tmp_path / "a.png")])

    def batch(model):
        cmd = extract.command_for(src, unit, model)
        return cmd[cmd.index("--batch-size") + 1]
    assert (batch("caption_vlm"), batch("clip")) == ("32", "64")


def test_caption_vlm_gpu_guard():
    problem = extract.caption_vlm_gpu_problem
    h100, a100 = "NVIDIA H100 80GB HBM3", "NVIDIA A100 80GB PCIe"
    assert problem({"clip"}, []) is None                       # no caption_vlm cell: no check
    assert problem({"caption_vlm"}, [h100], "H100") is None
    assert problem({"caption_vlm"}, [h100], "") is None        # override off
    msg = problem({"caption_vlm", "clip"}, [a100], "H100")
    assert "--skip-model caption_vlm" in msg and "A100" in msg
    assert "no GPU" in problem({"caption_vlm"}, [], "H100")
    assert problem({"caption_vlm"}, [h100, a100], "H100")      # every visible GPU must match
