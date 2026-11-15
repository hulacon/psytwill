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


# ---- AVCaps through the video path (viz2psy video sets) ------------------------

def _avcaps_media(tmp_path, split, names):
    d = tmp_path / "media" / "avcaps" / f"{split}_videos" / f"{split}_videos"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"x")
    return d


def test_avcaps_video_units_are_whole_clips_per_split(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "ROOTS", _roots(tmp_path), raising=False)
    monkeypatch.setattr(extract, "AVCAPS_VIDEO_UNIT", 2)
    tr = _avcaps_media(tmp_path, "train", ["30.mp4", "10.mp4", "20.mp4", "notes.txt"])
    va = _avcaps_media(tmp_path, "val", ["40.mp4"])
    units = extract._avcaps_video_units()
    assert [u.id for u in units] == ["train-000", "train-001", "val-000"]
    assert units[0].inputs == [str(tr / "10.mp4"), str(tr / "20.mp4")]
    assert units[1].inputs == [str(tr / "30.mp4")] and units[2].inputs == [str(va / "40.mp4")]
    assert units[0].extra == ["--id-pattern", "ext-avcaps-{stem}"]
    assert units[0].out_dir == tmp_path / "durable" / "avcaps" / "features" / "video" / "train-000"


def test_avcaps_clip_with_a_bad_native_id_fails(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(extract, "ROOTS", _roots(tmp_path), raising=False)
    _avcaps_media(tmp_path, "test", ["Bad_Name.mp4"])
    with pytest.raises(Exception, match="native id"):
        extract._avcaps_video_units()


def test_avcaps_video_source_is_framed_and_keeps_the_stills_source():
    srcs = {s.key: s for s in extract.build_sources()}
    assert srcs["avcaps/video"].models == srcs["avcaps/frames"].models
    assert "motion" not in srcs["avcaps/video"].models and "caption_vlm" not in srcs["avcaps/video"].models


def test_video_set_cell_command(tmp_path, monkeypatch):
    table = _table(tmp_path)
    monkeypatch.setattr(extract, "ROOTS", _roots(tmp_path, table), raising=False)
    tr = _avcaps_media(tmp_path, "train", ["10.mp4", "20.mp4"])
    src = {s.key: s for s in extract.build_sources()}["avcaps/video"]
    unit = extract._avcaps_video_units()[0]
    assert extract.is_video_cell(src, unit)
    cmd = extract.command_for(src, unit, "clip")
    assert cmd[cmd.index("clip") + 1: cmd.index("clip") + 3] == [str(tr / "10.mp4"), str(tr / "20.mp4")]
    assert cmd[cmd.index("--framing-table") + 1] == str(table)
    assert cmd[cmd.index("--frame-interval") + 1] == str(extract.GRID_HOP)
    assert cmd[cmd.index("--id-pattern") + 1] == "ext-avcaps-{stem}"


def test_video_set_done_only_for_exactly_its_clips(tmp_path, monkeypatch):
    import json

    sha = "new" * 8
    monkeypatch.setattr(extract, "ROOTS", _roots(tmp_path, _table(tmp_path, sha=sha)), raising=False)
    monkeypatch.setattr(extract, "check_sidecar", lambda sidecar, source: [])
    tr = _avcaps_media(tmp_path, "train", ["10.mp4", "20.mp4"])
    src = {s.key: s for s in extract.build_sources()}["avcaps/video"]
    unit = extract._avcaps_video_units()[0]
    stem = extract.stem_for(src, unit, "clip")
    stem.parent.mkdir(parents=True)
    a, b = str((tr / "10.mp4").resolve()), str((tr / "20.mp4").resolve())
    for listed, table_sha, want in [([a, b], sha, True), ([a], sha, False), ([b, a], sha, False),
                                    ([a, b], "old" * 8, False)]:
        side = {"input": {"type": "video_set", "framing": {"framing_table_sha256": table_sha},
                          "videos": [{"path": p} for p in listed]}}
        extract.sidecar_for(stem).write_text(json.dumps(side))
        assert extract.is_done(stem, src, unit) is want


def _qc_corpus(tmp_path, monkeypatch, texts):
    durable = tmp_path / "durable"
    unit = extract.Unit(id="ep01", out_dir=durable / "friends" / "features" / "ep01",
                        inputs=[str(durable / "friends" / "inputs" / "ep01_transcript.csv")])
    Path(unit.inputs[0]).parent.mkdir(parents=True)
    rows = ["stimulus_id,text,onset,offset"] + [f"ext-friends-ep01,{t},{i}.0,{i}.5" for i, t in enumerate(texts)]
    Path(unit.inputs[0]).write_text("\n".join(rows) + "\n")
    monkeypatch.setattr(extract, "ROOTS", types.SimpleNamespace(durable=durable), raising=False)
    monkeypatch.setattr(extract, "_whisper_transcript_units", lambda: {"friends": [unit], "blender": []})
    return durable, unit


def test_qc_then_exclusions_union_the_base_rows_with_loop_rows(tmp_path, monkeypatch):
    import pandas as pd

    durable, _ = _qc_corpus(tmp_path, monkeypatch, ["Hi."] + ["Airplane."] * 6)
    assert extract.cmd_qc(argparse.Namespace(corpus=None, dry_run=False, allow_partial=False)) == 0
    flags = pd.read_parquet(durable / "friends" / "qc" / "transcript_flags.parquet")
    assert flags["loop_exclude"].sum() == 5 and (flags["unit"] == "ep01").all()

    base = tmp_path / "base.parquet"
    pd.DataFrame({"stimulus_id": ["ext-nsd-000001"], "chunk_idx": [0]}).to_parquet(base)
    out = durable / "space" / "exclusions" / "L_test_exclude_rows.parquet"
    args = argparse.Namespace(base_rows=str(base), base_reason="caption_draw", out=str(out), dry_run=False)
    assert extract.cmd_exclusions(args) == 0
    rows = pd.read_parquet(out)
    assert rows["reason"].value_counts().to_dict() == {"transcript_loop": 5, "caption_draw": 1}
    assert sorted(rows.loc[rows.reason == "transcript_loop", "chunk_idx"]) == [2, 3, 4, 5, 6]
    assert extract.cmd_exclusions(args) == 0  # same rows again: a no-op


def test_exclusions_refuse_flags_older_than_their_inputs(tmp_path, monkeypatch):
    import pandas as pd
    import pytest

    durable, unit = _qc_corpus(tmp_path, monkeypatch, ["Hi."] * 5)
    extract.cmd_qc(argparse.Namespace(corpus=None, dry_run=False, allow_partial=False))
    with open(unit.inputs[0], "a") as f:
        f.write("ext-friends-ep01,Bye.,9.0,9.5\n")  # transcript re-derived after the flags
    base = tmp_path / "base.parquet"
    pd.DataFrame({"stimulus_id": ["x"], "chunk_idx": [0]}).to_parquet(base)
    args = argparse.Namespace(base_rows=str(base), base_reason="r", out=str(tmp_path / "o.parquet"), dry_run=False)
    with pytest.raises(SystemExit, match="predate"):
        extract.cmd_exclusions(args)


def test_exclusions_refuse_to_overwrite_a_written_list(tmp_path, monkeypatch):
    import pandas as pd
    import pytest

    _qc_corpus(tmp_path, monkeypatch, ["Hi."] * 5)
    extract.cmd_qc(argparse.Namespace(corpus=None, dry_run=False, allow_partial=False))
    base = tmp_path / "base.parquet"
    pd.DataFrame({"stimulus_id": ["x"], "chunk_idx": [0]}).to_parquet(base)
    out = tmp_path / "o.parquet"
    extract.cmd_exclusions(argparse.Namespace(base_rows=str(base), base_reason="r", out=str(out), dry_run=False))
    pd.DataFrame({"stimulus_id": ["y"], "chunk_idx": [0]}).to_parquet(base)
    with pytest.raises(SystemExit, match="immutable"):
        extract.cmd_exclusions(argparse.Namespace(base_rows=str(base), base_reason="r", out=str(out), dry_run=False))


def test_roots_resolve_the_nsd_brick(tmp_path, monkeypatch):
    # Regression: 0.50.0 left this assignment after the return in framing_sha256, so
    # Roots never had nsd_hdf5 and every nsd/image unit listing raised AttributeError.
    monkeypatch.delenv("PSYTWILL_NSD_HDF5", raising=False)
    assert _roots(tmp_path).nsd_hdf5 is None
    monkeypatch.setenv("PSYTWILL_NSD_HDF5", str(tmp_path / "nsd.hdf5"))
    roots = _roots(tmp_path)
    assert roots.nsd_hdf5 == tmp_path / "nsd.hdf5"
    monkeypatch.setattr(extract, "ROOTS", roots, raising=False)
    units = extract._nsd_units()
    assert units and units[0].inputs == [str(tmp_path / "nsd.hdf5")]


# ---- per-item corpora: many short units, each model loaded once per batch ------------------

def _per_item_corpus(tmp_path, monkeypatch, n=3, corpus="fma"):
    durable = tmp_path / "durable"
    inputs = durable / corpus / "inputs"
    inputs.mkdir(parents=True)
    rows = ["unit,stimulus_id,path,sr,duration_sec,n_segments,audio_sec"]
    for i in range(n):
        wav = tmp_path / "scratch" / corpus / "items" / f"x-{i}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"")
        rows.append(f"x-{i},ext-{corpus}-x-{i},{wav},44100,30.0,1,30.0")
    (inputs / "units.csv").write_text("\n".join(rows) + "\n")
    (inputs / "layout.json").write_text('{"layout": "per-item"}')
    monkeypatch.setattr(extract, "ROOTS", types.SimpleNamespace(durable=durable, scratch=tmp_path / "scratch",
                                                                nsd_hdf5=None, media=None, framing_table=None),
                        raising=False)
    monkeypatch.setattr(extract, "check_sidecar", lambda sidecar, source: [])
    monkeypatch.setattr(extract._per_item_layout, "cache", {}, raising=False)
    monkeypatch.setattr(extract, "gpu_names", lambda: [])  # no nvidia-smi through the fake run
    return durable


class _FakeAud2psy:
    """Stands in for `python -m aud2psy.cli MODEL --inputs-from M -o DIR`: writes each row's
    family (and a producer's table), records the call. ``fail`` = {(model, stimulus_id)}."""

    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    def __call__(self, cmd, **kw):
        import csv as _csv

        model = cmd[3]
        manifest = Path(cmd[cmd.index("--inputs-from") + 1])
        out = Path(cmd[cmd.index("-o") + 1])
        rows = list(_csv.DictReader(open(manifest)))
        self.calls.append((model, rows))
        for r in rows:
            if (model, r["stimulus_id"]) in self.fail:
                continue
            stem = out / r["output"]
            stem.parent.mkdir(parents=True, exist_ok=True)
            base = stem.parent / stem.stem
            Path(f"{base}.meta.json").write_text("{}")
            Path(f"{base}_frames.csv").write_text(f"stimulus_id,time,{model}_x\n{r['stimulus_id']},0.25,1\n")
            if model == "diarize":
                Path(f"{base}_speakers.csv").write_text("speaker,onset,offset\n")
            if model == "transcribe":
                Path(f"{base}_transcript_words.csv").write_text("word,onset,offset\n")
        return types.SimpleNamespace(returncode=0)


def _run_args(**kw):
    base = dict(corpus="fma", source="audio", model=None, skip_model=None, unit=None, unit_index=None,
                unit_slice=None, dry_run=False, redo=False, fail_fast=False, batch_items=500)
    base.update(kw)
    return argparse.Namespace(**base)


def test_layout_is_read_from_the_staging_record(tmp_path, monkeypatch):
    _per_item_corpus(tmp_path, monkeypatch)
    assert extract._per_item_layout("fma") is True
    assert extract._per_item_layout("musopen") is False  # nothing staged: the packed default


def test_per_item_cells_run_one_call_per_model_with_producers_first(tmp_path, monkeypatch):
    durable = _per_item_corpus(tmp_path, monkeypatch)
    fake = _FakeAud2psy()
    monkeypatch.setattr(extract.subprocess, "run", fake)
    assert extract.cmd_run(_run_args()) == 0
    models = [m for m, _ in fake.calls]
    assert sorted(models) == sorted(extract._ordered_audio())  # each model once, for all 3 items
    assert models.index("diarize") < min(models.index(m) for m in ("conversation", "egemaps", "speech_emotion"))
    assert models.index("transcribe") < models.index("speech_rate")
    feats = durable / "fma" / "features"
    for model, rows in fake.calls:
        assert [r["stimulus_id"] for r in rows] == [f"ext-fma-x-{i}" for i in range(3)]
        assert [r["output"] for r in rows] == [f"x-{i}/{model}.csv" for i in range(3)]
        if extract._AUDIO_DEPS.get(model) == "diarize":
            assert [r["speakers"] for r in rows] == [str(feats / f"x-{i}" / "diarize_speakers.csv") for i in range(3)]
        else:
            assert "speakers" not in rows[0]
        if model == "speech_rate":
            assert [r["words"] for r in rows] == [str(feats / f"x-{i}" / "transcribe_transcript_words.csv")
                                                  for i in range(3)]
    # every cell is done, so a second run has nothing to do
    fake.calls.clear()
    assert extract.cmd_run(_run_args()) == 0 and fake.calls == []


def test_batches_are_capped(tmp_path, monkeypatch):
    _per_item_corpus(tmp_path, monkeypatch, n=5)
    fake = _FakeAud2psy()
    monkeypatch.setattr(extract.subprocess, "run", fake)
    assert extract.cmd_run(_run_args(model="loudness", batch_items=2)) == 0
    assert [len(rows) for _, rows in fake.calls] == [2, 2, 1]


def test_unit_slice_selects_items_for_an_array_task(tmp_path, monkeypatch):
    _per_item_corpus(tmp_path, monkeypatch, n=5)
    fake = _FakeAud2psy()
    monkeypatch.setattr(extract.subprocess, "run", fake)
    assert extract.cmd_run(_run_args(model="loudness", unit_slice="1:3")) == 0
    assert [r["stimulus_id"] for r in fake.calls[0][1]] == ["ext-fma-x-1", "ext-fma-x-2"]


def test_a_consumer_skips_the_item_whose_producer_failed(tmp_path, monkeypatch, capsys):
    _per_item_corpus(tmp_path, monkeypatch)
    fake = _FakeAud2psy(fail={("diarize", "ext-fma-x-1")})
    monkeypatch.setattr(extract.subprocess, "run", fake)
    assert extract.cmd_run(_run_args(model=None, skip_model=None)) == 1
    eg = next(rows for m, rows in fake.calls if m == "egemaps")
    assert [r["stimulus_id"] for r in eg] == ["ext-fma-x-0", "ext-fma-x-2"]
    out = capsys.readouterr().out
    assert "FAILED fma/audio/diarize[x-1]" in out and "fma/audio/egemaps[x-1] (missing input)" in out


def test_dry_run_writes_no_manifest(tmp_path, monkeypatch, capsys):
    durable = _per_item_corpus(tmp_path, monkeypatch)
    monkeypatch.setattr(extract.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert extract.cmd_run(_run_args(dry_run=True)) == 0
    assert not (durable / "fma" / "inputs" / "_batch").exists()
    assert "--inputs-from" in capsys.readouterr().out


def test_packed_corpora_keep_one_call_per_cell(tmp_path, monkeypatch):
    durable = _per_item_corpus(tmp_path, monkeypatch)
    (durable / "fma" / "inputs" / "layout.json").unlink()
    extract._per_item_layout.cache.clear()
    calls = []
    monkeypatch.setattr(extract.subprocess, "run", lambda cmd, **k: calls.append(cmd) or types.SimpleNamespace(returncode=1))
    extract.cmd_run(_run_args(model="loudness"))
    assert len(calls) == 3 and all("--inputs-from" not in c for c in calls)


# ---- the centre rule: an exclusion list for per-item corpora --------------------------------

def _per_item_table(tmp_path, monkeypatch, items, corpus="twp-unpresented", layout="per-item"):
    """``items`` = {unit: (segments rows [(onset, offset, group)], frame centres)}."""
    import json

    import pandas as pd

    from psytwill.windows import record_table_segments

    durable = tmp_path / "durable"
    inputs = durable / corpus / "inputs"
    inputs.mkdir(parents=True)
    if layout:
        (inputs / "layout.json").write_text(json.dumps({"layout": layout}))
    rows = []
    for unit, (segs, centres) in items.items():
        sid = f"ext-{corpus}-{unit}"
        pd.DataFrame([(sid, on, off, g) for on, off, g in segs],
                     columns=["stimulus_id", "onset", "offset", "group"]).to_csv(inputs / f"{unit}_segments.csv",
                                                                                 index=False)
        for t in centres:
            for model in ("clap", "loudness"):
                rows.append((sid, t, model, f"{model}_000", 1.0))
    df = pd.DataFrame(rows, columns=["stimulus_id", "time", "model", "feature", "value"])
    table = durable / corpus / "groups" / f"{corpus}_audio_frames_features.parquet"
    table.parent.mkdir(parents=True)
    df.to_parquet(table, index=False)
    table.with_suffix(".meta.json").write_text("{}")
    record_table_segments(table, inputs)
    monkeypatch.setattr(extract, "ROOTS", types.SimpleNamespace(durable=durable), raising=False)
    return durable, table


def _centre_args(out, corpus=None, dry_run=False):
    return argparse.Namespace(corpus=corpus, out=str(out), dry_run=dry_run)


def test_centre_rule_lists_the_rows_outside_each_item_on_the_binned_key(tmp_path, monkeypatch):
    import json

    import pandas as pd

    durable, table = _per_item_table(tmp_path, monkeypatch, {
        "echo-abide": ([(0.0, 0.53, "abide")], [0.25, 0.75]),
        "echo-able": ([(0.0, 0.80, "able")], [0.25, 0.75])})
    out = durable / "space" / "exclusions" / "A_test_exclude_rows.parquet"
    assert extract.cmd_centre_rule(_centre_args(out)) == 0
    rows = pd.read_parquet(out)
    assert rows[["stimulus_id", "time"]].values.tolist() == [["ext-twp-unpresented-echo-abide", 0.5]]
    assert (rows["reason"] == "centre_outside_item").all()
    meta = json.loads(out.with_name(out.stem + ".meta.json").read_text())
    c = meta["corpora"]["twp-unpresented"]
    assert (c["n_units"], c["n_rows"], c["n_excluded"]) == (2, 4, 1)
    assert c["table"] == str(table) and meta["hop"] == 0.5 and meta["key"] == ["stimulus_id", "time"]
    assert extract.cmd_centre_rule(_centre_args(out)) == 0  # same rows again: a no-op


def test_centre_rule_refuses_a_packed_corpus(tmp_path, monkeypatch):
    import pytest

    _per_item_table(tmp_path, monkeypatch, {"u": ([(0.0, 0.5, "a"), (1.0, 1.5, "b")], [0.25, 1.25])},
                    layout=None)
    with pytest.raises(SystemExit, match="not staged per item"):
        extract.cmd_centre_rule(_centre_args(tmp_path / "o.parquet", corpus=["twp-unpresented"]))


def test_centre_rule_refuses_a_unit_holding_two_items(tmp_path, monkeypatch):
    import pytest

    _per_item_table(tmp_path, monkeypatch, {"u": ([(0.0, 0.5, "a"), (1.0, 1.5, "b")], [0.25, 1.25])})
    with pytest.raises(SystemExit, match="2 groups"):
        extract.cmd_centre_rule(_centre_args(tmp_path / "o.parquet"))


def test_centre_rule_refuses_a_table_without_recorded_segments(tmp_path, monkeypatch):
    import pytest

    _, table = _per_item_table(tmp_path, monkeypatch, {"u": ([(0.0, 0.5, "a")], [0.25])})
    table.with_suffix(".meta.json").write_text("{}")
    with pytest.raises(SystemExit, match="records no segments"):
        extract.cmd_centre_rule(_centre_args(tmp_path / "o.parquet"))


def test_centre_rule_refuses_to_overwrite_a_written_list(tmp_path, monkeypatch):
    import pytest

    _per_item_table(tmp_path, monkeypatch, {"u": ([(0.0, 0.53, "a")], [0.25, 0.75])})
    out = tmp_path / "o.parquet"
    extract.cmd_centre_rule(_centre_args(out))
    import pandas as pd

    pd.DataFrame({"stimulus_id": ["x"], "time": [0.0], "reason": ["r"]}).to_parquet(out)
    with pytest.raises(SystemExit, match="immutable"):
        extract.cmd_centre_rule(_centre_args(out))
