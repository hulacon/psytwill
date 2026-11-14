"""Offline checks on scripts/fitcorpus/stage.py's captioned-corpus adapters."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fitcorpus_stage", Path(__file__).parents[1] / "scripts" / "fitcorpus" / "stage.py")
stage = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def clotho_root(tmp_path):
    root = tmp_path / "scratch" / "clotho"
    root.mkdir(parents=True)
    meta = {"development": ["Car Wash.wav", "car_wash.wav", "Door.wav"],
            "evaluation": ["Door.wav", "Rain.wav"]}  # Door ships in both splits
    for split, names in meta.items():
        _write_csv(root / f"clotho_metadata_{split}.csv",
                   [{"file_name": n, "sound_id": "1"} for n in names])
        _write_csv(root / f"clotho_captions_{split}.csv",
                   [{"file_name": n, **{f"caption_{k}": f"{n} caption {k}" for k in range(1, 6)}}
                    for n in names])
    return root


def _args(tmp_path, **kw):
    return argparse.Namespace(scratch_root=tmp_path / "scratch", durable_root=tmp_path / "durable",
                              dry_run=False, **kw)


class TestClothoIds:
    def test_slug_colliding_names_get_distinct_ids(self):
        a, b = stage.clotho_native("dev", "Car Wash.wav"), stage.clotho_native("dev", "car_wash.wav")
        assert stage.slug("Car Wash") == stage.slug("car_wash")
        assert a != b

    def test_split_is_part_of_the_id(self):
        assert stage.clotho_native("dev", "x.wav").startswith("dev-")
        assert stage.clotho_native("eval", "x.wav") != stage.clotho_native("dev", "x.wav")

    def test_ids_pass_the_namespace_rule(self):
        stage.ext_id("clotho", stage.clotho_native("eval", "Oscar de Ávila (1).wav"))


class TestClothoClips:
    def test_eval_copy_of_a_dev_clip_is_dropped(self, clotho_root):
        clips = list(stage._clotho_clips(clotho_root))
        evals = [fn for _, tag, _, fn, _ in clips if tag == "eval"]
        assert evals == ["Rain.wav"]
        assert len(clips) == 4


class TestCaptions:
    def test_clotho_rows_follow_kept_clips(self, tmp_path, clotho_root):
        rows = stage.caption_rows("clotho", _args(tmp_path))
        assert len(rows) == 4 * 5
        assert {r["caption_type"] for r in rows} == {"audio"}
        assert not any(r["caption"].startswith("Door") and r["split"] == "evaluation" for r in rows)

    def test_avcaps_rows_keep_human_types_only(self, tmp_path):
        root = tmp_path / "scratch" / "avcaps"
        root.mkdir(parents=True)
        entry = {"audio_captions": ["a dog barks loudly outside", "two\nscenes pasted"],
                 "visual_captions": ["a dog on grass"], "audio_visual_captions": ["a dog barks on grass"],
                 "GPT_AV_captions": ["synthetic"]}
        for split in stage.AVCAPS_SPLITS:
            (root / f"{split}_captions.json").write_text(json.dumps({"123": entry} if split == "test" else {}))
        args = _args(tmp_path)
        rows = stage.caption_rows("avcaps", args)
        assert {r["caption_type"] for r in rows} == {"audio", "visual", "audio_visual"}
        assert all(r["stimulus_id"] == "ext-avcaps-123" for r in rows)
        stage.write_captions("avcaps", args)
        inputs = args.durable_root / "avcaps" / "inputs"
        kept = list(csv.DictReader(open(inputs / "captions.csv", encoding="utf-8")))
        dropped = list(csv.DictReader(open(inputs / "captions_dropped.csv", encoding="utf-8")))
        assert len(kept) == 3 and len(dropped) == 1
        assert dropped[0]["reason"] == "multi-line entry"

    @pytest.mark.parametrize("text,reason", [("A dog barks.", ""), ("  ", "empty"),
                                             ("one\n\nRevised: two", "multi-line entry")])
    def test_drop_reason(self, text, reason):
        assert stage.caption_drop_reason(text) == reason


def test_captioned_corpora_pack_with_the_wide_gap():
    seg = [stage.Segment(segment=f"c{k}", group=f"c{k}", duration=15.2) for k in range(2)]
    unit = stage.pack_groups([(s.group, [s]) for s in seg], target=3600, hop=stage.HOP,
                             gap=stage.GAPS["clotho"])[0]
    assert unit.onsets[1] - unit.offsets[0] >= 10.0


def test_frame_ids_parse_back_to_their_clip():
    from psytwill.fitcorpus import parse_ext_id

    sid = stage.frame_id("avcaps", "10001787725", 12.5)
    assert sid == "ext-avcaps-10001787725-t0012500"
    corpus, native = parse_ext_id(sid)
    assert corpus == "avcaps" and native.rsplit("-t", 1) == ["10001787725", "0012500"]


def test_frame_units_never_mix_sizes():
    clips = [("a", "640x480", 300.0), ("b", "480x640", 30.0), ("c", "640x480", 300.0),
             ("d", "640x480", 10.0), ("e", "480x640", 20.0)]
    units = stage.plan_frame_units(clips, per_unit=1000)
    size = {n: s for n, s, _ in clips}
    for uid, natives in units:
        assert {size[n] for n in natives} == {uid.rsplit("-", 1)[0]}
    assert [u for u, _ in units] == ["480x640-000", "640x480-000", "640x480-001"]
    assert sorted(n for _, ns in units for n in ns) == list("abcde")


# ---- LibriSpeech held-out splits ---------------------------------------------

def _librispeech_split(tmp_path, split, speaker="1089", chapter="134686"):
    """A one-chapter LibriSpeech tree. The .flac files are empty placeholders:
    the adapter only lists them and asks media_duration (stubbed by the
    tests), so no audio library is needed."""
    chap = tmp_path / "scratch" / "librispeech" / split / "LibriSpeech" / split / speaker / chapter
    chap.mkdir(parents=True)
    lines = []
    for k in range(2):
        uid = f"{speaker}-{chapter}-{k:04d}"
        (chap / f"{uid}.flac").write_bytes(b"")
        lines.append(f"{uid} HELLO WORLD {k}")
    (chap / f"{speaker}-{chapter}.trans.txt").write_text("\n".join(lines) + "\n")


class TestLibrispeechSplits:
    @pytest.fixture(autouse=True)
    def _no_audio_io(self, monkeypatch):
        monkeypatch.setattr(stage, "media_duration", lambda path: 1.0)

    def test_held_out_split_reads_its_own_tree(self, tmp_path):
        _librispeech_split(tmp_path, "test-clean")
        sr, plans, _ = stage.adapt_librispeech(_args(tmp_path, split="test-clean"))
        assert sr == 16000 and [p for p, _ in plans] == ["spk1089-"]
        segs = plans[0][1][0][1]
        assert [s.text for s in segs] == ["hello world 0", "hello world 1"]

    def test_default_split_is_the_fit_split(self, tmp_path):
        _librispeech_split(tmp_path, "train-clean-100", speaker="19", chapter="198")
        _, plans, _ = stage.adapt_librispeech(_args(tmp_path))
        assert [p for p, _ in plans] == ["spk19-"]

    def test_held_out_split_refused_without_the_marker(self, tmp_path):
        (tmp_path / "durable").mkdir()
        with pytest.raises(SystemExit, match="held out"):
            stage.check_heldout_root("librispeech", "test-clean", tmp_path / "durable")

    def test_held_out_split_allowed_with_the_marker(self, tmp_path):
        root = tmp_path / "durable"
        root.mkdir()
        (root / stage.HELDOUT_MARKER).write_text("")
        stage.check_heldout_root("librispeech", "test-other", root)

    def test_fit_split_refused_in_a_held_out_root(self, tmp_path):
        root = tmp_path / "durable"
        root.mkdir()
        (root / stage.HELDOUT_MARKER).write_text("")
        with pytest.raises(SystemExit, match="held-out root"):
            stage.check_heldout_root("librispeech", "train-clean-100", root)
        stage.check_heldout_root("gigaspeech", "train-clean-100", root)  # other corpora: no rule


# ---- per-item staging (E5 "re-extract per item", DECIDED 2026-11-13) -----------------------

ITEM_SR = 22050


def _tone(freqs, seconds, sr=ITEM_SR):
    t = np.arange(int(round(seconds * sr))) / sr
    return (0.2 * sum(np.sin(2 * np.pi * f * t) for f in freqs)).astype(np.float32)


def _item_adapter(items):
    """A fake corpus adapter: ``items`` = [(group, samples)], one file per group."""
    def adapter(args):
        groups = [(g, [stage.Segment(segment=g, group=g, duration=len(y) / ITEM_SR,
                                     extra={"genre": "test"})]) for g, y in items]
        ys = dict(items)
        return ITEM_SR, [("x-", groups)], lambda seg, sr: ys[seg.group]
    return adapter


def _stage_args(root, per_item, max_groups=None, **kw):
    return argparse.Namespace(scratch_root=root / "scratch", durable_root=root / "durable", dry_run=False,
                              force=False, target_minutes=25.0, per_item=per_item, max_groups=max_groups, **kw)


def _stage(monkeypatch, root, items, per_item):
    monkeypatch.setitem(stage.ADAPTERS, "fma", _item_adapter(items))
    args = _stage_args(root, per_item)
    rows = stage.stage_packed("fma", args)
    stage.write_manifest("fma", rows, args)
    return rows


def _read_wav(path):
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32")
    assert sr == ITEM_SR
    return y


A_ITEM, B_ITEM, C_ITEM = _tone([261.6, 329.6, 392.0], 2.2), _tone([1000.0], 4.0), _tone([466.2, 587.3], 4.0)


class TestPerItemStaging:
    def test_one_wav_per_item_holding_only_that_item(self, tmp_path, monkeypatch):
        rows = _stage(monkeypatch, tmp_path, [("a", A_ITEM), ("b", B_ITEM)], per_item=True)
        assert [r["unit"] for r in rows] == ["x-a", "x-b"]
        assert all(r["n_segments"] == 1 for r in rows)
        for r, y in zip(rows, (A_ITEM, B_ITEM)):
            assert Path(r["path"]).parent.name == "items"  # never beside packed units/
            wav = _read_wav(r["path"])
            assert len(wav) - len(y) < stage.HOP * ITEM_SR  # grid rounding only
            np.testing.assert_allclose(wav[:len(y)], y, atol=1 / 32768)
            assert not wav[len(y):].any()
        inputs = tmp_path / "durable" / "fma" / "inputs"
        segs = list(csv.DictReader(open(inputs / "x-a_segments.csv")))
        assert len(segs) == 1 and float(segs[0]["onset"]) == 0.0 and segs[0]["group"] == "a"
        assert segs[0]["genre"] == "test"  # per-corpus columns ride along as before
        assert json.loads((inputs / stage.LAYOUT_FILE).read_text())["layout"] == "per-item"

    def test_per_item_is_for_the_decided_corpora_only(self, tmp_path):
        assert stage.PER_ITEM_CORPORA == ("twp-unpresented", "fma", "musopen")
        with pytest.raises(SystemExit, match="per-item"):
            stage.check_layout("librispeech", _stage_args(tmp_path, per_item=True))

    @pytest.mark.parametrize("first,second", [(False, True), (True, False)])
    def test_a_root_is_never_staged_in_both_layouts(self, tmp_path, monkeypatch, first, second):
        _stage(monkeypatch, tmp_path, [("a", A_ITEM), ("b", B_ITEM)], per_item=first)
        with pytest.raises(SystemExit, match="layout"):
            _stage(monkeypatch, tmp_path, [("a", A_ITEM), ("b", B_ITEM)], per_item=second)

    def test_max_groups_stages_a_pilot_subset_and_says_so(self, tmp_path, monkeypatch):
        monkeypatch.setitem(stage.ADAPTERS, "fma", _item_adapter([("a", A_ITEM), ("b", B_ITEM), ("c", C_ITEM)]))
        args = _stage_args(tmp_path, per_item=True, max_groups=2)
        assert [r["unit"] for r in stage.stage_packed("fma", args)] == ["x-a", "x-b"]
        rec = json.loads((tmp_path / "durable" / "fma" / "inputs" / stage.LAYOUT_FILE).read_text())
        assert rec["max_groups"] == 2  # a pilot root is never mistaken for a full one

    def test_restaging_the_same_layout_is_idempotent(self, tmp_path, monkeypatch):
        _stage(monkeypatch, tmp_path, [("a", A_ITEM)], per_item=True)
        _stage(monkeypatch, tmp_path, [("a", A_ITEM)], per_item=True)


def _centred_window_mean(y, sr, center, w):
    """A windowed member as the producers compute it: a centred window clamped to the clip."""
    dur = len(y) / sr
    lo = 0.0 if dur <= w else min(max(center - w / 2, 0.0), dur - w)
    hi = min(lo + w, dur)
    return float(np.abs(y[int(lo * sr):int(hi * sr)]).mean())


class TestWindowNearAnItemBoundary:
    """A windowed member near an item's end reads only its own item once it is staged
    per item; packed, the same window reads the neighbour (the control)."""

    def _last_row(self, rows, unit_id):
        wav = _read_wav(next(r["path"] for r in rows if r["unit"] == unit_id))
        # the last grid row inside item a (2.2 s long): bin [1.5, 2.0), centre 1.75
        return wav, 1.75

    def test_synthetic_window(self, tmp_path, monkeypatch):
        got = {}
        for neighbour, y in (("b", B_ITEM), ("c", C_ITEM)):
            for per_item in (True, False):
                rows = _stage(monkeypatch, tmp_path / f"{neighbour}-{per_item}", [("a", A_ITEM), (neighbour, y)],
                              per_item=per_item)
                wav, c = self._last_row(rows, "x-a" if per_item else "x-000")
                got[neighbour, per_item] = _centred_window_mean(wav, ITEM_SR, c, 3.0)
        assert got["b", True] == got["c", True]  # per item: the neighbour is never read
        assert got["b", False] != got["c", False]  # packed: it is

    def test_real_windowed_member(self, tmp_path, monkeypatch):
        """aud2psy's tonal (a 3 s centred chroma window) on the same layouts."""
        pipeline = pytest.importorskip("aud2psy.pipeline")
        got = {}
        for neighbour, y in (("b", B_ITEM), ("c", C_ITEM)):
            for per_item in (True, False):
                rows = _stage(monkeypatch, tmp_path / f"{neighbour}-{per_item}", [("a", A_ITEM), (neighbour, y)],
                              per_item=per_item)
                path = next(r["path"] for r in rows if r["unit"] == ("x-a" if per_item else "x-000"))
                frames = pipeline.score_audio(path, ["tonal"], hop=stage.HOP, show_progress=False).frames_df
                got[neighbour, per_item] = frames[frames["time"] < 2.2].filter(like="tonal_").to_numpy(float)
        np.testing.assert_array_equal(got["b", True], got["c", True])
        assert not np.allclose(got["b", False], got["c", False], equal_nan=True)
        assert not np.allclose(got["b", True], got["b", False], equal_nan=True)
