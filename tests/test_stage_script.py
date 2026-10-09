"""Offline checks on scripts/fitcorpus/stage.py's captioned-corpus adapters."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

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
