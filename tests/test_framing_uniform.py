"""Video inputs framed under different rules are never stacked or fitted together.

viz2psy >= 0.17.0 records stage 0 of the framing rule in ``input.framing``
(Contract B 1.3). A features table refuses to mix video inputs whose rule
differs, records the shared rule as ``input_framing``, and the space loader
refuses member tables whose rules differ.
"""

import json

import numpy as np
import pandas as pd
import pytest

from psytwill.cli import _assert_tables_framing
from psytwill.exceptions import SpaceError
from psytwill.features import build_features, group_framing
from psytwill.sidecar import input_framing_rule

CLIP_CKPT = "ViT-B-32/laion2b_s34b_b79k"


def stage0(sha="a" * 64, kind="framing table"):
    return {"frame_source": "raw decode", "frame_cache": "memory", "stage0": kind,
            "stage0_version": "1", "framing_table_sha256": sha,
            "crop": {"left": 0, "right": 0, "top": 60, "bottom": 60}, "active_wh": [640, 360]}


def video_input(tmp_path, name, framing=None, sid=None):
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    df = pd.DataFrame({"stimulus_id": [sid or name] * 3, "time": [0.0, 0.5, 1.0],
                       **{f"clip_{i:03d}": rng.normal(size=3) for i in range(4)}})
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    inp = {"type": "video", "path": f"/media/{name}.mp4"}
    if framing is not None:
        inp["framing"] = framing
    side = {"schema_version": "1.3", "extractor": "viz2psy", "extractor_version": "0.17.0",
            "input": inp, "models": {"clip": {"checkpoint": CLIP_CKPT}}}
    path.with_suffix(".meta.json").write_text(json.dumps(side))
    return path


def still_input(tmp_path, name="stills"):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"stimulus_id": [f"nsd{i:05d}" for i in range(3)],
                       **{f"clip_{i:03d}": rng.normal(size=3) for i in range(4)}})
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    side = {"schema_version": "1.3", "extractor": "viz2psy", "extractor_version": "0.17.0",
            "input": {"type": "hdf5_brick", "framing": {"frame_source": "still image",
                                                        "stage0": "none: still image", "stage0_version": "1"}},
            "models": {"clip": {"checkpoint": CLIP_CKPT}}}
    path.with_suffix(".meta.json").write_text(json.dumps(side))
    return path


# ------------------------------------------------------------ the rule reader

def test_rule_reader_kinds():
    assert input_framing_rule({"input": {"type": "video", "framing": stage0()}})["framing_table_sha256"] == "a" * 64
    assert input_framing_rule({"input": {"type": "video"}})["stage0"] == "undeclared"
    assert input_framing_rule({"input": {"type": "video", "framing": {"frame_source": "raw decode"}}})["stage0"] == "undeclared"
    assert input_framing_rule({"input": {"type": "image_folder", "framing": {"frame_source": "still image"}}}) == \
        {"frame_source": "still image"}
    assert input_framing_rule({"input": {"type": "image_folder"}}) is None
    assert input_framing_rule({}) is None


# ------------------------------------------------------------ one table

def test_same_rule_stacks_and_is_recorded(tmp_path):
    a, b = video_input(tmp_path, "a", stage0()), video_input(tmp_path, "b", stage0())
    out = build_features([a, b], tmp_path / "f.csv")
    assert out["meta"]["input_framing"]["framing_table_sha256"] == "a" * 64
    assert group_framing(tmp_path / "f.csv")["stage0"] == "framing table"


def test_different_tables_refused(tmp_path):
    a, b = video_input(tmp_path, "a", stage0()), video_input(tmp_path, "b", stage0(sha="b" * 64))
    with pytest.raises(SpaceError, match="framing mismatch"):
        build_features([a, b], tmp_path / "f.csv")


def test_declared_and_legacy_video_refused(tmp_path):
    a = video_input(tmp_path, "a", stage0())
    b = video_input(tmp_path, "b", {"frame_source": "raw decode", "frame_cache": "memory"})
    with pytest.raises(SpaceError, match="before viz2psy 0.17.0"):
        build_features([a, b], tmp_path / "f.csv")


def test_legacy_only_passes(tmp_path):
    a, b = video_input(tmp_path, "a"), video_input(tmp_path, "b")
    out = build_features([a, b], tmp_path / "f.csv")
    assert out["meta"]["input_framing"]["stage0"] == "undeclared"


def test_stills_are_not_compared(tmp_path):
    out = build_features([still_input(tmp_path), video_input(tmp_path, "a", stage0())], tmp_path / "f.csv")
    assert out["meta"]["input_framing"]["stage0"] == "framing table"
    out = build_features([still_input(tmp_path)], tmp_path / "s.csv")
    assert out["meta"]["input_framing"] is None


# ------------------------------------------------------------ across member tables

def _table(tmp_path, name, *inputs):
    return str(build_features(list(inputs), tmp_path / f"{name}.csv")["output"])


def test_tables_same_rule_pass(tmp_path):
    t1 = _table(tmp_path, "t1", video_input(tmp_path, "a", stage0()))
    t2 = _table(tmp_path, "t2", video_input(tmp_path, "b", stage0()))
    t3 = _table(tmp_path, "t3", still_input(tmp_path))
    _assert_tables_framing([t1, t2, t3])


def test_tables_different_rule_refused(tmp_path):
    t1 = _table(tmp_path, "t1", video_input(tmp_path, "a", stage0()))
    t2 = _table(tmp_path, "t2", video_input(tmp_path, "b", stage0(sha="b" * 64)))
    with pytest.raises(SpaceError, match="across member tables"):
        _assert_tables_framing([t1, t2])


def test_tables_declared_and_legacy_refused(tmp_path):
    t1 = _table(tmp_path, "t1", video_input(tmp_path, "a", stage0()))
    t2 = _table(tmp_path, "t2", video_input(tmp_path, "b"))
    with pytest.raises(SpaceError, match="before viz2psy 0.17.0"):
        _assert_tables_framing([t1, t2])


def test_table_built_before_framing_records(tmp_path):
    t1 = _table(tmp_path, "t1", video_input(tmp_path, "a", stage0()))
    t2 = _table(tmp_path, "t2", video_input(tmp_path, "b", stage0()))
    meta = json.loads((tmp_path / "t2.meta.json").read_text())
    del meta["input_framing"]
    (tmp_path / "t2.meta.json").write_text(json.dumps(meta))
    assert group_framing(t2) == "unknown"
    with pytest.raises(SpaceError, match="rebuild"):
        _assert_tables_framing([t1, t2])
    # an older non-visual table is not a video table
    meta["inputs"] = [dict(e, extractor="aud2psy") for e in meta["inputs"]]
    (tmp_path / "t2.meta.json").write_text(json.dumps(meta))
    assert group_framing(t2) is None
    _assert_tables_framing([t1, t2])


def test_legacy_tables_only_pass(tmp_path):
    t1 = _table(tmp_path, "t1", video_input(tmp_path, "a"))
    t2 = _table(tmp_path, "t2", video_input(tmp_path, "b"))
    _assert_tables_framing([t1, t2])
