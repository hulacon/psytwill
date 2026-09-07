"""The clamped v0.1 battery: pins, two-sided checks, CLI verb."""

import json

import pytest

from psytwill.battery import (
    BATTERY,
    BATTERY_VERSION,
    EXTRACTOR_VERSIONS,
    NESTED_PREFIX_EXCEPTIONS,
    BatteryModel,
    _validate,
    check_registry,
    check_sidecar,
    models_seen,
    tie_groups,
    to_records,
)
from psytwill.cli import main
from psytwill.exceptions import BatteryError


class TestPins:
    def test_counts_by_extractor(self):
        by = {}
        for m in BATTERY.values():
            by[m.extractor] = by.get(m.extractor, 0) + 1
        assert by == {"viz2psy": 15, "aud2psy": 20, "word2psy": 13}
        assert len(BATTERY) == 48

    def test_every_learned_model_names_a_checkpoint(self):
        # §4.1: checkpoint REQUIRED for any model with learned weights. The
        # analytic set is closed and named here so a new learned model
        # cannot slip in with checkpoint None.
        analytic = {
            m.model for m in BATTERY.values() if m.checkpoint is None
        }
        assert analytic == {
            "gist", "llstat", "motion",
            "conversation", "egemaps", "loudness", "onsets", "pitch",
            "psychoacoustic", "rhythm", "spectral", "speech_rate", "timbre",
            "tonal",
            "interaction", "readability", "wordform",
        }

    def test_cross_modal_tie_groups(self):
        ties = tie_groups()
        assert ties["ViT-B-32/laion2b_s34b_b79k"] == ("clip", "clip_text")
        assert ties["encord-team/ebind-full"] == ("ebind", "ebind_audio", "ebind_text")
        assert ties["laion/larger_clap_music_and_speech"] == (
            "clap", "clap_text", "music_emotion", "sound_events"
        )
        assert len(ties) == 3

    def test_prefixes_unique_and_nesting_documented(self):
        prefixes = [p for m in BATTERY.values() for p in m.prefixes]
        assert len(prefixes) == len(set(prefixes)) == 49
        nested = {
            (c, p) for c in prefixes for p in prefixes
            if c != p and c.startswith(p + "_")
        }
        assert nested == set(NESTED_PREFIX_EXCEPTIONS)

    def test_validate_refuses_undocumented_nesting(self):
        bad = dict(BATTERY)
        bad["clip_probe"] = BatteryModel(
            "clip_probe", "viz2psy", "visual", "profile", 3, None
        )
        with pytest.raises(BatteryError, match="nests under 'clip'"):
            _validate(bad)

    def test_validate_refuses_duplicate_prefix(self):
        bad = dict(BATTERY)
        bad["places2"] = BatteryModel(
            "places2", "viz2psy", "visual", "profile", 3, None,
            extra_prefixes=("sunattr",),
        )
        with pytest.raises(BatteryError, match="owned by both"):
            _validate(bad)

    def test_records_export(self):
        rows = to_records()
        assert len(rows) == 48
        assert rows[0]["extractor"] == "viz2psy"
        assert all(r["battery_version"] == BATTERY_VERSION for r in rows)
        assert all(r["extractor_version"] == EXTRACTOR_VERSIONS[r["extractor"]] for r in rows)
        json.dumps(rows)  # plain types only


def _sidecar(model, checkpoint, extractor="viz2psy", **entry):
    return {
        "schema_version": "1.0",
        "extractor": extractor,
        "models": {model: {"checkpoint": checkpoint, **entry}},
    }


class TestCheckSidecar:
    def test_conforming_cell_is_clean(self):
        assert check_sidecar(_sidecar("clip", "ViT-B-32/laion2b_s34b_b79k")) == []
        assert check_sidecar(_sidecar("gist", None)) == []
        assert check_sidecar(_sidecar("gist", "None")) == []  # stringified null

    def test_checkpoint_bump_is_a_violation(self):
        out = check_sidecar(_sidecar("clip", "ViT-L-14/openai"), source="x.meta.json")
        assert len(out) == 1
        assert "x.meta.json: checkpoint_mismatch clip" in out[0]

    def test_unknown_model_is_a_violation(self):
        out = check_sidecar(_sidecar("videomae", "MCG-NJU/videomae-base"))
        assert out == [f"<sidecar>: unknown_model videomae (not in battery {BATTERY_VERSION})"]

    def test_extractor_and_prefix_mismatch(self):
        out = check_sidecar(
            _sidecar("places", "wideresnet18_places365", extractor="aud2psy",
                     prefixes=["places"])
        )
        kinds = sorted(line.split()[1] for line in out)
        assert kinds == ["extractor_mismatch", "prefix_mismatch"]

    def test_group_sidecar_shape(self):
        group = {
            "table": "features",
            "inputs": [
                {"path": "a/clip.csv", "extractor": "viz2psy",
                 "models": {"clip": {"checkpoint": "ViT-B-32/laion2b_s34b_b79k"}}},
                {"path": "a/ebind.csv", "extractor": "viz2psy",
                 "models": {"ebind": {"checkpoint": "encord-team/ebind-tiny"}}},
            ],
        }
        out = check_sidecar(group, source="g.meta.json")
        assert out == [
            "g.meta.json#a/ebind.csv: checkpoint_mismatch ebind found "
            "'encord-team/ebind-tiny', pinned 'encord-team/ebind-full'"
        ]
        assert models_seen([group]) == {"clip", "ebind"}


class TestCheckRegistry:
    def test_both_directions(self):
        live = {m.model for m in BATTERY.values() if m.extractor == "word2psy"}
        assert check_registry("word2psy", live) == (set(), set())
        live2 = (live - {"wordform"}) | {"pos_tags"}
        assert check_registry("word2psy", live2) == ({"pos_tags"}, {"wordform"})

    def test_unknown_extractor(self):
        with pytest.raises(BatteryError, match="unknown extractor"):
            check_registry("pliers", [])

    @pytest.mark.parametrize("pkg", ["viz2psy", "aud2psy", "word2psy"])
    def test_live_registry_matches_clamp(self, pkg):
        cli = pytest.importorskip(f"{pkg}.cli")
        assert check_registry(pkg, cli.MODEL_REGISTRY) == (set(), set()), (
            f"{pkg} registry drifted from battery {BATTERY_VERSION}; a new "
            "model enters via the admission test and a battery bump"
        )


class TestCli:
    def test_list_and_json(self, capsys):
        assert main(["battery"]) == 0
        text = capsys.readouterr().out
        assert "clip_text" in text and BATTERY_VERSION in text
        assert main(["battery", "--json"]) == 0
        rows = json.loads(capsys.readouterr().out)
        assert len(rows) == 48

    def test_check_files(self, tmp_path, capsys):
        good = tmp_path / "clip.meta.json"
        good.write_text(json.dumps(_sidecar("clip", "ViT-B-32/laion2b_s34b_b79k")))
        bad = tmp_path / "ebind.meta.json"
        bad.write_text(json.dumps(_sidecar("ebind", "encord-team/ebind-tiny")))
        assert main(["battery", "--check", str(good)]) == 0
        assert "0 violations" in capsys.readouterr().out
        assert main(["battery", "--check", str(good), str(bad)]) == 1
        err = capsys.readouterr()
        assert "checkpoint_mismatch ebind" in err.out
        assert "1 violation" in err.err

    def test_require_all_names_never_seen_models(self, tmp_path, capsys):
        good = tmp_path / "clip.meta.json"
        good.write_text(json.dumps(_sidecar("clip", "ViT-B-32/laion2b_s34b_b79k")))
        assert main(["battery", "--check", str(good), "--require-all"]) == 1
        err = capsys.readouterr().err
        assert "never emitted" in err and "diarize" in err
