"""psytwill.carry: slicing packed tables onto per-recording units (rule C, tables only)."""

import pytest

from psytwill import carry as cr
from psytwill.battery import check_sidecar

OLD, NEW = "ext-fma-x-000", "ext-fma-x-b"
PIECE = cr.Piece(OLD, NEW, shift=3.0, duration=2.5)  # old [3.0, 5.5)


def _sc(model, extractor="aud2psy", checkpoint=None, **extra):
    return {"extractor": extractor, "models": {model: {"checkpoint": checkpoint, **extra}}}


class TestVerdict:
    @pytest.mark.parametrize("sidecar, carried, why", [
        (_sc("minilm", "word2psy"), True, "text rows"),
        (_sc("diarize"), True, "producer"),
        (_sc("transcribe"), True, "producer"),
        # a grid label is no promise of locality (lattice phase, recurrent VAD, file-level dB floor)
        (_sc("loudness", labels={"default": {"window_sec": "grid"}}), False, "re-extract per recording"),
        (_sc("conversation", derived_from="diarize"), False, "re-extract per recording"),
        (_sc("clap", labels={"default": {"window_sec": 10.0}}), False, "re-extract per recording"),
        (_sc("clip", "viz2psy"), False, "not an audio-unit family"),
    ])
    def test_only_tables_carry(self, sidecar, carried, why):
        ok, reason = cr.verdict(sidecar)
        assert ok is carried and why in reason

    def test_a_multi_model_sidecar_is_not_carried(self):
        sc = {"extractor": "aud2psy", "models": {"diarize": {}, "transcribe": {}}}
        assert cr.verdict(sc) == (False, "2 models in one sidecar")


class TestSliceTurns:
    def test_overlapping_turns_are_clipped_renumbered_and_re_timed(self):
        turns = [{"stimulus_id": OLD, "turn_idx": str(i), "speaker": s, "onset": on, "offset": off}
                 for i, (s, on, off) in enumerate([("S0", "0.1", "2.0"), ("S1", "2.8", "3.4"),
                                                   ("S1", "3.6", "4.9"), ("S0", "5.2", "6.0"),
                                                   ("S0", "5.5", "6.0")])]
        rows, clipped = cr.slice_turns(turns, PIECE)
        assert [(r["turn_idx"], r["speaker"], r["onset"], r["offset"]) for r in rows] == [
            ("0", "S1", "0.0", "0.4"), ("1", "S1", "0.6", "1.9"), ("2", "S0", "2.2", "2.5")]
        assert clipped == 2 and {r["stimulus_id"] for r in rows} == {NEW}


class TestSliceTranscribe:
    def test_chunks_go_by_onset_and_take_their_words(self):
        chunks = [{"stimulus_id": OLD, "chunk_idx": str(i), "onset": on, "offset": off, "transcribe_text": t}
                  for i, (on, off, t) in enumerate([("0.1", "2.0", "a"), ("3.0", "4.0", "b"),
                                                    ("4.5", "5.8", "c"), ("6.0", "7.0", "d")])]
        words = [{"stimulus_id": OLD, "chunk_idx": c, "word_idx": w, "word": x, "onset": on, "offset": off}
                 for c, w, x, on, off in [("0", "0", "a", "0.1", "2.0"), ("1", "0", "b", "3.0", "3.5"),
                                          ("1", "1", "bb", "3.5", "4.0"), ("2", "0", "c", "4.5", "5.8")]]
        c, w, over = cr.slice_transcribe(chunks, words, PIECE)
        assert [(r["chunk_idx"], r["onset"], r["offset"], r["transcribe_text"]) for r in c] == [
            ("0", "0.0", "1.0", "b"), ("1", "1.5", "2.8", "c")]
        assert [(r["chunk_idx"], r["word_idx"], r["word"]) for r in w] == [
            ("0", "0", "b"), ("0", "1", "bb"), ("1", "0", "c")]
        assert over == 1  # chunk c runs past 5.5


class TestSliceText:
    def _tables(self):
        chunks = [{"chunk_idx": str(i), "chunk_label": OLD, "n_words": "2", "stimulus_id": OLD,
                   "onset": on, "offset": off, "minilm_000": f"e{i}"}
                  for i, (on, off) in enumerate([("0", "2.2"), ("3", "4.1"), ("4.5", "5.2")])]
        words = [{"stimulus_id": OLD, "word_idx": str(i), "word": f"w{i}", "sentence_idx": str(i // 2),
                  "chunk_idx": str(i // 2), "chunk_label": OLD, "onset": "", "offset": ""} for i in range(6)]
        return chunks, words

    def test_indices_restart_and_ids_follow(self):
        c, w = cr.slice_text(*self._tables(), PIECE)
        assert [(r["chunk_idx"], r["onset"], r["minilm_000"], r["chunk_label"]) for r in c] == [
            ("0", "0.0", "e1", NEW), ("1", "1.5", "e2", NEW)]
        assert [(r["word_idx"], r["sentence_idx"], r["chunk_idx"], r["word"]) for r in w] == [
            ("0", "0", "0", "w2"), ("1", "0", "0", "w3"), ("2", "1", "1", "w4"), ("3", "1", "1", "w5")]
        assert {r["onset"] for r in w} == {""}

    def test_a_sentence_across_chunks_refuses(self):
        chunks, words = self._tables()
        words[2]["sentence_idx"] = "0"
        with pytest.raises(ValueError, match="spans chunks"):
            cr.slice_text(chunks, words, PIECE)

    def test_a_chunk_without_an_onset_refuses(self):
        chunks, words = self._tables()
        chunks[0]["onset"] = ""
        with pytest.raises(ValueError, match="no onset"):
            cr.slice_text(chunks, words, PIECE)


class TestCarriedSidecar:
    OLD_DIR, NEW_DIR = "/old/fma/features/x-000", "/new/fma/features/x-b"

    def _diarize(self):
        return {"schema_version": "1.4", "extractor": "aud2psy",
                "input": {"type": "audio", "path": "/s/units/x-000.wav", "duration_sec": 12.0},
                "diarization": {"n_speakers": 2, "n_turns": 9, "speaker_time_sec": {"S0": 4.0, "S1": 3.0}},
                "output": {"speakers": {"path": f"{self.OLD_DIR}/diarize_speakers.csv", "rows": 9}},
                "models": {"diarize": {"checkpoint": "pyannote/speaker-diarization-community-1"}}}

    def _turns(self):
        return [{"stimulus_id": NEW, "speaker": "S1", "onset": "0.0", "offset": "0.4"},
                {"stimulus_id": NEW, "speaker": "S1", "onset": "1.0", "offset": "1.5"}]

    def _carry(self, sc, tables, **kw):
        return cr.carried_sidecar(sc, piece=PIECE, paths={self.OLD_DIR: self.NEW_DIR,
                                                          "/s/units/x-000.wav": "/s/recordings/x-b.wav"},
                                  tables=tables, old_unit="x-000", old_root="/old/fma", old_sha256="ab",
                                  unchanged=False, **kw)

    def test_paths_counts_and_provenance(self):
        sc = self._diarize()
        out = self._carry(sc, {"speakers": self._turns()}, notes={"clipped_turns": 1})
        assert out["input"] == {"type": "audio", "path": "/s/recordings/x-b.wav", "duration_sec": 2.5}
        assert out["output"]["speakers"] == {"path": f"{self.NEW_DIR}/diarize_speakers.csv", "rows": 2}
        assert out["carried_from"] == {"rule": cr.CARRY_RULE, "unit": "x-000", "stimulus_id": OLD,
                                       "root": "/old/fma", "shift_sec": 3.0, "sidecar_sha256": "ab",
                                       "unchanged": False, "clipped_turns": 1}
        assert sc["diarization"]["n_turns"] == 9  # the input is not modified
        assert check_sidecar(out) == []

    def test_diarization_and_transcription_summaries(self):
        sc = {"extractor": "aud2psy", "input": {"path": "/s/units/x-000.wav", "duration_sec": 12.0},
              "diarization": {"n_speakers": 4, "n_turns": 30, "speaker_time_sec": {"S9": 1.0}},
              "transcription": {"n_speech_segments": 7, "speech_timing": {"n_words": 80}},
              "output": {"speakers": {"path": f"{self.OLD_DIR}/diarize_speakers.csv", "rows": 30}},
              "models": {"diarize": {"checkpoint": "pyannote/speaker-diarization-community-1"}}}
        turns = [{"speaker": "S1", "onset": "0.0", "offset": "0.4"}, {"speaker": "S0", "onset": "1.0", "offset": "1.5"},
                 {"speaker": "S1", "onset": "2.0", "offset": "2.1"}]
        out = self._carry(sc, {"speakers": turns})
        assert out["diarization"] == {"n_speakers": 2, "n_turns": 3, "speaker_time_sec": {"S0": 0.5, "S1": 0.5}}
        assert out["output"]["speakers"]["rows"] == 3
        assert out["carried_from"]["packed_summaries"] == ["transcription.n_speech_segments",
                                                           "transcription.speech_timing"]

    def test_text_input_counts(self):
        sc = {"extractor": "word2psy", "input": {"type": "text", "n_words": 6, "n_chunks": 3, "n_sentences": 3,
                                                 "path": "/old/fma/inputs/x-000_transcript.csv"},
              "output": {"words": {"path": f"{self.OLD_DIR}/transcript_minilm_words.csv", "rows": 6},
                         "chunks": {"path": f"{self.OLD_DIR}/transcript_minilm_chunks.csv", "rows": 3}},
              "models": {"minilm": {"checkpoint": "sentence-transformers/all-MiniLM-L6-v2"}}}
        c, w = cr.slice_text(*TestSliceText()._tables(), PIECE)
        out = cr.carried_sidecar(sc, piece=PIECE, paths={self.OLD_DIR: self.NEW_DIR,
                                                         "/old/fma/inputs/x-000_transcript.csv": "/n/x-b_transcript.csv"},
                                 tables={"words": w, "chunks": c}, old_unit="x-000", old_root="/old/fma",
                                 old_sha256="ab", unchanged=False)
        assert out["input"] == {"type": "text", "n_words": 4, "n_chunks": 2, "n_sentences": 2,
                                "path": "/n/x-b_transcript.csv"}
        assert out["output"]["chunks"]["rows"] == 2

    def test_an_unchanged_unit_keeps_its_counts(self):
        out = cr.carried_sidecar(self._diarize(), piece=cr.Piece(OLD, OLD, 0.0, 12.0), paths={},
                                 tables={}, old_unit="x-000", old_root="/old/fma", old_sha256="ab", unchanged=True)
        assert out["diarization"]["n_turns"] == 9 and out["carried_from"]["unchanged"] is True


def test_slice_family_dispatches_and_refuses_unknown_tables():
    turns = [{"stimulus_id": OLD, "turn_idx": "0", "speaker": "S0", "onset": "2.0", "offset": "4.0"}]
    tables, notes = cr.slice_family({"extractor": "aud2psy"}, {"speakers": turns}, PIECE)
    assert len(tables["speakers"]) == 1 and notes == {"clipped_turns": 1}
    with pytest.raises(ValueError, match="no carry rule"):  # frame tables are re-extracted, never sliced
        cr.slice_family({"extractor": "aud2psy"}, {"frames": []}, PIECE)


def test_fmt_time():
    assert cr.fmt_time(5.25 - 3.0) == "2.25"
    assert cr.fmt_time(402.219 - 400.0) == "2.219"
    assert cr.fmt_time(3.0 - 3.0) == "0.0" and cr.fmt_time(-0.0) == "0.0"
