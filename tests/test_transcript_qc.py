"""psytwill.transcript_qc: loop, timing and style flags on synthetic transcripts."""

import pandas as pd
import pytest

from psytwill import transcript_qc as tq


def _tx(texts, sid="ext-c-a", dur=1.0, start=0.0):
    rows = [{"stimulus_id": sid, "text": t, "onset": start + i * 2.0, "offset": start + i * 2.0 + dur}
            for i, t in enumerate(texts)]
    return pd.DataFrame(rows)


def test_loop_keeps_the_first_chunk_and_excludes_the_rest():
    texts = ["Before."] + ["Hang it up."] * tq.N_LOOP + ["After."]
    f = tq.flag_chunks(_tx(texts))
    assert f["chunk_idx"].tolist() == list(range(len(texts)))
    loop = f.iloc[1:1 + tq.N_LOOP]
    assert loop["run_len"].eq(tq.N_LOOP).all()
    assert loop["run_pos"].tolist() == list(range(tq.N_LOOP))
    assert f["loop_exclude"].tolist() == [False, False] + [True] * (tq.N_LOOP - 1) + [False]
    assert tq.exclusion_rows(f)["chunk_idx"].tolist() == list(range(2, 1 + tq.N_LOOP))


def test_a_run_below_n_loop_is_flagged_not_excluded():
    f = tq.flag_chunks(_tx(["Hey."] * (tq.N_LOOP - 1)))
    assert f["run_len"].eq(tq.N_LOOP - 1).all()
    assert not f["loop_exclude"].any()


def test_runs_match_on_normalised_text_and_never_cross_stimuli():
    a = _tx(["Go ahead.", "go ahead", "GO AHEAD!"], sid="ext-c-a")
    b = _tx(["Go ahead."] * 2, sid="ext-c-b")
    f = tq.flag_chunks(pd.concat([a, b], ignore_index=True))
    assert f.loc[f.stimulus_id == "ext-c-a", "run_len"].eq(3).all()
    assert f.loc[f.stimulus_id == "ext-c-b", "run_len"].eq(2).all()
    assert f.loc[f.stimulus_id == "ext-c-b", "chunk_idx"].tolist() == [0, 1]


def test_text_that_normalises_to_nothing_does_not_extend_a_run():
    f = tq.flag_chunks(_tx(["...", "...", "...", "...", "...", "..."]))
    assert f["run_len"].eq(1).all()
    assert not f["loop_exclude"].any()


def test_timing_flag():
    fast = _tx(["one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"],
               dur=1.0)
    short = _tx(["Stop."], dur=tq.TIMING_MIN_DURATION / 2, sid="ext-c-b")
    fine = _tx(["A normal line of speech."], dur=2.0, sid="ext-c-c")
    f = tq.flag_chunks(pd.concat([fast, short, fine], ignore_index=True))
    assert f["timing_compressed"].tolist() == [True, True, False]


def test_style_flag_and_share():
    f = tq.flag_chunks(_tx(["so i said no", "So I said no.", "and then what"]))
    assert f["lower_style"].tolist() == [True, False, True]
    assert tq.style_share(f).loc["ext-c-a"] == pytest.approx(2 / 3)


def test_missing_columns_are_named():
    with pytest.raises(ValueError, match="offset"):
        tq.flag_chunks(pd.DataFrame({"stimulus_id": ["x"], "text": ["a"], "onset": [0.0]}))


def test_attach_asr_joins_by_time_and_text_and_keeps_rows():
    f = tq.flag_chunks(_tx(["Yes.", "No."]))
    asr = pd.DataFrame({"stimulus_id": ["ext-c-a"] * 3, "onset": [0.0, 1.5, 2.0], "offset": [1.0, 1.8, 3.0],
                        "transcribe_text": [" Yes.", "", " No."], "transcribe_asr_confidence": [.9, .1, .8]})
    m = tq.attach_asr(f, asr, text_column="transcribe_text", columns=("transcribe_asr_confidence",))
    assert len(m) == 2
    assert m["transcribe_asr_confidence"].tolist() == [.9, .8]


def test_attach_asr_refuses_an_ambiguous_decoder_table():
    f = tq.flag_chunks(_tx(["Yes."]))
    asr = pd.DataFrame({"stimulus_id": ["ext-c-a"] * 2, "onset": [0.0, 0.0], "offset": [1.0, 1.0],
                        "text": ["Yes.", "Yes."]})
    with pytest.raises(ValueError, match="sharing"):
        tq.attach_asr(f, asr)
