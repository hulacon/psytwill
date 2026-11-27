"""The pass manifest (psytwill.passmanifest) and its gate in scripts/fitcorpus/extract.py."""

import hashlib
import types

import pytest

from psytwill.exceptions import InputError
from psytwill.passmanifest import load_pass
from tests.test_extract_script import _per_item_corpus, _run_args, extract

HEAD = 'pass_schema_version = "1.0"\nname = "sp 0.3.0"\n'


def _write(tmp_path, body, head=HEAD, producers='[producers]\naud2psy = "9.9.9"\nviz2psy = "1.0.0"\n'):
    p = tmp_path / "pass.toml"
    p.write_text(head + producers + body)
    return p


def _cell(corpus="fma", source="audio", models='["*"]', stage="pass", added="2026-11-27", extra=""):
    return (f'\n[[cell]]\ncorpus = "{corpus}"\nsource = "{source}"\nmodels = {models}\nstage = "{stage}"\n'
            f'decided = "log 2026-11-26 class 2d"\nadded = "{added}"\n{extra}')


def test_a_manifest_loads_and_finds_cells(tmp_path):
    pm = load_pass(_write(tmp_path, _cell() + _cell(source="transcript", models='["minilm"]', extra='version = "0.1.0"\n')))
    assert pm.name == "sp 0.3.0" and len(pm.cells) == 2 and len(pm.sha256) == 64
    star = pm.cell_for("fma", "audio", "clap")
    assert star.key == "fma/audio" and pm.version_for(star, "aud2psy") == "9.9.9"
    named = pm.cell_for("fma", "transcript", "minilm")
    assert pm.version_for(named, "word2psy") == "0.1.0"  # the cell's own pin wins
    assert pm.cell_for("fma", "transcript", "clip_text") is None
    assert pm.cell_for("gigaspeech", "audio", "clap") is None
    assert pm.record()["n_cells"] == 2 and pm.record()["deviations"] == []


@pytest.mark.parametrize("body, match", [
    (_cell(stage="extra"), "stage"),
    (_cell(models='"clap"'), "list of model names"),
    (_cell(models='["*", "clap"]'), "stands alone"),
    (_cell(added="27 Nov"), "YYYY-MM-DD"),
    (_cell(stage="screen"), "pinned subset"),
    (_cell(extra='units = "ids.txt"\n'), "units_sha256"),
    (_cell() + _cell(), "already declared"),
    (_cell(extra='colour = "red"\n'), "unknown field"),
    ('\n[[cell]]\ncorpus = "fma"\n', "needs"),
])
def test_bad_cells_are_refused(tmp_path, body, match):
    with pytest.raises(InputError, match=match):
        load_pass(_write(tmp_path, body))


def test_header_is_checked(tmp_path):
    with pytest.raises(InputError, match="does not exist"):
        load_pass(tmp_path / "none.toml")
    with pytest.raises(InputError, match="pass_schema_version"):
        load_pass(_write(tmp_path, "", head='name = "x"\n'))
    with pytest.raises(InputError, match="needs a `name`"):
        load_pass(_write(tmp_path, "", head='pass_schema_version = "1.0"\n'))
    with pytest.raises(InputError, match="unknown producer"):
        load_pass(_write(tmp_path, "", producers='[producers]\npsytwill = "1"\n'))


def test_a_cell_added_after_the_freeze_needs_a_deviation(tmp_path):
    head = HEAD + 'frozen = "2026-12-01"\n'
    load_pass(_write(tmp_path, _cell(added="2026-12-01"), head=head))  # on the day is inside
    with pytest.raises(InputError, match="carries no deviation"):
        load_pass(_write(tmp_path, _cell(added="2026-12-02"), head=head))
    pm = load_pass(_write(tmp_path, _cell(added="2026-12-02", extra='deviation = "2026-12-02: a gap read"\n'),
                          head=head))
    assert pm.record()["deviations"] == [{"cell": "fma/audio", "models": ["*"], "deviation": "2026-12-02: a gap read"}]


def test_a_pinned_subset_is_checked_against_its_sha(tmp_path):
    ids = tmp_path / "screen_ids.txt"
    ids.write_text("x-0\nx-2\n")
    sha = hashlib.sha256(ids.read_bytes()).hexdigest()
    pm = load_pass(_write(tmp_path, _cell(stage="screen", extra=f'units = "screen_ids.txt"\nunits_sha256 = "{sha}"\n')))
    assert pm.cells[0].unit_ids() == {"x-0", "x-2"}
    ids.write_text("x-0\nx-1\n")
    with pytest.raises(InputError, match="changed since it was pinned"):
        pm.cells[0].unit_ids()


# --- the gate in extract.py --------------------------------------------------------


def _gate(tmp_path, monkeypatch, body, installed="9.9.9"):
    durable = _per_item_corpus(tmp_path, monkeypatch)
    pm = load_pass(_write(tmp_path, body))
    roots = extract.ROOTS
    monkeypatch.setattr(extract, "ROOTS", types.SimpleNamespace(**{**vars(roots), "pass_": lambda: pm}),
                        raising=False)
    monkeypatch.setattr(extract, "installed_version", lambda package: installed)
    ran = []
    monkeypatch.setattr(extract, "_run_batches",
                        lambda src, model, units, args: (ran.append((model, [u.id for u in units])) or (len(units), 0, [])))
    return durable, ran


def test_run_refuses_cells_outside_the_pass(tmp_path, monkeypatch, capsys):
    _, ran = _gate(tmp_path, monkeypatch, _cell(models='["diarize", "egemaps"]'))
    assert extract.cmd_run(_run_args(model=None)) == 1
    out = capsys.readouterr().out
    assert {m for m, _ in ran} == {"diarize", "egemaps"}
    assert "REFUSED fma/audio/clap[x-0]: not in pass sp 0.3.0" in out
    assert "refused by the pass" in out


def test_run_refuses_a_producer_off_its_pin(tmp_path, monkeypatch, capsys):
    _, ran = _gate(tmp_path, monkeypatch, _cell(), installed="9.9.8")
    assert extract.cmd_run(_run_args(model="diarize")) == 1
    assert ran == []
    assert "runs aud2psy 9.9.9; this interpreter has 9.9.8" in capsys.readouterr().out


def test_run_keeps_a_screen_to_its_pinned_units(tmp_path, monkeypatch, capsys):
    ids = tmp_path / "ids.txt"
    ids.write_text("x-1\n")
    sha = hashlib.sha256(ids.read_bytes()).hexdigest()
    _, ran = _gate(tmp_path, monkeypatch, _cell(stage="screen", extra=f'units = "ids.txt"\nunits_sha256 = "{sha}"\n'))
    assert extract.cmd_run(_run_args(model="diarize")) == 1  # x-0 and x-2 refused
    assert ran == [("diarize", ["x-1"])]
    assert "outside the pinned subset" in capsys.readouterr().out


def test_run_without_a_pass_says_so(tmp_path, monkeypatch, capsys):
    _per_item_corpus(tmp_path, monkeypatch)
    monkeypatch.setattr(extract, "_run_batches", lambda src, model, units, args: (len(units), 0, []))
    assert extract.cmd_run(_run_args(model="diarize")) == 0
    assert "no pass manifest" in capsys.readouterr().out


def test_plan_reports_cells_with_no_driver_source(tmp_path, monkeypatch, capsys):
    _gate(tmp_path, monkeypatch, _cell(models='["diarize"]') + _cell(corpus="coco", source="image")
          + _cell(source="audio2", models='["clap"]'))
    monkeypatch.setattr(extract, "is_done", lambda *a: False)
    args = _run_args(corpus=None, source=None, with_attached=False)
    assert extract.cmd_plan(args) == 0
    out = capsys.readouterr().out
    assert "coco/image" in out and "NO DRIVER SOURCE" in out
    assert "not in pass (carried if done; new extraction refused): fma/audio" in out and "clap" in out
