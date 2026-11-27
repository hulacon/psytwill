"""The typed corpus registry: declarations, the consumer usage file, the warnings a release records."""

from dataclasses import replace
import json

import pytest

from psytwill.cli import main
from psytwill.exceptions import CorpusError
from psytwill.fitcorpus import CORPORA, CorpusSpec, MemberOverlap, Role, _validate_registry
from psytwill.registry import (
    block_corpora,
    corpus_table,
    load_usage,
    record_registry,
    registry_warnings,
    release_uses,
)
from psytwill.release import build_release, write_release
from tests.conftest import write_usage


def _spec(key="t", **kw):
    return CorpusSpec(key, tier="a", serves="test", **{"licence": "CC0", "media": "fetch", **kw})


# --- declarations ----------------------------------------------------------------


def test_the_shipped_registry_is_valid_and_every_live_corpus_declares_a_role():
    _validate_registry(CORPORA)
    for key, spec in CORPORA.items():
        assert spec.dropped or spec.roles, f"{key} is live but declares no role"


@pytest.mark.parametrize("spec, match", [
    (_spec(roles=(Role("tune", "V"),)), "role kind"),
    (_spec(roles=(Role("fit", "X"),)), "needs one of"),
    (_spec(roles=(Role("member-training"),)), "names its target"),
    (_spec(roles=(Role("member-training", "V", target="m"),)), "names its target"),
    (_spec(roles=(Role("fit", "V"), Role("fit", "V"))), "declared twice"),
    (_spec(training_overlap=(MemberOverlap("clip", "maybe"),)), "is not one of"),
    (_spec(training_overlap=(MemberOverlap("clip", "present"),)), "names the split"),
    (_spec(training_overlap=(MemberOverlap("clip", "none", "dev"),)), "names the split"),
    (_spec(training_overlap=(MemberOverlap("clip", "none"), MemberOverlap("clip", "none"))), "declared twice"),
    (_spec(media="ftp"), "media"),
    (_spec(dropped="gone", roles=(Role("fit", "V"),)), "dropped corpus declares no roles"),
])
def test_bad_declarations_are_refused_at_import(spec, match):
    with pytest.raises(CorpusError, match=match):
        _validate_registry({spec.key: spec})


def test_member_overlap_falls_back_to_star_then_unchecked():
    s = _spec(training_overlap=(MemberOverlap("clip", "none"), MemberOverlap("*", "undisclosed")))
    assert s.overlap("clip").value == "none"
    assert s.overlap("dinov2").value == "undisclosed"
    assert _spec().overlap("clip").value == "unchecked"


# --- usage file ----------------------------------------------------------------


def test_usage_file_round_trips(tmp_path):
    p = write_usage(tmp_path, '[corpus.nsd]\nconsumer_overlap = [{grain = "recording", detail = "x"}]\n'
                              'read_by = [{split = "test", read = "a gate", release = "0.1.0"}]\n')
    u = load_usage(p)
    assert u["nsd"]["consumer_overlap"] == [{"grain": "recording", "detail": "x"}]
    assert u["nsd"]["read_by"] == [{"split": "test", "read": "a gate", "release": "0.1.0"}]
    t = corpus_table(usage=u)
    assert t["nsd"]["consumer_overlap"][0]["grain"] == "recording"
    assert t["librispeech"]["consumer_overlap"] is None  # not in the file = unchecked


def test_a_missing_usage_file_is_a_loud_error_naming_its_path(tmp_path):
    with pytest.raises(CorpusError, match=str(tmp_path / "nope.toml")):
        load_usage(tmp_path / "nope.toml")


@pytest.mark.parametrize("body, match", [
    ('[corpus.imagenet]\nread_by = []\n', "not registered"),
    ('[corpus.nsd]\nconsumer_overlap = [{grain = "pixel", detail = "x"}]\n', "grain"),
    ('[corpus.nsd]\nconsumer_overlap = [{grain = "item"}]\n', "names its `detail`"),
    ('[corpus.nsd]\nread_by = [{split = "test"}]\n', "names its `split` and its `read`"),
    ('[corpus.nsd]\nnotes = "x"\n', "unknown field"),
    ('[other]\n', "unknown top-level"),
])
def test_bad_usage_entries_are_refused(tmp_path, body, match):
    with pytest.raises(CorpusError, match=match):
        load_usage(write_usage(tmp_path, body))


def test_usage_schema_version_is_checked(tmp_path):
    p = tmp_path / "u.toml"
    p.write_text('usage_schema_version = "0.9"\n')
    with pytest.raises(CorpusError, match="usage_schema_version"):
        load_usage(p)


# --- uses a release's manifests show ---------------------------------------------


def test_block_corpora_from_row_counts_or_inputs():
    assert block_corpora({"criterion": {"corpus_rows": {"fma": 1, "friends": 2}}}) == ["fma", "friends"]
    assert block_corpora({"corpus_weights": {"corpus_rows": {"nsd": 3}}}) == ["nsd"]
    assert block_corpora({"inputs": ["/data/fit-corpora/nsd/groups/x.parquet"]}) == ["nsd"]
    assert block_corpora({"inputs": ["/data/unregistered/groups/x.parquet"]}) == []


def test_release_uses_reads_blocks_and_relation_registers():
    uses = release_uses({"V": {"inputs": ["/r/nsd/groups/v.parquet"]}},
                        {"VL": {"per_register": {"nsd": {"fit": True}, "avcaps": {"fit": False}}}})
    assert {(u["corpus"], u["product"], u["kind"]) for u in uses} == {
        ("nsd", "V", "fit"), ("nsd", "VL", "fit"), ("avcaps", "VL", "register")}


# --- the conflict matrix ----------------------------------------------------------


def _table(*specs, usage=None):
    return corpus_table({s.key: s for s in specs}, usage)


def _codes(ws):
    return sorted({w.split(":")[0] for w in ws})


def _warn(table, uses, members=None, sides=None, **kw):
    uses = [{"corpus": c, "product": p, "kind": k} for c, p, k in uses]
    return registry_warnings(table, uses, block_members=members or {"V": ["m"]}, relation_sides=sides or {}, **kw)


def test_a_clean_table_warns_nothing():
    t = _table(_spec(roles=(Role("fit", "V"),)), usage={"t": {"consumer_overlap": [{"grain": "none"}]}})
    assert _warn(t, [("t", "V", "fit")]) == []


def test_undeclared_unregistered_dropped_and_declared_not_used():
    t = _table(_spec(roles=(Role("fit", "V"), Role("register", "V"))),
               _spec("gone", dropped="never fetched"), _spec("idle", roles=(Role("fit", "A"),)),
               usage={"t": {"consumer_overlap": [{"grain": "none"}]},
                      "gone": {"consumer_overlap": [{"grain": "none"}]}})
    ws = _warn(t, [("t", "V", "fit"), ("t", "A", "fit"), ("x", "V", "fit"), ("gone", "V", "fit")],
               members={"V": ["m"], "A": ["n"]})
    assert any(w.startswith("undeclared-use: t is used as A fit") for w in ws)
    assert any(w.startswith("unregistered: V fit rows come from 'x'") for w in ws)
    assert any(w.startswith("dropped-used: gone") for w in ws)
    assert any(w.startswith("declared-not-used: t declares V register") for w in ws)
    assert not any("idle" in w and "declared-not-used" in w and "A" not in w for w in ws)


def test_a_role_for_a_product_outside_the_release_is_not_declared_not_used():
    t = _table(_spec(roles=(Role("fit", "A"),)))
    assert _warn(t, [], members={"V": ["m"]}) == []


def test_unrecorded_licence_and_media():
    t = _table(CorpusSpec("t", tier="a", serves="x", roles=(Role("dev", "V"),)))
    assert _codes(_warn(t, [])) == ["unrecorded"]


def test_dev_and_bar_on_meeting_splits():
    t = _table(_spec(roles=(Role("dev", "L", "dev"), Role("bar", "L", "test"))))
    assert _warn(t, []) == []
    t = _table(_spec(roles=(Role("dev", "L"), Role("bar", "L", "test"))))
    assert _codes(_warn(t, [])) == ["dev-and-bar"]


def test_same_rows_for_fit_and_gate():
    t = _table(_spec(roles=(Role("fit", "VL", "train"), Role("gate", "VL", "train"))))
    assert "same-rows" in _codes(_warn(t, [], sides={"VL": ("V", "L")}))


def test_a_bar_already_read_warns():
    t = _table(_spec(roles=(Role("bar", "ALs", "dev-clean"),)),
               usage={"t": {"read_by": [{"split": "dev-clean", "read": "a probe", "release": "0.2.1"}]}})
    ws = _warn(t, [])
    assert ws == ["bar-read: t ALs bar (dev-clean) is not unread: a probe (dev-clean, 0.2.1)"]
    t = _table(_spec(roles=(Role("bar", "ALs", "dev-clean"),)),
               usage={"t": {"read_by": [{"split": "test-clean", "read": "a gate"}]}})
    assert _warn(t, []) == []


def test_member_training_corpus_used_as_that_members_bar():
    t = _table(_spec(roles=(Role("member-training", target="probe"), Role("bar", "V"))))
    assert _codes(_warn(t, [], members={"V": ["probe"]})) == ["member-training"]
    assert _warn(t, [], members={"V": ["other"]}) == []


def test_consumer_overlap_unchecked_and_recording():
    t = _table(_spec(roles=(Role("fit", "V"), Role("fit", "L"))))
    ws = _warn(t, [("t", "V", "fit"), ("t", "L", "fit")], members={"V": ["m"], "L": ["n"]})
    assert ws == ["consumer-overlap: t fits L, V, and its overlap with the consumer's stimuli is unchecked"]
    t = _table(_spec(roles=(Role("fit", "V"),)),
               usage={"t": {"consumer_overlap": [{"grain": "recording", "detail": "the test images"}]}})
    assert "the test images" in _warn(t, [("t", "V", "fit")])[0]
    t = _table(_spec(roles=(Role("fit", "V"),)),
               usage={"t": {"consumer_overlap": [{"grain": "item", "detail": "same words"}]}})
    assert _warn(t, [("t", "V", "fit")]) == []


def _rel_table(gate_overlap=(), fit_overlap=(), gates=("g",)):
    none = {"consumer_overlap": [{"grain": "none"}]}
    specs = [_spec("f", roles=(Role("fit", "VL"),), training_overlap=fit_overlap)]
    specs += [_spec(g, roles=(Role("gate", "VL", "held"),), training_overlap=gate_overlap) for g in gates]
    return _table(*specs, usage={s.key: none for s in specs})


def _rel_warn(t, analytic=()):
    return _warn(t, [("f", "VL", "fit")], members={"V": ["clip", "gist"], "L": ["minilm"]},
                 sides={"VL": ("V", "L")}, analytic_members=analytic)


def test_relation_gate_count():
    assert any(w.startswith("no-gate: relation VL") for w in _rel_warn(_rel_table(gates=()), ["gist"]))
    assert any(w.startswith("gates: relation VL declares 2") for w in _rel_warn(_rel_table(gates=("g", "h"))))


def test_training_overlap_unchecked_skips_analytic_members():
    star = (MemberOverlap("*", "undisclosed"),)
    ws = _rel_warn(_rel_table(gate_overlap=star, fit_overlap=(MemberOverlap("clip", "none"),)), analytic=["gist"])
    assert ws == ["training-overlap: relation VL fit on f: 1 side member(s) unchecked (minilm)"]
    assert _rel_warn(_rel_table(gate_overlap=star, fit_overlap=star), analytic=["gist"]) == []


def test_training_overlap_present_on_the_gate_split():
    t = _rel_table(gate_overlap=(MemberOverlap("clip", "present", "held"), MemberOverlap("*", "none")),
                   fit_overlap=(MemberOverlap("*", "none"),))
    assert _rel_warn(t) == ["training-overlap: relation VL gate g (held) is in the training data of clip"]
    t = _rel_table(gate_overlap=(MemberOverlap("clip", "present", "train"), MemberOverlap("*", "none")),
                   fit_overlap=(MemberOverlap("*", "none"),))
    assert _rel_warn(t) == []


def test_without_a_usage_file_the_release_says_so():
    ws = registry_warnings(_table(), [], block_members={}, relation_sides={}, usage_recorded=False)
    assert _codes(ws) == ["no-usage-file"]


# --- a release records it ----------------------------------------------------------


def _block(tmp_path, block="V", members=("clip",), corpus="nsd"):
    (tmp_path / f"{block}.npz").write_bytes(b"w")
    m = tmp_path / f"{block}.json"
    m.write_text(json.dumps({"block": block, "k": 2, "members": list(members), "weights": f"{block}.npz",
                             "inputs": [str(tmp_path / corpus / "groups" / "x.parquet")]}))
    return m


def test_release_records_the_registry_and_requires_usage_under_its_data_root(tmp_path):
    m = _block(tmp_path)
    root = tmp_path / "fc"
    root.mkdir()
    with pytest.raises(CorpusError, match=str(root / "registry" / "usage.toml")):
        build_release(name="sp", version="0.3.0", blocks=[m], data_root=root)
    write_usage(root, '[corpus.nsd]\nconsumer_overlap = [{grain = "none"}]\n')
    rel = build_release(name="sp", version="0.3.0", blocks=[m], data_root=root)
    reg = rel["registry"]
    assert rel["release_schema_version"] == "1.6"
    assert reg["usage_file"] == str((root / "registry" / "usage.toml").resolve()) and len(reg["usage_sha256"]) == 64
    assert reg["uses"] == [{"corpus": "nsd", "product": "V", "kind": "fit"}]
    assert set(reg["corpora"]) == set(CORPORA)
    # nsd declares L and VL fits, and neither product is in this release
    assert not any(w.startswith("declared-not-used: nsd") for w in reg["warnings"])


def test_release_without_a_data_root_records_no_usage(tmp_path):
    rel = build_release(name="sp", version="0.3.0", blocks=[_block(tmp_path)])
    assert rel["registry"]["usage_file"] is None
    assert rel["registry"]["warnings"][0].startswith("no-usage-file")


def test_cli_prints_warnings_at_write_and_verify(tmp_path, capsys):
    m = _block(tmp_path, members=("clip", "gist"))
    usage = write_usage(tmp_path / "elsewhere")
    out = tmp_path / "rel"
    assert main(["space", "release", "write", "--name", "sp", "--version", "0.3.0", "--blocks", str(m),
                 "--registry-usage", str(usage), "-o", str(out)]) == 0
    written = capsys.readouterr().out
    assert "registry:" in written and "WARN consumer-overlap: nsd fits V" in written
    assert main(["space", "release", "verify", str(out / "sp_0.3.0.json")]) == 0
    verified = capsys.readouterr().out
    assert "every pinned file matches" in verified
    assert [l for l in verified.splitlines() if "WARN" in l] == [l for l in written.splitlines() if "WARN" in l]


def test_verify_says_when_a_release_predates_the_registry(tmp_path, capsys):
    rel = build_release(name="sp", version="0.2.9", blocks=[_block(tmp_path)])
    rel.pop("registry")
    rel["release_schema_version"] = "1.5"
    path = write_release(rel, tmp_path / "rel")
    assert main(["space", "release", "verify", str(path)]) == 0
    assert "none recorded (release schema < 1.6)" in capsys.readouterr().out


def test_record_registry_pins_the_usage_file(tmp_path):
    usage = write_usage(tmp_path)
    a = record_registry({}, {}, relation_sides={}, usage=usage)
    usage.write_text(usage.read_text() + '\n[corpus.nsd]\nconsumer_overlap = [{grain = "none"}]\n')
    b = record_registry({}, {}, relation_sides={}, usage=usage)
    assert a["usage_sha256"] != b["usage_sha256"]


def test_replace_keeps_specs_frozen():
    s = replace(_spec(), licence="CC BY")
    with pytest.raises(Exception):
        s.licence = "x"  # type: ignore[misc]
