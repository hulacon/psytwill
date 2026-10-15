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
    monkeypatch.setattr(extract, "is_done", lambda stem: (stem.parent.name, stem.stem) in done)
    monkeypatch.setattr(extract._dir_stems, "cache", {}, raising=False)
    args = argparse.Namespace(model=None, skip_model=None, unit=None, unit_index=None, corpus=None, source=None,
                              allow_partial=False, dry_run=True, redo=False, fail_fast=False)
    assert extract.cmd_aggregate(args) == 1
    assert "INCOMPLETE c/image/main" in capsys.readouterr().out
