"""Space releases: a pinned, immutable set of blocks, relations and absences.

What these pin: a relation can only be released beside the blocks it was
fitted on; a measured absence names real blocks and no relation; a release
file is never overwritten; and a release whose files changed after it was
written is refused on load.
"""

import json

import numpy as np
import pandas as pd
import pytest

from psytwill.cli import main
from psytwill.exceptions import SpaceError
from psytwill.release import build_release, load_release, parse_absent, write_release
from psytwill.relate import read_scores, relate_tables, save_relation


def _block(tmp_path, block, X, ids, tag="v1"):
    """A stand-in block fit (manifest + weights) and a score table projected through it."""
    np.savez(tmp_path / f"{block}_{tag}.npz", w=np.arange(3.0))
    manifest = tmp_path / f"{block}_{tag}.json"
    manifest.write_text(json.dumps({"block": block, "k": X.shape[1], "members": [f"{block.lower()}1"],
                                    "weights": f"{block}_{tag}.npz", "psytwill_version": "test"}))
    df = pd.DataFrame(X, columns=[f"{block}_{j:03d}" for j in range(X.shape[1])])
    df.insert(0, "stimulus_id", ids)
    scores = tmp_path / f"{block}_{tag}_scores.parquet"
    df.to_parquet(scores, index=False)
    (tmp_path / f"{block}_{tag}_scores.meta.json").write_text(json.dumps(
        {"space": str(manifest), "block": block, "k": X.shape[1], "key": "stimulus_id"}))
    return manifest, scores


@pytest.fixture
def fitted(tmp_path):
    rng = np.random.RandomState(0)
    z = rng.randn(300, 2)
    ids = [f"ext-t-{i:03d}" for i in range(300)]
    P = np.hstack([z, rng.randn(300, 6)]) @ rng.randn(8, 8)
    Q = np.hstack([z, rng.randn(300, 5)]) @ rng.randn(7, 7)
    mP, sP = _block(tmp_path, "P", P, ids)
    mQ, sQ = _block(tmp_path, "Q", Q, ids)
    mR, _ = _block(tmp_path, "R", rng.randn(300, 4), ids)
    fit = relate_tables(read_scores(sP), read_scores(sQ), name="PQ", join=["stimulus_id"], n_perm=0)
    _, mPQ = save_relation(fit, tmp_path / "rel", stem="PQ_v0.1")
    return dict(P=mP, Q=mQ, R=mR, PQ=mPQ, tmp=tmp_path)


def test_release_round_trip(fitted):
    f = fitted
    rel = build_release(name="sp", version="0.1.0", blocks=[f["P"], f["Q"], f["R"]], relations=[f["PQ"]],
                        absent=[parse_absent("P,R: log 2026-10-01")])
    assert rel["relations"]["PQ"]["a"] == "P" and rel["relations"]["PQ"]["b"] == "Q"
    assert rel["absent"] == [{"pair": ["P", "R"], "k": 0, "evidence": "log 2026-10-01"}]
    path = write_release(rel, f["tmp"] / "releases")
    back = load_release(path)
    assert back.label == "sp 0.1.0"
    assert back.relation("PQ").k == rel["relations"]["PQ"]["k"] == 2


def test_relation_needs_its_blocks_in_the_release(fitted):
    with pytest.raises(SpaceError, match="not one of this release's blocks"):
        build_release(name="sp", version="0.1.0", blocks=[fitted["P"], fitted["R"]], relations=[fitted["PQ"]])


def test_absences_are_checked(fitted):
    f = fitted
    with pytest.raises(SpaceError, match="not among this release's blocks"):
        build_release(name="sp", version="0.1.0", blocks=[f["P"], f["Q"]], absent=[parse_absent("P,Z: x")])
    with pytest.raises(SpaceError, match="also has a relation"):
        build_release(name="sp", version="0.1.0", blocks=[f["P"], f["Q"]], relations=[f["PQ"]],
                      absent=[parse_absent("Q,P: x")])
    for bad in ("P,R", "P: x", "P,R:   "):
        with pytest.raises(SpaceError, match="write it as"):
            parse_absent(bad)


def test_release_is_immutable_and_tamper_evident(fitted):
    f = fitted
    rel = build_release(name="sp", version="0.1.0", blocks=[f["P"], f["Q"]], relations=[f["PQ"]])
    path = write_release(rel, f["tmp"] / "releases")
    with pytest.raises(SpaceError, match="immutable"):
        write_release(rel, f["tmp"] / "releases")
    np.savez(f["tmp"] / "Q_v1.npz", w=np.arange(4.0))  # the Q weights change after release
    with pytest.raises(SpaceError, match="Q_v1.npz has changed"):
        load_release(path)


def test_cli_write_and_verify(fitted, capsys):
    f = fitted
    out = f["tmp"] / "releases"
    assert main(["space", "release", "write", "--name", "sp", "--version", "0.1.0",
                 "--blocks", str(f["P"]), str(f["Q"]), str(f["R"]), "--relations", str(f["PQ"]),
                 "--absent", "P,R: log", "Q,R: log", "-o", str(out)]) == 0
    assert "absent P<->R, Q<->R" in capsys.readouterr().out
    assert main(["space", "release", "verify", str(out / "sp_0.1.0.json")]) == 0
    assert "every pinned file matches" in capsys.readouterr().out
    (f["tmp"] / "R_v1.json").write_text("{}")
    assert main(["space", "release", "verify", str(out / "sp_0.1.0.json")]) == 1



# --- 0.2.0 write rules (release schema 1.2) -----------------------------------


def _new_block(tmp_path, block, key, windows, **extra):
    """A stand-in block manifest at space schema 1.14, members with declared windows."""
    np.savez(tmp_path / f"{block}_new.npz", w=np.arange(3.0))
    manifest = tmp_path / f"{block}_new.json"
    manifest.write_text(json.dumps({
        "block": block, "k": 2, "members": list(windows), "weights": f"{block}_new.npz",
        "psytwill_version": "test", "space_schema_version": "1.14", "key": key,
        "per_member": {m: {"window_sec": w, "window_source": src} for m, (w, src) in windows.items()},
        **extra}))
    return manifest


def test_member_windows_are_recorded_and_checked(tmp_path):
    ok = _new_block(tmp_path, "A", "stimulus_id,time", {"clap": (10.0, "battery"), "pitch": (None, "battery")})
    rel = build_release(name="sp", version="0.2.0", blocks=[ok])
    assert rel["release_schema_version"] == "1.2"
    assert rel["blocks"]["A"]["member_windows"] == {"clap": 10.0, "pitch": None}
    bad = _new_block(tmp_path, "B", "stimulus_id,time", {"x": (None, "undeclared")})
    with pytest.raises(SpaceError, match="x: no window declared"):
        build_release(name="sp", version="0.2.0", blocks=[bad])
    untimed = _new_block(tmp_path, "C", "stimulus_id", {"x": (2.0, "sidecar"), "y": (None, "undeclared")})
    with pytest.raises(SpaceError, match="keyed without time"):
        build_release(name="sp", version="0.2.0", blocks=[untimed])


def test_a_block_before_schema_1_14_is_recorded_as_undeclared(fitted):
    rel = build_release(name="sp", version="0.1.1", blocks=[fitted["P"]])
    assert rel["blocks"]["P"]["member_windows"] is None and rel["exclusions"] == {}


def test_exclusion_lists_must_live_under_the_data_root(tmp_path):
    root = tmp_path / "fit-corpora"
    (root / "space" / "exclusions").mkdir(parents=True)
    inside = root / "space" / "exclusions" / "ids.txt"
    inside.write_text("ext-t-000\n")
    outside = tmp_path / "repo" / "ids.txt"
    outside.parent.mkdir()
    outside.write_text("ext-t-000\n")
    w = {"pitch": (None, "battery")}
    good = _new_block(tmp_path, "A", "stimulus_id,time", w, exclude_ids_file=str(inside))
    with pytest.raises(SpaceError, match="give the release a data root"):
        build_release(name="sp", version="0.2.0", blocks=[good])
    rel = build_release(name="sp", version="0.2.0", blocks=[good], data_root=root)
    assert list(rel["exclusions"]) == [str(inside.resolve())] and len(rel["exclusions"][str(inside.resolve())]) == 64
    stray = _new_block(tmp_path, "B", "stimulus_id,time", w, exclude_rows_file=str(outside))
    with pytest.raises(SpaceError, match="is outside it"):
        build_release(name="sp", version="0.2.0", blocks=[good, stray], data_root=root)
    # pinned: verify notices an edited list
    path = write_release(rel, tmp_path / "releases")
    assert load_release(path).meta["data_root"] == str(root.resolve())
    inside.write_text("ext-t-000\next-t-001\n")
    with pytest.raises(SpaceError, match="exclusion list ids.txt has changed"):
        load_release(path)


def test_cli_write_takes_a_data_root(tmp_path, capsys):
    root = tmp_path / "fc"
    root.mkdir()
    ids = root / "ids.txt"
    ids.write_text("ext-t-000\n")
    m = _new_block(tmp_path, "A", "stimulus_id,time", {"pitch": (None, "battery")}, exclude_ids_file=str(ids))
    assert main(["space", "release", "write", "--name", "sp", "--version", "0.2.0", "--blocks", str(m),
                 "--data-root", str(root), "-o", str(tmp_path / "rel")]) == 0
    capsys.readouterr()
    assert main(["space", "release", "verify", str(tmp_path / "rel" / "sp_0.2.0.json")]) == 0
    assert "1 exclusion lists" in capsys.readouterr().out
