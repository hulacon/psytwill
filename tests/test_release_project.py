"""`space release project`: a held-out stimulus set placed in a release, as a §4.1 family.

What these pin: the family reproduces `space project` + `relate project` on the
same rows; ids land on the registry one to one and covering it; a row the fit
was trained on is refused (block and relation leak guards); a sub-stimulus key
the source numbers differently is remapped only through an explicit key map;
`psytwill features` reads the family with no special case; and a later
release moves the earlier family aside instead of overwriting it.
"""

import json

import numpy as np
import pandas as pd
import pytest

from psytwill.cli import main

N_IMG, N_HELD, N_CAP, LATENT = 240, 40, 2, 2
FIT = ["--k-schedule", "2,4", "--n-splits", "3", "--n-perm", "150", "--k-nn", "10", "--eval-n", "0", "--seed", "0"]


def _long(rows, extractor):
    df = pd.DataFrame(rows, columns=["stimulus_id", "chunk_idx", "model", "feature", "value"])
    df["value_str"] = pd.Series([None] * len(df), dtype="string")
    df["modality"], df["extractor"], df["extractor_version"] = "x", extractor, "0.0"
    return df


def _tables(tmp, rng):
    """V (image grain) and L (caption grain) tables driven by one latent per image."""
    Wv = {"a": rng.normal(size=(LATENT, 6)), "b": rng.normal(size=(LATENT, 3))}
    Wl = {"c": rng.normal(size=(LATENT, 5)), "d": rng.normal(size=(LATENT, 4))}
    v, l = [], []
    for i in range(N_IMG):
        sid, z = f"ext-t-{i:04d}", rng.normal(size=LATENT)
        for m, w in Wv.items():
            for j, x in enumerate(z @ w + 0.05 * rng.normal(size=w.shape[1])):
                v.append((sid, None, m, f"{m}_{j:02d}", float(x)))
        for c in range(N_CAP):
            zc = z + 0.1 * rng.normal(size=LATENT)
            for m, w in Wl.items():
                for j, x in enumerate(zc @ w + 0.05 * rng.normal(size=w.shape[1])):
                    # chunk_idx numbered across the whole source table, as NSD's are
                    l.append((sid, 1000 + N_CAP * i + c, m, f"{m}_{j:02d}", float(x)))
    vp, lp = tmp / "img_features.parquet", tmp / "cap_chunks_features.parquet"
    _long(v, "viz2psy").drop(columns="chunk_idx").to_parquet(vp, index=False)
    _long(l, "word2psy").to_parquet(lp, index=False)
    return vp, lp


@pytest.fixture(scope="module")
def space(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rp")
    vp, lp = _tables(tmp, np.random.default_rng(0))
    held = [f"ext-t-{i:04d}" for i in range(N_HELD)]
    ex = tmp / "held.txt"
    ex.write_text("\n".join(held) + "\n")
    sp = tmp / "space"
    assert main(["space", "fit", "--features", str(vp), "--block", "V", "--members", "a,b", "--exclude-ids", str(ex),
                 "-o", str(sp), "--stem", "V_t", *FIT]) == 0
    assert main(["space", "fit", "--features", str(lp), "--key", "stimulus_id,chunk_idx", "--block", "L",
                 "--members", "c,d", "--exclude-ids", str(ex), "-o", str(sp), "--stem", "L_t", *FIT]) == 0
    for blk, f, key in (("V", vp, "stimulus_id"), ("L", lp, "stimulus_id,chunk_idx")):
        assert main(["space", "project", "--features", str(f), "--key", key, "--exclude-ids", str(ex),
                     "--space", str(sp / f"{blk}_t.json"), "-o", str(sp / f"{blk}_fitrows.parquet")]) == 0
        # the reference: the same frozen block on every row, held-out ones included
        assert main(["space", "project", "--features", str(f), "--key", key,
                     "--space", str(sp / f"{blk}_t.json"), "-o", str(sp / f"{blk}_all.parquet")]) == 0
    assert main(["space", "relate", "fit", "--a", str(sp / "V_fitrows.parquet"), "--b", str(sp / "L_fitrows.parquet"),
                 "--name", "VL", "--pool-b", "mean", "--n-perm", "0", "-o", str(sp / "rel"), "--stem", "VL_t"]) == 0
    for blk, side in (("V", "a"), ("L", "b")):
        assert main(["space", "relate", "project", "--relation", str(sp / "rel" / "VL_t.json"), "--side", side,
                     "--scores", str(sp / f"{blk}_all.parquet"), "-o", str(sp / f"VL_{blk}_all.parquet")]) == 0
    rels = tmp / "releases"
    for version in ("0.1.0", "0.2.0"):
        assert main(["space", "release", "write", "--name", "sp", "--version", version,
                     "--blocks", str(sp / "V_t.json"), str(sp / "L_t.json"),
                     "--relations", str(sp / "rel" / "VL_t.json"), "-o", str(rels)]) == 0
    reg = tmp / "registry.tsv"
    pd.DataFrame({"stimulus_id": [f"img{i + 1:04d}" for i in range(N_HELD)], "tid": range(N_HELD)}).to_csv(
        reg, sep="\t", index=False)
    kmap = tmp / "chunk_map.csv"
    pd.DataFrame([{"src_stimulus_id": f"ext-t-{i:04d}", "src_chunk_idx": 1000 + N_CAP * i + c,
                   "stimulus_id": f"img{i + 1:04d}", "chunk_idx": N_CAP * i + c}
                  for i in range(N_HELD) for c in range(N_CAP)]).to_csv(kmap, index=False)
    return dict(tmp=tmp, vp=vp, lp=lp, sp=sp, rels=rels, reg=reg, kmap=kmap)


def _argv(s, out, *, version="0.1.0", registry=None, vp=None, kmap=True):
    argv = ["space", "release", "project", "--release", str(s["rels"] / f"sp_{version}.json"), "--set", "toy",
            "--registry", str(registry or s["reg"]), "--registry-join", "tid", "--id-pattern", r"ext-t-(\d+)",
            "--grain", "-", "V", "stimulus_id", str(vp or s["vp"]),
            "--grain", "chunks", "L", "stimulus_id,chunk_idx", str(s["lp"]), "-o", str(out)]
    if kmap:
        argv += ["--key-map", f"chunks={s['kmap']}"]
    return argv


@pytest.fixture(scope="module")
def family(space):
    out = space["tmp"] / "store" / "toy" / "psytwill_space"
    assert main(_argv(space, out)) == 0
    return out


def test_family_reproduces_space_and_relate_project(space, family):
    sp = space["sp"]
    img = pd.read_csv(family.parent / "psytwill_space.csv")
    cap = pd.read_csv(family.parent / "psytwill_space_chunks.csv")
    assert list(img["stimulus_id"]) == [f"img{i + 1:04d}" for i in range(N_HELD)]
    assert len(cap) == N_HELD * N_CAP and list(cap["chunk_idx"]) == list(range(N_HELD * N_CAP))
    to_src = {f"img{i + 1:04d}": f"ext-t-{i:04d}" for i in range(N_HELD)}
    for df, ref_v, ref_r, model, rel, keys in (
            (img, "V_all", "VL_V_all", "pspace_v", "pspace_vl", ["stimulus_id"]),
            (cap, "L_all", "VL_L_all", "pspace_l", "pspace_vl", ["stimulus_id", "chunk_idx"])):
        src = df.assign(stimulus_id=df["stimulus_id"].map(to_src))
        if "chunk_idx" in keys:
            src["chunk_idx"] = src["chunk_idx"] + 1000
        for ref, prefix, m in ((ref_v, ref_v[0], model), (ref_r, "VL", rel)):
            r = pd.read_parquet(sp / f"{ref}.parquet")
            r["stimulus_id"] = r["stimulus_id"].astype(str)
            if "chunk_idx" in keys:
                r["chunk_idx"] = r["chunk_idx"].astype(float).astype(int)
            got = src[keys].merge(r, on=keys, how="left")
            cols = sorted(c for c in r.columns if c.startswith(f"{prefix}_"))
            ours = df[[c for c in df.columns if c.startswith(f"{m}_")]].to_numpy()
            np.testing.assert_allclose(ours, got[cols].to_numpy(), rtol=1e-5, atol=1e-5)


def test_sidecar_pins_the_release_and_records_the_guards(space, family):
    meta = json.loads((family.parent / "psytwill_space.meta.json").read_text())
    assert meta["schema_version"] == "1.1" and meta["extractor"] == "psytwill"
    assert meta["input"]["release_version"] == "0.1.0"
    v, vl = meta["models"]["pspace_v"], meta["models"]["pspace_vl"]
    assert v["checkpoint"] == "sp@0.1.0:V_t" and v["modality"] == "visual" and list(v["tables"]) == ["base"]
    assert v["tables"]["base"]["leak_guard"]["rule"].startswith("every projected id")
    # one relation model, two tables: each side keeps its own facts
    assert {t: e["leak_guard"]["side"] for t, e in vl["tables"].items()} == {"base": "a", "chunks": "b"}
    assert set(vl["tables"]["chunks"]["leak_guard"]["scores"]) == {"a", "b"}
    assert v["nulls"]["pspace_v_000"]["means"] == "undefined"
    assert meta["input"]["grains"]["chunks"]["key_map"]["sha256"]


def test_features_reads_the_family_with_no_special_case(family, tmp_path):
    out = tmp_path / "toy_features.parquet"
    assert main(["features", str(family.parent / "psytwill_space.csv"),
                 str(family.parent / "psytwill_space_chunks.csv"), "-o", str(out)]) == 0
    df = pd.read_parquet(out)
    assert set(df["model"]) == {"pspace_v", "pspace_l", "pspace_vl"}
    mod = df.groupby("model")["modality"].first().to_dict()
    assert mod == {"pspace_v": "visual", "pspace_l": "text", "pspace_vl": "shared"}


def test_fitted_rows_are_refused(space, tmp_path, capsys):
    reg = tmp_path / "leaky.tsv"
    ids = list(range(N_HELD - 1)) + [N_HELD + 5]  # one id the fit trained on
    pd.DataFrame({"stimulus_id": [f"img{i:04d}" for i in ids], "tid": ids}).to_csv(reg, sep="\t", index=False)
    assert main(_argv(space, tmp_path / "f" / "psytwill_space", registry=reg, kmap=False)) == 1
    assert "rows of its fit" in capsys.readouterr().err


def test_registry_must_be_covered(space, tmp_path, capsys):
    reg = tmp_path / "wide.tsv"
    rows = [{"stimulus_id": f"img{i + 1:04d}", "tid": i} for i in range(N_HELD)] + [{"stimulus_id": "img9999", "tid": 99999}]
    pd.DataFrame(rows).to_csv(reg, sep="\t", index=False)
    assert main(_argv(space, tmp_path / "f" / "psytwill_space", registry=reg, kmap=False)) == 1
    assert "have no row in the input" in capsys.readouterr().err


def test_chunk_key_is_kept_without_a_key_map(space, tmp_path):
    out = tmp_path / "nomap" / "psytwill_space"
    assert main(_argv(space, out, kmap=False)) == 0
    cap = pd.read_csv(out.parent / "psytwill_space_chunks.csv")
    assert cap["chunk_idx"].min() == 1000  # the source numbering, untouched


def test_unplaced_rows_are_written_as_declared_nan(space, tmp_path):
    df = pd.read_parquet(space["vp"])
    drop = (df["stimulus_id"] == "ext-t-0003") & (df["model"] == "a")
    vp = tmp_path / "img_partial_features.parquet"
    df[~drop].to_parquet(vp, index=False)
    out = tmp_path / "partial" / "psytwill_space"
    assert main(_argv(space, out, vp=vp)) == 0
    img = pd.read_csv(out.parent / "psytwill_space.csv")
    meta = json.loads((out.parent / "psytwill_space.meta.json").read_text())
    assert len(img) == N_HELD  # the stimulus keeps its row either way
    n_nan = int(img.filter(like="pspace_v_").isna().any(axis=1).sum())
    assert meta["models"]["pspace_v"]["tables"]["base"]["n_unplaced"] == n_nan
    print(f"unplaced with member a absent: {n_nan}")


def test_a_later_release_moves_the_earlier_family_aside(space, tmp_path):
    out = tmp_path / "ver" / "psytwill_space"
    assert main(_argv(space, out)) == 0
    assert main(_argv(space, out)) == 0  # same release: rewritten in place
    assert not (out.parent / "psytwill_space_0.1.0").exists()
    assert main(_argv(space, out, version="0.2.0")) == 0
    old = out.parent / "psytwill_space_0.1.0"
    assert (old / "psytwill_space.meta.json").exists() and (old / "psytwill_space_chunks.csv").exists()
    meta = json.loads((out.parent / "psytwill_space.meta.json").read_text())
    assert meta["input"]["release_version"] == "0.2.0" and meta["input"]["archived_previous"] == str(old)


def test_table_names_must_be_findable(space, tmp_path, capsys):
    argv = _argv(space, tmp_path / "t" / "psytwill_space", kmap=False)
    argv[argv.index("chunks")] = "captions"
    assert main(argv) == 1
    assert "not a §4.1 table suffix" in capsys.readouterr().err


def test_skipped_member_is_masked_and_declared(space, tmp_path, capsys):
    out = tmp_path / "skip" / "psytwill_space"
    argv = _argv(space, out) + ["--skip-member=-=b"]
    assert main(argv) == 0
    img = pd.read_csv(out.parent / "psytwill_space.csv")
    meta = json.loads((out.parent / "psytwill_space.meta.json").read_text())
    assert meta["input"]["grains"]["base"]["members_absent"] == ["b"]
    assert meta["models"]["pspace_v"]["tables"]["base"]["members_absent"] == ["b"]
    # placed from member a alone: every row still has coordinates, and they differ from the full projection
    full = pd.read_csv(space["tmp"] / "store" / "toy" / "psytwill_space.csv")
    v = [c for c in img.columns if c.startswith("pspace_v_")]
    assert np.isfinite(img[v].to_numpy()).all()
    assert not np.allclose(img[v].to_numpy(), full[v].to_numpy())
    assert "members absent by declaration: b" in capsys.readouterr().out
    assert main(_argv(space, tmp_path / "bad" / "psytwill_space") + ["--skip-member=-=zz"]) == 1


def test_partial_coverage_lists_what_is_missing(space, tmp_path):
    reg = tmp_path / "wider.tsv"
    ids = list(range(N_HELD))
    rows = [{"stimulus_id": f"img{i + 1:04d}", "tid": i} for i in ids] + [{"stimulus_id": "img9999", "tid": 99999}]
    pd.DataFrame(rows).to_csv(reg, sep="\t", index=False)
    out = tmp_path / "partial_cov" / "psytwill_space"
    assert main(_argv(space, out, registry=reg) + ["--partial-coverage"]) == 0
    meta = json.loads((out.parent / "psytwill_space.meta.json").read_text())
    assert meta["input"]["grains"]["base"]["registry_uncovered"] == ["img9999"]


def test_baseline_release_writes_its_own_model_prefix(space, tmp_path, capsys):
    sp = space["sp"]
    rels = tmp_path / "rel"
    assert main(["space", "release", "write", "--name", "b0", "--version", "0.1.0", "--model-prefix", "b0",
                 "--blocks", str(sp / "V_t.json"), str(sp / "L_t.json"),
                 "--relations", str(sp / "rel" / "VL_t.json"), "-o", str(rels)]) == 0
    assert json.loads((rels / "b0_0.1.0.json").read_text())["model_prefix"] == "b0"
    out = tmp_path / "store" / "toy" / "b0"
    argv = _argv(space, out)
    argv[argv.index("--release") + 1] = str(rels / "b0_0.1.0.json")
    assert main(argv) == 0
    img = pd.read_csv(out.parent / "b0.csv")
    assert any(c.startswith("b0_v_") for c in img.columns) and any(c.startswith("b0_vl_") for c in img.columns)
    assert not any(c.startswith("pspace_") for c in img.columns)
    meta = json.loads((out.parent / "b0.meta.json").read_text())
    assert set(meta["models"]) == {"b0_v", "b0_vl", "b0_l"}
    capsys.readouterr()
    assert main(["space", "release", "write", "--name", "x", "--version", "0.1.0", "--model-prefix", "B-0",
                 "--blocks", str(sp / "V_t.json"), "-o", str(rels)]) == 1
