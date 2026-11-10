"""Offline checks on scripts/fitcorpus/fetch.py helpers (no network)."""

import hashlib
import importlib.util
import zipfile
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fitcorpus_fetch", Path(__file__).parents[1] / "scripts" / "fitcorpus" / "fetch.py")
fetch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fetch)


def _record(path: Path, digest: str | None = None) -> dict:
    data = path.read_bytes()
    return {path.name: (path.as_uri(), digest or hashlib.md5(data).hexdigest(), len(data))}


class TestDownloadZenodo:
    def test_verified_download_lands(self, tmp_path):
        src = tmp_path / "src" / "captions.json"
        src.parent.mkdir()
        src.write_text('{"a": 1}')
        out = tmp_path / "out"
        out.mkdir()
        got = fetch.download_zenodo(_record(src), "captions.json", out)
        assert got.read_text() == '{"a": 1}'
        assert not (out / "captions.json.part").exists()

    def test_present_and_verified_is_not_refetched(self, tmp_path, capsys):
        src = tmp_path / "captions.json"
        src.write_text("x")
        fetch.download_zenodo(_record(src), "captions.json", tmp_path)
        assert "present" in capsys.readouterr().out

    def test_stale_part_is_completed_or_restarted(self, tmp_path):
        # file:// ignores Range, so this exercises the "server ignored Range" path:
        # the partial must be overwritten from zero, never appended to.
        src = tmp_path / "src" / "audio.7z"
        src.parent.mkdir()
        src.write_bytes(b"0123456789")
        out = tmp_path / "out"
        out.mkdir()
        (out / "audio.7z.part").write_bytes(b"0123")
        got = fetch.download_zenodo(_record(src), "audio.7z", out)
        assert got.read_bytes() == b"0123456789"

    def test_md5_mismatch_refused(self, tmp_path):
        src = tmp_path / "src" / "captions.json"
        src.parent.mkdir()
        src.write_text("x")
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(SystemExit, match="failed after 1 attempts"):
            fetch.download_zenodo(_record(src, digest="0" * 32), "captions.json", out, attempts=1)
        assert not (out / "captions.json").exists()


class TestUnzip:
    def test_member_escaping_dest_refused(self, tmp_path):
        z = tmp_path / "evil.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("../outside.txt", "x")
        dest = tmp_path / "dest"
        dest.mkdir()
        with pytest.raises(SystemExit, match="outside"):
            fetch._unzip(z, dest)
        assert not (tmp_path / "outside.txt").exists()

    def test_rerun_keeps_existing(self, tmp_path):
        z = tmp_path / "ok.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("clips/a.mp4", "v")
        dest = tmp_path / "dest"
        assert fetch._unzip(z, dest) == 1
        assert fetch._unzip(z, dest) == 0


def test_cli_lists_new_corpora_as_scripted(capsys):
    assert fetch.main(["--list"]) == 0
    out = capsys.readouterr().out
    for key in ("clotho", "avcaps"):
        line = next(l for l in out.splitlines() if l.startswith(key))
        assert "[scripted]" in line


class TestBlender:
    def _films(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "Sintel.2010.1080p.mkv").write_bytes(b"sintel")
        with zipfile.ZipFile(src / "bbb.mov.zip", "w") as z:
            z.writestr("bbb/big_buck_bunny_1080p_h264.mov", b"bunny")
            z.writestr("__MACOSX/bbb/._big_buck_bunny_1080p_h264.mov", b"junk")
        return {
            "sintel": ((src / "Sintel.2010.1080p.mkv").as_uri(), 6, "file"),
            "big-buck-bunny": ((src / "bbb.mov.zip").as_uri(), (src / "bbb.mov.zip").stat().st_size, "zip"),
        }

    def test_one_file_per_film_with_a_manifest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch, "BLENDER_FILMS", self._films(tmp_path))
        d = fetch.fetch_blender(tmp_path / "media")
        names = sorted(p.name for p in d.iterdir())
        assert names == ["MANIFEST.sha256", "big-buck-bunny.mov", "sintel.mkv"]
        assert (d / "big-buck-bunny.mov").read_bytes() == b"bunny"
        assert fetch.manifest_ok(d)
        fetch.fetch_blender(tmp_path / "media")  # rerun: verified, nothing fetched
        assert sorted(p.name for p in d.iterdir()) == names

    def test_a_changed_file_is_a_loud_error(self, tmp_path, monkeypatch):
        films = self._films(tmp_path)
        url, _, how = films["sintel"]
        monkeypatch.setattr(fetch, "BLENDER_FILMS", {"sintel": (url, 999, how)})
        with pytest.raises(SystemExit, match="expected 999"):
            fetch.fetch_blender(tmp_path / "media")
