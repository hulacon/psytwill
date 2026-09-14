"""Write the movies timeline viewer bundle.

Layout (all static, works over ``file://``):

    <out>/index.html      the viewer page (film index embedded)
    <out>/data/<slug>.js  one lazy-loaded payload per film
    <out>/viewer.meta.json  build provenance

The page resolves media (frames, audio) relative to itself, so the bundle
must stay on the same filesystem as the films directory it was built
against; the default output location, ``<films-dir>/viz/timeline/``, keeps
that true by construction and follows the viewers-land-in-viz/ convention.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path, PurePosixPath

from psytwill import __version__
from psytwill.exceptions import InputError
from psytwill.viz import movies as movies_mod

TEMPLATE = "template_movies.html"


def _relative_media_root(out_dir: Path, films_dir: Path) -> str:
    """POSIX relative path from the bundle to the films directory."""
    import os

    rel = os.path.relpath(films_dir.resolve(), out_dir.resolve())
    return str(PurePosixPath(*Path(rel).parts))


def build_movies_bundle(features_dir: Path, films_dir: Path,
                        out_dir: Path | None = None,
                        registry: Path | None = None,
                        slugs: list[str] | None = None) -> Path:
    """Build the bundle; returns the path of the written page."""
    features_dir = Path(features_dir)
    films_dir = Path(films_dir)
    if not films_dir.is_dir():
        raise InputError(f"{films_dir} does not exist — pass the directory "
                         "holding the per-film feature folders (movies/<slug>/)")
    out_dir = Path(out_dir) if out_dir else films_dir / "viz" / "timeline"

    tables = movies_mod.load_tables(features_dir)
    available = movies_mod.film_slugs(tables)
    if slugs:
        unknown = sorted(set(slugs) - set(available))
        if unknown:
            raise InputError(f"stimulus_id(s) {unknown} not in the features "
                             f"tables under {features_dir}")
        available = [s for s in available if s in set(slugs)]

    registry_rows = movies_mod.load_registry(registry) if registry else {}

    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    for slug in available:
        if not (films_dir / slug).is_dir():
            warnings.warn(f"{films_dir / slug} missing — {slug} is in the "
                          "features tables but has no film directory; skipped")
            continue
        payload = movies_mod.film_payload(slug, tables, films_dir,
                                          registry_rows.get(slug))
        (data_dir / f"{slug}.js").write_text(movies_mod.payload_js(payload),
                                             encoding="utf-8")
        index.append({"slug": slug, "title": payload["title"],
                      "style": payload["style"],
                      "duration": payload["duration"],
                      "n_series": len(payload["series"]),
                      "audio": payload["media"]["audio"] is not None})
    if not index:
        raise InputError(f"no film in {features_dir} has a directory under "
                         f"{films_dir} — nothing to build")

    template = (resources.files("psytwill.viz") / TEMPLATE).read_text("utf-8")
    page = (template
            .replace("__PSYTWILL_INDEX__", json.dumps(index, separators=(",", ":")))
            .replace("__MEDIA_ROOT__", _relative_media_root(out_dir, films_dir))
            .replace("__PSYTWILL_VERSION__", __version__))
    page_path = out_dir / "index.html"
    page_path.write_text(page, encoding="utf-8")

    meta = {
        "generator": f"psytwill viz movies {__version__}",
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features_dir": str(features_dir),
        "films_dir": str(films_dir),
        "n_films": len(index),
        "tables": sorted(tables),
    }
    (out_dir / "viewer.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return page_path
