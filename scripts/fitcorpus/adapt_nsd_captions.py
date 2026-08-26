#!/usr/bin/env python3
"""Adapt COCO captions to the NSD fit-corpus id namespace.

Joins the fetched COCO caption annotations (fetch.py coco-captions) onto
the NSD stimulus info table and writes one tidy table keyed by
``ext-nsd-<nsdId>`` — the V<->L language-side input for psytwill-space,
ready for word2psy extraction once the block basis unlocks it.

Columns: stimulus_id, nsd_id, coco_id, coco_split, caption_idx, caption.

Usage:
    adapt_nsd_captions.py --durable-root <fit-corpora> \
        --nsd-stim-info <nsd_stim_info_merged.csv> [--out <path>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from psytwill.fitcorpus import ext_id  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--durable-root", required=True)
    parser.add_argument("--nsd-stim-info", required=True)
    parser.add_argument(
        "--out", default=None,
        help="output parquet (default: <durable-root>/nsd/inputs/coco_captions.parquet)",
    )
    parser.add_argument(
        "--check-registry", default=None,
        help="optional stimulus-registry TSV with 1-based nsdId + cocoId "
        "columns; every pair is asserted against the join",
    )
    args = parser.parse_args(argv)

    root = Path(args.durable_root)
    cap_dir = root / "coco-captions"
    out = Path(args.out) if args.out else root / "nsd" / "inputs" / "coco_captions.parquet"

    info = pd.read_csv(args.nsd_stim_info, index_col=0)
    # nsd_stim_info_merged nsdId is 0-based (0..72999); the namespace uses
    # the 1-based form to match the wider convention (registry filenames
    # like shared####_nsd#####). The two differ by exactly one — a known
    # off-by-one trap, hence the assert and the --check-registry flag.
    if not (info.nsdId.min() == 0 and info.nsdId.max() == len(info) - 1
            and info.nsdId.is_unique):
        raise SystemExit("nsd_stim_info nsdId is not the expected 0-based unique key")
    info = info.assign(nsdId=info.nsdId + 1)

    rows = []
    for split in ("train2017", "val2017"):
        path = cap_dir / f"captions_{split}.json"
        if not path.exists():
            raise SystemExit(
                f"{path} missing — run fetch.py coco-captions first"
            )
        with path.open() as f:
            anns = json.load(f)["annotations"]
        rows.append(pd.DataFrame(anns)[["image_id", "id", "caption"]])
    caps = pd.concat(rows, ignore_index=True).rename(columns={"image_id": "cocoId"})

    joined = info[["nsdId", "cocoId", "cocoSplit"]].merge(caps, on="cocoId", how="left")
    missing = joined[joined.caption.isna()]
    if len(missing):
        raise SystemExit(
            f"{missing.nsdId.nunique()} NSD images have no caption — "
            "the caption fetch is incomplete"
        )

    if args.check_registry:
        reg = pd.read_csv(args.check_registry, sep="\t")
        pairs = joined.drop_duplicates("nsdId").set_index("nsdId").cocoId
        bad = reg[reg.cocoId.values != pairs.loc[reg.nsdId].values]
        if len(bad):
            raise SystemExit(
                f"{len(bad)} registry rows disagree with the join "
                f"(first: {bad.iloc[0].to_dict()}) — check nsdId basing"
            )
        print(f"registry cross-check ok: {len(reg)} (nsdId, cocoId) pairs agree")

    # Deterministic caption_idx: annotation-id order within each image.
    joined = joined.sort_values(["nsdId", "id"]).reset_index(drop=True)
    joined["caption_idx"] = joined.groupby("nsdId").cumcount()
    joined["stimulus_id"] = [ext_id("nsd", int(n)) for n in joined.nsdId]
    tidy = joined.rename(columns={"nsdId": "nsd_id", "cocoId": "coco_id",
                                  "cocoSplit": "coco_split"})[
        ["stimulus_id", "nsd_id", "coco_id", "coco_split", "caption_idx", "caption"]
    ]
    tidy["caption"] = tidy.caption.str.strip()

    out.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_parquet(out, index=False)
    meta = {
        "source": "COCO 2017 caption annotations (CC-BY) x nsd_stim_info_merged",
        "n_images": int(tidy.nsd_id.nunique()),
        "n_captions": len(tidy),
        "id_namespace": "ext-nsd-<nsdId, 1-based, 6-digit>",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(tidy)} captions over {meta['n_images']} images -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
