"""Report layout health for every GDRBench split.

Usage:
  python scripts/check_data.py --root D:/fundus/GDRBench/FundusDG_mini
"""
from __future__ import annotations

import argparse
from pathlib import Path

from soda.data.verify_layout import verify_split

DATASETS = ["APTOS", "DDR", "DEEPDR", "EYEPACS", "FGADR", "IDRID", "MESSIDOR", "RLDR"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    splits = root / "splits"
    for ds in DATASETS:
        for kind in ("train", "crossval"):
            f = splits / f"{ds}_{kind}.txt"
            if not f.exists():
                print(f"{ds}_{kind}: MISSING split file")
                continue
            rep = verify_split(f, root)
            print(f"{ds}_{kind}: total={rep.total} missing={len(rep.missing)} "
                  f"labels={dict(sorted(rep.label_counts.items()))}")
            if rep.missing:
                print(f"    e.g. missing: {rep.missing[:3]}")
    masks = root / "masks"
    print(f"masks/ exists: {masks.exists()}")


if __name__ == "__main__":
    main()
