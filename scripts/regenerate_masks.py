"""Regenerate FOV masks for a GDRBench-style images tree.

Usage:
  python scripts/regenerate_masks.py --images-root D:/fundus/GDRBench/FundusDG_mini \
      --datasets APTOS DEEPDR FGADR IDRID MESSIDOR RLDR --out-root D:/fundus/GDRBench/FundusDG_mini/masks
Mirrors each <images-root>/<DATASET>/<label>/<file> to <out-root>/<DATASET>/<label>/<file>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from soda.data.masks import generate_fov_mask

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    root = Path(args.images_root)
    out = Path(args.out_root)
    total = 0
    failed: list[str] = []
    for ds in args.datasets:
        for path in (root / ds).rglob("*"):
            if path.suffix.lower() not in IMG_EXTS:
                continue
            rel = path.relative_to(root)
            dst = (out / rel).with_suffix(".png")
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(path) as im:
                    mask = generate_fov_mask(im)
                Image.fromarray(mask).save(dst)
                total += 1
            except Exception as exc:  # a bad file must not halt the whole batch
                failed.append(f"{rel}: {exc}")
                continue
            if total % 500 == 0:
                print(f"  {total} masks written...")
    print(f"Done. {total} masks written to {out}")
    print(f"Failed: {len(failed)}")
    for f in failed[:20]:
        print(f"  SKIP {f}")


if __name__ == "__main__":
    main()
