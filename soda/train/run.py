"""CLI: train one DG rotation and write a JSON result file.

Example (ResNet-50 validation gate — full fine-tune, SGD+nesterov, lr 1e-3):
  python -m soda.train.run --root /kaggle/working/gd --backbone resnet50 \
      --head softmax --target APTOS --epochs 100 --lr 1e-3 --optim sgd --out erm_APTOS.json
"""
from __future__ import annotations

import argparse, json
from soda.dg_config import dg_sources_for_target
from soda.train.engine import train_and_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--head", required=True, choices=["softmax", "corn"])
    ap.add_argument("--target", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optim", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = dict(root=a.root, backbone=a.backbone, head=a.head, target=a.target,
               sources=dg_sources_for_target(a.target), epochs=a.epochs,
               batch_size=a.batch_size, img_size=a.img_size, lr=a.lr, optim=a.optim,
               seed=a.seed, lora=a.lora)
    res = train_and_eval(cfg)
    res["config"] = {k: cfg[k]
                     for k in ("backbone", "head", "target", "epochs", "lora", "lr", "optim", "seed")}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
