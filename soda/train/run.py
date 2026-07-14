"""CLI: train one DG rotation (ERM/CORN or full SODA) and write a JSON result file."""
from __future__ import annotations

import argparse, json
from soda.dg_config import dg_sources_for_target
from soda.train.engine import train_and_eval, train_and_eval_soda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--head", required=True, choices=["softmax", "corn"])
    ap.add_argument("--target", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optim", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora", action="store_true")
    # SODA alignment flags
    ap.add_argument("--use-consist", action="store_true")
    ap.add_argument("--use-proto", action="store_true")
    ap.add_argument("--use-mono", action="store_true")
    ap.add_argument("--lambda1", type=float, default=0.1)
    ap.add_argument("--lambda2", type=float, default=0.1)
    ap.add_argument("--mono-margin", type=float, default=0.5)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--ema-momentum", type=float, default=0.9)
    ap.add_argument("--domains-per-batch", type=int, default=4)
    ap.add_argument("--per-domain", type=int, default=4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--ramp-epochs", type=int, default=5)
    ap.add_argument("--save-weights", action="store_true")
    ap.add_argument("--weights-out", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    soda_on = a.use_consist or a.use_proto or a.use_mono
    cfg = dict(root=a.root, backbone=a.backbone, head=a.head, target=a.target,
               sources=dg_sources_for_target(a.target), epochs=a.epochs,
               batch_size=a.batch_size, img_size=a.img_size, lr=a.lr, optim=a.optim,
               seed=a.seed, lora=a.lora,
               use_consist=a.use_consist, use_proto=a.use_proto, use_mono=a.use_mono,
               lambda1=a.lambda1, lambda2=a.lambda2, mono_margin=a.mono_margin,
               proj_dim=a.proj_dim, ema_momentum=a.ema_momentum,
               domains_per_batch=a.domains_per_batch, per_domain=a.per_domain,
               warmup_epochs=a.warmup_epochs, ramp_epochs=a.ramp_epochs,
               save_weights=a.save_weights,
               weights_out=a.weights_out or (a.out.rsplit(".", 1)[0] + ".pt"))
    res = train_and_eval_soda(cfg) if soda_on else train_and_eval(cfg)
    res["config"] = {k: cfg[k] for k in ("backbone", "head", "target", "epochs", "lora", "lr",
                     "optim", "seed", "use_consist", "use_proto", "use_mono", "lambda1", "lambda2")}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
