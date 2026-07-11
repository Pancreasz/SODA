"""Train loop + evaluation for the fresh SODA harness."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from soda.train.data import build_loaders
from soda.train.backbones import build_backbone
from soda.train.heads import SoftmaxHead, CORNHead, compute_loss, predict_class_probs, predict_grade
from soda.metrics.auc import macro_ovo_auc, bootstrap_ci
from soda.metrics.ordinal import (
    quadratic_weighted_kappa, catastrophic_error_rate, mean_absolute_grade_error,
)


class Net(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


@torch.no_grad()
def evaluate(net, loader, head_kind, device):
    net.eval()
    probs, grades, labels = [], [], []
    for xb, yb, _ in loader:
        logits = net(xb.to(device))
        probs.append(predict_class_probs(head_kind, logits))
        grades.append(predict_grade(head_kind, logits))
        labels.append(yb.numpy())
    probs = np.concatenate(probs); grades = np.concatenate(grades); labels = np.concatenate(labels)
    auc = macro_ovo_auc(labels, probs)
    qwk = quadratic_weighted_kappa(labels, grades)
    cat = catastrophic_error_rate(labels, grades)
    mae = mean_absolute_grade_error(labels, grades)
    acc = float((grades == labels).mean())
    _, qlo, qhi = bootstrap_ci(quadratic_weighted_kappa, labels, grades, n=1000, seed=0)
    return {"auc": auc, "qwk": qwk, "qwk_ci": [qlo, qhi], "cat": cat, "mae": mae, "acc": acc}


def train_and_eval(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te = build_loaders(cfg["root"], cfg["sources"], cfg["target"],
                               img_size=cfg.get("img_size", 224),
                               batch_size=cfg.get("batch_size", 16))
    backbone, feat = build_backbone(cfg["backbone"], lora=cfg.get("lora", False),
                                    lora_r=cfg.get("lora_r", 8))
    head_kind = cfg["head"]  # 'softmax' or 'corn'
    head = SoftmaxHead(feat, 5) if head_kind == "softmax" else CORNHead(feat, 5)
    net = Net(backbone, head).to(device)

    params = [p for p in net.parameters() if p.requires_grad]
    # ResNet validation mirrors DGDR (SGD+nesterov); DINOv2+LoRA uses AdamW.
    if cfg.get("optim", "adamw") == "sgd":
        opt = torch.optim.SGD(params, lr=cfg.get("lr", 1e-3), momentum=0.9,
                              weight_decay=cfg.get("wd", 5e-4), nesterov=True)
    else:
        opt = torch.optim.AdamW(params, lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("wd", 5e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    best_auc, best = -1.0, None
    for ep in range(1, cfg["epochs"] + 1):
        net.train()
        for xb, yb, _ in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = compute_loss(head_kind, net(xb), yb, 5)
            loss.backward(); opt.step()
        sched.step()
        if ep % cfg.get("val_every", 5) == 0 or ep == cfg["epochs"]:
            vm = evaluate(net, va, head_kind, device)
            if vm["auc"] > best_auc:
                best_auc = vm["auc"]; best = {k: v for k, v in vm.items()}
            print(f"epoch {ep} val_auc {vm['auc']:.4f}")
    test_metrics = evaluate(net, te, head_kind, device)
    test_metrics["val_best_auc"] = best_auc
    return test_metrics
