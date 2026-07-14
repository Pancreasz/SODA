"""Train loop + evaluation for the fresh SODA harness."""
from __future__ import annotations

import random

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
from soda.train.data import build_align_loaders
from soda.train.alignment import (
    ProjectionHead, PrototypeBank, torch_severity, l_consist, l_proto, l_mono,
)
from soda.train.checkpoint import save_trainable


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    try:
        auc = macro_ovo_auc(labels, probs)
    except ValueError:
        # A fold missing one of the 5 DR grades makes ovo AUC undefined; return NaN
        # rather than aborting a long run. NaN never wins the best-AUC comparison below.
        auc = float("nan")
    qwk = quadratic_weighted_kappa(labels, grades)
    cat = catastrophic_error_rate(labels, grades)
    mae = mean_absolute_grade_error(labels, grades)
    acc = float((grades == labels).mean())
    _, qlo, qhi = bootstrap_ci(quadratic_weighted_kappa, labels, grades, n=1000, seed=0)
    return {"auc": auc, "qwk": qwk, "qwk_ci": [qlo, qhi], "cat": cat, "mae": mae, "acc": acc}


def train_and_eval(cfg):
    set_seed(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te = build_loaders(cfg["root"], cfg["sources"], cfg["target"],
                               img_size=cfg.get("img_size", 224),
                               batch_size=cfg.get("batch_size", 16))
    backbone, feat = build_backbone(cfg["backbone"], lora=cfg.get("lora", False),
                                    lora_r=cfg.get("lora_r", 8))
    head_kind = cfg["head"]  # 'softmax' or 'corn'
    if head_kind == "softmax":
        head = SoftmaxHead(feat, 5)
    elif head_kind == "corn":
        head = CORNHead(feat, 5)
    else:
        raise ValueError(head_kind)
    net = Net(backbone, head).to(device)

    params = [p for p in net.parameters() if p.requires_grad]
    # ResNet validation mirrors DGDR (SGD+nesterov); DINOv2+LoRA uses AdamW.
    if cfg.get("optim", "adamw") == "sgd":
        opt = torch.optim.SGD(params, lr=cfg.get("lr", 1e-3), momentum=0.9,
                              weight_decay=cfg.get("wd", 5e-4), nesterov=True)
    else:
        opt = torch.optim.AdamW(params, lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("wd", 5e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    best_auc, best_state = -1.0, None
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
            if vm["auc"] > best_auc:  # NaN never wins this comparison
                best_auc = vm["auc"]
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            print(f"epoch {ep} val_auc {vm['auc']:.4f}")
    # DG model selection: restore the best source-validation checkpoint before the
    # held-out target test evaluation, rather than scoring the last-epoch weights.
    if best_state is not None:
        net.load_state_dict(best_state)
    test_metrics = evaluate(net, te, head_kind, device)
    test_metrics["val_best_auc"] = best_auc
    return test_metrics


class SodaNet(nn.Module):
    def __init__(self, backbone, head, proj):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.proj = proj

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat), self.proj(feat)


def _lambda_scale(epoch, warmup, ramp):
    if epoch <= warmup:
        return 0.0
    if epoch >= warmup + ramp:
        return 1.0
    return (epoch - warmup) / float(ramp)


@torch.no_grad()
def _eval_soda(net, loader, device):
    net.eval()
    probs, grades, labels = [], [], []
    for xb, yb, _ in loader:
        logits, _ = net(xb.to(device))
        probs.append(predict_class_probs("corn", logits))
        grades.append(predict_grade("corn", logits))
        labels.append(yb.numpy())
    probs = np.concatenate(probs); grades = np.concatenate(grades); labels = np.concatenate(labels)
    try:
        auc = macro_ovo_auc(labels, probs)
    except ValueError:
        auc = float("nan")
    qwk = quadratic_weighted_kappa(labels, grades)
    cat = catastrophic_error_rate(labels, grades)
    mae = mean_absolute_grade_error(labels, grades)
    acc = float((grades == labels).mean())
    return {"auc": auc, "qwk": qwk, "cat": cat, "mae": mae, "acc": acc}


def train_and_eval_soda(cfg):
    set_seed(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te, sampler = build_align_loaders(
        cfg["root"], cfg["sources"], cfg["target"], img_size=cfg.get("img_size", 224),
        domains_per_batch=cfg.get("domains_per_batch", 4), per_domain=cfg.get("per_domain", 4))
    num_domains = len(cfg["sources"])   # bank spans ALL source domains (dom idx 0..S-1), not domains-per-batch
    backbone, feat = build_backbone(cfg["backbone"], lora=True, lora_r=cfg.get("lora_r", 8))
    proj = ProjectionHead(feat, cfg.get("proj_dim", 128))
    head = CORNHead(feat, 5)
    net = SodaNet(backbone, head, proj).to(device)
    bank = PrototypeBank(num_domains, dim=cfg.get("proj_dim", 128),
                         momentum=cfg.get("ema_momentum", 0.9)).to(device)

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("wd", 5e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    use_c = cfg.get("use_consist", False)
    use_p = cfg.get("use_proto", False)
    use_m = cfg.get("use_mono", False)
    l1, l2 = cfg.get("lambda1", 0.1), cfg.get("lambda2", 0.1)
    margin = cfg.get("mono_margin", 0.5)
    warmup, ramp = cfg.get("warmup_epochs", 5), cfg.get("ramp_epochs", 5)

    best_auc, best_state = -1.0, None
    for ep in range(1, cfg["epochs"] + 1):
        sampler.set_epoch(ep)
        net.train()
        scale = _lambda_scale(ep, warmup, ramp)
        for batch in tr:
            x, x_style, y, dom = batch
            x, x_style, y, dom = x.to(device), x_style.to(device), y.to(device), dom.to(device)
            opt.zero_grad()
            logits, z = net(x)
            loss = compute_loss("corn", logits, y, 5)
            if scale > 0:
                if use_c:
                    s_style = torch_severity(net(x_style)[0])
                    loss = loss + l1 * scale * l_consist(torch_severity(logits), s_style)
                if use_p:
                    loss = loss + l1 * scale * l_proto(z, dom, y, bank)
                if use_m:
                    loss = loss + l2 * scale * l_mono(torch_severity(logits), dom, y, margin, num_domains)
            loss.backward(); opt.step()
            bank.update(z.detach(), dom, y)   # reuse the z already computed (pre-step; EMA tolerates staleness)
        sched.step()
        if ep % cfg.get("val_every", 5) == 0 or ep == cfg["epochs"]:
            vm = _eval_soda(net, va, device)
            if vm["auc"] > best_auc:
                best_auc = vm["auc"]
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            print(f"epoch {ep} val_auc {vm['auc']:.4f} scale {scale:.2f}")
    if best_state is not None:
        net.load_state_dict(best_state)
    out = _eval_soda(net, te, device)
    out["val_best_auc"] = best_auc
    if cfg.get("save_weights"):
        save_trainable(net, cfg["weights_out"], {k: cfg[k] for k in
                       ("backbone", "target", "epochs", "use_consist", "use_proto", "use_mono",
                        "lambda1", "lambda2", "seed")})
    return out
