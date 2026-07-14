"""Torch implementation of the SODA alignment terms (Kaggle only).

Mirrors soda.alignment_math (the numpy oracle). Batch prototypes are differentiable;
EMA-bank targets are detached. Backbone stays frozen — gradients reach only g + LoRA + head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_severity(logits):
    return torch.cumprod(torch.sigmoid(logits), dim=1).sum(dim=1)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class PrototypeBank(nn.Module):
    def __init__(self, num_domains, num_classes=5, dim=128, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("bank", torch.zeros(num_domains, num_classes, dim))
        self.register_buffer("filled", torch.zeros(num_domains, num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, z, domains, grades):
        z = z.detach()
        for d in domains.unique():
            dm = domains == d
            for k in grades[dm].unique():
                mask = dm & (grades == k)
                mean = z[mask].mean(0)
                if self.filled[d, k]:
                    self.bank[d, k] = self.momentum * self.bank[d, k] + (1 - self.momentum) * mean
                else:
                    self.bank[d, k] = mean
                    self.filled[d, k] = True


def l_consist(s, s_style):
    return ((s - s_style) ** 2).mean()


def l_proto(z, domains, grades, bank: PrototypeBank):
    num_domains = bank.bank.shape[0]
    terms = []
    for d in domains.unique():
        dm = domains == d
        for k in grades[dm].unique():
            mask = dm & (grades == k)
            p_hat = z[mask].mean(0)                     # differentiable
            acc = z.new_tensor(0.0)
            found = False
            for dp in range(num_domains):
                if dp == d or not bool(bank.filled[dp, k]):
                    continue
                acc = acc + ((p_hat - bank.bank[dp, k].detach()) ** 2).sum()
                found = True
            if found:
                terms.append(acc)
    if not terms:
        return z.new_tensor(0.0)
    return torch.stack(terms).mean()


def l_mono(s, domains, grades, margin, num_domains, num_classes=5):
    penalties = []
    for d in range(num_domains):
        dm = domains == d
        for k in range(num_classes - 1):
            lo = dm & (grades == k)
            hi = dm & (grades == k + 1)
            if lo.any() and hi.any():
                gap = s[hi].mean() - s[lo].mean()
                penalties.append(torch.clamp(margin - gap, min=0.0))
    if not penalties:
        return s.new_tensor(0.0)
    return torch.stack(penalties).mean()
