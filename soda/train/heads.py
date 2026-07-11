"""Classifier heads + loss/decoding for softmax and CORN ordinal outputs."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from soda.ordinal_encoding import corn_cumprobs_to_class_probs, corn_decode_grade


class SoftmaxHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class CORNHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes - 1)

    def forward(self, x):
        return self.fc(x)


def compute_loss(head_kind, logits, labels, num_classes):
    if head_kind == "softmax":
        return F.cross_entropy(logits, labels)
    if head_kind == "corn":
        from coral_pytorch.losses import corn_loss
        return corn_loss(logits, labels, num_classes)
    raise ValueError(head_kind)


def _corn_cumprobs(logits):
    # coral_pytorch CORN: chained conditional probs P(y>k)=prod sigmoid up to k
    probas = torch.sigmoid(logits)
    return torch.cumprod(probas, dim=1)


def predict_class_probs(head_kind, logits):
    if head_kind == "softmax":
        return F.softmax(logits, dim=1).detach().cpu().numpy()
    cum = _corn_cumprobs(logits).detach().cpu().numpy()
    return corn_cumprobs_to_class_probs(cum)


def predict_grade(head_kind, logits):
    if head_kind == "softmax":
        return logits.argmax(dim=1).detach().cpu().numpy()
    cum = _corn_cumprobs(logits).detach().cpu().numpy()
    return corn_decode_grade(cum)
