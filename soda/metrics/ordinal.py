"""Ordinal metrics for DR grading (grades 0..4)."""
from __future__ import annotations

import numpy as np


def quadratic_weighted_kappa(y_true, y_pred, num_classes: int = 5) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    O = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1
    idx = np.arange(num_classes)
    w = (idx[:, None] - idx[None, :]) ** 2 / (num_classes - 1) ** 2
    act = O.sum(axis=1)
    pred = O.sum(axis=0)
    n = O.sum()
    if n == 0:
        return 0.0
    E = np.outer(act, pred) / n
    den = (w * E).sum()
    if den == 0:
        return 0.0
    return float(1.0 - (w * O).sum() / den)


def catastrophic_error_rate(y_true, y_pred, threshold: int = 2) -> float:
    d = np.abs(np.asarray(y_true, dtype=int) - np.asarray(y_pred, dtype=int))
    return float((d >= threshold).mean())


def mean_absolute_grade_error(y_true, y_pred) -> float:
    d = np.abs(np.asarray(y_true, dtype=int) - np.asarray(y_pred, dtype=int))
    return float(d.mean())
