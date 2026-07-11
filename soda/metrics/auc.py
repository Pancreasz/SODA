"""Macro one-vs-one AUC and a generic bootstrap CI."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_ovo_auc(y_true, y_score, num_classes: int = 5) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    return float(
        roc_auc_score(
            y_true, y_score, average="macro", multi_class="ovo",
            labels=list(range(num_classes)),
        )
    )


def bootstrap_ci(metric_fn, y_true, y_hat, n: int = 1000, alpha: float = 0.05,
                 seed: int = 0):
    y_true = np.asarray(y_true)
    y_hat = np.asarray(y_hat)
    rng = np.random.default_rng(seed)
    m = len(y_true)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        stats.append(metric_fn(y_true[idx], y_hat[idx]))
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    point = float(metric_fn(y_true, y_hat))
    return point, lo, hi
