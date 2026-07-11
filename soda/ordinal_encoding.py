"""Pure-numpy CORN ordinal label/probability helpers (grades 0..K-1).

CORN outputs K-1 cumulative probabilities cum[:,k] = P(y > k), which are
monotonically non-increasing in k. These helpers convert to per-class
probabilities (for AUC), a predicted grade, and a scalar severity score.
"""
from __future__ import annotations

import numpy as np


def corn_encode_labels(y, num_classes: int = 5) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    ks = np.arange(num_classes - 1)[None, :]
    return (y[:, None] > ks).astype(int)


def corn_cumprobs_to_class_probs(cum) -> np.ndarray:
    cum = np.asarray(cum, dtype=float)
    n = cum.shape[0]
    k = cum.shape[1] + 1
    ones = np.ones((n, 1))
    zeros = np.zeros((n, 1))
    left = np.concatenate([ones, cum], axis=1)     # P(y > k-1), with P(y>-1)=1
    right = np.concatenate([cum, zeros], axis=1)    # P(y > k),   with P(y>K-1)=0
    probs = left - right                            # P(y = k)
    return np.clip(probs, 0.0, 1.0)


def corn_decode_grade(cum, threshold: float = 0.5) -> np.ndarray:
    cum = np.asarray(cum, dtype=float)
    return (cum > threshold).sum(axis=1).astype(int)


def corn_severity(cum) -> np.ndarray:
    cum = np.asarray(cum, dtype=float)
    return cum.sum(axis=1)
