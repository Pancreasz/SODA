"""Pure-numpy oracle for the SODA alignment loss math (grades 0..K-1).

These functions define the reference semantics for the three alignment terms and
the EMA prototype update. The Kaggle torch implementation in soda/train/alignment.py
mirrors them and is checked for numerical parity against these on Kaggle.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def severity_from_logits(logits) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    cum = np.cumprod(_sigmoid(logits), axis=1)
    return cum.sum(axis=1)


def l_consist_value(s, s_style) -> float:
    s = np.asarray(s, dtype=float)
    s_style = np.asarray(s_style, dtype=float)
    return float(np.mean((s - s_style) ** 2))


def l_proto_value(z, domains, grades, bank, filled) -> float:
    """Mean over (domain,grade) cells present in the batch of the summed squared
    distance from the batch prototype to same-grade prototypes in OTHER domains
    (read from the EMA bank). Returns 0.0 if no cross-domain partner exists."""
    z = np.asarray(z, dtype=float)
    domains = np.asarray(domains, dtype=int)
    grades = np.asarray(grades, dtype=int)
    bank = np.asarray(bank, dtype=float)
    filled = np.asarray(filled, dtype=bool)
    num_domains = bank.shape[0]
    terms = []
    for d in np.unique(domains):
        for k in np.unique(grades[domains == d]):
            mask = (domains == d) & (grades == k)
            p_hat = z[mask].mean(axis=0)
            acc = 0.0
            found = False
            for dp in range(num_domains):
                if dp == d or not filled[dp, k]:
                    continue
                acc += float(np.sum((p_hat - bank[dp, k]) ** 2))
                found = True
            if found:
                terms.append(acc)
    return float(np.mean(terms)) if terms else 0.0


def l_mono_value(s, domains, grades, margin, num_domains, num_classes: int = 5) -> float:
    """Per-domain hinge enforcing mean-severity(grade k+1) - mean-severity(grade k) >= margin.
    Averaged over contributing (domain, k) pairs (both k and k+1 present in that domain)."""
    s = np.asarray(s, dtype=float)
    domains = np.asarray(domains, dtype=int)
    grades = np.asarray(grades, dtype=int)
    penalties = []
    for d in range(num_domains):
        for k in range(num_classes - 1):
            lo = (domains == d) & (grades == k)
            hi = (domains == d) & (grades == k + 1)
            if lo.any() and hi.any():
                gap = s[hi].mean() - s[lo].mean()
                penalties.append(max(0.0, margin - gap))
    return float(np.mean(penalties)) if penalties else 0.0


def ema_update(bank, filled, z, domains, grades, momentum):
    """Momentum-update bank cells present in the batch. First observation of a cell
    initializes it directly; subsequent updates use m*old + (1-m)*batch_mean."""
    bank = np.array(bank, dtype=float, copy=True)
    filled = np.array(filled, dtype=bool, copy=True)
    z = np.asarray(z, dtype=float)
    domains = np.asarray(domains, dtype=int)
    grades = np.asarray(grades, dtype=int)
    for d in np.unique(domains):
        for k in np.unique(grades[domains == d]):
            mask = (domains == d) & (grades == k)
            batch_mean = z[mask].mean(axis=0)
            if filled[d, k]:
                bank[d, k] = momentum * bank[d, k] + (1.0 - momentum) * batch_mean
            else:
                bank[d, k] = batch_mean
                filled[d, k] = True
    return bank, filled
