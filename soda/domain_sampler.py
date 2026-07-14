"""Domain-balanced batch index sampler (pure python, no torch).

Each batch pulls `per_domain` indices from each of `domains_per_batch` distinct
source domains so cross-domain same-grade pairs exist for L_proto. Torch's
BatchSampler wrapper (soda/train/data.py) reuses this to order a DataLoader.
"""
from __future__ import annotations

import random
from collections import defaultdict


def domain_balanced_batches(domain_of_index, domains_per_batch, per_domain, seed=0):
    if per_domain < 1 or domains_per_batch < 1:
        raise ValueError("per_domain and domains_per_batch must be >= 1")
    rng = random.Random(seed)
    by_domain = defaultdict(list)
    for idx, dom in enumerate(domain_of_index):
        by_domain[dom].append(idx)
    pools = {d: rng.sample(v, len(v)) for d, v in by_domain.items()}  # shuffled copies

    batches = []
    while True:
        # domains that still have enough indices for one draw
        available = [d for d, pool in pools.items() if len(pool) >= per_domain]
        if len(available) < domains_per_batch:
            break
        chosen = rng.sample(available, domains_per_batch)
        batch = []
        for d in chosen:
            for _ in range(per_domain):
                batch.append(pools[d].pop())
        batches.append(batch)
    return batches
