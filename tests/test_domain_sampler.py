from soda.domain_sampler import domain_balanced_batches

import pytest


def test_batches_have_right_size_and_span_domains():
    # 3 domains, 8 indices each = 24 indices
    domain_of_index = [d for d in range(3) for _ in range(8)]
    batches = domain_balanced_batches(domain_of_index, domains_per_batch=2, per_domain=2, seed=0)
    for b in batches:
        assert len(b) == 4                                  # 2 domains * 2 per domain
        doms = {domain_of_index[i] for i in b}
        assert len(doms) == 2                               # spans exactly 2 domains
    flat = [i for b in batches for i in b]
    assert len(flat) == len(set(flat))                      # no index reused within an epoch


def test_all_indices_valid_and_from_declared_domains():
    domain_of_index = [0, 0, 0, 0, 1, 1, 1, 1]
    batches = domain_balanced_batches(domain_of_index, domains_per_batch=2, per_domain=2, seed=1)
    assert batches                                          # at least one full batch
    for b in batches:
        for i in b:
            assert 0 <= i < len(domain_of_index)


def test_deterministic_under_seed():
    doi = [d for d in range(4) for _ in range(6)]
    a = domain_balanced_batches(doi, 3, 2, seed=42)
    b = domain_balanced_batches(doi, 3, 2, seed=42)
    assert a == b


def test_degenerate_inputs_raise():
    with pytest.raises(ValueError):
        domain_balanced_batches([0, 0, 1, 1], domains_per_batch=0, per_domain=2, seed=0)
    with pytest.raises(ValueError):
        domain_balanced_batches([0, 0, 1, 1], domains_per_batch=2, per_domain=0, seed=0)
