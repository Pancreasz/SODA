import numpy as np
from soda.alignment_math import (
    severity_from_logits, l_consist_value, l_proto_value, l_mono_value, ema_update,
)
from soda.ordinal_encoding import corn_severity


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_severity_matches_corn_severity_oracle():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(7, 4))
    cum = np.cumprod(_sigmoid(logits), axis=1)
    np.testing.assert_allclose(severity_from_logits(logits), corn_severity(cum), atol=1e-12)


def test_consist_is_mean_squared_severity_gap():
    s = np.array([2.0, 1.0, 3.0])
    ss = np.array([2.5, 1.0, 2.0])
    # mean([0.25, 0, 1]) = 0.4166666...
    assert abs(l_consist_value(s, ss) - (0.25 + 0.0 + 1.0) / 3.0) < 1e-12


def test_proto_pulls_batch_proto_to_other_domain_bank():
    # D=2 embed dim. Grade 1 present in domain 0 (batch), bank has grade 1 in domain 1.
    z = np.array([[1.0, 0.0], [1.0, 0.0]])   # two domain-0 grade-1 samples -> batch proto [1,0]
    domains = np.array([0, 0])
    grades = np.array([1, 1])
    bank = np.zeros((2, 5, 2))
    bank[1, 1] = np.array([0.0, 1.0])        # domain-1 grade-1 prototype
    filled = np.zeros((2, 5), dtype=bool)
    filled[1, 1] = True
    # single cross-domain term: ||[1,0]-[0,1]||^2 = 2.0
    assert abs(l_proto_value(z, domains, grades, bank, filled) - 2.0) < 1e-12


def test_proto_zero_when_no_cross_domain_partner():
    z = np.array([[1.0, 0.0]])
    domains = np.array([0]); grades = np.array([2])
    bank = np.zeros((2, 5, 2)); filled = np.zeros((2, 5), dtype=bool)
    assert l_proto_value(z, domains, grades, bank, filled) == 0.0


def test_mono_hinge_penalizes_insufficient_gap():
    # domain 0: grade0 severity mean 0.5, grade1 severity mean 0.8 -> gap 0.3 < margin 0.5
    s = np.array([0.5, 0.8]); domains = np.array([0, 0]); grades = np.array([0, 1])
    # penalty max(0, 0.5-0.3)=0.2 ; one contributing pair -> 0.2
    assert abs(l_mono_value(s, domains, grades, margin=0.5, num_domains=2) - 0.2) < 1e-12


def test_mono_zero_when_gap_exceeds_margin():
    s = np.array([0.0, 1.0]); domains = np.array([0, 0]); grades = np.array([0, 1])
    assert l_mono_value(s, domains, grades, margin=0.5, num_domains=2) == 0.0


def test_ema_update_inits_then_momentum():
    bank = np.zeros((2, 5, 2)); filled = np.zeros((2, 5), dtype=bool)
    z = np.array([[1.0, 0.0]]); domains = np.array([0]); grades = np.array([1])
    bank, filled = ema_update(bank, filled, z, domains, grades, momentum=0.9)
    assert filled[0, 1]
    np.testing.assert_allclose(bank[0, 1], [1.0, 0.0])       # first obs initializes directly
    z2 = np.array([[0.0, 1.0]])
    bank, filled = ema_update(bank, filled, z2, domains, grades, momentum=0.9)
    # 0.9*[1,0] + 0.1*[0,1] = [0.9, 0.1]
    np.testing.assert_allclose(bank[0, 1], [0.9, 0.1], atol=1e-12)
