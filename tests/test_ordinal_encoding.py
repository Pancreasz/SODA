import numpy as np
from soda.ordinal_encoding import (
    corn_encode_labels, corn_cumprobs_to_class_probs, corn_decode_grade, corn_severity,
)


def test_encode_extended_binary():
    y = np.array([0, 3, 4])
    lv = corn_encode_labels(y, num_classes=5)
    assert lv.shape == (3, 4)
    assert lv[0].tolist() == [0, 0, 0, 0]   # grade 0: nothing above
    assert lv[1].tolist() == [1, 1, 1, 0]   # grade 3: >0,>1,>2 true
    assert lv[2].tolist() == [1, 1, 1, 1]   # grade 4: all


def test_cumprobs_to_class_probs_sum_to_one():
    cum = np.array([[0.9, 0.8, 0.7, 0.2]])   # monotone cumulative P(y>k)
    cp = corn_cumprobs_to_class_probs(cum)
    assert cp.shape == (1, 5)
    np.testing.assert_allclose(cp.sum(axis=1), 1.0, atol=1e-9)
    # P(y=0)=1-0.9=0.1 ; P(y=4)=0.2
    np.testing.assert_allclose(cp[0, 0], 0.1, atol=1e-9)
    np.testing.assert_allclose(cp[0, 4], 0.2, atol=1e-9)


def test_decode_grade_and_severity():
    cum = np.array([[0.9, 0.8, 0.7, 0.2], [0.3, 0.1, 0.05, 0.0]])
    np.testing.assert_array_equal(corn_decode_grade(cum), np.array([3, 0]))
    np.testing.assert_allclose(corn_severity(cum), np.array([2.6, 0.45]), atol=1e-9)
