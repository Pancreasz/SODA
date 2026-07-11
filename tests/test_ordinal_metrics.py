import numpy as np
from sklearn.metrics import cohen_kappa_score

from soda.metrics.ordinal import (
    quadratic_weighted_kappa,
    catastrophic_error_rate,
    mean_absolute_grade_error,
)

Y_TRUE = np.array([0, 1, 2, 3, 4, 2, 1, 0, 4, 3])
Y_PRED = np.array([0, 2, 2, 1, 4, 3, 1, 0, 2, 3])


def test_qwk_matches_sklearn_oracle():
    ours = quadratic_weighted_kappa(Y_TRUE, Y_PRED, num_classes=5)
    ref = cohen_kappa_score(Y_TRUE, Y_PRED, weights="quadratic", labels=[0, 1, 2, 3, 4])
    assert abs(ours - ref) < 1e-9


def test_qwk_perfect_is_one():
    assert abs(quadratic_weighted_kappa(Y_TRUE, Y_TRUE) - 1.0) < 1e-9


def test_catastrophic_error_rate():
    # diffs: 0,1,0,2,0,1,0,0,2,0 -> two entries with |diff|>=2 out of 10 -> 0.2
    assert abs(catastrophic_error_rate(Y_TRUE, Y_PRED) - 0.2) < 1e-9


def test_mae():
    assert abs(mean_absolute_grade_error(Y_TRUE, Y_PRED) - 0.6) < 1e-9
