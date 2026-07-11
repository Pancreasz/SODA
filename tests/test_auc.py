import numpy as np

from soda.metrics.auc import macro_ovo_auc, bootstrap_ci
from soda.metrics.ordinal import mean_absolute_grade_error


def _confident_scores(y_true, num_classes=5, correct_mass=0.9):
    n = len(y_true)
    s = np.full((n, num_classes), (1 - correct_mass) / (num_classes - 1))
    s[np.arange(n), y_true] = correct_mass
    return s


def test_auc_near_one_for_confident_correct():
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    auc = macro_ovo_auc(y, _confident_scores(y))
    assert auc > 0.99


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, 200)
    y_pred = y_true.copy()
    flip = rng.random(200) < 0.3
    y_pred[flip] = rng.integers(0, 5, flip.sum())
    point, lo, hi = bootstrap_ci(mean_absolute_grade_error, y_true, y_pred,
                                 n=500, seed=1)
    assert lo <= point <= hi
    assert hi - lo > 0
