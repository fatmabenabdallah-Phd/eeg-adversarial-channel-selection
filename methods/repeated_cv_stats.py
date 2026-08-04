"""
methods.repeated_cv_stats
============================
Piste 1 (highest-priority improvement direction): repeated k-fold
cross-validation, combined with a statistically CORRECTED significance
test -- not a naive t-test/Wilcoxon on the pooled repeat*fold
differences, which would treat correlated observations (the same
underlying data reused across repeats) as independent and inflate the
apparent significance.

Implements the Nadeau-Bengio correction (Nadeau & Bengio, 2003,
"Inference for the generalization error", Machine Learning 52(3):239-281),
the standard correction for this exact problem. Motivated directly by
this project's own findings: adversarial_isolated showed Cohen's d=0.87
(Nasrabadi connectivity, k=5, uncorrected p=0.016) and d~1.0 (TDBRAIN,
k=5-10), both promising but not surviving FDR correction at only 10
folds -- repeated CV is a legitimate way to increase effective power on
the SAME underlying comparison, without adding a new researcher degree
of freedom (unlike trying several different method variants until one
"works").
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def nadeau_bengio_corrected_ttest(
    differences: np.ndarray, n_train: int, n_test: int,
) -> dict:
    """Corrected paired t-test for repeated k-fold CV differences.

    differences: 1-D array of length n_repeats * n_folds, each entry
    being one fold's (method_a_score - method_b_score) from one
    repeat's cross-validation run. Must be the FULL set of repeat*fold
    differences, not pre-averaged per repeat.

    n_train, n_test: number of subjects in the training and test sets
    of a SINGLE fold (assumed constant across folds/repeats, as in
    standard k-fold CV with roughly equal-sized folds).

    Returns dict with corrected t-statistic, p-value, and the naive
    (uncorrected) t-test's p-value for direct comparison -- the naive
    p-value is ALWAYS more optimistic (smaller) than the corrected one
    for the same data, illustrating why the correction matters.
    """
    differences = np.asarray(differences, dtype=float)
    n = len(differences)
    if n < 2:
        raise ValueError("nadeau_bengio_corrected_ttest: need at least 2 differences.")

    mean_diff = differences.mean()
    var_diff = differences.var(ddof=1)

    if var_diff == 0:
        return {
            "corrected_t": np.inf if mean_diff != 0 else 0.0,
            "corrected_p": 0.0 if mean_diff != 0 else 1.0,
            "naive_p": 0.0 if mean_diff != 0 else 1.0,
            "mean_diff": float(mean_diff), "n": n,
        }

    correction_factor = (1.0 / n) + (n_test / n_train)
    corrected_t = mean_diff / np.sqrt(correction_factor * var_diff)
    corrected_p = 2 * (1 - stats.t.cdf(np.abs(corrected_t), df=n - 1))

    naive_t, naive_p = stats.ttest_1samp(differences, popmean=0.0)

    return {
        "corrected_t": float(corrected_t),
        "corrected_p": float(corrected_p),
        "naive_t": float(naive_t),
        "naive_p": float(naive_p),
        "mean_diff": float(mean_diff),
        "n": n,
    }
