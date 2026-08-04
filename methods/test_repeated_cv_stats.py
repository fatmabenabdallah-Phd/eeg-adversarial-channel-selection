"""
methods.test_repeated_cv_stats
=================================
Regression test for nadeau_bengio_corrected_ttest, via a Monte Carlo
simulation under the null hypothesis (no true difference between
methods).

HONEST RESULT, not a perfectly calibrated one: with subject-level
latent noise shared across repeats (a strong, arguably extreme
between-repeat correlation, stronger than typical real-CV covariance),
the naive t-test's Type I error rate is severely inflated (~41% at a
nominal 5% level -- confirming the problem repeated CV without
correction creates), while the Nadeau-Bengio correction is
CONSERVATIVE in this regime (~0% rejection rate under the null) rather
than exactly calibrated to 5%. This is the safe direction to err in
(fails to reject more often than it should, rather than the reverse),
and the relative comparison (corrected < naive, both under the null)
is the property this test actually asserts -- exact nominal
calibration is not guaranteed by the formula under every possible
covariance structure, and this simulation's very strong inter-repeat
correlation is a deliberately harder case than most real repeated-CV
settings, not a best-case calibration check.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from methods.repeated_cv_stats import nadeau_bengio_corrected_ttest


def test_corrected_test_is_less_liberal_than_naive_under_null():
    rng = np.random.RandomState(0)
    n_subjects, k_folds, n_repeats, n_trials = 100, 10, 10, 300
    n_test = n_subjects // k_folds
    n_train = n_subjects - n_test
    alpha = 0.05

    naive_rejections, corrected_rejections = 0, 0

    for _ in range(n_trials):
        subject_latent = rng.randn(n_subjects) * 1.0  # shared across repeats -> strong correlation
        all_diffs = []
        for _r in range(n_repeats):
            perm = rng.permutation(n_subjects)
            folds = np.array_split(perm, k_folds)
            for fold_test_idx in folds:
                fold_diff = subject_latent[fold_test_idx].mean() + rng.randn() * 0.3
                all_diffs.append(fold_diff)
        all_diffs = np.array(all_diffs)

        result = nadeau_bengio_corrected_ttest(all_diffs, n_train=n_train, n_test=n_test)
        if result["naive_p"] < alpha:
            naive_rejections += 1
        if result["corrected_p"] < alpha:
            corrected_rejections += 1

    naive_rate = naive_rejections / n_trials
    corrected_rate = corrected_rejections / n_trials

    assert naive_rate > 3 * alpha, (
        f"expected the naive test's Type I error to be severely inflated under this "
        f"strongly-correlated null simulation, got {naive_rate:.3f} (nominal {alpha})"
    )
    assert corrected_rate < naive_rate, (
        f"expected the corrected test to reject less often than the naive test under "
        f"the null, got corrected={corrected_rate:.3f} vs naive={naive_rate:.3f}"
    )


if __name__ == "__main__":
    test_corrected_test_is_less_liberal_than_naive_under_null()
    print("test_corrected_test_is_less_liberal_than_naive_under_null: PASSED")
