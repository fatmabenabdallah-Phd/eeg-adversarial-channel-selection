"""
methods.permutation_screening
================================
Piste 7: a cheap, high-power univariate screen for channel-level ADHD
signal, run BEFORE (or instead of) the expensive full ML pipeline
(retrain-a-model-per-subset that methods/adversarial_importance.py and
the other baselines all do). Inspired by cluster-permutation testing
in classical cognitive neuroscience (as used via MNE), but per-channel
rather than per-sensor-cluster-in-time, and on the same aggregated
band-power features as the rest of this project for direct
comparability.

Rationale: retraining an RF at every channel subset size, every fold,
every method is what made the real CGX/Nasrabadi/transfer runs take
hours. A univariate permutation test per channel is orders of
magnitude cheaper (no model retraining at all -- just a test
statistic and label-shuffling), so it can run at MUCH higher
permutation counts (thousands, not tens), giving it a chance to detect
a smaller true effect than the coarser subset-retraining pipeline
could resolve with only 5-10 folds.

This deliberately does NOT replace the existing pipeline -- it answers
a different, prior question: "is there ANY univariate channel-level
signal at all, at high statistical power, before spending hours
retraining models on subsets." A null result here would be strong
independent evidence that the channel-selection null result is real
and not a fold-count/power artifact; a positive result here despite
the subset-retraining pipeline's null result would suggest the model-
retraining approach itself (not the underlying data) is the
insensitive part.
"""

from __future__ import annotations

from typing import List

import numpy as np


def _channel_feature_slice(channel_idx: int, n_bands: int, n_total_features: int) -> slice:
    start = channel_idx * n_bands
    end = start + n_bands
    if end > n_total_features:
        raise ValueError(
            f"_channel_feature_slice: channel_idx={channel_idx} with n_bands={n_bands} "
            f"exceeds n_total_features={n_total_features}."
        )
    return slice(start, end)


def _channel_test_statistic(X_channel_block: np.ndarray, y: np.ndarray) -> float:
    """Test statistic for one channel's feature block (n_subjects,
    n_bands): the sum, across the channel's bands, of the squared
    difference in means between the two label groups, each normalized
    by pooled variance (a multivariate extension of a t-statistic,
    sensitive to any band in the block carrying signal, not just an
    omnibus mean)."""
    g0 = X_channel_block[y == 0]
    g1 = X_channel_block[y == 1]
    pooled_var = (g0.var(axis=0, ddof=1) + g1.var(axis=0, ddof=1)) / 2.0
    pooled_var = np.maximum(pooled_var, 1e-12)
    mean_diff_sq = (g0.mean(axis=0) - g1.mean(axis=0)) ** 2
    return float((mean_diff_sq / pooled_var).sum())


def permutation_screen_channels(
    X: np.ndarray, y: np.ndarray, channel_names: List[str], n_bands: int,
    n_permutations: int = 5000, seed: int = 42,
) -> "np.ndarray":
    """Runs a high-power (n_permutations, default 5000 -- two to three
    orders of magnitude more than the 5-10 folds the subset-retraining
    pipeline affords) univariate permutation test for every channel
    independently. Returns a structured array (channel_name,
    observed_stat, p_value), sorted by p_value ascending (most likely
    real signal first). No FDR correction is applied here (that is the
    caller's responsibility across channels, exactly as elsewhere in
    this project's Wilcoxon+FDR convention) -- returning raw p-values
    keeps this function's output composable with the project's
    existing multipletests-based correction step.
    """
    rng = np.random.RandomState(seed)
    n_channels = len(channel_names)
    n_total_features = X.shape[1]

    observed_stats = np.zeros(n_channels)
    for c in range(n_channels):
        ch_slice = _channel_feature_slice(c, n_bands, n_total_features)
        observed_stats[c] = _channel_test_statistic(X[:, ch_slice], y)

    # Permutation null: shuffle y (not X), recompute every channel's
    # statistic under each shuffle -- this is what makes the test cheap
    # (a shuffle + closed-form statistic, no model fit) even at
    # thousands of permutations.
    exceed_counts = np.zeros(n_channels)
    for p in range(n_permutations):
        y_perm = rng.permutation(y)
        for c in range(n_channels):
            ch_slice = _channel_feature_slice(c, n_bands, n_total_features)
            perm_stat = _channel_test_statistic(X[:, ch_slice], y_perm)
            if perm_stat >= observed_stats[c]:
                exceed_counts[c] += 1

    p_values = (exceed_counts + 1) / (n_permutations + 1)  # +1 avoids p=0

    dtype = [("channel_name", "U16"), ("observed_stat", "f8"), ("p_value", "f8")]
    out = np.zeros(n_channels, dtype=dtype)
    out["channel_name"] = channel_names
    out["observed_stat"] = observed_stats
    out["p_value"] = p_values
    return np.sort(out, order="p_value")
