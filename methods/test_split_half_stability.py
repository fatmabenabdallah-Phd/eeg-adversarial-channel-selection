"""
methods.test_split_half_stability
====================================
Regression test for split_half_stability_check.

IMPORTANT LESSON documented here (from an earlier failed assertion
attempt): a naive test asserting "spearman_rho(signal case) >
spearman_rho(noise case)" FAILED on first attempt, because with only
2 truly informative channels out of 15, the majority-noise channels'
arbitrary mutual ordering diluted the whole-ranking Spearman
correlation enough that it was NOT reliably higher than a pure-noise
control run. The two informative channels were, in fact, perfectly
recovered in BOTH halves' top-5 lists -- the right test is whether the
SPECIFIC known-informative channels appear in both halves' top_k, not
whether the aggregate Spearman correlation is high.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from methods.split_half_stability import split_half_stability_check


def test_split_half_recovers_known_informative_channels_in_both_halves():
    n_channels, n_bands, n_subjects = 15, 5, 121
    channel_names = [f"CH{i}" for i in range(n_channels)]

    rng = np.random.RandomState(5)
    X = rng.randn(n_subjects, n_channels * n_bands) * 1.0
    # CH1 (block 5:10) and CH10 (block 50:55) are the only informative channels
    y = (X[:, 5:10].sum(axis=1) + X[:, 50:55].sum(axis=1) + 0.5 * rng.randn(n_subjects) > 0).astype(int)

    result = split_half_stability_check(X, y, channel_names, n_bands, n_permutations=1000, seed=42, top_k=5)

    assert "CH1" in result["top_k_half_a"] and "CH1" in result["top_k_half_b"], (
        f"expected CH1 (a known informative channel) in both halves' top-5, "
        f"got A={result['top_k_half_a']}, B={result['top_k_half_b']}"
    )
    assert "CH10" in result["top_k_half_a"] and "CH10" in result["top_k_half_b"], (
        f"expected CH10 (a known informative channel) in both halves' top-5, "
        f"got A={result['top_k_half_a']}, B={result['top_k_half_b']}"
    )


if __name__ == "__main__":
    test_split_half_recovers_known_informative_channels_in_both_halves()
    print("test_split_half_recovers_known_informative_channels_in_both_halves: PASSED")
