"""
methods.test_permutation_screening
=====================================
Regression test for permutation_screen_channels, at the REAL scale
(30 channels, 121 subjects, matching CGX) to also document real
runtime -- confirmed ~4-5s for 2000 permutations, several orders of
magnitude cheaper than the subset-retraining pipeline's ~hours.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np

from methods.permutation_screening import permutation_screen_channels


def test_permutation_screen_channels_recovers_informative_channel_at_real_scale():
    rng = np.random.RandomState(2)
    n_channels, n_bands, n_subjects = 30, 5, 121  # real CGX scale
    channel_names = [f"CH{i}" for i in range(n_channels)]
    X = rng.randn(n_subjects, n_channels * n_bands) * 1.0
    y = (X[:, 35:40].sum(axis=1) + 0.8 * rng.randn(n_subjects) > 0).astype(int)  # CH7's block, modest effect

    t0 = time.time()
    result = permutation_screen_channels(X, y, channel_names, n_bands, n_permutations=2000, seed=42)
    elapsed = time.time() - t0

    assert result[0]["channel_name"] == "CH7", (
        f"expected CH7 (the only informative channel) to have the lowest p-value, "
        f"got {result[0]['channel_name']}"
    )
    assert result[0]["p_value"] < 0.01, f"expected a clearly significant p-value for CH7, got {result[0]['p_value']}"
    assert elapsed < 60, (
        f"expected this real-scale screen to stay cheap (~seconds), took {elapsed:.1f}s -- "
        f"if this regresses to minutes, the whole point of a 'cheap pre-screen' is lost"
    )


if __name__ == "__main__":
    test_permutation_screen_channels_recovers_informative_channel_at_real_scale()
    print("test_permutation_screen_channels_recovers_informative_channel_at_real_scale: PASSED")
