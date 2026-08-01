"""
methods.test_connectivity_features
=====================================
Regression test for extract_connectivity_features, using band-limited
noise (not pure sinusoids -- a first attempt using fixed-frequency
sine waves gave PLV~1 for every channel regardless of phase offset,
since two constant-frequency signals have a constant, hence perfectly
"locked", phase difference no matter the offset; band-limited noise is
the correct synthetic signal for a PLV test, since its phase
relationship only stays consistent over time if the signals share a
common source).
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/claude")  # for grn_balladeer -- see the tiret/underscore note elsewhere

import numpy as np
from scipy.signal import butter, filtfilt

from methods.connectivity_features import extract_connectivity_features, N_BANDS


def _bandlimited_noise(n, sfreq, lo=4.0, hi=8.0, rng=None):
    rng = rng or np.random
    x = rng.randn(n)
    b, a = butter(4, [lo / (sfreq / 2), hi / (sfreq / 2)], btype="band")
    return filtfilt(b, a, x)


def test_extract_connectivity_features_detects_independent_channel():
    rng = np.random.RandomState(1)
    n_channels, n_samples, sfreq = 5, 2000, 250.0
    channel_names = [f"CH{i}" for i in range(n_channels)]

    n_epochs = 5
    epochs = np.zeros((n_epochs, n_channels, n_samples))
    for e in range(n_epochs):
        shared_source = _bandlimited_noise(n_samples, sfreq, rng=rng)
        for c in range(n_channels - 1):
            epochs[e, c] = shared_source + 0.3 * _bandlimited_noise(n_samples, sfreq, rng=rng)
        epochs[e, n_channels - 1] = _bandlimited_noise(n_samples, sfreq, rng=rng)

    feats = extract_connectivity_features(epochs, channel_names, sfreq)
    assert feats.shape == (n_epochs, n_channels * N_BANDS)

    theta_band_idx = 1  # delta=0, theta=1, alpha=2, beta=3, gamma=4
    mean_plv_theta = feats[:, theta_band_idx::N_BANDS].mean(axis=0)

    assert mean_plv_theta[:4].mean() > mean_plv_theta[4] + 0.1, (
        f"expected the shared-source channels (CH0-CH3) to have clearly higher theta "
        f"PLV than the independent-noise channel (CH4), got {mean_plv_theta}"
    )


if __name__ == "__main__":
    test_extract_connectivity_features_detects_independent_channel()
    print("test_extract_connectivity_features_detects_independent_channel: PASSED")
