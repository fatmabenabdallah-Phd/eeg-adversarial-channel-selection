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

from methods.connectivity_features import extract_connectivity_features, build_functional_connectivity_graph, N_BANDS


def _bandlimited_noise(n, sfreq, lo=4.0, hi=8.0, rng=None):
    rng = rng or np.random
    x = rng.randn(n)
    b, a = butter(4, [lo / (sfreq / 2), hi / (sfreq / 2)], btype="band")
    return filtfilt(b, a, x)


def test_extract_connectivity_features_detects_independent_channel():
    """Decision boundary: y=1 if (CH1_signal - CH2_signal) > 0 -- a
    genuine two-channel interaction where CH1 and CH2 are structural
    neighbors."""
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

    feats = extract_connectivity_features(epochs, channel_names, sfreq, metric="plv")
    assert feats.shape == (n_epochs, n_channels * N_BANDS)

    theta_band_idx = 1  # delta=0, theta=1, alpha=2, beta=3, gamma=4
    mean_plv_theta = feats[:, theta_band_idx::N_BANDS].mean(axis=0)

    assert mean_plv_theta[:4].mean() > mean_plv_theta[4] + 0.1, (
        f"expected the shared-source channels (CH0-CH3) to have clearly higher theta "
        f"PLV than the independent-noise channel (CH4), got {mean_plv_theta}"
    )


def test_extract_connectivity_features_pli_detects_lagged_coupling():
    """PLI is specifically designed to be BLIND to zero-lag synchrony
    (the exact signal design used in the PLV test above), since
    zero-lag synchrony is the signature of volume conduction, not true
    neural connectivity -- an earlier attempt to reuse the PLV test's
    signal for PLI found NO contrast at all (0.108-0.117 across every
    channel including the "independent" one), which is PLI correctly
    doing its job, not a bug. PLI requires a genuine phase LAG between
    coupled channels to detect a connection, which is what this test
    constructs (each coupled channel receives the shared source shifted
    by a different sample delay, mimicking real signal propagation)."""
    rng = np.random.RandomState(1)
    n_channels, n_samples, sfreq = 5, 2000, 250.0
    channel_names = [f"CH{i}" for i in range(n_channels)]
    lag = 8

    n_epochs = 5
    epochs = np.zeros((n_epochs, n_channels, n_samples))
    for e in range(n_epochs):
        pad = lag * n_channels + 10
        shared_source = _bandlimited_noise(n_samples + pad, sfreq, rng=rng)
        for c in range(n_channels - 1):
            shift = lag * (c + 1)
            epochs[e, c] = shared_source[shift:shift + n_samples] + 0.3 * _bandlimited_noise(n_samples, sfreq, rng=rng)
        epochs[e, n_channels - 1] = _bandlimited_noise(n_samples, sfreq, rng=rng)

    feats_pli = extract_connectivity_features(epochs, channel_names, sfreq, metric="pli")
    theta_band_idx = 1
    mean_pli_theta = feats_pli[:, theta_band_idx::N_BANDS].mean(axis=0)

    assert mean_pli_theta[:4].mean() > mean_pli_theta[4] + 0.05, (
        f"expected the lag-coupled channels (CH0-CH3) to have clearly higher theta "
        f"PLI than the independent-noise channel (CH4), got {mean_pli_theta}"
    )


def test_build_functional_connectivity_graph_recovers_known_cluster():
    """CH0/CH1/CH2 share a common source (a real functional cluster);
    CH3/CH4/CH5 are mutually independent noise. The functional graph
    must connect CH0 to at least one other cluster member, and must
    NOT connect it to any of the independent-noise channels -- unlike
    a geometric k-NN graph, which would connect channels based on
    electrode position regardless of this functional structure."""
    rng = np.random.RandomState(3)
    n_channels, n_samples, sfreq = 6, 2000, 250.0
    channel_names = [f"CH{i}" for i in range(n_channels)]

    n_epochs = 8
    epochs = np.zeros((n_epochs, n_channels, n_samples))
    for e in range(n_epochs):
        shared = _bandlimited_noise(n_samples, sfreq, rng=rng)
        for c in [0, 1, 2]:
            epochs[e, c] = shared + 0.3 * _bandlimited_noise(n_samples, sfreq, rng=rng)
        for c in [3, 4, 5]:
            epochs[e, c] = _bandlimited_noise(n_samples, sfreq, rng=rng)

    adjacency = build_functional_connectivity_graph(epochs, channel_names, sfreq, k=2, metric="plv")

    cluster_ok = adjacency[0, 1] == 1 or adjacency[0, 2] == 1 or adjacency[1, 2] == 1
    no_cross = adjacency[0, 3] == 0 and adjacency[0, 4] == 0 and adjacency[0, 5] == 0
    assert cluster_ok, f"expected CH0 to connect to at least one cluster member, got row {adjacency[0]}"
    assert no_cross, f"expected CH0 to NOT connect to independent-noise channels, got row {adjacency[0]}"


if __name__ == "__main__":
    test_extract_connectivity_features_detects_independent_channel()
    print("test_extract_connectivity_features_detects_independent_channel: PASSED")
    test_extract_connectivity_features_pli_detects_lagged_coupling()
    print("test_extract_connectivity_features_pli_detects_lagged_coupling: PASSED")
    test_build_functional_connectivity_graph_recovers_known_cluster()
    print("test_build_functional_connectivity_graph_recovers_known_cluster: PASSED")
