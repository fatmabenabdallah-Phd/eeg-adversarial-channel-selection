"""
methods.connectivity_features
================================
Piste 5: instead of per-channel band POWER (extract_band_power_features),
extract per-channel connectivity STRENGTH -- a channel's mean Phase
Locking Value (PLV) to every other channel, per frequency band. If the
relevant ADHD signal lives in inter-channel relationships (network
role) rather than in a channel's own isolated power, this
representation can reveal importance invisible to the power-based
pipeline that found no significant channel-selection effect on either
CGX or Nasrabadi.

Design choice made for direct reuse of the entire existing pipeline
(methods.adversarial_importance, baselines.channel_selection_baselines):
produces exactly N_BANDS features PER CHANNEL, the same shape
convention as extract_band_power_features (n_channels*n_bands columns,
contiguous per-channel blocks) -- so rank_channels,
rank_channels_graph_constrained, and every baseline work UNCHANGED on
this feature matrix, with no code duplication.

Reuses grn_balladeer.connectivity.phase_connectivity (the ACTIVE
module -- connectivity/plv.py is explicitly marked a stale, unused
duplicate in its own docstring and must NOT be imported).

Convention note: phase_connectivity's functions operate on a SINGLE
epoch shaped (n_channels, n_samples) -- the TRANSPOSE of
extract_band_power_features's epoch convention
(n_epochs, n_channels, n_samples per epoch, i.e. epoch[e] is already
(n_channels, n_samples), so no transpose is actually needed at the
single-epoch level; only double-checked here since the OLD plv.py used
the opposite (n_samples, n_channels) convention and mixing them up
would silently corrupt every PLV value without erroring).
"""

from __future__ import annotations

from typing import List

import numpy as np

N_BANDS = 5
BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
]


def extract_connectivity_features(
    epochs: np.ndarray, channel_names: List[str], sfreq: float, metric: str = "plv",
) -> np.ndarray:
    """epochs: (n_epochs, n_channels, n_samples) -- same convention as
    baselines.baselines.extract_band_power_features. Returns
    (n_epochs, n_channels * N_BANDS): for each epoch, each channel, each
    band, the channel's MEAN connectivity to every other channel (a
    single scalar summarizing that channel's overall connectivity
    strength in that band -- not the full pairwise matrix, to keep the
    exact same per-channel-block shape convention as band-power
    features).

    metric: "plv" (default, Phase Locking Value) or "pli" (Phase Lag
    Index). PLI is less sensitive to volume conduction than PLV, since
    it discards zero-lag synchrony (a signal that spreads electrically
    across the scalp can create spurious near-zero-lag "synchrony"
    between nearby electrodes that has nothing to do with true neural
    connectivity; PLV can be inflated by this, PLI is specifically
    designed to be robust to it). Running the same channel-importance
    analysis with metric="pli" as a robustness check against a
    metric="plv" finding is recommended before treating any PLV-based
    result as reliable, precisely because of this volume-conduction
    risk.
    """
    from grn_balladeer.connectivity.phase_connectivity import (
        extract_band_signal, compute_instantaneous_phase, compute_plv_matrix, compute_pli_matrix,
    )

    if metric not in ("plv", "pli"):
        raise ValueError(f"extract_connectivity_features: metric must be 'plv' or 'pli', got {metric!r}")
    compute_matrix = compute_plv_matrix if metric == "plv" else compute_pli_matrix

    n_epochs, n_channels, n_samples = epochs.shape
    if n_channels != len(channel_names):
        raise ValueError(
            f"extract_connectivity_features: epochs has {n_channels} channels but "
            f"{len(channel_names)} channel_names were given."
        )

    out = np.zeros((n_epochs, n_channels * N_BANDS), dtype=np.float32)

    for e in range(n_epochs):
        epoch_data = epochs[e]  # (n_channels, n_samples) -- already the right convention
        for b_idx, (band_name, lo, hi) in enumerate(BANDS):
            band_signal = extract_band_signal(epoch_data, (lo, hi), sfreq)
            phases = compute_instantaneous_phase(band_signal)
            conn_matrix = compute_matrix(phases)  # (n_channels, n_channels)
            np.fill_diagonal(conn_matrix, 0.0)  # exclude self-connectivity from the per-channel mean
            # (PLV's diagonal is already 1 before this, PLI's is already 0 -- fill_diagonal(0) is a
            # no-op for PLI and the necessary correction for PLV; harmless either way)
            mean_conn_per_channel = conn_matrix.mean(axis=1)  # (n_channels,)
            for c in range(n_channels):
                out[e, c * N_BANDS + b_idx] = mean_conn_per_channel[c]

    return out
