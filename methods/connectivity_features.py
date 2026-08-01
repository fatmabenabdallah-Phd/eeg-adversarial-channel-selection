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
    epochs: np.ndarray, channel_names: List[str], sfreq: float,
) -> np.ndarray:
    """epochs: (n_epochs, n_channels, n_samples) -- same convention as
    baselines.baselines.extract_band_power_features. Returns
    (n_epochs, n_channels * N_BANDS): for each epoch, each channel, each
    band, the channel's MEAN PLV to every other channel (a single
    scalar summarizing that channel's overall connectivity strength in
    that band -- not the full pairwise matrix, to keep the exact same
    per-channel-block shape convention as band-power features).
    """
    from grn_balladeer.connectivity.phase_connectivity import (
        extract_band_signal, compute_instantaneous_phase, compute_plv_matrix,
    )

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
            plv = compute_plv_matrix(phases)  # (n_channels, n_channels), diagonal = 1
            np.fill_diagonal(plv, 0.0)  # exclude self-connectivity from the per-channel mean
            mean_plv_per_channel = plv.mean(axis=1)  # (n_channels,) -- mean PLV to all OTHER channels
            for c in range(n_channels):
                out[e, c * N_BANDS + b_idx] = mean_plv_per_channel[c]

    return out
