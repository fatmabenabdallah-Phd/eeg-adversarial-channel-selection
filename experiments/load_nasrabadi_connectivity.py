"""
experiments.load_nasrabadi_connectivity
==========================================
Identical to run_nasrabadi_experiment.load_all_subjects_nasrabadi
EXCEPT the final feature-extraction step: extract_connectivity_features
(PLV-based, piste 5) instead of extract_band_power_features.
build_subject_epoch_arrays already returns the raw (n_channels,
n_samples) epoch tensor needed, so only the last step differs.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.baselines import aggregate_epochs_to_subject
from experiments.load_nasrabadi import (
    load_nasrabadi_csv, build_subject_epoch_arrays, NASRABADI_CHANNELS_MODERN,
)
from methods.connectivity_features import extract_connectivity_features


def load_all_subjects_nasrabadi_connectivity(
    csv_path: str, sfreq: float = 128.0, window_samples: int = 512,
):
    """Same loading as load_all_subjects_nasrabadi, but extracts
    PLV-based connectivity features per epoch instead of band-power.
    Returns (X, y, subject_ids)."""
    df = load_nasrabadi_csv(csv_path)
    epochs_array, epoch_subject_ids, subject_labels = build_subject_epoch_arrays(
        df, sfreq=sfreq, window_samples=window_samples
    )
    connectivity_feats = extract_connectivity_features(epochs_array, NASRABADI_CHANNELS_MODERN, sfreq=sfreq)
    X_subject, subject_ids = aggregate_epochs_to_subject(connectivity_feats, epoch_subject_ids.tolist())
    y_subject = np.array([subject_labels[sid] for sid in subject_ids])
    return X_subject, y_subject, subject_ids


if __name__ == "__main__":
    print(
        "Real Colab usage:\n"
        "  from experiments.load_nasrabadi_connectivity import load_all_subjects_nasrabadi_connectivity\n"
        "  X_conn, y_conn, subject_ids_conn = load_all_subjects_nasrabadi_connectivity('/path/to/nasrabadi.csv')\n"
    )
