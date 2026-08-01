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
    equalize_epoch_count: bool = True, metric: str = "plv",
):
    """Same loading as load_all_subjects_nasrabadi, but extracts
    PLV-based connectivity features per epoch instead of band-power.

    equalize_epoch_count: CRITICAL, default True. Each individual
    epoch is a fixed window_samples=512-sample window, so the classic
    "PLV needs enough time samples within a window" bias does not
    directly apply here (every epoch's own PLV estimate has the same
    window length). The real, CONFIRMED risk on this dataset is
    different: subjects with more epochs have a LONGER recording
    session, and ADHD subjects have significantly more epochs than
    Controls (38.2+-14.5 vs 30.7+-7.4, Mann-Whitney p=0.003). A longer
    session can introduce session-duration-related drift (fatigue,
    electrode-gel drying/impedance change, accumulating movement
    artifact) that shifts the per-epoch PLV distribution over time,
    independent of any real neural ADHD difference -- and averaging
    over more/fewer epochs changes the resulting subject-level
    estimate's characteristics either way.
    An initial run without this correction found 15/19 channels
    "significant" after FDR, with p-values clustered at the
    permutation-count floor (0.0002) and near-uniform across nearly
    every channel -- the signature of a global artifact, not a
    localized neural effect, which is what motivated this fix.
    When True, every subject's epochs are TRUNCATED (kept in original,
    i.e. temporal, order -- not randomly subsampled) to the same fixed
    count (the minimum epoch count across all subjects), so every
    subject's PLV estimate is built from a directly comparable amount
    AND timing of recording, addressing the duration-drift confound
    more directly than random subsampling would.
    """
    df = load_nasrabadi_csv(csv_path)
    epochs_array, epoch_subject_ids, subject_labels = build_subject_epoch_arrays(
        df, sfreq=sfreq, window_samples=window_samples
    )

    if equalize_epoch_count:
        unique_subjects = sorted(set(epoch_subject_ids.tolist()))
        counts = {sid: int((epoch_subject_ids == sid).sum()) for sid in unique_subjects}
        min_count = min(counts.values())
        print(
            f"load_all_subjects_nasrabadi_connectivity: truncating every subject to their "
            f"first min_count={min_count} epochs (observed range: {min(counts.values())}-{max(counts.values())}) "
            f"to avoid a recording-duration confound (see this function's docstring)."
        )
        keep_idx = []
        for sid in unique_subjects:
            sid_idx = np.where(epoch_subject_ids == sid)[0]  # already in temporal order
            keep_idx.extend(sid_idx[:min_count].tolist())
        keep_idx = np.array(sorted(keep_idx))
        epochs_array = epochs_array[keep_idx]
        epoch_subject_ids = epoch_subject_ids[keep_idx]

    connectivity_feats = extract_connectivity_features(epochs_array, NASRABADI_CHANNELS_MODERN, sfreq=sfreq, metric=metric)
    X_subject, subject_ids = aggregate_epochs_to_subject(connectivity_feats, epoch_subject_ids.tolist())
    y_subject = np.array([subject_labels[sid] for sid in subject_ids])
    return X_subject, y_subject, subject_ids


if __name__ == "__main__":
    print(
        "Real Colab usage:\n"
        "  from experiments.load_nasrabadi_connectivity import load_all_subjects_nasrabadi_connectivity\n"
        "  X_conn, y_conn, subject_ids_conn = load_all_subjects_nasrabadi_connectivity('/path/to/nasrabadi.csv')\n"
    )
