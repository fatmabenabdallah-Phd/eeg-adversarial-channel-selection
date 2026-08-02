"""
experiments.load_emotiv_connectivity
=======================================
Identical subject/session discovery as experiments.load_emotiv, but
extracts PLV/PLI-based connectivity features instead of band-power.

IMPORTANT DIFFERENCE from the CGX/Nasrabadi connectivity loaders: an
Emotiv session is NOT pre-split into discrete epochs anywhere in this
project (grn_balladeer.data.epocx_features.extract_epocx_band_power_features
computes power over the WHOLE continuous session in one call). So each
subject contributes a SINGLE connectivity estimate (their whole
session treated as one "epoch"), not an average over many short
epochs like CGX/Nasrabadi.

This does NOT eliminate the duration-confound risk found on
Nasrabadi (ADHD subjects had significantly more discrete epochs than
Controls, p=0.003, which turned out to be a real, non-artifactual
finding on further robustness checks -- but the concern itself, that
recording duration could differ systematically by group and affect a
single-window PLV/PLI estimate's bias, remains a legitimate thing to
control for here too, since session DURATION (number of samples) can
still vary by subject). This loader equalizes duration by truncating
every subject's session to the same fixed number of samples (the
minimum across subjects) before computing connectivity, by default.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.load_emotiv import _find_epocx_file
from methods.connectivity_features import extract_connectivity_features


def load_all_subjects_emotiv_connectivity(
    label_df: pd.DataFrame, dataset_root: str, subject_dir_fn=None,
    metric: str = "plv", equalize_duration: bool = True,
):
    """Returns (X, y, subject_ids). See this module's docstring for why
    equalize_duration matters here too, by analogy with the confirmed
    Nasrabadi epoch-count confound."""
    from grn_balladeer.data.epocx_features import (
        load_epocx_recording, session_passes_quality_filter, EPOCX_CHANNELS, EPOCX_SFREQ,
    )

    if subject_dir_fn is None:
        def subject_dir_fn(user_id, root):
            return os.path.join(root, user_id, "AttentionRobotsDesktop")

    eeg_cols = [f"EEG.{ch}" for ch in EPOCX_CHANNELS]

    raw_sessions = {}  # user_id -> (n_channels, n_samples) array
    labels = {}
    n_total, n_epocx_found, n_quality_passed = 0, 0, 0

    for _, row in label_df.iterrows():
        user_id = row["user_id"]
        n_total += 1
        session_dir = subject_dir_fn(user_id, dataset_root)
        csv_path = _find_epocx_file(session_dir)
        if csv_path is None:
            continue
        n_epocx_found += 1

        try:
            eeg_df = load_epocx_recording(csv_path)
        except Exception:
            continue
        if not session_passes_quality_filter(eeg_df):
            continue
        if not all(c in eeg_df.columns for c in eeg_cols):
            continue
        n_quality_passed += 1

        # Keep only rows where EVERY channel is non-NaN, so the resulting
        # array is a clean rectangular (n_channels, n_samples) block.
        clean_df = eeg_df[eeg_cols].dropna(axis=0, how="any")
        if len(clean_df) < int(EPOCX_SFREQ * 2):
            continue  # too short to be usable at all

        raw_sessions[user_id] = clean_df.to_numpy(dtype=np.float64).T  # (n_channels, n_samples)
        labels[user_id] = int(row["label"])

    print(
        f"Emotiv (connectivity) coverage: {n_total} subjects in label_df -> {n_epocx_found} with an "
        f"EPOCX file found -> {n_quality_passed} passed the quality filter -> "
        f"{len(raw_sessions)} with a usable raw session."
    )

    if len(raw_sessions) == 0:
        raise ValueError("load_all_subjects_emotiv_connectivity: no usable subjects found.")

    lengths = {uid: arr.shape[1] for uid, arr in raw_sessions.items()}
    min_length = min(lengths.values())
    print(
        f"Session length (samples) range: {min(lengths.values())}-{max(lengths.values())} "
        f"({min(lengths.values())/EPOCX_SFREQ:.1f}s-{max(lengths.values())/EPOCX_SFREQ:.1f}s)."
    )

    all_features = []
    all_subject_ids = []
    all_labels = []
    for uid, arr in raw_sessions.items():
        if equalize_duration:
            arr = arr[:, :min_length]  # first min_length samples, temporal order (not random)
        epochs_array = arr[np.newaxis, :, :]  # (1, n_channels, n_samples) -- whole session as one epoch
        feats = extract_connectivity_features(epochs_array, EPOCX_CHANNELS, sfreq=EPOCX_SFREQ, metric=metric)
        all_features.append(feats[0])
        all_subject_ids.append(uid)
        all_labels.append(labels[uid])

    X = np.stack(all_features, axis=0)
    y = np.array(all_labels)
    return X, y, all_subject_ids


if __name__ == "__main__":
    print(
        "Real Colab usage:\n"
        "  from experiments.load_emotiv_connectivity import load_all_subjects_emotiv_connectivity\n"
        "  X_emotiv_conn, y_emotiv_conn, subject_ids_emotiv_conn = load_all_subjects_emotiv_connectivity(\n"
        "      label_df, dataset_root='/content/drive/MyDrive/BALLADEER ADHD DATASET', metric='plv')\n"
    )
