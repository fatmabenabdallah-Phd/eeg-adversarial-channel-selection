"""
experiments.load_cgx_connectivity
====================================
Identical to run_cgx_experiment.load_all_subjects_cgx EXCEPT the final
feature-extraction step: extract_connectivity_features (PLV-based,
piste 5) instead of extract_band_power_features. Reuses the exact same
per-subject loading loop (find_slackline_sessions,
build_subject_dataset_lightweight) since that already produces the raw
(n_channels, n_samples) epoch tensor needed for connectivity -- only
the LAST step differs.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.baselines import aggregate_epochs_to_subject
from methods.connectivity_features import extract_connectivity_features
from preprocessing.structural_graph import build_structural_knn_graph  # noqa: F401 (re-exported for convenience)


def load_all_subjects_cgx_connectivity(label_df: pd.DataFrame, dataset_root: str = None):
    """Same subject/session discovery as run_cgx_experiment.load_all_subjects_cgx,
    but extracts PLV-based connectivity features per epoch instead of
    band-power. Returns (X, y, subject_ids)."""
    from grn_balladeer.data.labels import DRIVE_DATASET_DIR, PATH_SLACKLINE_FLAGS
    from grn_balladeer.data.subject_files import find_slackline_sessions, SLACKLINE_LEVELS
    from grn_balladeer.data.build_dataset_lightweight import build_subject_dataset_lightweight
    from baselines.baselines import CGX_CHANNELS

    dataset_root = dataset_root or DRIVE_DATASET_DIR
    sfreq = 500.0  # CGX_SFREQ, confirmed in grn_balladeer.data.sync (NOT 250.0 -- caught before running)

    all_epoch_tensors = []
    all_epoch_subject_ids = []
    all_labels_per_subject = {}
    n_sessions_total, n_sessions_no_cgx = 0, 0

    for _, row in label_df.iterrows():
        user_id = row["user_id"]
        all_labels_per_subject[user_id] = int(row["label"])

        for level in SLACKLINE_LEVELS:
            sessions = find_slackline_sessions(dataset_root, user_id, level)
            for session in sessions:
                n_sessions_total += 1
                if session.cgx_path is None:
                    n_sessions_no_cgx += 1
                    continue

                dataset = build_subject_dataset_lightweight(
                    cgx_path=session.cgx_path,
                    flags_path=PATH_SLACKLINE_FLAGS,
                    level=f"Level{level}",
                    clean_bad_channels=True,
                )
                for raw_epoch_tensor, _band_power_feats in dataset:
                    all_epoch_tensors.append(raw_epoch_tensor)
                    all_epoch_subject_ids.append(user_id)

    print(
        f"CGX (connectivity) coverage: {n_sessions_total - n_sessions_no_cgx}/{n_sessions_total} "
        f"sessions had a CGX file."
    )

    epochs_array = np.stack(all_epoch_tensors, axis=0)  # (n_epochs, n_channels, n_samples)
    connectivity_feats = extract_connectivity_features(epochs_array, CGX_CHANNELS, sfreq=sfreq)
    X_subject, subject_ids = aggregate_epochs_to_subject(connectivity_feats, all_epoch_subject_ids)
    y_subject = np.array([all_labels_per_subject[sid] for sid in subject_ids])
    return X_subject, y_subject, subject_ids


if __name__ == "__main__":
    print(
        "Real Colab usage:\n"
        "  from experiments.load_cgx_connectivity import load_all_subjects_cgx_connectivity\n"
        "  X_conn, y_conn, subject_ids_conn = load_all_subjects_cgx_connectivity(label_df)\n"
    )
