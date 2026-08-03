"""
experiments.load_tdbrain_eeg
==============================
Loads real TDBRAIN BDF files (BIDS format, confirmed structure:
<root>/sub-<ID>/ses-1/eeg/sub-<ID>_ses-1_task-<restEC|restEO>_eeg.bdf),
extracts band-power and/or connectivity features using this project's
existing, already-tested extraction functions.

REAL channel list, confirmed directly from a channels.tsv file (NOT
guessed, NOT the 33 originally reported by the dataset's web
documentation, which apparently includes the 6 auxiliary channels
counted differently somewhere): 26 real EEG channels + 6 auxiliary
(VPVA/VNVB/HPHL/HNHR = EOG, Erbs = ECG, Mass = EMG) that must be
EXCLUDED from any channel-selection analysis, since they are not brain
signal. Sampling frequency confirmed 500 Hz (matching CGX's 500 Hz,
convenient for consistent epoch-duration choices across datasets).
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd

TDBRAIN_EEG_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4",
    "T7", "C3", "Cz", "C4", "T8", "CP3", "CPz", "CP4",
    "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2",
]
TDBRAIN_SFREQ = 500.0
TDBRAIN_WINDOW_SAMPLES = 2000  # 4.0s at 500Hz -- matches Nasrabadi's 512-samples-at-128Hz (also 4.0s),
                                 # for consistent epoch duration across datasets in this project


def find_tdbrain_subject_file(dataset_root: str, user_id: str, task: str = "restEC") -> str:
    """BIDS naming is fully deterministic here (confirmed against the
    real downloaded dataset, unlike CGX/Emotiv's irregular naming) --
    no session/file discovery loop needed."""
    path = os.path.join(dataset_root, user_id, "ses-1", "eeg", f"{user_id}_ses-1_task-{task}_eeg.bdf")
    if not os.path.exists(path):
        raise FileNotFoundError(f"find_tdbrain_subject_file: expected file not found at {path}")
    return path


def load_tdbrain_raw_epochs(bdf_path: str, window_samples: int = TDBRAIN_WINDOW_SAMPLES) -> np.ndarray:
    """Loads one subject's BDF file via MNE, keeps ONLY the 26 real EEG
    channels (explicitly, by name -- never by position or channel
    count, since the file also contains EOG/ECG/EMG channels and
    possibly an empty BioSemi-style Status channel per the dataset's
    own release notes), and splits into non-overlapping
    window_samples-length epochs. Returns (n_epochs, 26, window_samples).
    """
    import mne

    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    missing = [ch for ch in TDBRAIN_EEG_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise ValueError(
            f"load_tdbrain_raw_epochs: expected EEG channels {missing} not found in {bdf_path} "
            f"(available: {raw.ch_names})."
        )
    raw.pick(TDBRAIN_EEG_CHANNELS)  # explicit selection, drops EOG/ECG/EMG/Status entirely

    data = raw.get_data()  # (26, n_total_samples), already in the channel order requested
    n_total_samples = data.shape[1]
    n_windows = n_total_samples // window_samples
    if n_windows == 0:
        raise ValueError(f"load_tdbrain_raw_epochs: {bdf_path} is shorter than one window.")

    epochs = np.stack(
        [data[:, w * window_samples:(w + 1) * window_samples] for w in range(n_windows)], axis=0
    )
    return epochs


def load_all_subjects_tdbrain(
    label_df: pd.DataFrame, dataset_root: str, task: str = "restEC",
    feature_type: str = "band_power", metric: str = "plv",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loops over every subject in label_df (already filtered to
    confirmed diagnosis + age >= 18 by build_tdbrain_label_df), loads
    their BDF file, extracts either band-power or connectivity
    features per epoch, aggregates to one row per subject.

    feature_type: "band_power" (baselines.baselines.extract_band_power_features)
    or "connectivity" (methods.connectivity_features.extract_connectivity_features,
    metric="plv" or "pli").
    """
    from baselines.baselines import extract_band_power_features, aggregate_epochs_to_subject
    from methods.connectivity_features import extract_connectivity_features

    if feature_type not in ("band_power", "connectivity"):
        raise ValueError(f"load_all_subjects_tdbrain: feature_type must be 'band_power' or 'connectivity', got {feature_type!r}")

    all_features = []
    all_epoch_subject_ids = []
    all_labels_per_subject = {}
    n_total, n_loaded = 0, 0

    for _, row in label_df.iterrows():
        user_id = row["user_id"]
        n_total += 1
        all_labels_per_subject[user_id] = int(row["label"])

        try:
            bdf_path = find_tdbrain_subject_file(dataset_root, user_id, task=task)
            epochs_array = load_tdbrain_raw_epochs(bdf_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"  skipping {user_id}: {e}")
            continue
        n_loaded += 1

        if feature_type == "band_power":
            feats = extract_band_power_features(epochs_array, channels=TDBRAIN_EEG_CHANNELS, sfreq=TDBRAIN_SFREQ)
        else:
            feats = extract_connectivity_features(epochs_array, TDBRAIN_EEG_CHANNELS, sfreq=TDBRAIN_SFREQ, metric=metric)

        for feat_row in feats:
            all_features.append(feat_row)
            all_epoch_subject_ids.append(user_id)

    print(f"TDBRAIN ({task}, {feature_type}) coverage: {n_loaded}/{n_total} subjects loaded successfully.")

    features_arr = np.stack(all_features, axis=0)
    X_subject, subject_ids = aggregate_epochs_to_subject(features_arr, all_epoch_subject_ids)
    y_subject = np.array([all_labels_per_subject[sid] for sid in subject_ids])
    return X_subject, y_subject, subject_ids


if __name__ == "__main__":
    print(
        "Real Colab usage:\n"
        "  from experiments.load_tdbrain import build_tdbrain_label_df\n"
        "  from experiments.load_tdbrain_eeg import load_all_subjects_tdbrain\n"
        "  label_df = build_tdbrain_label_df('/path/to/TDBRAIN_participants_V3.xlsx')\n"
        "  X, y, subject_ids = load_all_subjects_tdbrain(label_df, '/path/to/TDBRAIN_Dataset_V3_1', task='restEC')\n"
    )
