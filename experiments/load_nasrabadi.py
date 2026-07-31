"""
experiments.load_nasrabadi
============================
Loads the public Nasrabadi ADHD EEG dataset (Kaggle mirror:
danizo/eeg-dataset-for-adhd; IEEE DataPort DOI 10.21227/rzfh-zn36) for
the band-power / RF / adversarial-channel-selection pipeline -- a
DIFFERENT loading path from grn-balladeer's
data/build_dataset_nasrabadi.py, which builds GRN-specific (CQT +
Laplacian) graph inputs, not band-power features.

TWO REAL DISCREPANCIES CAUGHT BEFORE WRITING ANY LOADING CODE (verified
via the dataset's own public documentation, not assumed):

1. Column ORDER. The actual CSV column order is:
   Fz, Cz, Pz, C3, T3, C4, T4, Fp1, Fp2, F3, F4, F7, F8, P3, P4, T5, T6,
   O1, O2, Class, ID
   This does NOT match grn-balladeer's NASRABADI_CHANNELS list order
   (which starts Fp1, Fp2, F3, F4, C3, C4, ...). Reading columns by
   POSITION instead of by NAME would silently mislabel channels.
   This loader always selects columns by name from the CSV header.

2. Channel NAMING convention. The dataset uses the older 10-20 labels
   T3, T4, T5, T6, not the modern equivalents T7, T8, P7, P8 used by
   grn-balladeer's NASRABADI_CHANNELS and by MNE's standard_1020
   montage lookups (preprocessing.structural_graph.build_structural_knn_graph
   requires the modern names to find electrode positions). This loader
   renames T3->T7, T4->T8, T5->P7, T6->P8 on load, matching the
   standard old->modern 10-20 equivalence.

NOT YET VERIFIED (no direct access to the actual Kaggle file from this
environment -- verify against the real downloaded CSV before trusting
this in production): whether the dataset ships as one CSV per subject
or one combined CSV with an ID column separating subjects (the public
description mentions both a "Class" and an "ID" column, consistent
with one combined file, but the per-subject file-splitting convention,
if any, has not been confirmed against the actual file contents).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

# Modern-nomenclature channel order this project standardizes on,
# matching grn-balladeer's NASRABADI_CHANNELS (kept as a separate
# constant here rather than imported, since this module must not
# depend on grn_balladeer's package-name/import quirks -- see the
# tiret/underscore lesson from the P0 module-copy task).
NASRABADI_CHANNELS_MODERN = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]

# The dataset's actual on-disk column names (old nomenclature for the
# 4 renamed electrodes), in the ACTUAL on-disk order -- this order is
# irrelevant to correctness here since we always select by name, but
# is recorded for documentation/verification purposes.
_RAW_COLUMN_ORDER = [
    "Fz", "Cz", "Pz", "C3", "T3", "C4", "T4", "Fp1", "Fp2", "F3", "F4",
    "F7", "F8", "P3", "P4", "T5", "T6", "O1", "O2", "Class", "ID",
]
_OLD_TO_MODERN_RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def load_nasrabadi_csv(csv_path: str) -> pd.DataFrame:
    """Loads the raw Nasrabadi CSV and renames old-nomenclature
    channels to their modern equivalents (T3->T7, T4->T8, T5->P7,
    T6->P8). Does NOT reorder or select channels -- returns every
    column present, renamed. Raises a clear error if an expected
    channel or the Class/ID columns are missing, rather than silently
    proceeding with a malformed frame.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns=_OLD_TO_MODERN_RENAME)

    missing_channels = [ch for ch in NASRABADI_CHANNELS_MODERN if ch not in df.columns]
    if missing_channels:
        raise ValueError(
            f"load_nasrabadi_csv: after renaming old->modern channel names, "
            f"{missing_channels} are still missing from {csv_path}'s columns "
            f"({list(df.columns)}). The real file's column names may differ "
            f"from what this loader assumes -- verify against the actual "
            f"downloaded CSV header before proceeding."
        )
    for required_col in ("Class", "ID"):
        if required_col not in df.columns:
            raise ValueError(
                f"load_nasrabadi_csv: expected a '{required_col}' column, not found "
                f"in {csv_path} (columns: {list(df.columns)})."
            )
    return df


def build_subject_epoch_arrays(
    df: pd.DataFrame, sfreq: float = 128.0, window_samples: int = 512,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Splits the combined (all-subjects) DataFrame into one
    (n_channels, n_samples) continuous recording per subject (by 'ID'),
    then windows each into non-overlapping window_samples-length
    segments (default 512 samples = 4.0s at 128Hz, matching
    grn-balladeer's build_dataset_nasrabadi.py windowing convention for
    a directly comparable RF-vs-GRN/EEGNet setup on the same data).

    Returns (epochs_array, epoch_subject_ids, subject_labels):
      - epochs_array: (n_total_epochs, n_channels, window_samples), channel
        order = NASRABADI_CHANNELS_MODERN, ready for
        baselines.baselines.extract_band_power_features
      - epoch_subject_ids: (n_total_epochs,) array of subject ID per epoch
      - subject_labels: dict {subject_id: 0/1}, 1 = ADHD ('ADHD' in the
        Class column), 0 = Control -- ASSUMES the Class column's ADHD
        label spelling; verify against the real file (the public
        description says "Class: ADHD/Control" but the exact string
        values, casing, etc. have not been confirmed against the actual
        file contents from this environment).
    """
    all_epochs = []
    epoch_subject_ids = []
    subject_labels = {}

    for subject_id, group in df.groupby("ID"):
        channel_data = group[NASRABADI_CHANNELS_MODERN].values.T  # (n_channels, n_samples)
        n_samples = channel_data.shape[1]
        n_windows = n_samples // window_samples
        if n_windows == 0:
            continue  # recording shorter than one window -- skip, don't crash

        class_values = group["Class"].unique()
        if len(class_values) != 1:
            raise ValueError(
                f"build_subject_epoch_arrays: subject {subject_id} has inconsistent "
                f"Class values within its own rows: {class_values} -- expected a "
                f"single label per subject."
            )
        label_str = str(class_values[0]).strip().upper()
        subject_labels[subject_id] = 1 if label_str.startswith("ADHD") else 0

        for w in range(n_windows):
            start = w * window_samples
            all_epochs.append(channel_data[:, start:start + window_samples])
            epoch_subject_ids.append(subject_id)

    epochs_array = np.stack(all_epochs, axis=0)
    return epochs_array, np.array(epoch_subject_ids), subject_labels
