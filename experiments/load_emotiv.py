"""
experiments.load_emotiv
=========================
Loads BALLADEER Emotiv EPOCX recordings (AttentionRobotsDesktop
session) for the band-power / RF / adversarial-channel-selection
pipeline, reusing grn_balladeer.data.epocx_features UNCHANGED (already
handles the metadata-header-skip quirk, the per-channel CQ 0-4 scale
quality filter, and Welch band-power extraction on raw signal --
consistent methodology with CGX and Nasrabadi, not the vendor's own
POW.* columns, per the P0 decision to keep methodology identical
across hardware for a valid cross-hardware comparison).

IMPORTANT CORRECTION, twice over: the project's earlier "35 valid
subjects" figure could not be reconstructed or verified. A later
re-derivation of the quality filter suggested "~97-100 subjects at any
reasonable threshold" -- but that re-derivation apparently did not
account for most subjects' AttentionRobotsDesktop folder containing
only GAME_DATA/EYE_TRACKING_DATA CSVs with NO actual EPOCX (EEG) file
at all (see _find_epocx_file's docstring). Directly counting subjects
with a real EPOCX file present gives a THIRD number, confirmed by
scanning the real dataset: 52/138 subjects. This loader's own
coverage summary will report a further-reduced count after the
quality filter is also applied (CQ-based signal-quality check) -- so
even 52 is an upper bound on the subject count this experiment will
ultimately run on, not the final number.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def _find_epocx_file(session_root: str):
    """Searches session_root and its subfolders (e.g.
    AttentionRobotsDesktop/1686235332/) for a file whose name contains
    'EPOCX' -- NOT just any .csv, since the real folder structure also
    contains GAME_DATA and EYE_TRACKING_DATA CSVs for every subject,
    but the EPOCX (actual EEG signal) file only for a fraction of
    them. Confirmed empirically: of 8 sampled subjects, only 1
    (UB0021, file 'UB0021_EPOCX_96566_...csv') had a matching file --
    the other 7 only had GAME_DATA/EYE_TRACKING_DATA, no EEG at all.
    Returns None, not a wrong file, if no EPOCX file is found anywhere
    under session_root.
    """
    if not os.path.isdir(session_root):
        return None
    for root, _dirs, files in os.walk(session_root):
        for f in files:
            if "epocx" in f.lower() and f.lower().endswith(".csv"):
                return os.path.join(root, f)
    return None


def load_all_subjects_emotiv(
    label_df: pd.DataFrame, dataset_root: str, subject_dir_fn=None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loops over every subject in label_df, finds their EPOCX
    AttentionRobotsDesktop recording (searching for a filename
    containing 'EPOCX', not just any CSV in the session folder -- see
    _find_epocx_file's docstring for why this distinction is real and
    necessary: most subjects' AttentionRobotsDesktop folder contains
    only GAME_DATA/EYE_TRACKING_DATA CSVs, no actual EEG file), applies
    the quality filter (skip, don't error, on failure), and extracts
    band-power features. Returns (X, y, subject_ids) for subjects that
    passed.

    Prints a coverage summary at the end (n_epocx_file_found /
    n_quality_passed / n_in_label_df) so the real subject count is
    always visible, never assumed to be either the unverifiable "35"
    or the re-derived "~97-100" without actually checking on this run
    (that re-derivation counted differently -- e.g. possibly without
    requiring the label to be resolvable -- so this run's own count may
    still differ from both prior figures; report what THIS run finds).
    """
    from grn_balladeer.data.epocx_features import (
        load_epocx_recording, session_passes_quality_filter,
        extract_epocx_band_power_features, EPOCX_FEATURE_NAMES,
    )

    if subject_dir_fn is None:
        def subject_dir_fn(user_id, root):
            return os.path.join(root, user_id, "AttentionRobotsDesktop")

    all_features = []
    all_labels = []
    all_subject_ids = []
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
        n_quality_passed += 1

        try:
            feats = extract_epocx_band_power_features(eeg_df)
        except ValueError:
            continue

        all_features.append(feats)
        all_labels.append(int(row["label"]))
        all_subject_ids.append(user_id)

    print(
        f"Emotiv coverage: {n_total} subjects in label_df -> {n_epocx_found} with an "
        f"EPOCX file found -> {n_quality_passed} passed the quality filter -> "
        f"{len(all_subject_ids)} with valid extracted features."
    )

    X = np.stack(all_features, axis=0)
    y = np.array(all_labels)
    return X, y, all_subject_ids
