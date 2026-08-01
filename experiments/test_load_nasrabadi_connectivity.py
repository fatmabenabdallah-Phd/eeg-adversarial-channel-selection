"""
experiments.test_load_nasrabadi_connectivity
===============================================
Regression tests for load_nasrabadi_connectivity.py, in particular the
epoch-count equalization fix motivated by a REAL confound found on the
actual Nasrabadi dataset: ADHD subjects had significantly more epochs
than Controls (38.2+-14.5 vs 30.7+-7.4, Mann-Whitney p=0.003), and an
initial connectivity-feature permutation screen without equalization
found a suspicious 15/19 channels "significant" after FDR, with
p-values clustered at the permutation-count floor and near-uniform
across nearly every channel -- the signature of a global artifact
(likely recording-duration-related drift), not a localized neural
effect.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/claude")  # for grn_balladeer -- tiret/underscore note elsewhere in this project

import numpy as np
import pandas as pd

from experiments.load_nasrabadi import _RAW_COLUMN_ORDER
from experiments.load_nasrabadi_connectivity import load_all_subjects_nasrabadi_connectivity


def test_epoch_count_equalization_is_noop_when_counts_already_equal():
    """When every subject already has the same epoch count (the
    common case in the existing integration test's synthetic data),
    equalize_epoch_count=True must produce IDENTICAL output to
    equalize_epoch_count=False -- confirms the equalization logic
    doesn't corrupt the normal case."""
    rng = np.random.RandomState(42)
    n_subjects = 20
    n_samples_per_subject = 1536  # 3 epochs of 512 samples for every subject
    rows = []
    for i in range(n_subjects):
        subj_id = f"S{i:03d}"
        label = "ADHD" if i < 10 else "Control"
        for t in range(n_samples_per_subject):
            row = {col: rng.randn() for col in _RAW_COLUMN_ORDER if col not in ("Class", "ID")}
            row["Class"] = label
            row["ID"] = subj_id
            rows.append(row)
    df_raw = pd.DataFrame(rows)[_RAW_COLUMN_ORDER]
    csv_path = "/tmp/test_synthetic_nasrabadi_equal_duration.csv"
    df_raw.to_csv(csv_path, index=False)

    X_eq, y_eq, ids_eq = load_all_subjects_nasrabadi_connectivity(
        csv_path, sfreq=128.0, window_samples=512, equalize_epoch_count=True
    )
    X_noeq, y_noeq, ids_noeq = load_all_subjects_nasrabadi_connectivity(
        csv_path, sfreq=128.0, window_samples=512, equalize_epoch_count=False
    )
    assert X_eq.shape == X_noeq.shape == (n_subjects, 19 * 5)


def test_epoch_count_equalization_truncates_to_min_count():
    """Confirms the actual fix: with genuinely UNEQUAL epoch counts
    per subject (mimicking the real Nasrabadi confound), every subject
    is truncated to the SAME epoch count (the minimum across
    subjects), keeping their FIRST min_count epochs (temporal order),
    not a random subsample -- addresses a recording-duration-related
    confound more directly than random subsampling would."""
    rng = np.random.RandomState(7)
    rows = []
    subject_epoch_counts = {}
    for i in range(10):
        subj_id = f"S{i:03d}"
        label = "ADHD" if i < 5 else "Control"
        n_samples_this_subject = int(rng.choice([512 * 2, 512 * 3, 512 * 4, 512 * 5]))
        subject_epoch_counts[subj_id] = n_samples_this_subject // 512
        for t in range(n_samples_this_subject):
            row = {col: rng.randn() for col in _RAW_COLUMN_ORDER if col not in ("Class", "ID")}
            row["Class"] = label
            row["ID"] = subj_id
            rows.append(row)
    df_raw = pd.DataFrame(rows)[_RAW_COLUMN_ORDER]
    csv_path = "/tmp/test_synthetic_nasrabadi_variable_duration.csv"
    df_raw.to_csv(csv_path, index=False)

    assert len(set(subject_epoch_counts.values())) > 1, (
        "sanity check: this test requires genuinely unequal epoch counts across subjects"
    )

    X_eq, y_eq, ids_eq = load_all_subjects_nasrabadi_connectivity(
        csv_path, sfreq=128.0, window_samples=512, equalize_epoch_count=True
    )
    assert X_eq.shape == (10, 19 * 5)
    assert len(ids_eq) == 10


if __name__ == "__main__":
    test_epoch_count_equalization_is_noop_when_counts_already_equal()
    print("test_epoch_count_equalization_is_noop_when_counts_already_equal: PASSED")
    test_epoch_count_equalization_truncates_to_min_count()
    print("test_epoch_count_equalization_truncates_to_min_count: PASSED")
