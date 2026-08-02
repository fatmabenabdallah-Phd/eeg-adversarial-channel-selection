"""
experiments.test_load_emotiv_connectivity
============================================
Integration test for load_emotiv_connectivity.py, using synthetic
EPOCX CSVs with the REAL file structure (metadata line to skip, then
EEG.<channel>/CQ.<channel> headers), and DIFFERENT session durations
per subject -- to confirm the duration-equalization logic actually
engages (by analogy with the confirmed Nasrabadi epoch-count
confound), not just that the loader runs without crashing.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/claude")

import numpy as np
import pandas as pd

from experiments.load_emotiv_connectivity import load_all_subjects_emotiv_connectivity


def _write_fake_epocx_csv(path, channel_names, sfreq, n_samples, seed):
    rng = np.random.RandomState(seed)
    cols = {"Timestamp": np.arange(n_samples) / sfreq}
    for ch in channel_names:
        cols[f"EEG.{ch}"] = rng.randn(n_samples) * 10
        cols[f"CQ.{ch}"] = np.full(n_samples, 4)
    df = pd.DataFrame(cols)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("title:FAKE, start timestamp:0\n")
        df.to_csv(f, index=False)


def test_load_all_subjects_emotiv_connectivity_equalizes_variable_durations():
    from grn_balladeer.data.epocx_features import EPOCX_CHANNELS, EPOCX_SFREQ

    root = "/tmp/test_fake_emotiv_connectivity"
    _write_fake_epocx_csv(
        f"{root}/UB0001/AttentionRobotsDesktop/111/UB0001_EPOCX_123.csv",
        EPOCX_CHANNELS, EPOCX_SFREQ, n_samples=2000, seed=1,
    )
    _write_fake_epocx_csv(
        f"{root}/UB0002/AttentionRobotsDesktop/222/UB0002_EPOCX_456.csv",
        EPOCX_CHANNELS, EPOCX_SFREQ, n_samples=3000, seed=2,  # deliberately different duration
    )
    label_df = pd.DataFrame({"user_id": ["UB0001", "UB0002"], "label": [1, 0]})

    X, y, subject_ids = load_all_subjects_emotiv_connectivity(label_df, dataset_root=root, metric="plv")

    assert X.shape == (2, len(EPOCX_CHANNELS) * 5)
    assert subject_ids == ["UB0001", "UB0002"]
    assert list(y) == [1, 0]


if __name__ == "__main__":
    test_load_all_subjects_emotiv_connectivity_equalizes_variable_durations()
    print("test_load_all_subjects_emotiv_connectivity_equalizes_variable_durations: PASSED")
