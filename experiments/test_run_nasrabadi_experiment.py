"""
experiments.test_run_nasrabadi_experiment
============================================
INTEGRATION test (synthetic data, real column-naming/windowing
conventions) for run_nasrabadi_experiment.py -- same purpose as
test_run_cgx_experiment.py: catch wiring bugs before touching the real
dataset or burning real compute time.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from experiments.load_nasrabadi import _RAW_COLUMN_ORDER
from experiments.run_nasrabadi_experiment import (
    load_all_subjects_nasrabadi, run_nasrabadi_experiment,
)


def _make_synthetic_nasrabadi_csv(path: str, seed: int = 42):
    """Writes a synthetic CSV with the EXACT real column order/names
    (old nomenclature T3/T4/T5/T6, Class/ID columns) documented for the
    real Kaggle dataset -- so this test exercises the real renaming and
    column-selection logic, not a simplified stand-in."""
    rng = np.random.RandomState(seed)
    n_subjects = 20  # small, for test speed -- real dataset has 121
    n_samples_per_subject = 1536  # 3 windows of 512 samples at 128Hz

    rows = []
    for i in range(n_subjects):
        subj_id = f"S{i:03d}"
        label = "ADHD" if i < 10 else "Control"
        # Inject a genuine (if crude) signal so the RF has something to
        # learn: Fp1 column's mean differs by label -- integration test
        # only needs SOME learnable structure, not a realistic one.
        offset = 0.5 if label == "ADHD" else -0.5
        for t in range(n_samples_per_subject):
            row = {col: rng.randn() for col in _RAW_COLUMN_ORDER if col not in ("Class", "ID")}
            row["Fp1"] = row["Fp1"] + offset
            row["Class"] = label
            row["ID"] = subj_id
            rows.append(row)

    df_raw = pd.DataFrame(rows)[_RAW_COLUMN_ORDER]
    df_raw.to_csv(path, index=False)


def test_load_all_subjects_nasrabadi_end_to_end():
    csv_path = "/tmp/synthetic_nasrabadi.csv"
    _make_synthetic_nasrabadi_csv(csv_path)

    X, y, subject_ids = load_all_subjects_nasrabadi(csv_path, sfreq=128.0, window_samples=512)
    assert X.shape[0] == len(subject_ids) == len(y)
    assert X.shape[0] == 20, f"expected 20 subjects, got {X.shape[0]}"
    assert X.shape[1] == 19 * 5 + 1, f"expected 96 features (19ch*5 bands + 1 ratio), got {X.shape[1]}"
    assert set(np.unique(y)) == {0, 1}
    return X, y, subject_ids


def test_run_nasrabadi_experiment_end_to_end_runs_without_error():
    X, y, subject_ids = test_load_all_subjects_nasrabadi_end_to_end()

    import experiments.run_nasrabadi_experiment as m
    m.CHANNEL_SUBSET_SIZES = [5, 10]  # smaller than the real [5,10,15,19] for test speed
    m.NASRABADI_GRAPH_K = 4

    results_df = run_nasrabadi_experiment(
        X=X, y=y, subject_ids=subject_ids, k_folds=2, seed=42,
        adversarial_max_samples=8, greedy_max_k=5,
        epsilon_init=0.1, growth_factor=2.2, max_epsilon=3.0, n_directions=8, n_refine_steps=3,
    )

    assert len(results_df) > 0
    for col in ["fold", "method", "n_channels", "balanced_accuracy", "auc"]:
        assert col in results_df.columns

    expected_methods = {
        "all_channels", "random", "mutual_information", "anova_fscore",
        "correlation", "permutation_importance", "adversarial_isolated",
        "adversarial_graph_constrained", "greedy_forward",
    }
    seen_methods = set(results_df["method"].unique())
    missing = expected_methods - seen_methods
    assert not missing, f"expected methods not found: {missing}"
    assert results_df["balanced_accuracy"].between(0, 1).all()

    return results_df


if __name__ == "__main__":
    import time
    t0 = time.time()
    test_load_all_subjects_nasrabadi_end_to_end()
    print(f"test_load_all_subjects_nasrabadi_end_to_end: PASSED ({time.time()-t0:.1f}s)")
    t0 = time.time()
    results_df = test_run_nasrabadi_experiment_end_to_end_runs_without_error()
    print(f"test_run_nasrabadi_experiment_end_to_end_runs_without_error: PASSED ({time.time()-t0:.1f}s)")
