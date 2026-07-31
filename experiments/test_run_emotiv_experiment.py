"""
experiments.test_run_emotiv_experiment
=========================================
INTEGRATION test (synthetic data) for run_emotiv_experiment.py.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from baselines.baselines import EMOTIV_CHANNELS
from experiments.run_emotiv_experiment import run_emotiv_experiment


def _make_synthetic_emotiv_dataset(seed: int = 42):
    rng = np.random.RandomState(seed)
    n_bands = 5
    channel_names = EMOTIV_CHANNELS  # real channel names -- needed for build_structural_knn_graph
    n_channels = len(channel_names)
    n_subjects = 30

    X = rng.randn(n_subjects, n_channels * n_bands + 1) * 1.0
    y = (X[:, 0:n_bands].sum(axis=1) > 0).astype(int)

    subject_ids = [f"sub{i:03d}" for i in range(n_subjects)]
    label_df = pd.DataFrame({
        "user_id": subject_ids,
        "label": y,
        "sex": (["M", "F"] * (n_subjects // 2 + 1))[:n_subjects],
        "age_bin": (["young", "old"] * (n_subjects // 2 + 1))[:n_subjects],
    })
    return X, y, subject_ids, label_df, channel_names, n_bands


def test_run_emotiv_experiment_end_to_end_runs_without_error():
    X, y, subject_ids, label_df, channel_names, n_bands = _make_synthetic_emotiv_dataset()

    import experiments.run_emotiv_experiment as m
    m.N_BANDS = n_bands
    m.CHANNEL_SUBSET_SIZES = [3, 6]
    m.EMOTIV_GRAPH_K = 3

    results_df = run_emotiv_experiment(
        X=X, y=y, subject_ids=subject_ids, label_df=label_df,
        channel_names=channel_names, n_bands=n_bands, k_folds=2, seed=42,
        adversarial_max_samples=10, greedy_max_k=3,
        epsilon_init=0.1, growth_factor=2.0, max_epsilon=3.0, n_directions=8, n_refine_steps=3,
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

    # manual_frontal should be ABSENT: EMOTIV_CHANNELS only has F7 of the
    # 5-channel cluster, so manual_frontal_baseline() must have raised
    # and been caught (not silently produce a degraded 1-channel result).
    assert "manual_frontal" not in seen_methods, (
        "manual_frontal should not appear for Emotiv (only F7 of the cluster is present)"
    )
    return results_df


if __name__ == "__main__":
    import time
    t0 = time.time()
    df = test_run_emotiv_experiment_end_to_end_runs_without_error()
    print(f"test_run_emotiv_experiment_end_to_end_runs_without_error: PASSED ({time.time()-t0:.1f}s)")
    print("Methods seen:", sorted(df.method.unique()))
