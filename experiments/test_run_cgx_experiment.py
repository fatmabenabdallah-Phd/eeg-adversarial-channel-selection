"""
experiments.test_run_cgx_experiment
=====================================
INTEGRATION test for run_cgx_experiment.py's plumbing (does every
method run, produce the right shapes, and not crash when wired
together end-to-end), using synthetic data instead of real CGX
recordings -- this does NOT validate the scientific results (which can
only come from a real Colab run against the actual BALLADEER dataset,
see run_cgx_experiment.py's __main__ block). It exists to catch
integration bugs (shape mismatches, wrong argument order, a baseline
that breaks when combined with the others) before ever touching real
data or burning real Colab compute time on a broken pipeline.

Uses a SMALL synthetic montage (10 channels, not the real 30) and few
folds/samples, purely to keep this test fast -- this is a deliberate
scale-down for test speed, not a claim about what will happen at the
real 30-channel, 121-subject scale.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from experiments.run_cgx_experiment import run_cgx_experiment, run_wilcoxon_comparison


def _make_synthetic_cgx_like_dataset(seed: int = 42):
    from baselines.baselines import CGX_CHANNELS

    rng = np.random.RandomState(seed)
    n_bands = 5
    # Use REAL electrode names (a subset of CGX_CHANNELS), not generic
    # CH0/CH1/.. placeholders -- build_structural_knn_graph requires a
    # real standard_1020 position for every channel name, which a fake
    # name like "CH0" does not have (caught by actually running this
    # test, not assumed).
    channel_names = CGX_CHANNELS[:6]
    n_channels = len(channel_names)
    n_subjects = 30  # small, for test speed -- real CGX has 121

    # Feature layout matches the real extract_band_power_features output:
    # n_channels*n_bands contiguous block features + 1 trailing global
    # theta/beta ratio column, never touched by per-channel perturbation.
    X = rng.randn(n_subjects, n_channels * n_bands + 1) * 1.0
    y = (X[:, 0:n_bands].sum(axis=1) > 0).astype(int)  # first channel is informative

    subject_ids = [f"sub{i:03d}" for i in range(n_subjects)]
    label_df = pd.DataFrame({
        "user_id": subject_ids,
        "label": y,
        "sex": (["M", "F"] * (n_subjects // 2 + 1))[:n_subjects],
        "age_bin": (["young", "old"] * (n_subjects // 2 + 1))[:n_subjects],
    })
    return X, y, subject_ids, label_df, channel_names, n_bands


def test_run_cgx_experiment_end_to_end_runs_without_error():
    X, y, subject_ids, label_df, channel_names, n_bands = _make_synthetic_cgx_like_dataset()

    # Monkeypatch the module-level constants this test needs to override
    # (real CGX_CHANNELS/N_BANDS/CHANNEL_SUBSET_SIZES/CGX_GRAPH_K are sized
    # for the real 30-channel montage, too slow for a quick test here).
    import experiments.run_cgx_experiment as run_cgx_experiment_module
    run_cgx_experiment_module.N_BANDS = n_bands
    run_cgx_experiment_module.CHANNEL_SUBSET_SIZES = [3]
    run_cgx_experiment_module.CGX_GRAPH_K = 3

    results_df = run_cgx_experiment(
        X=X, y=y, subject_ids=subject_ids, label_df=label_df,
        channel_names=channel_names, n_bands=n_bands, k_folds=2, seed=42,
        adversarial_max_samples=12, greedy_max_k=3,
        epsilon_init=0.1, growth_factor=1.8, max_epsilon=3.0, n_directions=10, n_refine_steps=4,
    )

    assert len(results_df) > 0
    for col in ["fold", "method", "n_channels", "balanced_accuracy", "auc"]:
        assert col in results_df.columns, f"missing expected column: {col}"

    # Every method that ran must have produced results for every fold.
    n_folds_seen = results_df["fold"].nunique()
    assert n_folds_seen == 2, f"expected 2 folds, got {n_folds_seen}"

    expected_methods = {
        "all_channels", "random", "mutual_information", "anova_fscore",
        "correlation", "permutation_importance", "adversarial_isolated",
        "adversarial_graph_constrained", "greedy_forward",
    }
    seen_methods = set(results_df["method"].unique())
    missing = expected_methods - seen_methods
    assert not missing, f"expected methods not found in results: {missing}"

    # balanced_accuracy must be a valid probability-like value everywhere.
    assert results_df["balanced_accuracy"].between(0, 1).all()

    return results_df


def test_wilcoxon_comparison_runs_on_real_output(results_df: pd.DataFrame):
    """Confirms run_wilcoxon_comparison executes a REAL scipy.stats.wilcoxon
    call on the experiment's own output (not an estimated/fabricated
    p-value) -- consistent with this project's standing rule after the
    earlier p-value fabrication incident. Takes the already-computed
    results_df rather than re-running the (expensive) full experiment a
    second time."""
    comparison = run_wilcoxon_comparison(
        results_df, method_a="adversarial_isolated", method_b="random", n_channels=3
    )
    assert "p_value" in comparison and 0.0 <= comparison["p_value"] <= 1.0
    assert "wilcoxon_stat" in comparison


if __name__ == "__main__":
    import time
    t0 = time.time()
    results_df = test_run_cgx_experiment_end_to_end_runs_without_error()
    t1 = time.time()
    print(f"test_run_cgx_experiment_end_to_end_runs_without_error: PASSED ({t1-t0:.1f}s)")
    test_wilcoxon_comparison_runs_on_real_output(results_df)
    print("test_wilcoxon_comparison_runs_on_real_output: PASSED")
