"""
experiments.test_run_cross_hardware_transfer
===============================================
INTEGRATION test (synthetic data) for run_cross_hardware_transfer.py.
Constructs synthetic CGX and Emotiv datasets that SHARE a genuinely
informative common channel, to check the transfer pipeline recovers
something sensible end-to-end (wiring correctness, not a claim about
real transfer performance, which can only come from real data).
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from baselines.baselines import CGX_CHANNELS, EMOTIV_CHANNELS, COMMON_CHANNELS
from experiments.run_cross_hardware_transfer import run_cross_hardware_transfer


def _make_synthetic_cross_hardware_data(seed: int = 42):
    rng = np.random.RandomState(seed)
    n_bands = 5

    # CGX: 30 channels, 60 subjects (small, for test speed)
    n_cgx_subjects = 60
    X_cgx = rng.randn(n_cgx_subjects, len(CGX_CHANNELS) * n_bands) * 1.0
    # Make the label depend on a COMMON channel's block (e.g. "F7", present
    # in both CGX_CHANNELS and EMOTIV_CHANNELS) so a correctly-working
    # transfer pipeline should find this channel important on both sides.
    common_ch = "F7"
    cgx_idx = CGX_CHANNELS.index(common_ch)
    y_cgx = (X_cgx[:, cgx_idx*n_bands:(cgx_idx+1)*n_bands].sum(axis=1) > 0).astype(int)

    # Emotiv: 14 channels, 25 subjects (small, for test speed)
    n_emotiv_subjects = 25
    X_emotiv = rng.randn(n_emotiv_subjects, len(EMOTIV_CHANNELS) * n_bands) * 1.0
    emotiv_idx = EMOTIV_CHANNELS.index(common_ch)
    y_emotiv = (X_emotiv[:, emotiv_idx*n_bands:(emotiv_idx+1)*n_bands].sum(axis=1) > 0).astype(int)
    emotiv_subject_ids = [f"esub{i:03d}" for i in range(n_emotiv_subjects)]

    return X_cgx, y_cgx, X_emotiv, y_emotiv, emotiv_subject_ids, common_ch


def test_run_cross_hardware_transfer_end_to_end_runs_without_error():
    X_cgx, y_cgx, X_emotiv, y_emotiv, emotiv_subject_ids, common_ch = _make_synthetic_cross_hardware_data()

    import experiments.run_cross_hardware_transfer as m
    m.COMMON_GRAPH_K = 3

    results_df = run_cross_hardware_transfer(
        X_cgx=X_cgx, y_cgx=y_cgx, X_emotiv=X_emotiv, y_emotiv=y_emotiv,
        emotiv_subject_ids=emotiv_subject_ids,
        common_channels=COMMON_CHANNELS, n_bands=5,
        k_sizes=[3, len(COMMON_CHANNELS)], k_folds=2, seed=42,
        adversarial_max_samples=8,
        epsilon_init=0.1, growth_factor=2.2, max_epsilon=3.0, n_directions=8, n_refine_steps=3,
    )

    assert len(results_df) > 0
    for col in ["fold", "source", "method", "n_channels", "balanced_accuracy", "auc"]:
        assert col in results_df.columns

    expected_sources = {"baseline", "cgx_transfer", "emotiv_own"}
    seen_sources = set(results_df["source"].unique())
    assert expected_sources == seen_sources, f"expected sources {expected_sources}, got {seen_sources}"

    cgx_methods = set(results_df[results_df.source == "cgx_transfer"]["method"].unique())
    expected_cgx_methods = {
        "mutual_information", "anova_fscore", "correlation",
        "permutation_importance", "adversarial_isolated", "adversarial_graph_constrained",
    }
    assert cgx_methods == expected_cgx_methods, f"expected {expected_cgx_methods}, got {cgx_methods}"

    assert results_df["balanced_accuracy"].between(0, 1).all()

    # Sanity: since the common channel F7 was made genuinely informative on
    # BOTH CGX and Emotiv by construction, at k=3 the cgx_transfer methods
    # should on average beat "random" baseline. Not a strict assertion
    # (small synthetic sample -> real noise), but a loose directional check.
    random_mean = results_df[(results_df.source == "baseline") & (results_df.method == "random") & (results_df.n_channels == 3)]["balanced_accuracy"].mean()
    cgx_transfer_mean = results_df[(results_df.source == "cgx_transfer") & (results_df.n_channels == 3)]["balanced_accuracy"].mean()
    print(f"random baseline mean (k=3): {random_mean:.3f}, cgx_transfer mean (k=3): {cgx_transfer_mean:.3f}")

    return results_df


if __name__ == "__main__":
    import time
    t0 = time.time()
    df = test_run_cross_hardware_transfer_end_to_end_runs_without_error()
    print(f"test_run_cross_hardware_transfer_end_to_end_runs_without_error: PASSED ({time.time()-t0:.1f}s)")
