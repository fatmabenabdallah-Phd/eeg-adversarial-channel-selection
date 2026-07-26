"""
baselines.test_channel_selection_baselines
============================================
Regression tests for the channel-selection baseline suite, using the
same synthetic-with-known-ground-truth approach as
methods/test_adversarial_importance.py: a single informative channel
(CH3) is constructed by design, and each baseline is checked to
actually recover it -- not just to run without crashing.

Run with: python -m baselines.test_channel_selection_baselines
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from baselines.channel_selection_baselines import (
    all_channels_baseline,
    random_subset_baseline,
    manual_frontal_baseline,
    mutual_information_baseline,
    anova_fscore_baseline,
    correlation_baseline,
    greedy_forward_selection,
    permutation_importance_baseline,
)
from baselines.baselines import CGX_CHANNELS, EMOTIV_CHANNELS


def _make_synthetic_dataset(seed: int = 5):
    rng = np.random.RandomState(seed)
    n_channels, n_bands, n_samples = 8, 4, 500
    channel_names = [f"CH{i}" for i in range(n_channels)]
    X = rng.randn(n_samples, n_channels * n_bands) * 1.0
    y = (X[:, 12:16].sum(axis=1) > 0).astype(int)  # CH3's block is the only informative one
    return X, y, channel_names, n_bands


def test_all_channels_baseline_returns_everything():
    _, _, channel_names, _ = _make_synthetic_dataset()
    assert all_channels_baseline(channel_names) == channel_names


def test_random_subset_baseline_shape_and_reproducibility():
    _, _, channel_names, _ = _make_synthetic_dataset()
    draws = random_subset_baseline(channel_names, k=3, seed=42, n_repeats=2)
    assert len(draws) == 2
    assert all(len(d) == 3 for d in draws)
    # same seed -> same draw (reproducibility, needed since results must be
    # exactly reproducible for a Q1/Q2 submission, never re-estimated)
    draws_again = random_subset_baseline(channel_names, k=3, seed=42, n_repeats=2)
    assert draws == draws_again


def test_manual_frontal_baseline_full_on_cgx_partial_on_emotiv():
    cgx_result = manual_frontal_baseline(CGX_CHANNELS)
    assert set(cgx_result) == {"AF7", "Fp1", "Fpz", "F7", "Fz"}, (
        "expected the full manual frontal cluster to be available on CGX (30 channels)"
    )
    try:
        manual_frontal_baseline(EMOTIV_CHANNELS)
        raised = False
    except ValueError:
        raised = True
    assert raised, (
        "expected manual_frontal_baseline to raise on Emotiv (14 channels), where only "
        "F7 of the 5-channel cluster is present -- a silent 1-channel baseline would be "
        "a materially weaker, undocumented comparison"
    )


def test_mutual_information_baseline_recovers_informative_channel():
    X, y, channel_names, n_bands = _make_synthetic_dataset()
    selected = mutual_information_baseline(X, y, channel_names, n_bands, k=2)
    assert selected[0] == "CH3"


def test_anova_fscore_baseline_recovers_informative_channel():
    X, y, channel_names, n_bands = _make_synthetic_dataset()
    selected = anova_fscore_baseline(X, y, channel_names, n_bands, k=2)
    assert selected[0] == "CH3"


def test_correlation_baseline_recovers_informative_channel():
    X, y, channel_names, n_bands = _make_synthetic_dataset()
    selected = correlation_baseline(X, y, channel_names, n_bands, k=2)
    assert selected[0] == "CH3"


def test_greedy_forward_selection_recovers_informative_channel():
    X, y, channel_names, n_bands = _make_synthetic_dataset()
    selected = greedy_forward_selection(
        model_factory=lambda: RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        X=X, y=y, channel_names=channel_names, n_bands=n_bands, k=2, cv_folds=5,
    )
    assert selected[0] == "CH3"


def test_permutation_importance_baseline_recovers_informative_channel():
    X, y, channel_names, n_bands = _make_synthetic_dataset()
    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X, y)
    selected = permutation_importance_baseline(
        clf, X, y, channel_names, n_bands, k=2, n_repeats=10, seed=42
    )
    assert selected[0] == "CH3"


if __name__ == "__main__":
    test_all_channels_baseline_returns_everything()
    print("test_all_channels_baseline_returns_everything: PASSED")
    test_random_subset_baseline_shape_and_reproducibility()
    print("test_random_subset_baseline_shape_and_reproducibility: PASSED")
    test_manual_frontal_baseline_full_on_cgx_partial_on_emotiv()
    print("test_manual_frontal_baseline_full_on_cgx_partial_on_emotiv: PASSED")
    test_mutual_information_baseline_recovers_informative_channel()
    print("test_mutual_information_baseline_recovers_informative_channel: PASSED")
    test_anova_fscore_baseline_recovers_informative_channel()
    print("test_anova_fscore_baseline_recovers_informative_channel: PASSED")
    test_correlation_baseline_recovers_informative_channel()
    print("test_correlation_baseline_recovers_informative_channel: PASSED")
    test_greedy_forward_selection_recovers_informative_channel()
    print("test_greedy_forward_selection_recovers_informative_channel: PASSED")
    test_permutation_importance_baseline_recovers_informative_channel()
    print("test_permutation_importance_baseline_recovers_informative_channel: PASSED")
