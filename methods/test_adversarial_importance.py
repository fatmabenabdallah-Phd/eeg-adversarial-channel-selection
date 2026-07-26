"""
methods.test_adversarial_importance
=====================================
Regression tests for the black-box adversarial channel-importance
method, using synthetic data where the truly informative channel is
known in advance -- so the ranking's correctness can actually be
checked, not just its runtime behavior.

Run with: python -m pytest methods/test_adversarial_importance.py -v
(or: python methods/test_adversarial_importance.py to run standalone)
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from methods.adversarial_importance import rank_channels, rank_channels_graph_constrained


def test_recovers_single_informative_channel_equal_scale():
    """All channels share the same natural scale -- the simplest case.
    The method must rank the one truly informative channel first."""
    rng = np.random.RandomState(0)
    n_channels, n_bands, n_samples = 6, 5, 300
    channel_names = [f"CH{i}" for i in range(n_channels)]

    X = rng.randn(n_samples, n_channels * n_bands) * 0.5
    y = (X[:, 10:15].sum(axis=1) > 0).astype(int)  # CH2's block

    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X, y)
    assert clf.score(X, y) > 0.9, "sanity check: RF must actually learn the synthetic task"

    ranking = rank_channels(
        model=clf, X=X, channel_names=channel_names, n_bands=n_bands,
        max_samples=25, seed=42,
        epsilon_init=0.05, growth_factor=1.5, max_epsilon=5.0,
        n_directions=15, n_refine_steps=6,
    )
    assert ranking[0]["channel_name"] == "CH2", (
        f"expected CH2 (the only informative channel) to rank first, "
        f"got {ranking[0]['channel_name']}"
    )


def test_scale_correction_needed_for_heterogeneous_amplitudes():
    """Channels have very different natural amplitudes (as in real EEG,
    where delta power is typically ~50x larger than gamma power).
    use_feature_scale=True must recover the true informative channel;
    use_feature_scale=False is EXPECTED to fail this -- asserting that
    failure here so a future accidental fix doesn't silently mask a
    regression in the opposite direction (i.e. this test also documents
    why the correction exists)."""
    rng = np.random.RandomState(1)
    n_channels, n_bands, n_samples = 4, 5, 300
    channel_names = [f"CH{i}" for i in range(n_channels)]
    scales = np.array([100.0, 50.0, 10.0, 2.0])
    X = np.hstack([rng.randn(n_samples, n_bands) * s for s in scales])
    y = (X[:, 5:10].sum(axis=1) > 0).astype(int)  # CH1's block (mid-scale, not extreme)

    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X, y)
    assert clf.score(X, y) > 0.9

    ranking_corrected = rank_channels(
        model=clf, X=X, channel_names=channel_names, n_bands=n_bands,
        max_samples=20, seed=42, use_feature_scale=True,
        epsilon_init=0.05, growth_factor=1.5, max_epsilon=6.0,
        n_directions=15, n_refine_steps=6,
    )
    assert ranking_corrected[0]["channel_name"] == "CH1", (
        "with feature_scale correction, the method must recover the true "
        "informative channel despite heterogeneous channel amplitudes"
    )

    ranking_uncorrected = rank_channels(
        model=clf, X=X, channel_names=channel_names, n_bands=n_bands,
        max_samples=20, seed=42, use_feature_scale=False,
        epsilon_init=0.05, growth_factor=1.5, max_epsilon=6.0,
        n_directions=15, n_refine_steps=6,
    )
    # Documents the failure mode this correction fixes -- if this
    # assertion ever fails, it likely means max_epsilon/epsilon_init
    # changed enough to accidentally cover the scale gap, not that the
    # underlying bias went away.
    assert ranking_uncorrected[0]["channel_name"] != "CH1" or ranking_uncorrected["pct_censored"].mean() == 1.0, (
        "expected the uncorrected search to fail to recover CH1 (or to be "
        "fully censored) on heterogeneous-scale channels -- if it now "
        "succeeds, re-check whether the scale-bias problem was actually fixed "
        "or just coincidentally avoided by the current epsilon range"
    )


def test_graph_constrained_beats_isolated_on_difference_interaction():
    """Decision boundary: y=1 if (CH1_signal - CH2_signal) > 0 -- a
    genuine two-channel interaction where CH1 and CH2 are structural
    neighbors. Theory: to increase (CH1-CH2) by D, perturbing CH1 alone
    costs L2 norm D, but perturbing CH1 by +D/2 AND CH2 by -D/2 jointly
    costs D/sqrt(2) ~= 0.707*D -- strictly cheaper. The graph-constrained
    search (which perturbs CH1+CH2 jointly) must therefore find a
    smaller epsilon than the isolated single-channel search, and this
    must hold reproducibly (not just by chance on one run).

    Note: an earlier AND-based interaction test (y=1 if CH1>0 AND
    CH2>0) did NOT show a clear gap, because an AND condition can often
    be flipped by moving either channel alone when the other is already
    on the right side -- documented here so a future reader doesn't
    assume ANY two-channel interaction produces this gap; specifically
    a difference/ratio-type interaction is needed.
    """
    rng = np.random.RandomState(3)
    n_channels, n_bands, n_samples = 5, 3, 400
    channel_names = [f"CH{i}" for i in range(n_channels)]

    X = rng.randn(n_samples, n_channels * n_bands) * 1.0
    ch1_signal = X[:, 3:6].sum(axis=1)
    ch2_signal = X[:, 6:9].sum(axis=1)
    y = ((ch1_signal - ch2_signal) > 0).astype(int)

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X, y)
    assert clf.score(X, y) > 0.95

    adjacency = np.zeros((n_channels, n_channels))
    adjacency[1, 2] = 1
    adjacency[2, 1] = 1

    kwargs = dict(epsilon_init=0.05, growth_factor=1.4, max_epsilon=4.0, n_directions=20, n_refine_steps=8)

    ranking_isolated = rank_channels(
        model=clf, X=X, channel_names=channel_names, n_bands=n_bands, max_samples=25, seed=42, **kwargs
    )
    ranking_graph = rank_channels_graph_constrained(
        model=clf, X=X, channel_names=channel_names, n_bands=n_bands,
        adjacency=adjacency, max_samples=25, seed=42, **kwargs
    )

    eps_ch1_isolated = ranking_isolated[ranking_isolated["channel_name"] == "CH1"]["mean_epsilon"][0]
    eps_ch1_graph = ranking_graph[ranking_graph["channel_name"] == "CH1"]["mean_epsilon"][0]

    # Conservative bound: require at least a 5% reduction, well below the
    # ~41% theoretical ceiling (random-direction search is not an exact
    # optimizer, so the empirical gain is smaller than the theoretical
    # ratio -- observed ~22% in the reference run this test is based on).
    assert eps_ch1_graph < eps_ch1_isolated * 0.95, (
        f"expected the graph-constrained search to find a meaningfully smaller "
        f"epsilon for CH1 than the isolated search on this difference-interaction "
        f"task (isolated={eps_ch1_isolated:.3f}, graph={eps_ch1_graph:.3f})"
    )


if __name__ == "__main__":
    test_recovers_single_informative_channel_equal_scale()
    print("test_recovers_single_informative_channel_equal_scale: PASSED")
    test_scale_correction_needed_for_heterogeneous_amplitudes()
    print("test_scale_correction_needed_for_heterogeneous_amplitudes: PASSED")
    test_graph_constrained_beats_isolated_on_difference_interaction()
    print("test_graph_constrained_beats_isolated_on_difference_interaction: PASSED")
