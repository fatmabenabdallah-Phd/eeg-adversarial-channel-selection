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

from methods.adversarial_importance import (
    rank_channels,
    rank_channels_graph_constrained,
    compute_epsilon_matrix,
    per_subject_top_k_channels,
    subgroup_ranking_agreement,
)


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


def test_subject_adaptive_detects_subgroup_dependent_importance():
    """Two subgroups (A, B) rely on DIFFERENT channels for the same
    task -- CH0 decides the label for group A, CH4 decides it for
    group B, with CH5 acting as an implicit group indicator (mirroring
    a real systematic subgroup difference, e.g. age-related EEG
    amplitude, WITHOUT giving the model an explicit group label
    feature). This directly parallels the project's own empirical
    finding on the RF's built-in feature importances (Spearman
    rho=0.004 between two demographic subgroups on BALLADEER) -- here
    we ask the same question of the adversarial epsilon-based ranking.

    IMPORTANT LESSON (documented from an earlier failed attempt at this
    test): the epsilon matrix MUST be computed on held-out test points,
    not training points. An unconstrained-depth RF trained on this kind
    of task reaches 100% train accuracy by partly memorizing individual
    training points via splits unrelated to the intended CH0/CH4 signal,
    which fully masked the expected subgroup divergence (first attempt:
    rho=0.86, no divergence). Evaluating on genuinely held-out test
    points (with a shallow-enough model, max_depth=6, that actually
    generalizes -- verified here via test accuracy > 0.95 per subgroup)
    is what makes the subgroup divergence visible (rho drops to ~0.38,
    and per-subject top-1 correctly diverges to CH0 for group A / CH4
    for group B in the large majority of subjects).
    """
    from collections import Counter
    from sklearn.model_selection import train_test_split

    rng_seed = 4
    np.random.seed(rng_seed)
    n_channels, n_bands = 6, 1
    channel_names = [f"CH{i}" for i in range(n_channels)]
    n_per_group = 300
    n_samples = 2 * n_per_group

    X = np.random.randn(n_samples, n_channels * n_bands) * 1.0
    group = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    X[:n_per_group, 5] += 5.0  # CH5 = implicit group indicator (A)
    X[n_per_group:, 5] -= 5.0  # CH5 = implicit group indicator (B)

    y = np.zeros(n_samples, dtype=int)
    y[:n_per_group] = (X[:n_per_group, 0] > 0).astype(int)  # CH0 decides for group A
    y[n_per_group:] = (X[n_per_group:, 4] > 0).astype(int)  # CH4 decides for group B

    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        X, y, group, test_size=0.5, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    for g in ["A", "B"]:
        mask = group_test == g
        assert clf.score(X_test[mask], y_test[mask]) > 0.95, (
            f"sanity check: RF must genuinely generalize (not just memorize) "
            f"the group-{g} rule on held-out test data"
        )

    eps_matrix, censored_matrix, idx = compute_epsilon_matrix(
        model=clf, X=X_test, channel_names=channel_names, n_bands=n_bands,
        max_samples=60, seed=42,
        epsilon_init=0.05, growth_factor=1.5, max_epsilon=4.0, n_directions=15, n_refine_steps=6,
    )
    group_sampled = group_test[idx]

    agreement = subgroup_ranking_agreement(eps_matrix, group_sampled)
    rho_ab = agreement[("A", "B")]
    assert rho_ab < 0.7, (
        f"expected a clearly reduced Spearman agreement between subgroups relying on "
        f"different channels, got rho={rho_ab:.3f} (too high -- suggests the subgroup "
        f"divergence was not detected)"
    )

    top1 = [chs[0] for chs in per_subject_top_k_channels(eps_matrix, channel_names, k=1)]
    top1_a = Counter(top1[i] for i in range(len(idx)) if group_sampled[i] == "A")
    top1_b = Counter(top1[i] for i in range(len(idx)) if group_sampled[i] == "B")
    assert top1_a.most_common(1)[0][0] == "CH0", (
        f"expected CH0 to be group A's most common top-1 channel, got {top1_a.most_common(3)}"
    )
    assert top1_b.most_common(1)[0][0] == "CH4", (
        f"expected CH4 to be group B's most common top-1 channel, got {top1_b.most_common(3)}"
    )


if __name__ == "__main__":
    test_recovers_single_informative_channel_equal_scale()
    print("test_recovers_single_informative_channel_equal_scale: PASSED")
    test_scale_correction_needed_for_heterogeneous_amplitudes()
    print("test_scale_correction_needed_for_heterogeneous_amplitudes: PASSED")
    test_graph_constrained_beats_isolated_on_difference_interaction()
    print("test_graph_constrained_beats_isolated_on_difference_interaction: PASSED")
    test_subject_adaptive_detects_subgroup_dependent_importance()
    print("test_subject_adaptive_detects_subgroup_dependent_importance: PASSED")
