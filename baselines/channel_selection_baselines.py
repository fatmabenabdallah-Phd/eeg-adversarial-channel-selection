"""
baselines.channel_selection_baselines
=======================================
The comparison suite for the adversarial channel-importance method
(methods/adversarial_importance.py). Every function here returns a
selected subset of channel names (or a full ranking, where relevant),
using the same feature layout as extract_band_power_features
(contiguous n_bands-wide blocks per channel, optionally a trailing
global ratio column that no per-channel selection method here touches).

Baselines implemented (P3 of the project plan):
  1. all_channels_baseline          -- upper-reference, no selection
  2. random_subset_baseline         -- random channels, same size k
  3. manual_frontal_baseline        -- the AF7/Fp1/Fpz/F7/Fz cluster
                                        this project aims to replace
  4. mutual_information_baseline    -- filter method
  5. anova_fscore_baseline          -- filter method
  6. correlation_baseline           -- filter method
  7. greedy_forward_selection       -- wrapper method
  8. permutation_importance_baseline -- surrogate/model-agnostic
                                        importance (the family NeuroXAI
                                        belongs to -- see note below)

Note on "NeuroXAI-style": this is a permutation-importance-based
surrogate ranking (shuffle a channel's feature block, measure the
resulting accuracy/AUC drop), representative of the surrogate-analysis
family of channel-selection methods that NeuroXAI (Lee et al., Expert
Systems with Applications, 2025) belongs to. It is NOT a literal
reimplementation of NeuroXAI's specific published algorithm -- if the
paper needs to claim a head-to-head against NeuroXAI itself rather
than "a representative surrogate-importance baseline", the original
method should be reimplemented from the paper directly before
submission, and this should be renamed/labeled accordingly to avoid
overclaiming in the manuscript.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np


MANUAL_FRONTAL_CLUSTER = ["AF7", "Fp1", "Fpz", "F7", "Fz"]


def _channel_feature_slice(channel_idx: int, n_bands: int, n_total_features: int) -> slice:
    """Same convention as methods.adversarial_importance._channel_feature_slice
    -- duplicated here (rather than imported) to keep the baselines
    module independently usable without a dependency on methods/."""
    start = channel_idx * n_bands
    end = start + n_bands
    if end > n_total_features:
        raise ValueError(
            f"_channel_feature_slice: channel_idx={channel_idx} with n_bands={n_bands} "
            f"exceeds n_total_features={n_total_features}."
        )
    return slice(start, end)


def all_channels_baseline(channel_names: List[str]) -> List[str]:
    """Upper-reference baseline: every channel, unmodified. Included so
    every other selection method's accuracy/AUC can be reported as a
    fraction of this ceiling."""
    return list(channel_names)


def random_subset_baseline(
    channel_names: List[str], k: int, seed: int = 42, n_repeats: int = 1
) -> List[List[str]]:
    """Random baseline: k channels chosen uniformly at random, with no
    information about the task. Returns `n_repeats` independent draws
    (different seeds) so downstream evaluation can report a
    distribution (mean +/- std accuracy) rather than a single lucky/
    unlucky draw -- a single random draw is not a meaningful baseline
    on its own for a Q1/Q2 submission.
    """
    if k > len(channel_names):
        raise ValueError(
            f"random_subset_baseline: k={k} exceeds the number of available "
            f"channels ({len(channel_names)})."
        )
    draws = []
    for r in range(n_repeats):
        rng = np.random.RandomState(seed + r)
        chosen = rng.choice(channel_names, size=k, replace=False)
        draws.append(list(chosen))
    return draws


def manual_frontal_baseline(channel_names: List[str]) -> List[str]:
    """The manual, literature-justified frontal cluster this project
    aims to replace with an automatic, empirically-validated method
    (AF7, Fp1, Fpz, F7, Fz -- frontal theta/beta ratio, a known but
    contested ADHD biomarker).

    IMPORTANT: this cluster is only fully available on the CGX montage
    (30 channels). On Emotiv (14 channels), only F7 is present --
    AF7, Fp1, Fpz, and Fz do not exist on that hardware. Rather than
    silently returning a partial/empty list, this function returns
    whichever of the 5 target channels are actually present in
    `channel_names` and raises if that intersection is suspiciously
    small (< 2 channels), since a "manual frontal baseline" that
    degenerates to 1 channel is a different, weaker comparison than
    intended and should be flagged, not silently run.
    """
    available = [ch for ch in MANUAL_FRONTAL_CLUSTER if ch in channel_names]
    missing = [ch for ch in MANUAL_FRONTAL_CLUSTER if ch not in channel_names]
    if len(available) < 2:
        raise ValueError(
            f"manual_frontal_baseline: only {available} of the target cluster "
            f"{MANUAL_FRONTAL_CLUSTER} are present in the given channel_names "
            f"(missing: {missing}) -- this montage may not support a meaningful "
            f"comparison against the manual frontal baseline. On Emotiv (14 ch), "
            f"only F7 is present; this is expected and should be reported as a "
            f"named limitation rather than silently run as a 1-channel baseline."
        )
    return available


def mutual_information_baseline(
    X: np.ndarray, y: np.ndarray, channel_names: List[str], n_bands: int, k: int, seed: int = 42
) -> List[str]:
    """Filter method: rank channels by the mutual information between
    their feature block and the label, aggregated (summed) over the
    channel's n_bands features, then take the top k. Uses sklearn's
    mutual_info_classif (nearest-neighbor-based continuous-feature MI
    estimator), a standard, well-established filter baseline in the
    EEG channel-selection literature.
    """
    from sklearn.feature_selection import mutual_info_classif

    mi_per_feature = mutual_info_classif(X, y, random_state=seed)
    n_channels = len(channel_names)
    mi_per_channel = np.zeros(n_channels)
    for c in range(n_channels):
        ch_slice = _channel_feature_slice(c, n_bands, X.shape[1])
        mi_per_channel[c] = mi_per_feature[ch_slice].sum()

    order = np.argsort(mi_per_channel)[::-1]
    return [channel_names[c] for c in order[:k]]


def anova_fscore_baseline(
    X: np.ndarray, y: np.ndarray, channel_names: List[str], n_bands: int, k: int
) -> List[str]:
    """Filter method: rank channels by the ANOVA F-score between their
    feature block and the label, aggregated (summed) over the
    channel's n_bands features, then take the top k."""
    from sklearn.feature_selection import f_classif

    f_scores, _ = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0)  # a constant feature yields F=nan
    n_channels = len(channel_names)
    f_per_channel = np.zeros(n_channels)
    for c in range(n_channels):
        ch_slice = _channel_feature_slice(c, n_bands, X.shape[1])
        f_per_channel[c] = f_scores[ch_slice].sum()

    order = np.argsort(f_per_channel)[::-1]
    return [channel_names[c] for c in order[:k]]


def correlation_baseline(
    X: np.ndarray, y: np.ndarray, channel_names: List[str], n_bands: int, k: int
) -> List[str]:
    """Filter method: rank channels by the sum of |Pearson correlation|
    between each of their n_bands features and the (0/1) label,
    aggregated over the channel's block, then take the top k."""
    n_channels = len(channel_names)
    n_total_features = X.shape[1]
    corr_per_feature = np.zeros(n_total_features)
    y_float = y.astype(float)
    for f in range(n_total_features):
        col = X[:, f]
        if np.std(col) == 0:
            corr_per_feature[f] = 0.0
        else:
            corr_per_feature[f] = abs(np.corrcoef(col, y_float)[0, 1])

    corr_per_channel = np.zeros(n_channels)
    for c in range(n_channels):
        ch_slice = _channel_feature_slice(c, n_bands, n_total_features)
        corr_per_channel[c] = corr_per_feature[ch_slice].sum()

    order = np.argsort(corr_per_channel)[::-1]
    return [channel_names[c] for c in order[:k]]


def greedy_forward_selection(
    model_factory: Callable[[], "object"],
    X: np.ndarray,
    y: np.ndarray,
    channel_names: List[str],
    n_bands: int,
    k: int,
    cv_folds: int = 5,
    seed: int = 42,
) -> List[str]:
    """Wrapper method: greedily adds one channel at a time (out of the
    channels not yet selected), each time picking whichever addition
    gives the highest cross-validated accuracy, until k channels are
    selected. Classic, strong wrapper baseline -- more expensive than
    the filter methods above (retrains a model at every candidate
    channel, at every step) but typically stronger, so it is an
    important baseline to include, not an optional one.

    model_factory: a zero-argument callable returning a fresh unfitted
    model instance (e.g. `lambda: RandomForestClassifier(n_estimators=200,
    class_weight='balanced', random_state=42)`), so a fresh model is
    trained at every candidate evaluation.
    """
    from sklearn.model_selection import cross_val_score

    n_channels = len(channel_names)
    n_total_features = X.shape[1]
    remaining = list(range(n_channels))
    selected: List[int] = []

    for _ in range(k):
        best_score = -np.inf
        best_channel = None
        for c in remaining:
            candidate_channels = selected + [c]
            feature_idx = np.concatenate(
                [
                    np.arange(
                        _channel_feature_slice(cc, n_bands, n_total_features).start,
                        _channel_feature_slice(cc, n_bands, n_total_features).stop,
                    )
                    for cc in candidate_channels
                ]
            )
            model = model_factory()
            scores = cross_val_score(
                model, X[:, feature_idx], y, cv=cv_folds, scoring="balanced_accuracy"
            )
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_channel = c

        selected.append(best_channel)
        remaining.remove(best_channel)

    return [channel_names[c] for c in selected]


def permutation_importance_baseline(
    model,
    X: np.ndarray,
    y: np.ndarray,
    channel_names: List[str],
    n_bands: int,
    k: int,
    n_repeats: int = 10,
    seed: int = 42,
    scoring: str = "balanced_accuracy",
) -> List[str]:
    """Surrogate/model-agnostic importance baseline -- see the module
    docstring's note on why this is labeled "permutation_importance",
    not "NeuroXAI", pending an exact reimplementation of that paper's
    algorithm if a literal head-to-head is required for submission.

    For each channel, shuffles that channel's feature block across
    samples (breaking its relationship with y while preserving its
    marginal distribution and every other channel's values), and
    measures the resulting drop in `scoring` on a fitted `model`,
    averaged over `n_repeats` independent shuffles. Channels whose
    shuffling causes the largest score drop are ranked most important.

    model: an already-FITTED model (unlike greedy_forward_selection,
    which fits fresh models internally -- permutation importance is
    evaluated against one fixed fitted model, consistent with how
    sklearn's own permutation_importance works).
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    scorers = {
        "balanced_accuracy": lambda yt, yp: balanced_accuracy_score(yt, yp),
    }
    if scoring not in scorers:
        raise ValueError(f"permutation_importance_baseline: unsupported scoring={scoring!r}")
    scorer = scorers[scoring]

    baseline_score = scorer(y, model.predict(X))

    n_channels = len(channel_names)
    n_total_features = X.shape[1]
    importance = np.zeros(n_channels)

    for c in range(n_channels):
        ch_slice = _channel_feature_slice(c, n_bands, n_total_features)
        drops = []
        for r in range(n_repeats):
            rng = np.random.RandomState(seed + c * 1000 + r)
            X_perm = X.copy()
            perm_idx = rng.permutation(X.shape[0])
            X_perm[:, ch_slice] = X_perm[perm_idx][:, ch_slice]
            perm_score = scorer(y, model.predict(X_perm))
            drops.append(baseline_score - perm_score)
        importance[c] = np.mean(drops)

    order = np.argsort(importance)[::-1]
    return [channel_names[c] for c in order[:k]]
