"""
methods.adversarial_importance
================================
Black-box (query-based) adversarial channel-importance for tabular
band-power features (as produced by baselines.baselines.extract_band_power_features),
scored against a non-differentiable model (Random Forest).

Core idea (adapted from FIMAP, Chapman-Rounds et al., AAAI 2021 --
"feature importance = minimal adversarial perturbation" -- ported here
to a query-based / gradient-free setting, and applied at the
electrode-channel level rather than the individual-feature level):

    importance_c = 1 / epsilon_c

where epsilon_c is the smallest L2-norm perturbation, restricted to the
feature block belonging to channel c, that flips the model's predicted
class for a given sample. Channels requiring only a tiny perturbation
to flip the prediction are the ones the model relies on most.

No gradient is used anywhere -- the model is queried as a black box
(model.predict / model.predict_proba only), since RandomForestClassifier
is not differentiable. This trades query-efficiency for model-agnosticism:
works identically for RF, SVM, or (later) a graph-constrained group of
channels.

Search strategy per channel, per sample:
  1. Coarse phase -- geometrically increasing radius (starting at
     `epsilon_init`, multiplied by `growth_factor` each step), trying
     `n_directions` random unit directions restricted to the channel's
     feature block at each radius, until at least one direction flips
     the prediction or `max_epsilon` is exceeded.
  2. Refinement phase -- binary search between the last non-flipping
     radius and the first flipping radius (along the flipping
     direction found), for `n_refine_steps` iterations, to tighten the
     epsilon estimate.

If no flip is found within `max_epsilon`, epsilon_c is recorded as
`max_epsilon` (a right-censored value -- the channel is at least this
robust) rather than left undefined, so downstream ranking always has a
finite number per channel/sample and censored cases are explicitly
flagged for reporting/QA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


@dataclass
class ChannelPerturbationResult:
    channel_idx: int
    channel_name: str
    epsilon: float
    censored: bool  # True if no flip was found within max_epsilon
    n_queries: int


@dataclass
class SampleImportanceResult:
    sample_idx: int
    original_pred: int
    per_channel: List[ChannelPerturbationResult] = field(default_factory=list)


def _channel_feature_slice(channel_idx: int, n_bands: int, n_total_features: int) -> slice:
    """Feature-block slice for channel `channel_idx`, assuming the
    layout produced by extract_band_power_features: features are laid
    out as [ch0_band0..ch0_band{n_bands-1}, ch1_band0.., ..., theta_beta_ratio].
    The trailing global theta_beta_ratio column (if present, i.e. when
    n_total_features == n_channels*n_bands + 1) is never part of any
    single channel's block and is left untouched by any perturbation.
    """
    start = channel_idx * n_bands
    end = start + n_bands
    if end > n_total_features:
        raise ValueError(
            f"_channel_feature_slice: channel_idx={channel_idx} with n_bands={n_bands} "
            f"exceeds n_total_features={n_total_features}."
        )
    return slice(start, end)


def _predict_class(predict_fn: Callable[[np.ndarray], np.ndarray], x_row: np.ndarray) -> int:
    """predict_fn must accept a 2D array (n_samples, n_features) and
    return an array of predicted class labels (0/1). We always pass a
    (1, n_features) row and take the single result."""
    return int(predict_fn(x_row.reshape(1, -1))[0])


def find_minimal_epsilon_for_channel(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x_row: np.ndarray,
    channel_idx: int,
    n_bands: int,
    original_pred: int,
    epsilon_init: float = 1e-3,
    growth_factor: float = 2.0,
    max_epsilon: float = 10.0,
    n_directions: int = 20,
    n_refine_steps: int = 8,
    seed: int = 42,
    feature_scale: Optional[np.ndarray] = None,
) -> ChannelPerturbationResult:
    """Query-based minimal-perturbation search restricted to one
    channel's feature block, for a single sample `x_row`.

    feature_scale: REQUIRED for meaningful cross-channel/cross-band
    comparison on real EEG band-power features. Pass the per-feature
    standard deviation across the training set (same length as
    x_row -- typically `X_train.std(axis=0)`). Perturbation directions
    are scaled elementwise by feature_scale before being added, and
    epsilon is then interpreted in "typical-std units" rather than raw
    feature units. Without this, epsilon comparisons across bands are
    not meaningful: EEG band power follows an approximate 1/f law
    (delta power is typically ~50x larger in magnitude than gamma
    power), so a fixed raw-unit epsilon represents a tiny fraction of a
    standard deviation for high-amplitude bands (e.g. delta) and a huge
    fraction for low-amplitude bands (e.g. gamma) -- biasing the
    resulting ranking toward low-amplitude bands regardless of their
    true importance to the model. If omitted, defaults to all-ones
    (raw units) with no correction, for backward-compatible use on
    already-standardized inputs (e.g. if the caller standardized X
    globally with sklearn.preprocessing.StandardScaler before calling
    this function -- in which case feature_scale=None is correct,
    since the scale correction has already been applied upstream).
    """
    rng = np.random.RandomState(seed + channel_idx)
    n_total_features = x_row.shape[0]
    ch_slice = _channel_feature_slice(channel_idx, n_bands, n_total_features)
    block_dim = ch_slice.stop - ch_slice.start

    if feature_scale is None:
        block_scale = np.ones(block_dim)
    else:
        block_scale = feature_scale[ch_slice]
        if np.any(block_scale <= 0):
            raise ValueError(
                f"find_minimal_epsilon_for_channel: feature_scale contains non-positive "
                f"values for channel_idx={channel_idx} -- check for a constant/zero-variance "
                f"feature (e.g. a flat channel that should have been interpolated upstream)."
            )

    n_queries = 0
    epsilon = epsilon_init
    last_non_flip_epsilon = 0.0
    flip_found = False
    flip_direction = None
    flip_epsilon = None

    while epsilon <= max_epsilon and not flip_found:
        # Batch all n_directions perturbations for this epsilon into a
        # single predict() call instead of querying one row at a time --
        # sklearn's per-call Python overhead dominates for single-row
        # predictions, so batching is a large real speedup, not just a
        # style preference, at the sample/channel counts this project
        # needs (121+35 subjects x up to 30 channels).
        directions = rng.randn(n_directions, block_dim)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        directions = directions / norms

        batch = np.tile(x_row, (n_directions, 1))
        # Scale the (unit) direction by the feature's natural std BEFORE
        # applying epsilon, so epsilon is in comparable "std units"
        # across bands/channels of very different natural amplitude.
        batch[:, ch_slice] = batch[:, ch_slice] + epsilon * directions * block_scale
        preds = np.asarray(predict_fn(batch))
        n_queries += n_directions

        flips = np.where(preds != original_pred)[0]
        if len(flips) > 0:
            flip_found = True
            flip_direction = directions[flips[0]]
            flip_epsilon = epsilon
            break

        last_non_flip_epsilon = epsilon
        epsilon *= growth_factor

    if not flip_found:
        return ChannelPerturbationResult(
            channel_idx=channel_idx,
            channel_name="",  # filled in by caller
            epsilon=max_epsilon,
            censored=True,
            n_queries=n_queries,
        )

    # Binary-search refinement between last_non_flip_epsilon (no flip)
    # and flip_epsilon (flip), along the direction that produced the flip.
    lo, hi = last_non_flip_epsilon, flip_epsilon
    for _ in range(n_refine_steps):
        mid = (lo + hi) / 2.0
        perturbed = x_row.copy()
        perturbed[ch_slice] = perturbed[ch_slice] + mid * flip_direction * block_scale
        pred = _predict_class(predict_fn, perturbed)
        n_queries += 1
        if pred != original_pred:
            hi = mid
        else:
            lo = mid

    return ChannelPerturbationResult(
        channel_idx=channel_idx,
        channel_name="",  # filled in by caller
        epsilon=hi,
        censored=False,
        n_queries=n_queries,
    )


def compute_sample_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x_row: np.ndarray,
    channel_names: List[str],
    n_bands: int,
    sample_idx: int = 0,
    feature_scale: Optional[np.ndarray] = None,
    **search_kwargs,
) -> SampleImportanceResult:
    """Runs find_minimal_epsilon_for_channel for every channel on one
    sample, returning per-channel epsilon (and hence importance=1/epsilon).

    feature_scale: see find_minimal_epsilon_for_channel -- pass the
    per-feature std across the training set for meaningful cross-band
    comparison on raw (non-standardized) band-power features.
    """
    original_pred = _predict_class(predict_fn, x_row)
    n_channels = len(channel_names)

    per_channel = []
    for c in range(n_channels):
        result = find_minimal_epsilon_for_channel(
            predict_fn=predict_fn,
            x_row=x_row,
            channel_idx=c,
            n_bands=n_bands,
            original_pred=original_pred,
            feature_scale=feature_scale,
            **search_kwargs,
        )
        result.channel_name = channel_names[c]
        per_channel.append(result)

    return SampleImportanceResult(
        sample_idx=sample_idx, original_pred=original_pred, per_channel=per_channel
    )


def rank_channels(
    model,
    X: np.ndarray,
    channel_names: List[str],
    n_bands: int,
    max_samples: Optional[int] = None,
    seed: int = 42,
    use_feature_scale: bool = True,
    **search_kwargs,
) -> "np.ndarray":
    """Runs compute_sample_importance across (a subset of) X and
    aggregates into a single per-channel importance ranking.

    model: any object with .predict(X) -> array of 0/1 (e.g. a fitted
    sklearn RandomForestClassifier). Only .predict is used -- no
    gradient, no .predict_proba dependency, so this works unchanged for
    any black-box classifier.

    use_feature_scale: if True (default), computes feature_scale as
    X.std(axis=0) and passes it through so epsilon is comparable across
    bands/channels of different natural amplitude (see
    find_minimal_epsilon_for_channel's docstring for why this matters
    on raw EEG band-power features). Set to False only if X has already
    been standardized upstream (e.g. via sklearn's StandardScaler).

    Returns a structured array with fields (channel_name, mean_epsilon,
    median_epsilon, pct_censored, mean_importance), sorted by
    mean_importance descending (most important channel first).
    """
    n_samples = X.shape[0]
    if max_samples is not None and max_samples < n_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_samples, size=max_samples, replace=False)
    else:
        idx = np.arange(n_samples)

    feature_scale = X.std(axis=0) if use_feature_scale else None

    n_channels = len(channel_names)
    all_epsilons = np.zeros((len(idx), n_channels))
    all_censored = np.zeros((len(idx), n_channels), dtype=bool)

    for row_i, sample_i in enumerate(idx):
        result = compute_sample_importance(
            predict_fn=model.predict,
            x_row=X[sample_i],
            channel_names=channel_names,
            n_bands=n_bands,
            sample_idx=int(sample_i),
            seed=seed,
            feature_scale=feature_scale,
            **search_kwargs,
        )
        for c, pc in enumerate(result.per_channel):
            all_epsilons[row_i, c] = pc.epsilon
            all_censored[row_i, c] = pc.censored

    mean_epsilon = all_epsilons.mean(axis=0)
    median_epsilon = np.median(all_epsilons, axis=0)
    pct_censored = all_censored.mean(axis=0)
    mean_importance = 1.0 / np.maximum(mean_epsilon, 1e-12)

    dtype = [
        ("channel_name", "U16"),
        ("mean_epsilon", "f8"),
        ("median_epsilon", "f8"),
        ("pct_censored", "f8"),
        ("mean_importance", "f8"),
    ]
    out = np.zeros(n_channels, dtype=dtype)
    out["channel_name"] = channel_names
    out["mean_epsilon"] = mean_epsilon
    out["median_epsilon"] = median_epsilon
    out["pct_censored"] = pct_censored
    out["mean_importance"] = mean_importance

    return np.sort(out, order="mean_importance")[::-1]
