"""
methods.split_half_stability
===============================
Second robustness check requested alongside the PLI check: split the
subjects into two random halves, run permutation_screen_channels
independently on each half, and compare the resulting channel
rankings. A genuine, stable effect should produce similar rankings in
both halves (high Spearman correlation, meaningful overlap in the
top-ranked channels); an artifact or an effect driven by a handful of
influential subjects should be unstable across the split (low or
near-zero correlation, inconsistent top channels).
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.stats import spearmanr

from methods.permutation_screening import permutation_screen_channels


def split_half_stability_check(
    X: np.ndarray, y: np.ndarray, channel_names: List[str], n_bands: int,
    n_permutations: int = 2000, seed: int = 42, top_k: int = 5,
) -> dict:
    """Splits subjects into two random halves (stratified on y to keep
    class balance comparable in both halves), runs the permutation
    screen independently on each half, and returns:
      - spearman_rho: rank correlation between the two halves'
        observed_stat rankings across all channels
      - spearman_p: the correlation's own p-value
      - top_k_half_a / top_k_half_b: each half's top-k channels by
        p-value
      - top_k_overlap: how many of the top-k channels are shared
        between the two halves (out of top_k)

    IMPORTANT CAVEAT on spearman_rho, found empirically while
    validating this function: when only a SMALL number of channels
    carry real signal and the rest are pure noise (the realistic case
    for this project, e.g. 2 real channels out of 15-30), spearman_rho
    computed across ALL channels can be LOW even when the truly
    informative channels are perfectly recovered in both halves' top
    ranks -- because the remaining majority-noise channels' mutual
    ordering is arbitrary and independently randomized in each half,
    which dilutes the overall rank correlation. In this sparse-signal
    regime, top_k_overlap (and manually checking whether the SAME
    specific channels appear in both halves' top_k lists) is the more
    informative and directly interpretable stability metric;
    spearman_rho should be read as a secondary, whole-ranking summary,
    not as the primary evidence of stability.
    """
    rng = np.random.RandomState(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    half_a_idx = np.concatenate([idx0[: len(idx0) // 2], idx1[: len(idx1) // 2]])
    half_b_idx = np.concatenate([idx0[len(idx0) // 2:], idx1[len(idx1) // 2:]])

    result_a = permutation_screen_channels(
        X[half_a_idx], y[half_a_idx], channel_names, n_bands, n_permutations=n_permutations, seed=seed
    )
    result_b = permutation_screen_channels(
        X[half_b_idx], y[half_b_idx], channel_names, n_bands, n_permutations=n_permutations, seed=seed + 1
    )

    # Re-align both results to the same channel order before correlating
    stat_a = {row["channel_name"]: row["observed_stat"] for row in result_a}
    stat_b = {row["channel_name"]: row["observed_stat"] for row in result_b}
    stats_a_ordered = np.array([stat_a[ch] for ch in channel_names])
    stats_b_ordered = np.array([stat_b[ch] for ch in channel_names])

    rho, p = spearmanr(stats_a_ordered, stats_b_ordered)

    top_k_half_a = list(result_a["channel_name"][:top_k])
    top_k_half_b = list(result_b["channel_name"][:top_k])
    overlap = len(set(top_k_half_a) & set(top_k_half_b))

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "top_k_half_a": top_k_half_a,
        "top_k_half_b": top_k_half_b,
        "top_k_overlap": overlap,
        "top_k": top_k,
        "n_half_a": len(half_a_idx),
        "n_half_b": len(half_b_idx),
    }
