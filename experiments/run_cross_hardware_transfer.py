"""
experiments.run_cross_hardware_transfer
==========================================
THE central remaining positive-result candidate for this project: does
a channel subset selected on CGX (30-channel montage) still perform
well when transferred to Emotiv (14-channel montage), using only the
12 channels physically common to both (COMMON_CHANNELS)?

Design (avoids leakage in both directions):
  - CGX-side channel selection is computed ONCE from ALL 121 CGX
    subjects (CGX is a completely separate dataset from Emotiv's CV
    folds -- no leakage risk from using all of it).
  - The "same-hardware ceiling" comparison (Emotiv's own channel
    selection) is instead computed PER FOLD from that fold's own
    Emotiv training subjects only -- consistent with how
    run_emotiv_experiment.py avoids leakage, so the ceiling comparison
    is fair, not inflated by peeking at test data.
  - Every method (CGX-transferred selections, Emotiv-own selections,
    random-common, all-common) is evaluated identically: a FRESH RF is
    trained on that fold's Emotiv training subjects (restricted to the
    given channel subset's features) and evaluated on that fold's
    Emotiv test subjects -- so every method's accuracy number reflects
    genuine held-out Emotiv performance, never CGX performance.

The headline comparison this experiment is FOR: does CGX-transferred
adversarial_isolated/adversarial_graph_constrained perform comparably
to Emotiv-own adversarial selection (the same-hardware ceiling), and
better than random-common or all-common? If yes, that is the paper's
strongest remaining claim, independent of whether channel selection
shows a same-hardware accuracy advantage on either dataset alone
(which, per the CGX and Nasrabadi within-hardware results, it does
not, with negligible-to-modest, non-significant effect sizes).
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.baselines import COMMON_CHANNELS, CGX_CHANNELS, EMOTIV_CHANNELS, train_rf_baseline
from baselines.channel_selection_baselines import (
    random_subset_baseline, mutual_information_baseline, anova_fscore_baseline,
    correlation_baseline, permutation_importance_baseline,
)
from experiments.run_cgx_experiment import evaluate_channel_subset
from methods.adversarial_importance import rank_channels, rank_channels_graph_constrained
from preprocessing.structural_graph import build_structural_knn_graph

N_BANDS = 5
COMMON_GRAPH_K = 4  # 12 channels total -- same reasoning as Emotiv's k=4 (k=8 would be near-complete)


def _restrict_to_common_channels(
    X: np.ndarray, source_channel_names: List[str], common_channels: List[str], n_bands: int,
) -> np.ndarray:
    """Extracts, from a full-montage feature matrix X (n_subjects,
    n_source_channels*n_bands [+1 ratio col, dropped here]), just the
    feature blocks belonging to `common_channels`, in `common_channels`
    order (so CGX-side and Emotiv-side restricted matrices have
    IDENTICALLY ordered columns, required for the RF trained on one
    montage's restricted features to mean the same thing when applied
    conceptually to the other -- in practice a fresh RF is always
    retrained on the target hardware, but consistent column order
    still matters for channel-name bookkeeping downstream).
    """
    feature_idx = []
    for ch in common_channels:
        c = source_channel_names.index(ch)
        feature_idx.extend(range(c * n_bands, c * n_bands + n_bands))
    return X[:, feature_idx]


def select_channels_from_cgx(
    X_cgx: np.ndarray, y_cgx: np.ndarray, common_channels: List[str] = COMMON_CHANNELS,
    n_bands: int = N_BANDS, adversarial_max_samples: int = 40, seed: int = 42,
    **adversarial_search_kwargs,
) -> dict:
    """Computes every channel-selection method's FULL ranking of
    `common_channels`, using ALL of X_cgx/y_cgx (CGX's 121 subjects,
    restricted to the 12 common-channel feature blocks). Returns
    {method_name: [channel_names sorted most-to-least important]}.
    """
    X_common = _restrict_to_common_channels(X_cgx, CGX_CHANNELS, common_channels, n_bands)
    clf = train_rf_baseline(X_common, y_cgx)
    adjacency = build_structural_knn_graph(common_channels, k=COMMON_GRAPH_K)

    rankings = {
        "mutual_information": mutual_information_baseline(X_common, y_cgx, common_channels, n_bands, k=len(common_channels)),
        "anova_fscore": anova_fscore_baseline(X_common, y_cgx, common_channels, n_bands, k=len(common_channels)),
        "correlation": correlation_baseline(X_common, y_cgx, common_channels, n_bands, k=len(common_channels)),
        "permutation_importance": permutation_importance_baseline(
            clf, X_common, y_cgx, common_channels, n_bands, k=len(common_channels), n_repeats=10, seed=seed
        ),
        "adversarial_isolated": [
            row["channel_name"] for row in
            rank_channels(clf, X_common, common_channels, n_bands, max_samples=adversarial_max_samples,
                          seed=seed, **adversarial_search_kwargs)
        ],
        "adversarial_graph_constrained": [
            row["channel_name"] for row in
            rank_channels_graph_constrained(clf, X_common, common_channels, n_bands, adjacency=adjacency,
                                             max_samples=adversarial_max_samples, seed=seed, **adversarial_search_kwargs)
        ],
    }
    return rankings


def run_cross_hardware_transfer(
    X_cgx: np.ndarray, y_cgx: np.ndarray,
    X_emotiv: np.ndarray, y_emotiv: np.ndarray, emotiv_subject_ids: List[str],
    common_channels: List[str] = COMMON_CHANNELS, n_bands: int = N_BANDS,
    k_sizes: List[int] = (5, 8, 12), k_folds: int = 5, seed: int = 42,
    adversarial_max_samples: int = 40, verbose: bool = True, checkpoint_path: str = None,
    **adversarial_search_kwargs,
) -> pd.DataFrame:
    """Runs the full transfer comparison. Returns a long-format
    DataFrame with columns [fold, source, method, n_channels,
    balanced_accuracy, auc]:
      - source='cgx_transfer': channel subset selected on CGX, evaluated on Emotiv
      - source='emotiv_own': channel subset selected on THIS FOLD's Emotiv
        training subjects (same-hardware ceiling)
      - source='baseline': random-common or all-common (no selection method)
    """
    import time
    t_start = time.time()

    def log(msg):
        if verbose:
            print(f"[{time.time()-t_start:7.1f}s] {msg}", flush=True)

    log("Computing CGX-side channel rankings on ALL 121 CGX subjects (one-time, no per-fold leakage risk)...")
    cgx_rankings = select_channels_from_cgx(
        X_cgx, y_cgx, common_channels, n_bands, adversarial_max_samples, seed, **adversarial_search_kwargs
    )
    log("CGX-side rankings computed: " + ", ".join(cgx_rankings.keys()))

    X_emotiv_common = _restrict_to_common_channels(X_emotiv, EMOTIV_CHANNELS, common_channels, n_bands)
    adjacency_common = build_structural_knn_graph(common_channels, k=COMMON_GRAPH_K)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    records = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_emotiv_common, y_emotiv)):
        log(f"--- Fold {fold_i+1}/{k_folds} (Emotiv-side CV) ---")
        X_train, y_train = X_emotiv_common[train_idx], y_emotiv[train_idx]
        X_test, y_test = X_emotiv_common[test_idx], y_emotiv[test_idx]
        log(f"  train={len(train_idx)} Emotiv subjects, test={len(test_idx)} Emotiv subjects")

        # --- source=emotiv_own: rank channels from THIS FOLD's Emotiv
        # training subjects only (no leakage into this fold's test set) ---
        emotiv_clf = train_rf_baseline(X_train, y_train)
        emotiv_rankings = {
            "adversarial_isolated": [
                row["channel_name"] for row in
                rank_channels(emotiv_clf, X_train, common_channels, n_bands,
                              max_samples=adversarial_max_samples, seed=seed, **adversarial_search_kwargs)
            ],
            "adversarial_graph_constrained": [
                row["channel_name"] for row in
                rank_channels_graph_constrained(emotiv_clf, X_train, common_channels, n_bands,
                                                 adjacency=adjacency_common, max_samples=adversarial_max_samples,
                                                 seed=seed, **adversarial_search_kwargs)
            ],
            "anova_fscore": anova_fscore_baseline(X_train, y_train, common_channels, n_bands, k=len(common_channels)),
        }
        log("  emotiv_own (per-fold) rankings computed")

        for k in k_sizes:
            # source = baseline (random-common always; all-common only
            # meaningful at k == len(common_channels), the full set)
            baseline_selections = {"random": random_subset_baseline(common_channels, k=k, seed=seed, n_repeats=1)[0]}
            if k == len(common_channels):
                baseline_selections["all_common"] = list(common_channels)
            for method_name, selected in baseline_selections.items():
                result = evaluate_channel_subset(X_train, y_train, X_test, y_test, selected, common_channels, n_bands)
                records.append({"fold": fold_i, "source": "baseline", "method": method_name, **result})

            # source = cgx_transfer
            for method_name, ranking in cgx_rankings.items():
                selected = ranking[:k]
                result = evaluate_channel_subset(X_train, y_train, X_test, y_test, selected, common_channels, n_bands)
                records.append({"fold": fold_i, "source": "cgx_transfer", "method": method_name, **result})

            # source = emotiv_own (ceiling)
            for method_name, ranking in emotiv_rankings.items():
                selected = ranking[:k]
                result = evaluate_channel_subset(X_train, y_train, X_test, y_test, selected, common_channels, n_bands)
                records.append({"fold": fold_i, "source": "emotiv_own", "method": method_name, **result})

        log(f"  fold {fold_i+1} evaluated for k_sizes={list(k_sizes)}")

        if checkpoint_path is not None:
            fold_df = pd.DataFrame.from_records([r for r in records if r["fold"] == fold_i])
            write_header = not os.path.exists(checkpoint_path)
            fold_df.to_csv(checkpoint_path, mode="a", header=write_header, index=False)
            log(f"  fold {fold_i+1} results appended to {checkpoint_path}")

        log(f"  fold {fold_i+1} complete (total elapsed {time.time()-t_start:.1f}s)")

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    print(
        "Needs real X_cgx/y_cgx (from run_cgx_experiment.load_all_subjects_cgx) and "
        "real X_emotiv/y_emotiv/emotiv_subject_ids (from load_emotiv.load_all_subjects_emotiv) "
        "-- see test_run_cross_hardware_transfer.py for an integration test on synthetic data."
    )
