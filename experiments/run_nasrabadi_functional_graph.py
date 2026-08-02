"""
experiments.run_nasrabadi_functional_graph
=============================================
Re-runs the graph-constrained adversarial method on Nasrabadi
connectivity features, but with a FUNCTIONAL connectivity graph
(methods.connectivity_features.build_functional_connectivity_graph)
instead of the geometric k-NN graph
(preprocessing.structural_graph.build_structural_knn_graph) used
everywhere else in this project.

Motivation: on connectivity features, a channel's structurally
relevant "neighbors" for the graph-constrained perturbation are
plausibly its most FUNCTIONALLY connected channels, not its
geometrically nearest electrodes -- the geometric graph was designed
for band-power features (where nearby electrodes tend to share
volume-conducted signal), and applying it unchanged to connectivity
features may test the wrong hypothesis about which channels interact.

Requires the RAW epochs (not the already-aggregated band-power-shaped
X matrix) to build the functional graph, since build_functional_connectivity_graph
needs the full pairwise connectivity matrix, which extract_connectivity_features
does not retain (it collapses to a per-channel mean). This script
therefore needs BOTH:
  - the aggregated X (n_subjects, n_channels*n_bands) connectivity
    features, for the actual adversarial ranking (already available
    from load_nasrabadi_connectivity.load_all_subjects_nasrabadi_connectivity)
  - the raw epochs_array, to build the ONE functional graph used
    across all folds (built once from ALL subjects' epochs, same
    no-leakage rationale as CGX's one-time source-side ranking in
    run_cross_hardware_transfer.py: this is a fixed structural prior,
    not a per-fold-fitted quantity)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.baselines import train_rf_baseline
from experiments.load_nasrabadi import load_nasrabadi_csv, build_subject_epoch_arrays, NASRABADI_CHANNELS_MODERN
from experiments.run_cgx_experiment import evaluate_channel_subset
from methods.connectivity_features import extract_connectivity_features, build_functional_connectivity_graph
from methods.adversarial_importance import rank_channels_graph_constrained
from sklearn.model_selection import StratifiedKFold


def build_nasrabadi_functional_graph(
    csv_path: str, sfreq: float = 128.0, window_samples: int = 512, k: int = 4, metric: str = "plv",
) -> np.ndarray:
    """Builds the functional connectivity graph from ALL subjects'
    (epoch-count-equalized) epochs -- a fixed structural prior computed
    once, analogous to how CGX's channel ranking is computed once from
    all 121 subjects in run_cross_hardware_transfer.py."""
    df = load_nasrabadi_csv(csv_path)
    epochs_array, epoch_subject_ids, _ = build_subject_epoch_arrays(df, sfreq=sfreq, window_samples=window_samples)

    # Equalize epoch count per subject, same rationale/method as load_nasrabadi_connectivity.py
    unique_subjects = sorted(set(epoch_subject_ids.tolist()))
    counts = {sid: int((epoch_subject_ids == sid).sum()) for sid in unique_subjects}
    min_count = min(counts.values())
    keep_idx = []
    for sid in unique_subjects:
        sid_idx = np.where(epoch_subject_ids == sid)[0]
        keep_idx.extend(sid_idx[:min_count].tolist())
    keep_idx = np.array(sorted(keep_idx))
    epochs_array = epochs_array[keep_idx]

    return build_functional_connectivity_graph(epochs_array, NASRABADI_CHANNELS_MODERN, sfreq, k=k, metric=metric)


def run_with_functional_graph(
    X: np.ndarray, y: np.ndarray, functional_adjacency: np.ndarray,
    channel_names=NASRABADI_CHANNELS_MODERN, n_bands: int = 5,
    k_sizes=(5, 10, 15, 19), k_folds: int = 10, seed: int = 42,
    adversarial_max_samples: int = 40, verbose: bool = True,
    **adversarial_search_kwargs,
) -> pd.DataFrame:
    """Compares adversarial_graph_constrained using the FUNCTIONAL
    graph against the same method using the GEOMETRIC graph (built
    fresh here for direct comparison) and against random, at every
    k_size, over k_folds subject-disjoint folds."""
    import time
    t_start = time.time()

    def log(msg):
        if verbose:
            print(f"[{time.time()-t_start:7.1f}s] {msg}", flush=True)

    from preprocessing.structural_graph import build_structural_knn_graph
    from baselines.channel_selection_baselines import random_subset_baseline

    geometric_adjacency = build_structural_knn_graph(channel_names, k=4)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    records = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        log(f"--- Fold {fold_i+1}/{k_folds} ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        clf = train_rf_baseline(X_train, y_train)

        t0 = time.time()
        ranking_functional = rank_channels_graph_constrained(
            clf, X_train, channel_names, n_bands, adjacency=functional_adjacency,
            max_samples=adversarial_max_samples, seed=seed, **adversarial_search_kwargs,
        )
        log(f"  functional-graph ranking done ({time.time()-t0:.1f}s)")

        t0 = time.time()
        ranking_geometric = rank_channels_graph_constrained(
            clf, X_train, channel_names, n_bands, adjacency=geometric_adjacency,
            max_samples=adversarial_max_samples, seed=seed, **adversarial_search_kwargs,
        )
        log(f"  geometric-graph ranking done ({time.time()-t0:.1f}s)")

        functional_full = [row["channel_name"] for row in ranking_functional]
        geometric_full = [row["channel_name"] for row in ranking_geometric]

        for k in k_sizes:
            selections = {
                "graph_functional": functional_full[:k],
                "graph_geometric": geometric_full[:k],
                "random": random_subset_baseline(channel_names, k=k, seed=seed, n_repeats=1)[0],
            }
            for method_name, selected in selections.items():
                result = evaluate_channel_subset(X_train, y_train, X_test, y_test, selected, channel_names, n_bands)
                records.append({"fold": fold_i, "method": method_name, **result})

        log(f"  fold {fold_i+1} complete (total elapsed {time.time()-t_start:.1f}s)")

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    print(
        "Real Colab usage:\n"
        "  from experiments.run_nasrabadi_functional_graph import build_nasrabadi_functional_graph, run_with_functional_graph\n"
        "  functional_adj = build_nasrabadi_functional_graph('/path/to/nasrabadi.csv')\n"
        "  # X_nas_conn, y_nas_conn already loaded via load_all_subjects_nasrabadi_connectivity\n"
        "  results_df = run_with_functional_graph(X_nas_conn, y_nas_conn, functional_adj)\n"
    )
