"""
experiments.run_repeated_cv_adversarial_vs_random
=====================================================
Piste 1: repeated k-fold cross-validation for the ONE comparison
closest to significance in the whole investigation --
adversarial_isolated (and adversarial_graph_constrained) vs. random,
at k=5, on Nasrabadi's connectivity features (Cohen's d=0.87,
uncorrected p=0.016, not surviving FDR at 10 folds).

Deliberately does NOT recompute every baseline (MI, ANOVA, correlation,
permutation_importance, greedy_forward) at every repeat -- those are
already well-characterized from the single-run pipeline, and
recomputing them here would multiply runtime for no new information
relevant to THIS specific improvement direction. Only the two
adversarial methods and random are computed per fold, per repeat,
keeping repeated-CV computationally tractable.

Uses nadeau_bengio_corrected_ttest (methods/repeated_cv_stats.py), NOT
a naive t-test or Wilcoxon on the pooled repeat*fold differences, since
folds across repeats reuse the same underlying subjects and are
therefore correlated -- treating them as independent would inflate
significance (confirmed via simulation in
methods/test_repeated_cv_stats.py: naive Type I error ~41% vs nominal
5% under a strongly-correlated null).

IMPORTANT: this targets a SINGLE, pre-specified comparison (adversarial
vs random at k=5), decided BEFORE running this experiment, precisely to
avoid the multiple-comparisons/researcher-degrees-of-freedom risk of
trying several "improvement" ideas until one reaches significance. If
this does not reach significance either, that is real evidence the
effect is not merely an artifact of insufficient CV repetitions, and
should be reported as such -- not followed by trying a different
improvement idea on the same data without correction.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.baselines import train_rf_baseline
from baselines.channel_selection_baselines import random_subset_baseline
from methods.adversarial_importance import rank_channels, rank_channels_graph_constrained
from methods.repeated_cv_stats import nadeau_bengio_corrected_ttest
from preprocessing.structural_graph import build_structural_knn_graph
from sklearn.model_selection import StratifiedKFold


def evaluate_channel_subset(X_train, y_train, X_test, y_test, selected_channels, channel_names, n_bands):
    """Local copy of the same evaluation logic used elsewhere in this
    project (train a fresh RF restricted to the selected channels'
    feature indices, evaluate balanced accuracy on the held-out fold).
    Duplicated here rather than imported to keep this script
    self-contained and independently auditable, given its role in a
    headline result."""
    feature_idx = []
    for ch in selected_channels:
        c = channel_names.index(ch)
        feature_idx.extend(range(c * n_bands, c * n_bands + n_bands))
    clf = train_rf_baseline(X_train[:, feature_idx], y_train)
    from sklearn.metrics import balanced_accuracy_score
    y_pred = clf.predict(X_test[:, feature_idx])
    return balanced_accuracy_score(y_test, y_pred)


def run_repeated_cv(
    X: np.ndarray, y: np.ndarray, channel_names, n_bands: int = 5,
    k: int = 5, k_folds: int = 10, n_repeats: int = 5,
    adversarial_max_samples: int = 40, graph_k: int = 6, verbose: bool = True,
    checkpoint_path: str = None, **adversarial_search_kwargs,
) -> pd.DataFrame:
    """Runs n_repeats independent k_folds-fold CV splits (different
    seed each repeat), computing adversarial_isolated,
    adversarial_graph_constrained, and random at montage size k only.
    Returns a long-format DataFrame [repeat, fold, method,
    balanced_accuracy] -- feed the adversarial-vs-random paired
    differences into nadeau_bengio_corrected_ttest for the final test.
    """
    import time
    t_start = time.time()

    def log(msg):
        if verbose:
            print(f"[{time.time()-t_start:7.1f}s] {msg}", flush=True)

    adjacency = build_structural_knn_graph(channel_names, k=graph_k)
    records = []

    for repeat in range(n_repeats):
        seed = 1000 + repeat  # distinct, deterministic seed per repeat
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            clf = train_rf_baseline(X_train, y_train)

            iso_ranking = [row["channel_name"] for row in rank_channels(
                clf, X_train, channel_names, n_bands, max_samples=adversarial_max_samples,
                seed=seed, **adversarial_search_kwargs)]
            graph_ranking = [row["channel_name"] for row in rank_channels_graph_constrained(
                clf, X_train, channel_names, n_bands, adjacency=adjacency,
                max_samples=adversarial_max_samples, seed=seed, **adversarial_search_kwargs)]
            random_sel = random_subset_baseline(channel_names, k=k, seed=seed, n_repeats=1)[0]

            for method_name, selected in [
                ("adversarial_isolated", iso_ranking[:k]),
                ("adversarial_graph_constrained", graph_ranking[:k]),
                ("random", random_sel),
            ]:
                acc = evaluate_channel_subset(X_train, y_train, X_test, y_test, selected, channel_names, n_bands)
                records.append({"repeat": repeat, "fold": fold_i, "method": method_name, "balanced_accuracy": acc})

            log(f"  repeat {repeat+1}/{n_repeats}, fold {fold_i+1}/{k_folds} done")

            if checkpoint_path is not None:
                pd.DataFrame.from_records(records).to_csv(checkpoint_path, index=False)

        log(f"repeat {repeat+1}/{n_repeats} complete (total elapsed {time.time()-t_start:.1f}s)")

    return pd.DataFrame.from_records(records)


def analyze_repeated_cv(results_df: pd.DataFrame, n_train: int, n_test: int) -> dict:
    """Applies nadeau_bengio_corrected_ttest to the adversarial-vs-random
    paired differences, for each adversarial method separately."""
    out = {}
    for method in ["adversarial_isolated", "adversarial_graph_constrained"]:
        merged = results_df[results_df.method.isin([method, "random"])].pivot_table(
            index=["repeat", "fold"], columns="method", values="balanced_accuracy"
        )
        diffs = (merged[method] - merged["random"]).values
        result = nadeau_bengio_corrected_ttest(diffs, n_train=n_train, n_test=n_test)
        result["method"] = method
        result["mean_method"] = float(merged[method].mean())
        result["mean_random"] = float(merged["random"].mean())
        out[method] = result
    return out


if __name__ == "__main__":
    print(
        "Real Colab usage (Nasrabadi connectivity, closest-to-significance comparison):\n"
        "  from experiments.run_repeated_cv_adversarial_vs_random import run_repeated_cv, analyze_repeated_cv\n"
        "  results_df = run_repeated_cv(X_nas_conn, y_nas_conn, NASRABADI_CHANNELS_MODERN, k=5, n_repeats=5,\n"
        "                                checkpoint_path='/content/drive/.../repeated_cv_checkpoint.csv')\n"
        "  results_df.to_csv('/content/drive/.../repeated_cv_results.csv', index=False)\n"
        "  n_test = len(y_nas_conn) // 10; n_train = len(y_nas_conn) - n_test\n"
        "  analysis = analyze_repeated_cv(results_df, n_train=n_train, n_test=n_test)\n"
        "  print(analysis)\n"
    )
