"""
experiments.run_emotiv_experiment
====================================
Same protocol as run_cgx_experiment.py / run_nasrabadi_experiment.py,
applied to BALLADEER's Emotiv EPOCX data (14 channels). Serves two
purposes: (1) a same-hardware baseline comparison (this file), and
(2) the channel set/RF trained here feed the cross-hardware transfer
test (CGX -> Emotiv, the project's central remaining positive-result
candidate -- see run_cross_hardware_transfer.py).

No manual_frontal baseline: EPOCX_CHANNELS contains only F7 of the
5-channel manual cluster (AF7, Fp1, Fpz, F7, Fz are not all present on
this montage), so manual_frontal_baseline() will raise -- caught and
skipped, same as CGX's handling for hardware where the cluster isn't
fully available.

NASRABADI_GRAPH_K-style choice: k=4 for Emotiv's k-NN structural graph
(not CGX's k=8), per the real, tested finding from the P0 module-copy
task that k=8 on 14 channels gives a near-complete graph (mean degree
~8.6/13 possible), losing most topological structure.
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.baselines import EMOTIV_CHANNELS, train_rf_baseline, stratified_subject_kfold
from baselines.channel_selection_baselines import (
    random_subset_baseline, manual_frontal_baseline, mutual_information_baseline,
    anova_fscore_baseline, correlation_baseline, permutation_importance_baseline,
    greedy_forward_selection,
)
from experiments.run_cgx_experiment import evaluate_channel_subset, run_wilcoxon_comparison
from methods.adversarial_importance import rank_channels, rank_channels_graph_constrained
from preprocessing.structural_graph import build_structural_knn_graph

N_BANDS = 5
CHANNEL_SUBSET_SIZES = [5, 10, 14]  # 14 = Emotiv's full montage
EMOTIV_GRAPH_K = 4


def run_emotiv_experiment(
    X: np.ndarray, y: np.ndarray, subject_ids: List[str], label_df: pd.DataFrame,
    channel_names: List[str] = EMOTIV_CHANNELS, n_bands: int = N_BANDS,
    k_folds: int = 5, seed: int = 42, adversarial_max_samples: int = 40,
    greedy_max_k: int = 10, verbose: bool = True, checkpoint_path: str = None,
    **adversarial_search_kwargs,
) -> pd.DataFrame:
    """Same protocol as run_cgx_experiment -- see that function's
    docstring for full rationale on greedy_max_k, checkpoint_path,
    verbose."""
    import time
    t_start = time.time()

    def log(msg):
        if verbose:
            print(f"[{time.time()-t_start:7.1f}s] {msg}", flush=True)

    folds = stratified_subject_kfold(label_df, k=k_folds, seed=seed)
    id_to_row = {sid: i for i, sid in enumerate(subject_ids)}
    adjacency = build_structural_knn_graph(channel_names, k=EMOTIV_GRAPH_K)
    log(f"Starting: {len(folds)} folds, {len(channel_names)} channels (Emotiv), "
        f"greedy_max_k={greedy_max_k}, adversarial_max_samples={adversarial_max_samples}")

    records = []
    for fold_i, fold in enumerate(folds):
        log(f"--- Fold {fold_i+1}/{len(folds)} ---")
        train_idx = [id_to_row[uid] for uid in fold["train_ids"] if uid in id_to_row]
        test_idx = [id_to_row[uid] for uid in fold["val_ids"] if uid in id_to_row]
        if len(train_idx) == 0 or len(test_idx) == 0:
            log(f"  SKIPPING fold {fold_i+1}: no subjects in this fold made it into "
                f"the Emotiv-loaded subject set (train={len(train_idx)}, test={len(test_idx)})")
            continue
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        log(f"  train={len(train_idx)} subjects, test={len(test_idx)} subjects")

        full_clf = train_rf_baseline(X_train, y_train)
        log("  full_clf trained")

        result = evaluate_channel_subset(X_train, y_train, X_test, y_test, channel_names, channel_names, n_bands)
        records.append({"fold": fold_i, "method": "all_channels", **result})

        mi_full = mutual_information_baseline(X_train, y_train, channel_names, n_bands, k=len(channel_names))
        anova_full = anova_fscore_baseline(X_train, y_train, channel_names, n_bands, k=len(channel_names))
        corr_full = correlation_baseline(X_train, y_train, channel_names, n_bands, k=len(channel_names))
        log("  filter baselines (MI/ANOVA/corr) done")

        perm_full = permutation_importance_baseline(
            full_clf, X_train, y_train, channel_names, n_bands, k=len(channel_names), n_repeats=10, seed=seed
        )
        log("  permutation_importance done")

        t0 = time.time()
        adv_isolated_full = [
            row["channel_name"] for row in
            rank_channels(full_clf, X_train, channel_names, n_bands,
                          max_samples=adversarial_max_samples, seed=seed, **adversarial_search_kwargs)
        ]
        log(f"  adversarial_isolated done ({time.time()-t0:.1f}s)")

        t0 = time.time()
        adv_graph_full = [
            row["channel_name"] for row in
            rank_channels_graph_constrained(full_clf, X_train, channel_names, n_bands, adjacency=adjacency,
                                             max_samples=adversarial_max_samples, seed=seed, **adversarial_search_kwargs)
        ]
        log(f"  adversarial_graph_constrained done ({time.time()-t0:.1f}s)")

        manual = None
        try:
            manual = manual_frontal_baseline(channel_names)
        except ValueError:
            pass  # expected on Emotiv: only F7 of the 5-channel cluster is present

        for k in CHANNEL_SUBSET_SIZES:
            selections = {
                "random": random_subset_baseline(channel_names, k=k, seed=seed, n_repeats=1)[0],
                "mutual_information": mi_full[:k],
                "anova_fscore": anova_full[:k],
                "correlation": corr_full[:k],
                "permutation_importance": perm_full[:k],
                "adversarial_isolated": adv_isolated_full[:k],
                "adversarial_graph_constrained": adv_graph_full[:k],
            }
            if manual is not None and k == len(manual):
                selections["manual_frontal"] = manual
            for method_name, selected in selections.items():
                result = evaluate_channel_subset(X_train, y_train, X_test, y_test, selected, channel_names, n_bands)
                records.append({"fold": fold_i, "method": method_name, **result})
        log(f"  fixed-size baselines done for k={CHANNEL_SUBSET_SIZES}")

        t0 = time.time()
        greedy_selected = greedy_forward_selection(
            model_factory=lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1),
            X=X_train, y=y_train, channel_names=channel_names, n_bands=n_bands,
            k=min(greedy_max_k, len(channel_names)), cv_folds=5, seed=seed,
        )
        log(f"  greedy_forward (k={greedy_max_k}) done ({time.time()-t0:.1f}s)")
        for k in [k for k in CHANNEL_SUBSET_SIZES if k <= greedy_max_k]:
            result = evaluate_channel_subset(X_train, y_train, X_test, y_test, greedy_selected[:k], channel_names, n_bands)
            records.append({"fold": fold_i, "method": "greedy_forward", **result})

        if checkpoint_path is not None:
            fold_df = pd.DataFrame.from_records([r for r in records if r["fold"] == fold_i])
            write_header = not os.path.exists(checkpoint_path)
            fold_df.to_csv(checkpoint_path, mode="a", header=write_header, index=False)
            log(f"  fold {fold_i+1} results appended to {checkpoint_path}")

        log(f"  fold {fold_i+1} complete (total elapsed {time.time()-t_start:.1f}s)")

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    print(
        "This script needs real Emotiv data loaded via experiments.load_emotiv -- "
        "see test_run_emotiv_experiment.py for an integration test on synthetic data, "
        "runnable right now without Drive access."
    )
