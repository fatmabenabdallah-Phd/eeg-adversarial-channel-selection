"""
experiments.run_nasrabadi_experiment
======================================
Same protocol as run_cgx_experiment.py (P4), applied to the public
Nasrabadi ADHD dataset instead of BALLADEER/CGX. Purpose: diagnostic,
not narrative extension -- Nasrabadi's signal is already confirmed
exploitable in the parent project (RF~0.746 AUC, EEGNet~0.869 AUC), so
if the adversarial channel-selection method ALSO finds no significant
advantage over random here, that points to a method-level limitation;
if it DOES find a significant advantage here (unlike on CGX), that
supports the parent project's existing conclusion that CGX/BALLADEER's
data-quality issues, not the selection method itself, are the
bottleneck.

Reuses methods/adversarial_importance.py and
baselines/channel_selection_baselines.py UNCHANGED -- only the data
loading (experiments/load_nasrabadi.py) and the per-subject feature
aggregation differ from run_cgx_experiment.py.

IMPORTANT SCOPE NOTE for the manuscript: Nasrabadi is already used in
the parent GRN/BSPC paper for cross-dataset validation of the
classifier architecture. This experiment asks a DIFFERENT question
(channel-selection method validity, not architecture generalization) --
state this explicitly in the manuscript to avoid the appearance of
redundant publication from the same dataset.

NOT YET RUN ON REAL DATA from this environment (no Kaggle access here)
-- integration-tested on synthetic data with the real column-naming/
windowing conventions (see experiments/load_nasrabadi.py's own
docstring for the 2 real discrepancies already caught: column order,
and old (T3/T4/T5/T6) vs modern (T7/T8/P7/P8) channel nomenclature).
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.baselines import extract_band_power_features, aggregate_epochs_to_subject
from experiments.load_nasrabadi import (
    load_nasrabadi_csv, build_subject_epoch_arrays, NASRABADI_CHANNELS_MODERN,
)
from experiments.run_cgx_experiment import (
    evaluate_channel_subset, run_wilcoxon_comparison,
)
from baselines.channel_selection_baselines import (
    all_channels_baseline, random_subset_baseline, mutual_information_baseline,
    anova_fscore_baseline, correlation_baseline, permutation_importance_baseline,
    greedy_forward_selection,
)
from methods.adversarial_importance import rank_channels, rank_channels_graph_constrained
from preprocessing.structural_graph import build_structural_knn_graph

N_BANDS = 5
CHANNEL_SUBSET_SIZES = [5, 10, 15, 19]  # 19 is Nasrabadi's full montage, not 30
NASRABADI_GRAPH_K = 6  # smaller than CGX's k=8 -- 19 channels total, k=8 would be
                        # nearly-complete-graph dense here too (see the k=8-on-14-
                        # channels Emotiv lesson from the P0 module-copy task)


def load_all_subjects_nasrabadi(
    csv_path: str, sfreq: float = 128.0, window_samples: int = 512,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads the Nasrabadi CSV, builds per-epoch band-power features
    (same extract_band_power_features used for CGX), aggregates to one
    row per subject. Returns (X, y, subject_ids)."""
    df = load_nasrabadi_csv(csv_path)
    epochs_array, epoch_subject_ids, subject_labels = build_subject_epoch_arrays(
        df, sfreq=sfreq, window_samples=window_samples
    )
    features_arr = extract_band_power_features(
        epochs_array, channels=NASRABADI_CHANNELS_MODERN, sfreq=sfreq
    )
    X_subject, subject_ids = aggregate_epochs_to_subject(features_arr, epoch_subject_ids.tolist())
    y_subject = np.array([subject_labels[sid] for sid in subject_ids])
    return X_subject, y_subject, subject_ids


def stratified_subject_kfold_simple(y: np.ndarray, k: int = 5, seed: int = 42) -> List[dict]:
    """Simpler stand-in for baselines.baselines.stratified_subject_kfold
    (which stratifies on label+sex+age_bin via a label_df) -- Nasrabadi's
    public release does not include the same demographic columns
    BALLADEER's label_df does, so this stratifies on label only. Returns
    the same {train_ids, val_ids}-list-of-dicts shape (here, ids are row
    INDICES into X/y/subject_ids, not user_id strings, since Nasrabadi
    subject IDs are unrelated to BALLADEER's)."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    indices = np.arange(len(y))
    for train_idx, val_idx in skf.split(indices, y):
        folds.append({"train_ids": train_idx.tolist(), "val_ids": val_idx.tolist()})
    return folds


def run_nasrabadi_experiment(
    X: np.ndarray, y: np.ndarray, subject_ids: List[str],
    channel_names: List[str] = NASRABADI_CHANNELS_MODERN, n_bands: int = N_BANDS,
    k_folds: int = 5, seed: int = 42, adversarial_max_samples: int = 40,
    greedy_max_k: int = 10, verbose: bool = True, checkpoint_path: str = None,
    **adversarial_search_kwargs,
) -> pd.DataFrame:
    """Same protocol as run_cgx_experiment (see that function's
    docstring for full rationale on greedy_max_k, checkpoint_path,
    verbose) -- no manual_frontal baseline here (that cluster is a
    CGX/BALLADEER-specific convention motivated by this project's own
    prior work, not something Nasrabadi's independent literature uses;
    including it here would not be a meaningful comparison)."""
    import time
    t_start = time.time()

    def log(msg):
        if verbose:
            print(f"[{time.time()-t_start:7.1f}s] {msg}", flush=True)

    folds = stratified_subject_kfold_simple(y, k=k_folds, seed=seed)
    adjacency = build_structural_knn_graph(channel_names, k=NASRABADI_GRAPH_K)
    log(f"Starting: {len(folds)} folds, {len(channel_names)} channels (Nasrabadi), "
        f"greedy_max_k={greedy_max_k}, adversarial_max_samples={adversarial_max_samples}")

    records = []
    for fold_i, fold in enumerate(folds):
        log(f"--- Fold {fold_i+1}/{len(folds)} ---")
        train_idx, test_idx = fold["train_ids"], fold["val_ids"]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        log(f"  train={len(train_idx)} subjects, test={len(test_idx)} subjects")

        from baselines.baselines import train_rf_baseline
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
            for method_name, selected in selections.items():
                result = evaluate_channel_subset(X_train, y_train, X_test, y_test, selected, channel_names, n_bands)
                records.append({"fold": fold_i, "method": method_name, **result})
        log(f"  fixed-size baselines done for k={CHANNEL_SUBSET_SIZES}")

        t0 = time.time()
        from sklearn.ensemble import RandomForestClassifier
        greedy_selected = greedy_forward_selection(
            model_factory=lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1),
            X=X_train, y=y_train, channel_names=channel_names, n_bands=n_bands,
            k=greedy_max_k, cv_folds=5, seed=seed,
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
    # Real Colab/local entry point:
    # X, y, subject_ids = load_all_subjects_nasrabadi("/path/to/nasrabadi.csv")
    # results_df = run_nasrabadi_experiment(
    #     X, y, subject_ids, k_folds=5,
    #     checkpoint_path="nasrabadi_results_checkpoint.csv",
    # )
    # results_df.to_csv("nasrabadi_channel_selection_results.csv", index=False)
    print(
        "This script needs the real Nasrabadi CSV (Kaggle: danizo/eeg-dataset-for-adhd) -- "
        "see test_run_nasrabadi_experiment.py for an integration test on synthetic data "
        "with the real column-naming/windowing conventions, runnable right now."
    )
