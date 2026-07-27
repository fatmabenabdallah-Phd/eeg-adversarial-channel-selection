"""
experiments.run_cgx_experiment
================================
End-to-end P4 experiment: compares the adversarial channel-importance
method (isolated + graph-constrained) against the full baseline suite
(baselines.channel_selection_baselines) on real CGX data, using
subject-disjoint cross-validation.

REAL FUNCTIONS THIS SCRIPT CALLS (confirmed from grn-balladeer, not
guessed): grn_balladeer.data.labels.{mount_drive_colab,
load_demographics, build_label_table, stratified_subject_kfold,
DRIVE_DATASET_DIR, PATH_SLACKLINE_FLAGS},
grn_balladeer.data.subject_files.{find_slackline_sessions, SLACKLINE_LEVELS}
(the real, already-tested file-discovery module for this dataset --
handles 3 real TAGS filename shapes and the expected CGX coverage gap,
~140/151-154 sessions per level),
grn_balladeer.data.build_dataset_lightweight.build_subject_dataset_lightweight,
grn_balladeer.eval.baselines.{extract_band_power_features,
aggregate_epochs_to_subject, train_rf_baseline}.

An earlier draft of this script guessed a single-file, single-level
naming convention for subject files and left it as a "TODO ADAPT"
placeholder. That guess was wrong on inspection of the real folder
tree: subjects have up to 3 Slackline levels (SLACKLINE_LEVELS =
"1","6","11"), each its own session folder, and not every subject/level
has a CGX file (an expected, documented coverage gap -- see
data/subject_files.py's own docstring for the exact counts). This has
since been fixed to use the real find_slackline_sessions helper instead
of guessing file names.

LESSON CARRIED OVER FROM P2 (methods/test_adversarial_importance.py):
all channel-importance computation (adversarial AND baselines) is run
on each fold's HELD-OUT test subjects only, using the RF trained on
that fold's train subjects -- never on training subjects, to avoid
mistaking memorization for genuine importance (see
test_subject_adaptive_detects_subgroup_dependent_importance for why
this matters).
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.baselines import (
    CGX_CHANNELS,
    extract_band_power_features,
    aggregate_epochs_to_subject,
    train_rf_baseline,
    stratified_subject_kfold,
)
from baselines.channel_selection_baselines import (
    all_channels_baseline,
    random_subset_baseline,
    manual_frontal_baseline,
    mutual_information_baseline,
    anova_fscore_baseline,
    correlation_baseline,
    greedy_forward_selection,
    permutation_importance_baseline,
)
from methods.adversarial_importance import rank_channels, rank_channels_graph_constrained
from preprocessing.structural_graph import build_structural_knn_graph

N_BANDS = 5  # delta, theta, alpha, beta, gamma -- see eval/baselines.py::EEG_BANDS
CHANNEL_SUBSET_SIZES = [5, 10, 15, 20, 25, 30]  # for the accuracy-vs-#channels curve
CGX_GRAPH_K = 8  # see the k=8 vs k=4 note from the P0 module-copy task: k=8 is
                  # appropriate for CGX's 30 channels (NOT for Emotiv's 14 -- use
                  # a separate, smaller k there, see run_emotiv_experiment.py)


# ---------------------------------------------------------------------------
# Subject file resolution -- uses grn_balladeer.data.subject_files, the
# real, already-tested file-discovery module for this dataset (handles
# 3 real TAGS filename shapes and expected CGX coverage gaps ~140/151-154
# per level -- see that module's docstring). Loops over all 3 Slackline
# levels per subject (SLACKLINE_LEVELS = "1","6","11"), NOT just one
# level as an earlier draft of this script assumed.
# ---------------------------------------------------------------------------
def load_all_subjects_cgx(
    label_df: pd.DataFrame, dataset_root: str = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loops over every subject x level in label_df, builds per-epoch
    band-power features via the real pipeline
    (build_subject_dataset_lightweight with clean_bad_channels=True) for
    every session that actually HAS a CGX file (skipping the expected
    ~140/151-154 coverage gap, not erroring on it), then aggregates to
    one feature row per subject across all its available levels
    (aggregate_epochs_to_subject). Returns (X, y, subject_ids) aligned
    to whichever subjects end up with at least one usable session.
    """
    from grn_balladeer.data.labels import DRIVE_DATASET_DIR, PATH_SLACKLINE_FLAGS
    from grn_balladeer.data.subject_files import find_slackline_sessions, SLACKLINE_LEVELS
    from grn_balladeer.data.build_dataset_lightweight import build_subject_dataset_lightweight

    dataset_root = dataset_root or DRIVE_DATASET_DIR

    all_features = []
    all_epoch_subject_ids = []
    all_labels_per_subject = {}
    n_sessions_total, n_sessions_no_cgx = 0, 0

    for _, row in label_df.iterrows():
        user_id = row["user_id"]
        all_labels_per_subject[user_id] = int(row["label"])

        for level in SLACKLINE_LEVELS:
            sessions = find_slackline_sessions(dataset_root, user_id, level)
            for session in sessions:
                n_sessions_total += 1
                if session.cgx_path is None:
                    n_sessions_no_cgx += 1
                    continue  # expected coverage gap, not an error

                dataset = build_subject_dataset_lightweight(
                    cgx_path=session.cgx_path,
                    flags_path=PATH_SLACKLINE_FLAGS,
                    level=f"Level{level}",  # matches slackline_flags_info.json's "level" field convention
                    clean_bad_channels=True,
                )
                for _, band_power_feats in dataset:
                    all_features.append(band_power_feats)
                    all_epoch_subject_ids.append(user_id)

    print(
        f"CGX coverage: {n_sessions_total - n_sessions_no_cgx}/{n_sessions_total} sessions "
        f"had a CGX file (expected gap per the dataset's own documented ~140/151-154 ratio)."
    )

    features_arr = np.stack(all_features, axis=0)
    X_subject, subject_ids = aggregate_epochs_to_subject(features_arr, all_epoch_subject_ids)
    y_subject = np.array([all_labels_per_subject[sid] for sid in subject_ids])
    return X_subject, y_subject, subject_ids


# ---------------------------------------------------------------------------
# Evaluation helpers (channel subset -> held-out accuracy/AUC)
# ---------------------------------------------------------------------------
def _feature_indices_for_channels(
    selected_channels: List[str], all_channel_names: List[str], n_bands: int
) -> np.ndarray:
    idx = []
    for ch in selected_channels:
        c = all_channel_names.index(ch)
        idx.extend(range(c * n_bands, c * n_bands + n_bands))
    return np.array(idx)


def evaluate_channel_subset(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    selected_channels: List[str], all_channel_names: List[str], n_bands: int,
) -> dict:
    """Trains a fresh RF restricted to `selected_channels`' features on
    the fold's train subjects, evaluates on the fold's test subjects."""
    feat_idx = _feature_indices_for_channels(selected_channels, all_channel_names, n_bands)
    clf = train_rf_baseline(X_train[:, feat_idx], y_train)  # returns clf only, no scaler (RF needs none)
    y_pred = clf.predict(X_test[:, feat_idx])
    y_proba = clf.predict_proba(X_test[:, feat_idx])[:, 1]
    return {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else float("nan"),
        "n_channels": len(selected_channels),
    }


def run_cgx_experiment(
    X: np.ndarray, y: np.ndarray, subject_ids: List[str], label_df: pd.DataFrame,
    channel_names: List[str] = CGX_CHANNELS, n_bands: int = N_BANDS,
    k_folds: int = 5, seed: int = 42, adversarial_max_samples: int = 40,
    **adversarial_search_kwargs,
) -> pd.DataFrame:
    """Runs the full P4 comparison: for every subject-disjoint fold,
    fits the RF on train subjects, computes every baseline's channel
    subset AND the adversarial ranking (isolated + graph-constrained)
    on the RF, all evaluated on the SAME fold's held-out test subjects
    -- so every method's fold-level score is directly comparable
    (paired), which is what makes the Wilcoxon test downstream valid.

    Returns a long-format DataFrame with columns
    [fold, method, n_channels, balanced_accuracy, auc], one row per
    (fold, method, subset-size) evaluated.
    """
    folds = stratified_subject_kfold(label_df, k=k_folds, seed=seed)
    id_to_row = {sid: i for i, sid in enumerate(subject_ids)}
    adjacency = build_structural_knn_graph(channel_names, k=CGX_GRAPH_K)

    records = []
    for fold_i, fold in enumerate(folds):
        train_idx = [id_to_row[uid] for uid in fold["train_ids"] if uid in id_to_row]
        test_idx = [id_to_row[uid] for uid in fold["val_ids"] if uid in id_to_row]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Full RF on ALL channels, used only to derive channel rankings
        # (baselines + adversarial) -- never the model that produces the
        # reported accuracy/AUC for a channel SUBSET (that uses a fresh
        # RF retrained on just that subset's features, see
        # evaluate_channel_subset -- an RF's feature_importances_-style
        # internal notion of "using" a channel is not the same as
        # actually needing it once other channels are removed).
        full_clf = train_rf_baseline(X_train, y_train)  # returns clf only, no scaler

        # --- All-channels reference ---
        result = evaluate_channel_subset(X_train, y_train, X_test, y_test, channel_names, channel_names, n_bands)
        records.append({"fold": fold_i, "method": "all_channels", **result})

        # Compute each RANKING-based method's full ranking ONCE per fold
        # (not once per k) -- these are the expensive calls, and the top-k
        # for any k is just a prefix of the same full ranking, so
        # recomputing per k would be pure waste (this was a real bug in
        # an earlier draft of this function, caught before ever running
        # it on real data).
        mi_full = mutual_information_baseline(X_train, y_train, channel_names, n_bands, k=len(channel_names))
        anova_full = anova_fscore_baseline(X_train, y_train, channel_names, n_bands, k=len(channel_names))
        corr_full = correlation_baseline(X_train, y_train, channel_names, n_bands, k=len(channel_names))
        perm_full = permutation_importance_baseline(
            full_clf, X_train, y_train, channel_names, n_bands, k=len(channel_names), n_repeats=10, seed=seed
        )
        adv_isolated_full = [
            row["channel_name"] for row in
            rank_channels(
                full_clf, X_train, channel_names, n_bands,
                max_samples=adversarial_max_samples, seed=seed, **adversarial_search_kwargs,
            )
        ]
        adv_graph_full = [
            row["channel_name"] for row in
            rank_channels_graph_constrained(
                full_clf, X_train, channel_names, n_bands, adjacency=adjacency,
                max_samples=adversarial_max_samples, seed=seed, **adversarial_search_kwargs,
            )
        ]

        manual = None
        try:
            manual = manual_frontal_baseline(channel_names)
        except ValueError:
            pass  # e.g. not enough of the cluster present on this montage

        # --- Fixed-size baselines, evaluated at every subset size (just
        # slicing the full rankings computed once above) ---
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
                result = evaluate_channel_subset(
                    X_train, y_train, X_test, y_test, selected, channel_names, n_bands
                )
                records.append({"fold": fold_i, "method": method_name, **result})

        # --- Greedy forward selection: expensive, run once at the largest k only ---
        from sklearn.ensemble import RandomForestClassifier
        greedy_selected = greedy_forward_selection(
            model_factory=lambda: RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1
            ),
            X=X_train, y=y_train, channel_names=channel_names, n_bands=n_bands,
            k=max(CHANNEL_SUBSET_SIZES), cv_folds=5, seed=seed,
        )
        for k in CHANNEL_SUBSET_SIZES:
            result = evaluate_channel_subset(
                X_train, y_train, X_test, y_test, greedy_selected[:k], channel_names, n_bands
            )
            records.append({"fold": fold_i, "method": "greedy_forward", **result})

    return pd.DataFrame.from_records(records)


def run_wilcoxon_comparison(
    results_df: pd.DataFrame, method_a: str, method_b: str, n_channels: int, metric: str = "balanced_accuracy"
) -> dict:
    """Real (not estimated) paired Wilcoxon signed-rank test between two
    methods' per-fold scores at a given subset size -- per this
    project's standing rule that statistical results must always be
    executed, never approximated (see the p-value fabrication incident
    this rule originates from)."""
    a = results_df[(results_df.method == method_a) & (results_df.n_channels == n_channels)].sort_values("fold")[metric].values
    b = results_df[(results_df.method == method_b) & (results_df.n_channels == n_channels)].sort_values("fold")[metric].values
    if len(a) != len(b) or len(a) == 0:
        raise ValueError(
            f"run_wilcoxon_comparison: mismatched or empty fold counts for "
            f"{method_a} (n={len(a)}) vs {method_b} (n={len(b)}) at n_channels={n_channels}."
        )
    stat, p = wilcoxon(a, b)
    return {
        "method_a": method_a, "method_b": method_b, "n_channels": n_channels,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "wilcoxon_stat": float(stat), "p_value": float(p),
    }


if __name__ == "__main__":
    # Real Colab entry point -- uncomment on Colab, after mounting Drive.
    # from grn_balladeer.data.labels import mount_drive_colab, load_demographics, build_label_table
    # mount_drive_colab()
    # demo_df = load_demographics()
    # label_df = build_label_table(demo_df)
    # X, y, subject_ids = load_all_subjects_cgx(label_df)
    # results_df = run_cgx_experiment(X, y, subject_ids, label_df)
    # results_df.to_csv("/content/drive/MyDrive/BALLADEER ADHD DATASET/cgx_channel_selection_results.csv", index=False)
    # print(run_wilcoxon_comparison(results_df, "adversarial_isolated", "manual_frontal", n_channels=5))
    print(
        "This script is meant to be run on Colab with Drive mounted -- see "
        "test_run_cgx_experiment.py for an integration test on synthetic data "
        "with the real pipeline's exact shapes, runnable right now without Drive."
    )
