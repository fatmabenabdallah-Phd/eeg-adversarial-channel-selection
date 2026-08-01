"""
experiments.run_permutation_screening
========================================
Runs the cheap, high-power permutation screen (methods/permutation_screening.py)
on real CGX and Nasrabadi data, with Benjamini-Hochberg FDR correction
across channels (this project's standing statistical convention).

Meant to run in minutes, not hours -- a fast diagnostic BEFORE deciding
whether to re-run the expensive subset-retraining pipeline (run_cgx_experiment.py,
run_nasrabadi_experiment.py) with the connectivity-feature representation
(piste 5) instead of band-power.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.permutation_screening import permutation_screen_channels


def _benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Minimal BH-FDR implementation (no statsmodels dependency needed
    on a bare Colab runtime) -- returns a boolean array, same order as
    input, True where the null is rejected at the given alpha."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranked_p = p_values[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = ranked_p <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool)
    max_rank = np.max(np.where(below)[0])
    reject_sorted = np.zeros(n, dtype=bool)
    reject_sorted[: max_rank + 1] = True
    reject = np.zeros(n, dtype=bool)
    reject[order] = reject_sorted
    return reject


def run_screen(
    X: np.ndarray, y: np.ndarray, channel_names, n_bands: int = 5,
    n_permutations: int = 5000, seed: int = 42, alpha: float = 0.05,
    label: str = "dataset",
) -> pd.DataFrame:
    import time
    t0 = time.time()
    result = permutation_screen_channels(X, y, channel_names, n_bands, n_permutations=n_permutations, seed=seed)
    elapsed = time.time() - t0

    df = pd.DataFrame(result)
    df["significant_fdr"] = _benjamini_hochberg(df["p_value"].values, alpha=alpha)

    n_sig = df["significant_fdr"].sum()
    print(f"[{label}] {elapsed:.1f}s for {n_permutations} permutations x {len(channel_names)} channels "
          f"x {X.shape[0]} subjects -- {n_sig}/{len(channel_names)} channels significant after FDR (alpha={alpha})")
    print(df.head(10).to_string(index=False))
    print()
    return df


if __name__ == "__main__":
    print(
        "Real Colab usage:\n"
        "  from experiments.run_permutation_screening import run_screen\n"
        "  from baselines.baselines import CGX_CHANNELS\n"
        "  # X_cgx, y_cgx already loaded via load_all_subjects_cgx(label_df)\n"
        "  df_cgx = run_screen(X_cgx, y_cgx, CGX_CHANNELS, n_permutations=5000, label='CGX')\n"
        "  df_cgx.to_csv('/content/drive/MyDrive/.../cgx_permutation_screen.csv', index=False)\n"
    )
