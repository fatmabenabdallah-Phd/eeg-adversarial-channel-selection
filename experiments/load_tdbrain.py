"""
experiments.load_tdbrain
==========================
TDBRAIN participants-file handling and the two confounds/data-quality
issues that MUST be addressed before any ADHD-vs-Healthy comparison on
this dataset.

CONFOUND/ISSUE 1 -- diagnosis confirmation. The raw 'indication'
column contains 200 ADHD-labeled subjects, but only 101 of these
(verified directly from TDBRAIN_participants_V3.xlsx) have a
CONFIRMED 'formal_status' of ADHD or ADD; the other 99 are
formal_status='UNKNOWN' (matching the dataset's own V3.1 release notes,
which report "a formal diagnostic status for N=624, including N=104
ADD/ADHD" -- close to the 101 found here, the small difference likely
being a V3 vs V3.1 version difference). Using the loose 'indication'
field instead of 'formal_status' would silently include ~half
unconfirmed diagnoses. This module uses formal_status, not indication.

CONFOUND/ISSUE 2 -- age. The formal_status=ADHD/ADD group is age 6-55.6
(mean ~23, includes children); the formal_status=HEALTHY group is age
18-82.7 (mean 40.3, adults only, no children at all). Any group
difference found on the unrestricted groups could trivially reflect
age-related EEG maturation/aging rather than ADHD itself. Restricting
both groups to age >= 18 controls this.
"""

from __future__ import annotations

import pandas as pd


def build_tdbrain_label_df(
    participants_xlsx_path: str, min_age: float = 18.0,
) -> pd.DataFrame:
    """Loads TDBRAIN_participants_V3.xlsx and returns a label_df with
    columns [user_id, label, age, gender], restricted to CONFIRMED
    ADHD/ADD (formal_status, not the looser 'indication' field) and
    confirmed HEALTHY, both filtered to age >= min_age. label: 1 = ADHD
    (or ADD), 0 = HEALTHY (matching this project's convention
    elsewhere).

    Prints the before/after counts and age comparison at every
    filtering step, so both confounds and their corrections are always
    visible, never silently applied.
    """
    df = pd.read_excel(participants_xlsx_path)

    n_indication_adhd = (df["indication"] == "ADHD").sum()
    n_formal_adhd = df["formal_status"].isin(["ADHD", "ADD"]).sum()
    print(
        f"Diagnosis confirmation check: 'indication'==ADHD gives {n_indication_adhd} subjects, "
        f"but only {n_formal_adhd} have a CONFIRMED formal_status of ADHD/ADD -- using "
        f"formal_status (the stricter, confirmed field), not indication."
    )

    df = df[df["formal_status"].isin(["ADHD", "ADD", "HEALTHY"])].copy()

    adhd_before = df[df["formal_status"].isin(["ADHD", "ADD"])]["age"]
    healthy_before = df[df["formal_status"] == "HEALTHY"]["age"]
    print(
        f"Before age restriction: ADHD/ADD age {adhd_before.min():.1f}-{adhd_before.max():.1f} "
        f"(mean {adhd_before.mean():.1f}, n={len(adhd_before)}); "
        f"HEALTHY age {healthy_before.min():.1f}-{healthy_before.max():.1f} "
        f"(mean {healthy_before.mean():.1f}, n={len(healthy_before)})."
    )

    filtered = df[df["age"] >= min_age].copy()

    adhd_after = filtered[filtered["formal_status"].isin(["ADHD", "ADD"])]["age"]
    healthy_after = filtered[filtered["formal_status"] == "HEALTHY"]["age"]
    print(
        f"After restricting to age >= {min_age}: ADHD/ADD age {adhd_after.min():.1f}-{adhd_after.max():.1f} "
        f"(mean {adhd_after.mean():.1f}, n={len(adhd_after)}); "
        f"HEALTHY age {healthy_after.min():.1f}-{healthy_after.max():.1f} "
        f"(mean {healthy_after.mean():.1f}, n={len(healthy_after)})."
    )

    label_df = pd.DataFrame({
        "user_id": filtered["TDBRAIN_ID"].values,
        "label": filtered["formal_status"].isin(["ADHD", "ADD"]).astype(int).values,
        "age": filtered["age"].values,
        "gender": filtered["gender"].values,
    }).reset_index(drop=True)

    n_before_dedup = len(label_df)
    n_duplicate_ids = n_before_dedup - label_df["user_id"].nunique()
    if n_duplicate_ids > 0:
        dup_ids = label_df[label_df.duplicated(subset="user_id", keep=False)]["user_id"].unique()
        print(
            f"Found {n_duplicate_ids} duplicate user_id rows in the participants file "
            f"(subjects: {list(dup_ids)}) -- likely repeated clinical assessments (e.g. pre/post "
            f"treatment) referencing the SAME single EEG session (this project's TDBRAIN loader "
            f"only reads ses-1). Keeping the first row per subject and dropping the rest, so the "
            f"EEG-loading step doesn't redundantly reload and average the identical file with itself."
        )
        label_df = label_df.drop_duplicates(subset="user_id", keep="first").reset_index(drop=True)

    return label_df


def build_tdbrain_severity_regression_df(
    participants_xlsx_path: str, target: str = "ADHD_pre_Att_leading",
) -> pd.DataFrame:
    """Point 6: continuous-severity regression sub-cohort. Returns a
    label_df with columns [user_id, label, age, gender], where 'label'
    is now the CONTINUOUS severity score (0-9 scale), not a binary
    diagnosis. Restricted to the ~72 subjects who have this score.

    IMPORTANT, verified directly from the participants file: this is a
    WITHIN-ADHD regression, not an ADHD-vs-Healthy comparison. 100% of
    subjects with a non-null severity score are ADHD/ADD-diagnosed
    (69 ADHD + 3 ADD, all from the 'ADHD_NF' neurofeedback treatment
    sub-cohort) -- no Healthy subjects have this score at all. This
    sub-cohort also spans age 6-55 (children included, mean 26.3),
    unlike the adult-only convention used for the main ADHD-vs-Healthy
    comparison elsewhere in this project -- age restriction is not
    applied here since there is no control group to age-match against,
    but age itself may still be worth checking as a covariate if a
    result is found (younger children's EEG differs substantially from
    adults', and severity ratings may correlate with age too).

    target: "ADHD_pre_Att_leading" (Attention, default) or
    "ADHD_pre_Hyp_leading" (Hyperactivity), both 0-9 scale, pre-treatment.
    Post-treatment columns (ADHD_post_*) also exist but are not used by
    default (would require deciding whether pre or post better reflects
    baseline severity for this kind of channel-importance analysis).
    """
    if target not in ("ADHD_pre_Att_leading", "ADHD_pre_Hyp_leading"):
        raise ValueError(f"build_tdbrain_severity_regression_df: unsupported target {target!r}")

    df = pd.read_excel(participants_xlsx_path)
    has_severity = df[target].notna()
    filtered = df[has_severity].copy()

    print(
        f"Severity regression sub-cohort ({target}): n={len(filtered)}, "
        f"formal_status breakdown: {filtered['formal_status'].value_counts().to_dict()}, "
        f"age {filtered['age'].min():.1f}-{filtered['age'].max():.1f} (mean {filtered['age'].mean():.1f})."
    )

    n_before_dedup = len(filtered)
    if filtered["TDBRAIN_ID"].duplicated().any():
        dup_ids = filtered[filtered["TDBRAIN_ID"].duplicated(keep=False)]["TDBRAIN_ID"].unique()
        print(f"Found duplicate user_id rows: {list(dup_ids)} -- keeping first occurrence.")
        filtered = filtered.drop_duplicates(subset="TDBRAIN_ID", keep="first")

    label_df = pd.DataFrame({
        "user_id": filtered["TDBRAIN_ID"].values,
        "label": filtered[target].values,  # CONTINUOUS, not binary -- caller must use regression, not classification
        "age": filtered["age"].values,
        "gender": filtered["gender"].values,
    })
    return label_df.reset_index(drop=True)


if __name__ == "__main__":
    label_df = build_tdbrain_label_df("/path/to/TDBRAIN_participants_V3.xlsx")
    print(label_df.label.value_counts())
