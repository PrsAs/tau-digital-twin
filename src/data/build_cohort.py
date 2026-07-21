"""
Stage 1 -- Cohort Construction (v3)
=======================================
Selects ADNI subjects eligible for individualized tau-progression modeling
under the "group connectome + individual tau" design.

Data sources (LONI / IDA downloads):
  1. UC Berkeley - Tau PET PVC 6mm Res analysis [ADNI2,3,4]
       -> data/raw/UCBERKELEYAV1451_PVC.csv
     Confirmed real column set (from user's actual download) includes:
       RID, VISCODE, VISCODE2, SCANDATE, PROCESSDATE, TRACER,
       META_TEMPORAL_SUVR, and per-region *_SUVR / *_VOLUME columns for
       every Desikan-Killiany cortical region (CTX_LH_*, CTX_RH_*) plus
       subcortical structures (HIPPOCAMPUS, AMYGDALA, THALAMUS_PROPER, etc).
     NOTE: the date column is SCANDATE, not EXAMDATE (this differs from the
     ADNI online data dictionary page, which lists EXAMDATE -- the actual
     PVC file uses SCANDATE. Always verify against the real header, as
     done here.)

  2. ADNIMERGE.csv -> data/raw/ADNIMERGE.csv
     Per-visit RID, VISCODE, EXAMDATE, DX_bl, AGE, PTGENDER, and FreeSurfer
     volumetric columns used as a proxy for "has baseline MRI".

Output:
  data/processed/cohort_manifest.csv
    columns: subject_id, n_tau_visits, visit_dates,
             inter_visit_interval_days, baseline_diagnosis, age, sex,
             has_baseline_mri
  data/processed/regional_tau_long.csv
    long-format per-subject-visit regional SUVR table (all CTX_LH_*/CTX_RH_*
    + META_TEMPORAL_SUVR columns), for direct use in Stage 2 loaders.py.

Design decisions locked in:
  - Connectome: GROUP-LEVEL normative structural connectome (Desikan-
    Killiander parcellation to match this tau-PET file's region naming).
  - Tau input: PVC, AV-1451, regional SUVR (bilateral CTX_LH_/CTX_RH_
    columns give 68 cortical regions directly -- no re-parcellation needed).
  - Time handling: inter-visit interval stored explicitly per subject.
"""

import re
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TAU_PET_FILE = RAW_DIR / "UCBERKELEYAV1451_PVC.csv"
ADNIMERGE_FILE = RAW_DIR / "ADNIMERGE.csv"

MIN_TAU_VISITS = 2
FS_VOLUME_MARKER_COLS = ["ST10CV", "ICV"]

REGION_SUVR_PATTERN = re.compile(r"^CTX_(LH|RH)_.*_SUVR$")


def get_region_suvr_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if REGION_SUVR_PATTERN.match(c)]


def load_tau_pet_visits(path: Path = TAU_PET_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["SCANDATE"])
    df = df.rename(columns={"SCANDATE": "EXAMDATE"})
    return df


def load_adnimerge(path: Path = ADNIMERGE_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    keep = [c for c in ["RID", "VISCODE", "EXAMDATE", "DX_bl", "AGE", "PTGENDER"] + FS_VOLUME_MARKER_COLS if c in df.columns]
    df = df[keep].copy()
    if "EXAMDATE" in df.columns:
        df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    return df


def derive_has_mri(adni_df: pd.DataFrame) -> pd.DataFrame:
    marker_cols = [c for c in FS_VOLUME_MARKER_COLS if c in adni_df.columns]
    baseline = adni_df[adni_df["VISCODE"] == "bl"].copy() if "VISCODE" in adni_df.columns else adni_df.copy()
    if marker_cols:
        baseline["HAS_MRI"] = baseline[marker_cols].notna().any(axis=1)
    else:
        baseline["HAS_MRI"] = False
    return baseline[["RID", "HAS_MRI"]].drop_duplicates(subset="RID")


def build_cohort(tau_path: Path = TAU_PET_FILE, adni_path: Path = ADNIMERGE_FILE):
    tau = load_tau_pet_visits(tau_path)
    adni = load_adnimerge(adni_path)

    visit_counts = tau.groupby("RID")["EXAMDATE"].apply(list).reset_index()
    visit_counts["n_tau_visits"] = visit_counts["EXAMDATE"].apply(len)
    eligible = visit_counts[visit_counts["n_tau_visits"] >= MIN_TAU_VISITS].copy()

    def interval_days(dates):
        dates = sorted(dates)
        return (dates[1] - dates[0]).days

    eligible["inter_visit_interval_days"] = eligible["EXAMDATE"].apply(interval_days)

    mri_flags = derive_has_mri(adni)
    dx_baseline = (
        adni[adni["VISCODE"] == "bl"][["RID", "DX_bl", "AGE", "PTGENDER"]].drop_duplicates(subset="RID")
        if "VISCODE" in adni.columns
        else adni[["RID", "DX_bl", "AGE", "PTGENDER"]].drop_duplicates(subset="RID")
    )

    cohort = eligible.merge(mri_flags, on="RID", how="left")
    cohort = cohort.merge(dx_baseline, on="RID", how="left")
    cohort = cohort[cohort["HAS_MRI"] == True]  # noqa: E712

    cohort = cohort.rename(columns={
        "RID": "subject_id", "EXAMDATE": "visit_dates",
        "DX_bl": "baseline_diagnosis", "AGE": "age", "PTGENDER": "sex",
        "HAS_MRI": "has_baseline_mri",
    })
    cohort = cohort[[
        "subject_id", "n_tau_visits", "visit_dates",
        "inter_visit_interval_days", "baseline_diagnosis", "age", "sex",
        "has_baseline_mri",
    ]]

    region_cols = get_region_suvr_columns(tau)
    id_cols = ["RID", "VISCODE", "EXAMDATE"]
    extra_cols = [c for c in ["META_TEMPORAL_SUVR"] if c in tau.columns]
    regional_long = tau[id_cols + extra_cols + region_cols].copy()
    regional_long = regional_long[regional_long["RID"].isin(cohort["subject_id"])]

    return cohort, regional_long


def report_cohort_stats(cohort: pd.DataFrame) -> None:
    print(f"Total eligible subjects: {len(cohort)}")
    print(cohort["n_tau_visits"].value_counts().sort_index())
    print(cohort["inter_visit_interval_days"].describe())
    print(cohort["baseline_diagnosis"].value_counts())


if __name__ == "__main__":
    cohort_df, regional_long_df = build_cohort()
    report_cohort_stats(cohort_df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cohort_df.to_csv(PROCESSED_DIR / "cohort_manifest.csv", index=False)
    regional_long_df.to_csv(PROCESSED_DIR / "regional_tau_long.csv", index=False)
    print(f"Saved -> {PROCESSED_DIR / 'cohort_manifest.csv'}")
    print(f"Saved -> {PROCESSED_DIR / 'regional_tau_long.csv'}")
