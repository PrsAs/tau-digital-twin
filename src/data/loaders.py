"""
src/data/loaders.py (v2)
=============================
Extends v1 to add amyloid-PET (UCBERKELEYAV45) and FreeSurfer cortical
thickness (UCSFFSX7) as real covariates, replacing the zero-filled
placeholders used in gnn.py v1.

Region-naming across FOUR sources, all resolved to a single canonical
68-region order (the connectome's order):
  1. Connectome labels   : "L_bankssts", "R_superiorfrontal", ...
  2. Tau-PET columns      : "CTX_LH_BANKSSTS_SUVR", "CTX_RH_SUPERIORFRONTAL_SUVR"
  3. Amyloid-PET columns  : "CTX_LH_BANKSSTS_SUVR", ... (SAME convention as
     tau-PET -- both are UC Berkeley pipeline outputs, confirmed by user's
     actual column dump)
  4. Thickness columns    : "ST13TA", "ST72TA", ... (FreeSurfer ST-codes,
     mapped via the official ADNI UCSFFSX7 data dictionary the user
     supplied -- NOT guessed. See THICKNESS_ST_CODE_MAP below, built by
     cross-referencing every ST*TA "Thickness Average aparc.stats of
     <Region>" entry in UCSFFSX7_DICT.csv against the 68 DK cortical
     regions.)
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
EXTERNAL_DIR = Path("data/external")
PROCESSED_DIR = Path("data/processed")

CONNECTOME_PATH = EXTERNAL_DIR / "enigma_hcp_dk_connectome.csv"
LABELS_PATH = EXTERNAL_DIR / "enigma_hcp_dk_labels.csv"
COHORT_MANIFEST_PATH = PROCESSED_DIR / "cohort_manifest.csv"
REGIONAL_TAU_PATH = PROCESSED_DIR / "regional_tau_long.csv"
AMYLOID_PATH = RAW_DIR / "UCBERKELEYAV45.csv"
THICKNESS_PATH = RAW_DIR / "UCSFFSX7.csv"

# ST-code -> canonical "Hemi_region" name, built from the official ADNI
# UCSFFSX7 data dictionary (Thickness Average, aparc.stats). Verified
# against all 68 DK cortical ROIs -- no fuzzy matching used.
THICKNESS_ST_CODE_MAP = {
    "ST13TA": "Left_bankssts", "ST72TA": "Right_bankssts",
    "ST14TA": "Left_caudalanteriorcingulate", "ST73TA": "Right_caudalanteriorcingulate",
    "ST15TA": "Left_caudalmiddlefrontal", "ST74TA": "Right_caudalmiddlefrontal",
    "ST23TA": "Left_cuneus", "ST82TA": "Right_cuneus",
    "ST24TA": "Left_entorhinal", "ST83TA": "Right_entorhinal",
    "ST25TA": "Left_frontalpole", "ST84TA": "Right_frontalpole",
    "ST26TA": "Left_fusiform", "ST85TA": "Right_fusiform",
    "ST31TA": "Left_inferiorparietal", "ST90TA": "Right_inferiorparietal",
    "ST32TA": "Left_inferiortemporal", "ST91TA": "Right_inferiortemporal",
    "ST34TA": "Left_isthmuscingulate", "ST93TA": "Right_isthmuscingulate",
    "ST35TA": "Left_lateraloccipital", "ST94TA": "Right_lateraloccipital",
    "ST36TA": "Left_lateralorbitofrontal", "ST95TA": "Right_lateralorbitofrontal",
    "ST38TA": "Left_lingual", "ST97TA": "Right_lingual",
    "ST39TA": "Left_medialorbitofrontal", "ST98TA": "Right_medialorbitofrontal",
    "ST40TA": "Left_middletemporal", "ST99TA": "Right_middletemporal",
    "ST43TA": "Left_paracentral", "ST102TA": "Right_paracentral",
    "ST44TA": "Left_parahippocampal", "ST103TA": "Right_parahippocampal",
    "ST45TA": "Left_parsopercularis", "ST104TA": "Right_parsopercularis",
    "ST46TA": "Left_parsorbitalis", "ST105TA": "Right_parsorbitalis",
    "ST47TA": "Left_parstriangularis", "ST106TA": "Right_parstriangularis",
    "ST48TA": "Left_pericalcarine", "ST107TA": "Right_pericalcarine",
    "ST49TA": "Left_postcentral", "ST108TA": "Right_postcentral",
    "ST50TA": "Left_posteriorcingulate", "ST109TA": "Right_posteriorcingulate",
    "ST51TA": "Left_precentral", "ST110TA": "Right_precentral",
    "ST52TA": "Left_precuneus", "ST111TA": "Right_precuneus",
    "ST54TA": "Left_rostralanteriorcingulate", "ST113TA": "Right_rostralanteriorcingulate",
    "ST55TA": "Left_rostralmiddlefrontal", "ST114TA": "Right_rostralmiddlefrontal",
    "ST56TA": "Left_superiorfrontal", "ST115TA": "Right_superiorfrontal",
    "ST57TA": "Left_superiorparietal", "ST116TA": "Right_superiorparietal",
    "ST58TA": "Left_superiortemporal", "ST117TA": "Right_superiortemporal",
    "ST59TA": "Left_supramarginal", "ST118TA": "Right_supramarginal",
    "ST60TA": "Left_temporalpole", "ST119TA": "Right_temporalpole",
    "ST62TA": "Left_transversetemporal", "ST121TA": "Right_transversetemporal",
    "ST129TA": "Left_insula", "ST130TA": "Right_insula",
}

REGION_SUVR_PATTERN = re.compile(r"^CTX_(LH|RH)_.*_SUVR$")


def load_connectome():
    adj = pd.read_csv(CONNECTOME_PATH, header=None).values
    labels = pd.read_csv(LABELS_PATH, header=None)[0].tolist()
    assert adj.shape == (len(labels), len(labels))
    return adj, labels


def connectome_label_to_tau_column(label: str) -> str:
    m = re.match(r"^([LR])_(.+)$", label)
    hemi, region = m.groups()
    hemi_code = "LH" if hemi == "L" else "RH"
    return f"CTX_{hemi_code}_{region.upper()}_SUVR"


def connectome_label_to_thickness_st(label: str) -> str:
    """
    'L_bankssts' -> 'ST13TA'  (reverse lookup via THICKNESS_ST_CODE_MAP)
    """
    m = re.match(r"^([LR])_(.+)$", label)
    hemi, region = m.groups()
    hemi_word = "Left" if hemi == "L" else "Right"
    target = f"{hemi_word}_{region.lower()}"
    for st_code, name in THICKNESS_ST_CODE_MAP.items():
        if name.lower() == target.lower():
            return st_code
    raise ValueError(f"No thickness ST-code found for connectome label: {label}")


def get_region_suvr_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if REGION_SUVR_PATTERN.match(c)]


def build_region_alignment(region_labels: list, tau_columns: list, amyloid_columns: list) -> pd.DataFrame:
    rows = []
    for label in region_labels:
        tau_col = connectome_label_to_tau_column(label)
        amy_col = connectome_label_to_tau_column(label)  # same naming convention
        thick_col = connectome_label_to_thickness_st(label)
        rows.append({
            "connectome_label": label,
            "tau_column": tau_col,
            "amyloid_column": amy_col,
            "thickness_column": thick_col,
            "tau_matched": tau_col in tau_columns,
            "amyloid_matched": amy_col in amyloid_columns,
        })
    alignment = pd.DataFrame(rows)
    for col in ["tau_matched", "amyloid_matched"]:
        n_bad = (~alignment[col]).sum()
        if n_bad > 0:
            raise ValueError(f"{n_bad} regions failed {col}: {alignment.loc[~alignment[col], 'connectome_label'].tolist()}")
    return alignment


def load_cohort_and_regional_tau():
    cohort = pd.read_csv(COHORT_MANIFEST_PATH)
    regional_tau = pd.read_csv(REGIONAL_TAU_PATH, parse_dates=["EXAMDATE"])
    return cohort, regional_tau


def load_amyloid():
    df = pd.read_csv(AMYLOID_PATH, parse_dates=["SCANDATE"])
    return df.rename(columns={"SCANDATE": "EXAMDATE"})


def load_thickness():
    df = pd.read_csv(THICKNESS_PATH, parse_dates=["EXAMDATE"])
    return df


def get_subject_baseline_covariates(amyloid_df: pd.DataFrame, thickness_df: pd.DataFrame,
                                     subject_id: int, alignment: pd.DataFrame):
    """
    Returns (amyloid_vec[68], thickness_vec[68]) for a subject's BASELINE
    visit (first available), ordered to match connectome region order.
    Falls back to NaN (later imputed) if subject missing from a source --
    tau-PET, amyloid-PET, and structural MRI are not always acquired on
    the exact same visit schedule in ADNI.
    """
    amy_rows = amyloid_df[amyloid_df["RID"] == subject_id].sort_values("EXAMDATE")
    thick_rows = thickness_df[thickness_df["RID"] == subject_id].sort_values("EXAMDATE")

    if len(amy_rows) > 0:
        amy_row = amy_rows.iloc[0]
        amy_vec = amy_row[alignment["amyloid_column"]].values.astype(float)
    else:
        amy_vec = np.full(len(alignment), np.nan)

    if len(thick_rows) > 0:
        thick_row = thick_rows.iloc[0]
        thick_vec = thick_row[alignment["thickness_column"]].values.astype(float)
    else:
        thick_vec = np.full(len(alignment), np.nan)

    return amy_vec, thick_vec


def get_subject_tau_vector(regional_tau: pd.DataFrame, subject_id: int,
                            visit_index: int, alignment: pd.DataFrame) -> np.ndarray:
    subj_visits = regional_tau[regional_tau["RID"] == subject_id].sort_values("EXAMDATE")
    if visit_index >= len(subj_visits):
        raise IndexError(f"Subject {subject_id} has only {len(subj_visits)} visits")
    row = subj_visits.iloc[visit_index]
    return row[alignment["tau_column"]].values.astype(float)


def build_subject_dataset(interval_group: str = None):
    adj, region_labels = load_connectome()
    cohort, regional_tau = load_cohort_and_regional_tau()
    amyloid_df = load_amyloid()
    thickness_df = load_thickness()

    tau_columns = [c for c in regional_tau.columns if c.endswith("_SUVR") and c.startswith("CTX_")]
    amyloid_columns = [c for c in amyloid_df.columns if c.endswith("_SUVR") and c.startswith("CTX_")]
    alignment = build_region_alignment(region_labels, tau_columns, amyloid_columns)

    if interval_group is not None:
        cohort = cohort[cohort["interval_group"] == interval_group]

    return adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment


if __name__ == "__main__":
    adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment = build_subject_dataset()
    print(f"Connectome shape: {adj.shape}")
    print(f"Tau + amyloid alignment OK: {alignment['tau_matched'].all() and alignment['amyloid_matched'].all()}")
    print(f"Thickness ST-codes resolved for all {len(alignment)} regions (no exceptions raised).")
    example_rid = cohort["subject_id"].iloc[0]
    tau_vec = get_subject_tau_vector(regional_tau, example_rid, 0, alignment)
    amy_vec, thick_vec = get_subject_baseline_covariates(amyloid_df, thickness_df, example_rid, alignment)
    print(f"Example subject {example_rid}: tau {tau_vec.shape}, amyloid {amy_vec.shape}, thickness {thick_vec.shape}")
    print(f"Amyloid NaN count: {np.isnan(amy_vec).sum()}, Thickness NaN count: {np.isnan(thick_vec).sum()}")
