"""
scripts/stage6_pick_case_study_subjects_v1.py

Picks concrete subject IDs for the Stage 6 case studies, per Stage 7's
guidance ("2-3 visualized subject case studies... one clear success
case, one partial/ambiguous case, discussed honestly").

Uses the already-saved Stage 4 outputs:
  - stage4_per_subject_predictions_v1.csv (per-region predictions)
  - stage4_hit_rate_breakdown_v1.csv (severity-tercile breakdown)

SELECTION CRITERIA:

"Success" case: a low-or-medium baseline-severity subject where the
GNN's top-3 hit rate is a clean win (predicted top-3 regions overlap
with true top-3 regions) AND regional MAE is low relative to cohort
median -- i.e. the model got both WHERE and HOW MUCH right for this
person.

"Ambiguous" case: a high baseline-severity subject where the GNN
missed at top-5 hit rate specifically (consistent with the finding
that persistence wins on top-5 in high-severity subjects), but where
regional MAE is still reasonable -- i.e. the model's magnitude
estimate isn't bad, but its region-ranking is where it struggles.
Chosen from the population where GNN misses but persistence hits at
top-5 (n=136 from the earlier breakdown), to make this case
representative of the documented boundary condition rather than a
one-off outlier.
"""

import numpy as np
import pandas as pd

PREDICTIONS_PATH = "results/figures/stage4_per_subject_predictions_v1.csv"
OUT_PATH = "results/figures/stage6_case_study_subjects_v1.csv"


def per_subject_topk_hit(df: pd.DataFrame, pred_col: str, k: int) -> pd.Series:
    df = df.copy()
    df["true_increase"] = df["y_true_suvr"] - df["baseline_suvr"]
    df["pred_increase"] = df[pred_col] - df["baseline_suvr"]
    results = {}
    for rid, sub in df.groupby("subject_id"):
        if len(sub) < k:
            continue
        true_top_k = set(sub.nlargest(k, "true_increase")["region_label"])
        pred_top_k = set(sub.nlargest(k, "pred_increase")["region_label"])
        results[rid] = len(true_top_k & pred_top_k) > 0
    return pd.Series(results, name=f"hit_top_{k}")


def per_subject_mae(df: pd.DataFrame) -> pd.Series:
    return df.groupby("subject_id").apply(
        lambda sub: np.mean(np.abs(sub["y_pred_suvr"] - sub["y_true_suvr"]))
    )


def main():
    df = pd.read_csv(PREDICTIONS_PATH)

    subject_baseline = df.groupby("subject_id")["baseline_suvr"].mean()
    severity_tercile = pd.qcut(subject_baseline, q=3, labels=["low", "medium", "high"])

    mae = per_subject_mae(df)
    cohort_median_mae = mae.median()

    hit_top3_gnn = per_subject_topk_hit(df, "y_pred_suvr", 3)
    hit_top5_gnn = per_subject_topk_hit(df, "y_pred_suvr", 5)
    hit_top5_persistence = per_subject_topk_hit(df, "baseline_suvr", 5)

    summary = pd.DataFrame({
        "severity_tercile": severity_tercile,
        "mae": mae,
        "hit_top3_gnn": hit_top3_gnn,
        "hit_top5_gnn": hit_top5_gnn,
        "hit_top5_persistence": hit_top5_persistence,
    }).dropna()

    success_candidates = summary[
        (summary["severity_tercile"].isin(["low", "medium"])) &
        (summary["hit_top3_gnn"]) &
        (summary["mae"] < cohort_median_mae)
    ].sort_values("mae")

    ambiguous_candidates = summary[
        (summary["severity_tercile"] == "high") &
        (~summary["hit_top5_gnn"]) &
        (summary["hit_top5_persistence"]) &
        (summary["mae"] < cohort_median_mae * 1.5)
    ].sort_values("mae")

    print(f"Cohort median regional MAE: {cohort_median_mae:.4f}")
    print(f"\nSuccess-case candidates found: {len(success_candidates)}")
    print(success_candidates.head(5))
    print(f"\nAmbiguous-case candidates found: {len(ambiguous_candidates)}")
    print(ambiguous_candidates.head(5))

    chosen = []
    if len(success_candidates) > 0:
        rid = success_candidates.index[0]
        chosen.append({"case_type": "success", "subject_id": rid, **success_candidates.loc[rid].to_dict()})
    if len(ambiguous_candidates) > 0:
        rid = ambiguous_candidates.index[0]
        chosen.append({"case_type": "ambiguous", "subject_id": rid, **ambiguous_candidates.loc[rid].to_dict()})

    chosen_df = pd.DataFrame(chosen)
    chosen_df.to_csv(OUT_PATH, index=False)

    print(f"\n=== CHOSEN CASE STUDY SUBJECTS ===")
    print(chosen_df.to_string(index=False))
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
