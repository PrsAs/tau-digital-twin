"""
scripts/stage4_hit_rate_breakdown_v1.py

Follow-up to run_stage4_validation_v1.py: digs into WHY top-5 hit rate
favors persistence (0.491) over the GNN (0.441), while the GNN wins on
regional MAE, top-1, and top-3. Breaks the comparison down along three
axes to find where the GNN's advantage holds vs. where it disappears:

  1. By baseline tau severity (low/medium/high baseline SUVR) --
     hypothesis: persistence may do better for subjects who are already
     near ceiling/floor and genuinely change little, while the GNN's
     advantage concentrates in subjects with more dynamic spread.
  2. By interval_group (short vs. long follow-up gap, if available in
     the per-subject predictions) -- hypothesis: over longer intervals,
     more regions genuinely change, diluting any top-5 "easy win" for
     persistence.
  3. By per-subject hit vs. miss -- identifies whether the GNN's top-5
     losses are concentrated in a small subset of subjects (a few bad
     outliers) or spread evenly across the cohort.

Reuses the already-saved stage4_per_subject_predictions_v1.csv rather
than re-running evaluate_gnn_per_subject() from scratch.
"""

import numpy as np
import pandas as pd

IN_PATH = "results/figures/stage4_per_subject_predictions_v1.csv"
OUT_PATH = "results/figures/stage4_hit_rate_breakdown_v1.csv"

K_VALUES = [1, 3, 5]
SEVERITY_BINS = [0, 1, 2, 3]  # tercile-based, computed below
SEVERITY_LABELS = ["low", "medium", "high"]


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


def main():
    df = pd.read_csv(IN_PATH)

    subject_baseline = df.groupby("subject_id")["baseline_suvr"].mean()
    severity_tercile = pd.qcut(subject_baseline, q=3, labels=SEVERITY_LABELS)

    has_interval_group = "interval_group" in df.columns
    subject_interval = df.groupby("subject_id")["interval_group"].first() if has_interval_group else None

    hit_tables = {}
    for k in K_VALUES:
        gnn_hits = per_subject_topk_hit(df, "y_pred_suvr", k)
        pers_hits = per_subject_topk_hit(df, "baseline_suvr", k)
        hit_tables[k] = pd.DataFrame({
            "gnn_hit": gnn_hits, "persistence_hit": pers_hits
        })

    breakdown_rows = []

    for k in K_VALUES:
        merged = hit_tables[k].join(severity_tercile.rename("severity_tercile"))
        if has_interval_group:
            merged = merged.join(subject_interval.rename("interval_group"))

        for tercile in SEVERITY_LABELS:
            sub = merged[merged["severity_tercile"] == tercile]
            if len(sub) == 0:
                continue
            breakdown_rows.append({
                "breakdown_dim": "baseline_severity_tercile",
                "group": tercile, "k": k, "n_subjects": len(sub),
                "gnn_hit_rate": sub["gnn_hit"].mean(),
                "persistence_hit_rate": sub["persistence_hit"].mean(),
                "gnn_minus_persistence": sub["gnn_hit"].mean() - sub["persistence_hit"].mean(),
            })

        if has_interval_group:
            for grp, sub in merged.groupby("interval_group"):
                breakdown_rows.append({
                    "breakdown_dim": "interval_group",
                    "group": grp, "k": k, "n_subjects": len(sub),
                    "gnn_hit_rate": sub["gnn_hit"].mean(),
                    "persistence_hit_rate": sub["persistence_hit"].mean(),
                    "gnn_minus_persistence": sub["gnn_hit"].mean() - sub["persistence_hit"].mean(),
                })

    breakdown_df = pd.DataFrame(breakdown_rows)
    breakdown_df.to_csv(OUT_PATH, index=False)

    print("=== Stage 4 hit-rate breakdown ===")
    print(breakdown_df.to_string(index=False))

    k5 = hit_tables[5]
    gnn_miss_persistence_hit = k5[(~k5["gnn_hit"]) & (k5["persistence_hit"])]
    both_miss = k5[(~k5["gnn_hit"]) & (~k5["persistence_hit"])]
    print(f"\nTop-5: subjects where GNN misses but persistence hits: {len(gnn_miss_persistence_hit)} of {len(k5)}")
    print(f"Top-5: subjects where BOTH miss: {len(both_miss)} of {len(k5)}")
    print(f"\nSaved breakdown -> {OUT_PATH}")


if __name__ == "__main__":
    main()
