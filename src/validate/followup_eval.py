"""
scripts/run_stage4_validation_v1.py

Stage 4 -- Validation Against Real Follow-Up, per the thesis plan:

"Core metric: for each subject, does the model's predicted region(s) of
next spread match where tau actually appeared at their real follow-up
scan? Report both a discrete top-k predicted region hit rate and a
continuous regional-error metric. Compare this individualized
validation metric against the aggregate-only validation your existing
repo already reports -- the gap between good on average and good for
this specific person is the thesis's central empirical contribution."

Reuses evaluate_gnn_per_subject() from gnn_v3.py directly (5-fold
cross-validated, current RESIDUAL/delta model -- NOT the old absolute-
SUVR model that the gnn_v3.py docstring reports as losing to persistence
on top-3 hit rate: 0.060 vs 0.564). This script adds the missing top-k
hit-rate computation on top of that function's output, since
evaluate_gnn_per_subject() only returns raw predictions, not hit-rate
metrics.

TOP-K HIT RATE DEFINITION:
For each subject, rank regions by PREDICTED regional increase
(y_pred_suvr - baseline_suvr) descending -- i.e. "where does the model
think tau will increase the most". Rank regions by TRUE regional
increase (y_true_suvr - baseline_suvr) descending -- i.e. "where did
tau actually increase the most". A "hit" at k means the predicted
top-k regions overlap with the true top-k regions (any overlap, per
the standard top-k hit-rate convention used in the Stage 2 baseline
comparison referenced in gnn_v3.py's docstring).

This is computed for GNN predictions AND for the persistence baseline
(predicted increase = 0 for every region under persistence, so
persistence's "top-k" is effectively undefined/random -- reported as a
sanity floor, matching how gnn_v3.py's own docstring frames the
original Stage 2 comparison: 0.060 GNN vs 0.564 persistence top-3 hit
rate, using the OLD absolute-SUVR model).
"""

import numpy as np
import pandas as pd

from src.model.gnn_v3 import evaluate_gnn_per_subject

TOP_K_VALUES = [1, 3, 5]
OUT_PREDICTIONS_PATH = "results/figures/stage4_per_subject_predictions_v1.csv"
OUT_METRICS_PATH = "results/figures/stage4_validation_metrics_v1.csv"


def compute_top_k_hit_rate(df: pd.DataFrame, pred_col: str, k: int) -> float:
    """
    For each subject, computes whether the top-k regions by PREDICTED
    increase overlap with the top-k regions by TRUE increase. Returns
    the fraction of subjects with at least one overlapping region.
    """
    df = df.copy()
    df["true_increase"] = df["y_true_suvr"] - df["baseline_suvr"]
    df["pred_increase"] = df[pred_col] - df["baseline_suvr"]

    hits = []
    for rid, sub in df.groupby("subject_id"):
        if len(sub) < k:
            continue
        true_top_k = set(sub.nlargest(k, "true_increase")["region_label"])
        pred_top_k = set(sub.nlargest(k, "pred_increase")["region_label"])
        hits.append(len(true_top_k & pred_top_k) > 0)

    return float(np.mean(hits)) if hits else float("nan")


def compute_regional_mae(df: pd.DataFrame, pred_col: str) -> float:
    return float(np.mean(np.abs(df[pred_col] - df["y_true_suvr"])))


def main():
    print("Running evaluate_gnn_per_subject() with current residual-formulation gnn_v3 model...")
    preds_df = evaluate_gnn_per_subject()
    preds_df.to_csv(OUT_PREDICTIONS_PATH, index=False)
    print(f"Saved per-subject predictions: {preds_df.shape} -> {OUT_PREDICTIONS_PATH}")

    metrics_rows = []

    mae_gnn = compute_regional_mae(preds_df, "y_pred_suvr")
    mae_persistence = compute_regional_mae(preds_df, "baseline_suvr")
    metrics_rows.append({"metric": "regional_MAE", "gnn": mae_gnn, "persistence": mae_persistence})

    for k in TOP_K_VALUES:
        hit_gnn = compute_top_k_hit_rate(preds_df, "y_pred_suvr", k)
        # persistence predicts zero change everywhere, so its "predicted
        # top-k increase" is an arbitrary tie-break across all regions --
        # reported as a random-baseline sanity check, consistent with how
        # gnn_v3.py's docstring frames the ORIGINAL (pre-fix) comparison.
        hit_persistence = compute_top_k_hit_rate(preds_df, "baseline_suvr", k)
        metrics_rows.append({
            "metric": f"top_{k}_hit_rate", "gnn": hit_gnn, "persistence": hit_persistence
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_METRICS_PATH, index=False)

    print("\n=== Stage 4 Validation Results (current residual gnn_v3 model) ===")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved metrics -> {OUT_METRICS_PATH}")


if __name__ == "__main__":
    main()
