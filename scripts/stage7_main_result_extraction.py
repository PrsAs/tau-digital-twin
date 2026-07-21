"""
scripts/stage7_main_result_extraction_v1.py

Extracts EVERYTHING needed for the Stage 7 "main result" section in one
run: cohort size + inter-visit interval stats (Stage 1 reporting
requirement), individualized per-subject MAE + top-3/top-5 region hit
rate for gnn_v3 vs. persistence/ridge/xgboost/diffusion/growth-diffusion
baselines (Stage 4's core metric), and a paired significance test
(Wilcoxon) so the write-up can report a p-value, not just a mean
difference.

NOTE ON THE SCREENSHOT TABLE ALREADY IN HAND: that table (gnn MAE
0.0850, persistence 0.0893, diffusion 0.0916, growth_diffusion 0.0901)
is the AGGREGATE-level comparison from the original tau-gnn-progression
repo's baselines.py/diffusion.py/diffusion_growth.py -- useful as
"prior work context" but it is NOT the individualized validation result
your thesis's central claim rests on (no top-k hit rate, no ridge/xgboost,
no per-subject breakdown, uses the OLD aggregate-only GNN, not gnn_v3).
This script produces the INDIVIDUALIZED counterpart so both can be
placed side-by-side in the write-up as "aggregate baseline (prior work)"
vs. "individualized result (this thesis)".
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.data.loaders_v2 import build_subject_dataset
from src.model.gnn_v3 import (
    build_normalized_adjacency, build_individualized_tensors,
    scale_non_baseline_features, train_one_fold, TauDigitalTwinGNN, DEVICE,
)
import torch

OUT_DIR = "results/figures/stage7"
N_FOLDS = 5
SEED = 42
TOP_K_VALUES = [3, 5]


def report_cohort_stats(cohort, dt):
    n_subjects = len(cohort)
    dt_years = dt if dt is not None else None
    stats = {
        "n_subjects": n_subjects,
        "inter_visit_interval_years_mean": float(np.mean(dt_years)),
        "inter_visit_interval_years_std": float(np.std(dt_years)),
        "inter_visit_interval_years_min": float(np.min(dt_years)),
        "inter_visit_interval_years_max": float(np.max(dt_years)),
    }
    return stats


def top_k_hit_rate(y_true_delta, y_pred_delta, k):
    """
    For each subject, does the model's top-k predicted regions (by predicted
    delta magnitude, i.e. largest predicted INCREASE) overlap with the
    true top-k regions of largest actual increase? Returns mean hit rate
    across subjects (fraction of true top-k regions recovered in predicted top-k).
    """
    n_subj, n_nodes = y_true_delta.shape
    hits = []
    for i in range(n_subj):
        true_top = set(np.argsort(-y_true_delta[i])[:k])
        pred_top = set(np.argsort(-y_pred_delta[i])[:k])
        hits.append(len(true_top & pred_top) / k)
    return float(np.mean(hits))


def evaluate_gnn_v3_individualized(cohort, region_labels, adj, regional_tau,
                                    amyloid_df, thickness_df, alignment,
                                    n_folds=N_FOLDS, seed=SEED):
    torch.manual_seed(seed)
    A_norm = build_normalized_adjacency(adj)
    X, y, dt, rids = build_individualized_tensors(cohort, regional_tau, amyloid_df, thickness_df, alignment, region_labels)
    baseline = X[:, :, 0]  # baseline_tau feature

    gkf = GroupKFold(n_splits=n_folds)
    fold_mae, fold_hits = {k: [] for k in TOP_K_VALUES}, {k: [] for k in TOP_K_VALUES}
    fold_gnn_mae = []
    per_subject_rows = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, rids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_s, X_test_s = scale_non_baseline_features(X_train, X_test, baseline_feature_idx=0)

        model = train_one_fold(X_train_s, y_train, A_norm, n_epochs=200, lr=0.01, weight_decay=1e-4)
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE)
            preds = model(X_test_t, A_norm.to(DEVICE)).cpu().numpy()

        mae = mean_absolute_error(y_test.ravel(), preds.ravel())
        fold_gnn_mae.append(mae)

        y_true_delta = y_test - X_test[:, :, 0]
        y_pred_delta = preds - X_test[:, :, 0]
        for k in TOP_K_VALUES:
            fold_hits[k].append(top_k_hit_rate(y_true_delta, y_pred_delta, k))

        for i, rid in enumerate(rids[test_idx]):
            per_subject_rows.append({
                "subject_id": rid, "fold": fold_i,
                "gnn_mae": mean_absolute_error(y_test[i], preds[i]),
                "persistence_mae": mean_absolute_error(y_test[i], X_test[i, :, 0]),
            })

    result = {
        "gnn_individualized_mae_mean": float(np.mean(fold_gnn_mae)),
        "gnn_individualized_mae_std": float(np.std(fold_gnn_mae)),
    }
    for k in TOP_K_VALUES:
        result[f"gnn_top{k}_hit_rate_mean"] = float(np.mean(fold_hits[k]))
        result[f"gnn_top{k}_hit_rate_std"] = float(np.std(fold_hits[k]))

    return result, pd.DataFrame(per_subject_rows), fold_gnn_mae


def evaluate_baselines_individualized(cohort, region_labels, regional_tau, amyloid_df,
                                       thickness_df, alignment, n_folds=N_FOLDS, seed=SEED):
    """Persistence, Ridge, XGBoost -- same per-subject/per-region individualized
    framing as the GNN, for a fair apples-to-apples comparison (not the
    aggregate-only baselines.py numbers)."""
    from src.model.gnn_v3 import build_individualized_tensors
    X, y, dt, rids = build_individualized_tensors(cohort, regional_tau, amyloid_df, thickness_df, alignment, region_labels)
    n_subj, n_nodes, n_feat = X.shape

    gkf = GroupKFold(n_splits=n_folds)
    persistence_maes, ridge_maes, xgb_maes = [], [], []
    persistence_hits = {k: [] for k in TOP_K_VALUES}
    ridge_hits = {k: [] for k in TOP_K_VALUES}
    xgb_hits = {k: [] for k in TOP_K_VALUES}

    for train_idx, test_idx in gkf.split(X, y, rids):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pred_persist = X_test[:, :, 0]
        persistence_maes.append(mean_absolute_error(y_test.ravel(), pred_persist.ravel()))
        true_delta = y_test - X_test[:, :, 0]
        persist_delta = pred_persist - X_test[:, :, 0]
        for k in TOP_K_VALUES:
            persistence_hits[k].append(top_k_hit_rate(true_delta, persist_delta, k))

        Xf_train, yf_train = X_train.reshape(-1, n_feat), y_train.ravel()
        Xf_test, yf_test = X_test.reshape(-1, n_feat), y_test.ravel()

        ridge = Ridge(alpha=1.0)
        ridge.fit(Xf_train, yf_train)
        pred_ridge = ridge.predict(Xf_test).reshape(y_test.shape)
        ridge_maes.append(mean_absolute_error(yf_test, pred_ridge.ravel()))
        ridge_delta = pred_ridge - X_test[:, :, 0]
        for k in TOP_K_VALUES:
            ridge_hits[k].append(top_k_hit_rate(true_delta, ridge_delta, k))

        xgb = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42)
        xgb.fit(Xf_train, yf_train)
        pred_xgb = xgb.predict(Xf_test).reshape(y_test.shape)
        xgb_maes.append(mean_absolute_error(yf_test, pred_xgb.ravel()))
        xgb_delta = pred_xgb - X_test[:, :, 0]
        for k in TOP_K_VALUES:
            xgb_hits[k].append(top_k_hit_rate(true_delta, xgb_delta, k))

    result = {
        "persistence_mae_mean": float(np.mean(persistence_maes)), "persistence_mae_std": float(np.std(persistence_maes)),
        "ridge_mae_mean": float(np.mean(ridge_maes)), "ridge_mae_std": float(np.std(ridge_maes)),
        "xgboost_mae_mean": float(np.mean(xgb_maes)), "xgboost_mae_std": float(np.std(xgb_maes)),
    }
    for k in TOP_K_VALUES:
        result[f"persistence_top{k}_hit_rate_mean"] = float(np.mean(persistence_hits[k]))
        result[f"ridge_top{k}_hit_rate_mean"] = float(np.mean(ridge_hits[k]))
        result[f"xgboost_top{k}_hit_rate_mean"] = float(np.mean(xgb_hits[k]))

    return result, {"persistence": persistence_maes, "ridge": ridge_maes, "xgboost": xgb_maes}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading cohort + connectome...")
    adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment = build_subject_dataset()

    from src.model.gnn_v3 import build_individualized_tensors
    _, _, dt, rids = build_individualized_tensors(cohort, regional_tau, amyloid_df, thickness_df, alignment, region_labels)

    print("\n=== Stage 1: cohort stats ===")
    cohort_stats = report_cohort_stats(cohort, dt)
    for k, v in cohort_stats.items():
        print(f"  {k}: {v}")
    pd.DataFrame([cohort_stats]).to_csv(f"{OUT_DIR}/stage7_cohort_stats_v1.csv", index=False)

    print("\n=== Stage 4: individualized GNN (gnn_v3) result ===")
    gnn_result, per_subject_df, gnn_fold_maes = evaluate_gnn_v3_individualized(
        cohort, region_labels, adj, regional_tau, amyloid_df, thickness_df, alignment)
    for k, v in gnn_result.items():
        print(f"  {k}: {v:.4f}")
    per_subject_df.to_csv(f"{OUT_DIR}/stage7_gnn_per_subject_v1.csv", index=False)

    print("\n=== Stage 4: individualized baselines (persistence/ridge/xgboost) ===")
    baseline_result, baseline_fold_maes = evaluate_baselines_individualized(
        cohort, region_labels, regional_tau, amyloid_df, thickness_df, alignment)
    for k, v in baseline_result.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Paired significance tests (Wilcoxon, GNN vs. each baseline, per-fold MAE) ===")
    sig_results = {}
    for name, maes in baseline_fold_maes.items():
        stat, p_value = wilcoxon(gnn_fold_maes, maes)
        sig_results[name] = {"wilcoxon_stat": float(stat), "p_value": float(p_value), "significant_at_0.05": bool(p_value < 0.05)}
        print(f"  GNN vs {name}: stat={stat:.4f}, p={p_value:.4f}, significant={p_value < 0.05}")

    summary_rows = [
        {"model": "gnn_v3_individualized", "mae_mean": gnn_result["gnn_individualized_mae_mean"], "mae_std": gnn_result["gnn_individualized_mae_std"],
         "top3_hit_rate": gnn_result["gnn_top3_hit_rate_mean"], "top5_hit_rate": gnn_result["gnn_top5_hit_rate_mean"], "p_vs_gnn": None},
        {"model": "persistence", "mae_mean": baseline_result["persistence_mae_mean"], "mae_std": baseline_result["persistence_mae_std"],
         "top3_hit_rate": baseline_result["persistence_top3_hit_rate_mean"], "top5_hit_rate": baseline_result["persistence_top5_hit_rate_mean"],
         "p_vs_gnn": sig_results["persistence"]["p_value"]},
        {"model": "ridge", "mae_mean": baseline_result["ridge_mae_mean"], "mae_std": baseline_result["ridge_mae_std"],
         "top3_hit_rate": baseline_result["ridge_top3_hit_rate_mean"], "top5_hit_rate": baseline_result["ridge_top5_hit_rate_mean"],
         "p_vs_gnn": sig_results["ridge"]["p_value"]},
        {"model": "xgboost", "mae_mean": baseline_result["xgboost_mae_mean"], "mae_std": baseline_result["xgboost_mae_std"],
         "top3_hit_rate": baseline_result["xgboost_top3_hit_rate_mean"], "top5_hit_rate": baseline_result["xgboost_top5_hit_rate_mean"],
         "p_vs_gnn": sig_results["xgboost"]["p_value"]},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{OUT_DIR}/stage7_main_result_summary_v1.csv", index=False)

    print("\n=== FINAL SUMMARY TABLE (save this for the write-up) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved -> {OUT_DIR}/stage7_main_result_summary_v1.csv")
    print(f"Saved -> {OUT_DIR}/stage7_cohort_stats_v1.csv")
    print(f"Saved -> {OUT_DIR}/stage7_gnn_per_subject_v1.csv")


if __name__ == "__main__":
    main()
