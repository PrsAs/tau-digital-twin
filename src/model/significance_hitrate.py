"""
src/model/significance_hitrate_v4.py
=========================================
Focused significance test: gnn_v4 (ranking-loss GNN) vs. persistence on
TOP-3 REGION HIT RATE specifically (not MAE -- that was already
confirmed significant in compare_models_v4.py). Diffusion and
growth-diffusion are DELIBERATELY EXCLUDED here since they already
underperformed even persistence on MAE in the prior test -- no reason to
re-test them on hit-rate, and skipping them keeps this fast (no scipy
optimization per fold, just GNN training).

Runs 10-fold GroupKFold (same fold-count reasoning as compare_models_v4:
5 folds cannot mathematically produce p<0.05 even with a perfect sweep).
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold

from src.data.loaders_v2 import build_subject_dataset
from src.model.gnn_v3 import build_normalized_adjacency, scale_non_baseline_features
from src.model.gnn_v4 import build_individualized_tensors, train_one_fold_ranking, DEVICE


def top3_hit_rate_batch(preds: np.ndarray, targets: np.ndarray) -> float:
    hits = []
    for i in range(preds.shape[0]):
        true_top3 = set(np.argsort(targets[i])[-3:])
        pred_top3 = set(np.argsort(preds[i])[-3:])
        hits.append(len(true_top3 & pred_top3) / 3.0)
    return np.mean(hits)


def paired_fold_hitrate(gnn_kwargs=None, n_folds=10, seed=42):
    gnn_kwargs = gnn_kwargs or {}
    torch.manual_seed(seed)

    adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment = build_subject_dataset()
    A_norm = build_normalized_adjacency(adj)
    X, y, dt, rids = build_individualized_tensors(cohort, regional_tau, amyloid_df, thickness_df, alignment, region_labels)

    gkf = GroupKFold(n_splits=n_folds)
    gnn_hits, persistence_hits = [], []

    for train_idx, test_idx in gkf.split(X, y, rids):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        baseline_test = X_test[:, :, 0]

        X_train_s, X_test_s = scale_non_baseline_features(X_train, X_test, baseline_feature_idx=0)
        model = train_one_fold_ranking(X_train_s, y_train, A_norm, **gnn_kwargs)
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE), A_norm.to(DEVICE)).cpu().numpy()

        gnn_hits.append(top3_hit_rate_batch(preds, y_test))
        persistence_hits.append(top3_hit_rate_batch(baseline_test, y_test))

    return np.array(gnn_hits), np.array(persistence_hits)


if __name__ == "__main__":
    gnn_kwargs = {"hidden_dim": 32, "n_epochs": 400, "lr": 0.01, "weight_decay": 1e-4,
                  "mse_weight": 1.0, "rank_weight": 0.5, "n_pairs": 20, "margin": 0.01}

    gnn_hits, persistence_hits = paired_fold_hitrate(gnn_kwargs=gnn_kwargs, n_folds=10)

    print(f"GNN-v4 hit-rate per fold: {gnn_hits}")
    print(f"Persistence hit-rate per fold: {persistence_hits}")
    print(f"GNN-v4 mean: {gnn_hits.mean():.4f} | Persistence mean: {persistence_hits.mean():.4f}")

    stat, p = wilcoxon(gnn_hits, persistence_hits)
    print(f"Wilcoxon stat: {stat}, p-value: {p:.4f}, significant at 0.05: {p < 0.05}")

    pd.DataFrame({
        "fold": range(len(gnn_hits)), "gnn_hit3": gnn_hits, "persistence_hit3": persistence_hits
    }).to_csv("results/figures/stage2_hitrate_significance_v4.csv", index=False)
    print("Saved results/figures/stage2_hitrate_significance_v4.csv")
