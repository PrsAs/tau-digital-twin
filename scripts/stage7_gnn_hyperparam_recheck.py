"""
scripts/stage7_gnn_hyperparam_recheck_v1.py

RECHECK before finalizing the Stage 7 framing: stage7_main_result_extraction_v1.py
used gnn_v3's DEFAULT hyperparameters (hidden_dim=16, dropout=0.3, n_epochs=200,
lr=0.01, weight_decay=1e-4) for the individualized comparison. compare_models.py's
existing grid search only ever optimized for MAE, on the OLD aggregate-only GNN --
never for top-k hit rate, and never on gnn_v3's residual/delta formulation.

Since top-k hit rate (not MAE) is where gnn_v3 currently loses to XGBoost
(0.082 vs 0.195 top-3), this grid search explicitly optimizes hidden_dim,
dropout, n_epochs, lr, and weight_decay AGAINST top-3 hit rate, to check
whether the gap is a genuine architectural ceiling or just a suboptimal
default configuration that was never tuned for this specific metric.
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

from src.data.loaders_v2 import build_subject_dataset
from src.model.gnn_v3 import (
    build_normalized_adjacency, build_individualized_tensors,
    scale_non_baseline_features, train_one_fold, TauDigitalTwinGNN, DEVICE,
)

OUT_DIR = "results/figures/stage7"
N_FOLDS = 5
SEED = 42
TOP_K_VALUES = [3, 5]

GRID = [
    {"hidden_dim": 16, "dropout": 0.3, "n_epochs": 200, "lr": 0.01, "weight_decay": 1e-4},   # current default
    {"hidden_dim": 32, "dropout": 0.3, "n_epochs": 200, "lr": 0.01, "weight_decay": 1e-4},
    {"hidden_dim": 64, "dropout": 0.3, "n_epochs": 200, "lr": 0.01, "weight_decay": 1e-4},
    {"hidden_dim": 32, "dropout": 0.1, "n_epochs": 200, "lr": 0.01, "weight_decay": 1e-4},
    {"hidden_dim": 32, "dropout": 0.5, "n_epochs": 200, "lr": 0.01, "weight_decay": 1e-4},
    {"hidden_dim": 32, "dropout": 0.3, "n_epochs": 400, "lr": 0.005, "weight_decay": 1e-3},
    {"hidden_dim": 32, "dropout": 0.3, "n_epochs": 200, "lr": 0.01, "weight_decay": 1e-2},
    {"hidden_dim": 64, "dropout": 0.2, "n_epochs": 400, "lr": 0.005, "weight_decay": 1e-3},
]


def top_k_hit_rate(y_true_delta, y_pred_delta, k):
    n_subj = y_true_delta.shape[0]
    hits = []
    for i in range(n_subj):
        true_top = set(np.argsort(-y_true_delta[i])[:k])
        pred_top = set(np.argsort(-y_pred_delta[i])[:k])
        hits.append(len(true_top & pred_top) / k)
    return float(np.mean(hits))


def evaluate_one_config(X, y, rids, A_norm, hidden_dim, dropout, n_epochs, lr, weight_decay,
                         n_folds=N_FOLDS, seed=SEED):
    torch.manual_seed(seed)
    baseline = X[:, :, 0]
    gkf = GroupKFold(n_splits=n_folds)

    fold_mae, fold_hits = [], {k: [] for k in TOP_K_VALUES}

    for train_idx, test_idx in gkf.split(X, y, rids):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_s, X_test_s = scale_non_baseline_features(X_train, X_test, baseline_feature_idx=0)

        model = TauDigitalTwinGNN(in_dim=X_train_s.shape[-1], hidden_dim=hidden_dim,
                                   dropout=dropout, baseline_feature_idx=0).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.MSELoss()
        Xt = torch.tensor(X_train_s, dtype=torch.float32).to(DEVICE)
        yt = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
        A_norm_dev = A_norm.to(DEVICE)

        model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            preds = model(Xt, A_norm_dev)
            loss = loss_fn(preds, yt)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE)
            preds = model(X_test_t, A_norm_dev).cpu().numpy()

        fold_mae.append(mean_absolute_error(y_test.ravel(), preds.ravel()))
        y_true_delta = y_test - X_test[:, :, 0]
        y_pred_delta = preds - X_test[:, :, 0]
        for k in TOP_K_VALUES:
            fold_hits[k].append(top_k_hit_rate(y_true_delta, y_pred_delta, k))

    result = {"mae_mean": float(np.mean(fold_mae)), "mae_std": float(np.std(fold_mae))}
    for k in TOP_K_VALUES:
        result[f"top{k}_hit_rate_mean"] = float(np.mean(fold_hits[k]))
        result[f"top{k}_hit_rate_std"] = float(np.std(fold_hits[k]))
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading cohort + connectome (once, reused across all configs)...")
    adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment = build_subject_dataset()
    A_norm = build_normalized_adjacency(adj)
    X, y, dt, rids = build_individualized_tensors(cohort, regional_tau, amyloid_df, thickness_df, alignment, region_labels)

    print(f"\nRunning grid search over {len(GRID)} configs, optimizing for top-3 hit rate...")
    all_results = []
    for i, params in enumerate(GRID):
        print(f"\n[{i+1}/{len(GRID)}] {params}")
        result = evaluate_one_config(X, y, rids, A_norm, **params)
        row = {**params, **result}
        all_results.append(row)
        print(f"    -> MAE={result['mae_mean']:.4f}  top3={result['top3_hit_rate_mean']:.4f}  top5={result['top5_hit_rate_mean']:.4f}")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("top3_hit_rate_mean", ascending=False)
    results_df.to_csv(f"{OUT_DIR}/stage7_gnn_hyperparam_recheck_v1.csv", index=False)

    print("\n=== GRID SEARCH RESULTS (sorted by top-3 hit rate, best first) ===")
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest config by top-3 hit rate: {best.to_dict()}")
    print(f"\nFor reference, XGBoost baseline top3_hit_rate was 0.1954, top5 was 0.2414.")
    print(f"Saved -> {OUT_DIR}/stage7_gnn_hyperparam_recheck_v1.csv")


if __name__ == "__main__":
    main()
