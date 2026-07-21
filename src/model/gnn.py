"""
src/model/gnn_v4.py
========================
Adds a RANKING-AWARE loss term to address the loss-metric mismatch
found after v3: MSELoss optimizes average magnitude accuracy across all
68 regions equally, with zero explicit incentive to get RELATIVE
ORDERING between regions correct -- which is exactly what top-3 hit
rate measures. v3 tied persistence on hit-rate (0.560 vs 0.564) despite
significantly beating it on MAE, because MSE and "getting the ranking
right" are different objectives.

Architecture is UNCHANGED from gnn_v3.py (same TauDigitalTwinGNN,
same residual connection, same GCNLayer). Only the LOSS FUNCTION changes:

    total_loss = mse_weight * MSELoss(pred, true)
               + rank_weight * PairwiseRankingLoss(pred, true)

PairwiseRankingLoss: for each subject, sample pairs of regions (i, j)
where true_suvr[i] > true_suvr[j], and penalize the model whenever
pred_suvr[i] is not also greater than pred_suvr[j] by at least a margin.
This directly optimizes for the ORDERING the top-3 hit rate metric
cares about, not just absolute magnitude closeness.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

from src.data.loaders_v2 import build_subject_dataset, get_subject_tau_vector, get_subject_baseline_covariates
from src.model.gnn_v3 import (
    build_normalized_adjacency, GCNLayer, TauDigitalTwinGNN,
    build_individualized_tensors, scale_non_baseline_features, DEVICE
)


def pairwise_ranking_loss(preds: torch.Tensor, targets: torch.Tensor, n_pairs: int = 20, margin: float = 0.01):
    """
    preds, targets: [batch, n_nodes]. For each subject in the batch,
    samples n_pairs random region-pairs, identifies which region has
    higher TRUE suvr, and penalizes the model (margin ranking loss) if
    predicted ordering disagrees. Averaged across batch and pairs.
    """
    batch_size, n_nodes = preds.shape
    device = preds.device
    total_loss = 0.0
    for b in range(batch_size):
        idx_a = torch.randint(0, n_nodes, (n_pairs,), device=device)
        idx_b = torch.randint(0, n_nodes, (n_pairs,), device=device)
        true_a, true_b = targets[b, idx_a], targets[b, idx_b]
        pred_a, pred_b = preds[b, idx_a], preds[b, idx_b]

        sign = torch.sign(true_a - true_b)  # +1 if a>b, -1 if a<b, 0 if tie
        pair_loss = torch.relu(margin - sign * (pred_a - pred_b))
        total_loss = total_loss + pair_loss.mean()
    return total_loss / batch_size


def train_one_fold_ranking(X_train, y_train, A_norm, hidden_dim=32, n_epochs=400,
                            lr=0.01, weight_decay=1e-4, mse_weight=1.0, rank_weight=0.5,
                            n_pairs=20, margin=0.01):
    model = TauDigitalTwinGNN(in_dim=X_train.shape[-1], hidden_dim=hidden_dim, baseline_feature_idx=0).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    A_norm = A_norm.to(DEVICE)

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        preds = model(X_t, A_norm)
        mse = mse_loss_fn(preds, y_t)
        rank = pairwise_ranking_loss(preds, y_t, n_pairs=n_pairs, margin=margin)
        loss = mse_weight * mse + rank_weight * rank
        loss.backward()
        optimizer.step()
    return model


def evaluate_gnn_ranking(n_folds: int = 10, seed: int = 42, gnn_kwargs: dict = None,
                          interval_group: str = None) -> pd.DataFrame:
    gnn_kwargs = gnn_kwargs or {}
    torch.manual_seed(seed)
    adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment = build_subject_dataset(
        interval_group=interval_group
    )
    A_norm = build_normalized_adjacency(adj)
    X, y, dt, rids = build_individualized_tensors(cohort, regional_tau, amyloid_df, thickness_df, alignment, region_labels)
    interval_lookup = cohort.set_index("subject_id")["interval_group"].to_dict()

    gkf = GroupKFold(n_splits=n_folds)
    all_rows = []
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, rids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_s, X_test_s = scale_non_baseline_features(X_train, X_test, baseline_feature_idx=0)
        model = train_one_fold_ranking(X_train_s, y_train, A_norm, **gnn_kwargs)
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE), A_norm.to(DEVICE)).cpu().numpy()

        for i, rid in enumerate(rids[test_idx]):
            for node_idx, region_label in enumerate(region_labels):
                all_rows.append({
                    "subject_id": rid, "region_index": node_idx, "region_label": region_label,
                    "y_true_suvr": y_test[i, node_idx], "y_pred_suvr": preds[i, node_idx],
                    "baseline_suvr": X_test[i, node_idx, 0], "dt_years": dt[test_idx][i],
                    "interval_group": interval_lookup.get(rid), "fold": fold_i,
                })
    return pd.DataFrame(all_rows)


def quick_metrics(df: pd.DataFrame, label: str = "gnn_v4_ranking"):
    mae = np.mean(np.abs(df["y_pred_suvr"] - df["y_true_suvr"]))
    mae_pers = np.mean(np.abs(df["baseline_suvr"] - df["y_true_suvr"]))

    def top3_rate(group, col):
        true_top3 = set(group.nlargest(3, "y_true_suvr")["region_label"])
        pred_top3 = set(group.nlargest(3, col)["region_label"])
        return len(true_top3 & pred_top3) / 3.0

    hit_model = df.groupby("subject_id").apply(lambda g: top3_rate(g, "y_pred_suvr")).mean()
    hit_pers = df.groupby("subject_id").apply(lambda g: top3_rate(g, "baseline_suvr")).mean()

    print(f"[{label}] MAE: {mae:.4f} | Persistence MAE: {mae_pers:.4f} | beats persistence MAE: {mae < mae_pers}")
    print(f"[{label}] Top-3 hit rate: {hit_model:.3f} | Persistence hit rate: {hit_pers:.3f} | beats persistence hit-rate: {hit_model > hit_pers}")
    return {"MAE": mae, "persistence_MAE": mae_pers, "hit3": hit_model, "persistence_hit3": hit_pers}


if __name__ == "__main__":
    gnn_kwargs = {"hidden_dim": 32, "n_epochs": 400, "lr": 0.01, "weight_decay": 1e-4,
                  "mse_weight": 1.0, "rank_weight": 0.5, "n_pairs": 20, "margin": 0.01}
    preds_df = evaluate_gnn_ranking(gnn_kwargs=gnn_kwargs)
    preds_df.to_csv("results/figures/gnn_per_subject_predictions_v4.csv", index=False)
    print(f"Saved: {preds_df.shape}")
    quick_metrics(preds_df)
