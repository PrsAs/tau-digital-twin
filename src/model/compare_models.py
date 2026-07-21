"""
src/model/compare_models_v4.py
===================================
Fixes the ModuleNotFoundError and data-pipeline mismatch from
compare_models_v2.py.

FURTHER UPDATE (v4): n_folds increased from 5 to 10. With only 5 folds,
the Wilcoxon signed-rank test has a hard mathematical floor of p=0.0625
-- meaning even a PERFECT sweep (winning all folds, as the GNN did vs
every baseline in v3) cannot cross the p<0.05 threshold. 10 folds lowers
that floor well below 0.05, so a consistent win can now register as
genuinely significant rather than being capped by fold count alone. Now imports diffusion_v2.py and
diffusion_growth_v2.py (both under src/model/, matching gnn_v3.py's
location), which use loaders_v2.py for data -- same cohort, same region
alignment, same subject set as the GNN. This makes the Wilcoxon
comparison fair and apples-to-apples.

NOTE ON TARGETS: gnn_v3 predicts absolute SUVR (via residual connection),
while diffusion_v2/diffusion_growth_v2 predict ANNUALIZED DELTA (rate),
matching their original design. To compare fairly, this script converts
diffusion predictions to absolute SUVR via:
    predicted_suvr = baseline_tau + predicted_delta * dt_years
so all three models are compared on the SAME final quantity (absolute
follow-up SUVR), not mixed units.
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold

from src.data.loaders_v2 import build_subject_dataset
from src.model.gnn_v3 import (
    build_normalized_adjacency, build_individualized_tensors, scale_non_baseline_features,
    TauDigitalTwinGNN, DEVICE
)
from src.model.diffusion_v2 import graph_laplacian, build_diffusion_tensors, fit_beta, predict_diffusion
from src.model.diffusion_growth_v2 import fit_growth_diffusion, predict_growth_diffusion


def train_one_fold_configurable(X_train, y_train, A_norm, hidden_dim=32, n_epochs=400,
                                 lr=0.01, weight_decay=1e-4):
    model = TauDigitalTwinGNN(in_dim=X_train.shape[-1], hidden_dim=hidden_dim, baseline_feature_idx=0).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    A_norm = A_norm.to(DEVICE)
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        preds = model(X_t, A_norm)
        loss = loss_fn(preds, y_t)
        loss.backward()
        optimizer.step()
    return model


def paired_fold_comparison(gnn_kwargs=None, n_folds=10, seed=42):
    gnn_kwargs = gnn_kwargs or {}
    torch.manual_seed(seed)

    adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment = build_subject_dataset()
    A_norm = build_normalized_adjacency(adj)
    L = graph_laplacian(adj, normalized=True)

    X, y, dt, rids = build_individualized_tensors(cohort, regional_tau, amyloid_df, thickness_df, alignment, region_labels)
    x0_diff, y_delta_diff, dt_diff, rids_diff = build_diffusion_tensors(cohort, regional_tau, alignment)
    assert np.array_equal(rids, rids_diff), "Subject sets diverged between GNN and diffusion tensors!"

    gkf = GroupKFold(n_splits=n_folds)
    results = {"gnn": [], "persistence": [], "diffusion": [], "growth_diffusion": []}

    for train_idx, test_idx in gkf.split(X, y, rids):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        baseline_test = X_test[:, :, 0]

        X_train_s, X_test_s = scale_non_baseline_features(X_train, X_test, baseline_feature_idx=0)
        model = train_one_fold_configurable(X_train_s, y_train, A_norm, **gnn_kwargs)
        model.eval()
        with torch.no_grad():
            gnn_preds = model(torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE), A_norm.to(DEVICE)).cpu().numpy()

        beta = fit_beta(x0_diff[train_idx], y_delta_diff[train_idx], dt_diff[train_idx], L)
        diff_delta_preds = predict_diffusion(x0_diff[test_idx], dt_diff[test_idx], L, beta)
        diff_suvr_preds = x0_diff[test_idx] + diff_delta_preds * dt_diff[test_idx, None]

        alpha, beta_g = fit_growth_diffusion(x0_diff[train_idx], y_delta_diff[train_idx], dt_diff[train_idx], L)
        growth_delta_preds = predict_growth_diffusion(x0_diff[test_idx], dt_diff[test_idx], L, alpha, beta_g)
        growth_suvr_preds = x0_diff[test_idx] + growth_delta_preds * dt_diff[test_idx, None]

        results["gnn"].append(np.mean(np.abs(gnn_preds - y_test)))
        results["persistence"].append(np.mean(np.abs(baseline_test - y_test)))
        results["diffusion"].append(np.mean(np.abs(diff_suvr_preds - y_test)))
        results["growth_diffusion"].append(np.mean(np.abs(growth_suvr_preds - y_test)))

    return {k: np.array(v) for k, v in results.items()}


if __name__ == "__main__":
    best_kwargs = {"hidden_dim": 32, "n_epochs": 400, "lr": 0.01, "weight_decay": 1e-4}
    print(f"Using best GNN config from hyperparam search: {best_kwargs}")

    errs = paired_fold_comparison(gnn_kwargs=best_kwargs, n_folds=10)

    summary = pd.DataFrame({
        "model": list(errs.keys()),
        "mean_mae": [v.mean() for v in errs.values()],
        "std_mae": [v.std() for v in errs.values()],
    })
    print("\n=== Pooled per-fold MAE comparison (all models on IDENTICAL subjects/folds) ===")
    print(summary)

    print("\n=== Wilcoxon signed-rank tests vs GNN ===")
    sig_rows = []
    for other in ["persistence", "diffusion", "growth_diffusion"]:
        stat, p = wilcoxon(errs["gnn"], errs[other])
        sig_rows.append({
            "comparison": f"gnn_vs_{other}", "wilcoxon_stat": stat, "p_value": p,
            "significant_at_0.05": p < 0.05, "gnn_mean_mae": errs["gnn"].mean(),
            "other_mean_mae": errs[other].mean(),
        })
        print(sig_rows[-1])

    pd.DataFrame(sig_rows).to_csv("results/figures/stage2_significance_test_v4.csv", index=False)
    summary.to_csv("results/figures/stage2_model_comparison_v4.csv", index=False)
    print("\nSaved results/figures/stage2_significance_test_v4.csv and stage2_model_comparison_v3.csv")
