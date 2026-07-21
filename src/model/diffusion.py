"""
src/model/diffusion_v2.py
==============================
Adapted from the ORIGINAL diffusion.py (Phase 4, item 4: classical
network diffusion model, x(t) = expm(-beta * L * t) @ x(0)). The
underlying MATH IS COMPLETELY UNCHANGED -- graph_laplacian(), fit_beta(),
predict_diffusion() are copied verbatim.

WHAT CHANGED: the data interface. The original imported from
src.data.loader (singular, older module) and read
data/processed/model_ready_table_clean.csv, a wide-format table from a
different pipeline. This version uses loaders_v2.py's
build_subject_dataset() / get_subject_tau_vector() instead, so the
diffusion model evaluates on the EXACT SAME 517-subject cohort, same
region alignment, and same GroupKFold splits as gnn_v3.py -- required
for the Wilcoxon significance test in compare_models_v2.py to be a fair,
apples-to-apples comparison.
"""

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.loaders_v2 import build_subject_dataset, get_subject_tau_vector


def graph_laplacian(adj: np.ndarray, normalized: bool = True) -> np.ndarray:
    """UNCHANGED from original diffusion.py."""
    deg = np.diag(adj.sum(axis=1))
    L = deg - adj
    if normalized:
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(deg), 1e-8)))
        L = d_inv_sqrt @ L @ d_inv_sqrt
    return L


def fit_beta(x0, y_true_delta, dt, L):
    """UNCHANGED from original diffusion.py."""
    def loss(beta):
        preds_delta = np.zeros_like(y_true_delta)
        for i in range(len(x0)):
            propagator = expm(-beta * L * dt[i])
            x_t = propagator @ x0[i]
            preds_delta[i] = (x_t - x0[i]) / dt[i]
        return np.mean((preds_delta - y_true_delta) ** 2)

    res = minimize_scalar(loss, bounds=(1e-4, 5.0), method="bounded")
    return res.x


def predict_diffusion(x0, dt, L, beta):
    """UNCHANGED from original diffusion.py."""
    preds_delta = np.zeros_like(x0)
    for i in range(len(x0)):
        propagator = expm(-beta * L * dt[i])
        x_t = propagator @ x0[i]
        preds_delta[i] = (x_t - x0[i]) / dt[i]
    return preds_delta


def build_diffusion_tensors(cohort, regional_tau, alignment):
    """
    NEW (replaces pivot_to_subject_vectors from original diffusion.py).
    Builds x0 [n_subj, 68], delta_tau [n_subj, 68] (annualized rate, to
    match the original model's target units), dt [n_subj] in years, and
    rids -- using loaders_v2's tau extraction so region order and subject
    set are identical to gnn_v3.py.
    """
    subject_ids = cohort["subject_id"].values
    x0_list, delta_list, dt_list, kept_rids = [], [], [], []

    for rid in subject_ids:
        try:
            x0 = get_subject_tau_vector(regional_tau, rid, 0, alignment)
            y1 = get_subject_tau_vector(regional_tau, rid, 1, alignment)
        except IndexError:
            continue
        row = cohort[cohort["subject_id"] == rid].iloc[0]
        dt_years = row["inter_visit_interval_days"] / 365.25

        annualized_delta = (y1 - x0) / dt_years
        x0_list.append(x0)
        delta_list.append(annualized_delta)
        dt_list.append(dt_years)
        kept_rids.append(rid)

    return (np.stack(x0_list), np.stack(delta_list), np.array(dt_list), np.array(kept_rids))


def evaluate_diffusion_v2(n_folds: int = 5) -> dict:
    adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment = build_subject_dataset()
    L = graph_laplacian(adj, normalized=True)

    x0, y, dt, rids = build_diffusion_tensors(cohort, regional_tau, alignment)

    gkf = GroupKFold(n_splits=n_folds)
    maes, rmses, betas = [], [], []
    for train_idx, test_idx in gkf.split(x0, y, rids):
        beta = fit_beta(x0[train_idx], y[train_idx], dt[train_idx], L)
        preds = predict_diffusion(x0[test_idx], dt[test_idx], L, beta)
        maes.append(mean_absolute_error(y[test_idx].ravel(), preds.ravel()))
        rmses.append(np.sqrt(mean_squared_error(y[test_idx].ravel(), preds.ravel())))
        betas.append(beta)

    return {
        "model": "network_diffusion_v2", "MAE": np.mean(maes), "MAE_std": np.std(maes),
        "RMSE": np.mean(rmses), "RMSE_std": np.std(rmses),
        "beta_mean": np.mean(betas), "n_subjects": len(rids),
        "per_fold_mae": maes,
    }


if __name__ == "__main__":
    result = evaluate_diffusion_v2()
    print(result)
