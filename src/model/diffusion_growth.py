"""
src/model/diffusion_growth_v2.py
=====================================
Adapted from the ORIGINAL diffusion_growth.py (Phase 4 diagnostic:
network diffusion + local growth model, dx/dt = (alpha*I - beta*L) @ x).
Math (fit_growth_diffusion, predict_growth_diffusion) is UNCHANGED --
only the data interface is swapped to loaders_v2.py, matching
diffusion_v2.py's approach, so this model evaluates on the identical
517-subject cohort and region alignment as gnn_v3.py.
"""

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.loaders_v2 import build_subject_dataset
from src.model.diffusion_v2 import graph_laplacian, build_diffusion_tensors


def fit_growth_diffusion(x0, y_true_delta, dt, L, init=(0.01, 0.001)):
    """UNCHANGED from original diffusion_growth.py."""
    def loss(params):
        alpha, beta = params
        if alpha < -1 or alpha > 1 or beta < 0 or beta > 5:
            return 1e6
        A = alpha * np.eye(L.shape[0]) - beta * L
        preds_delta = np.zeros_like(y_true_delta)
        for i in range(len(x0)):
            propagator = expm(A * dt[i])
            x_t = propagator @ x0[i]
            preds_delta[i] = (x_t - x0[i]) / dt[i]
        return np.mean((preds_delta - y_true_delta) ** 2)

    res = minimize(loss, x0=np.array(init), method="Nelder-Mead",
                    options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 300})
    return res.x[0], res.x[1]


def predict_growth_diffusion(x0, dt, L, alpha, beta):
    """UNCHANGED from original diffusion_growth.py."""
    A = alpha * np.eye(L.shape[0]) - beta * L
    preds_delta = np.zeros_like(x0)
    for i in range(len(x0)):
        propagator = expm(A * dt[i])
        x_t = propagator @ x0[i]
        preds_delta[i] = (x_t - x0[i]) / dt[i]
    return preds_delta


def evaluate_growth_diffusion_v2(n_folds: int = 5) -> dict:
    adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment = build_subject_dataset()
    L = graph_laplacian(adj, normalized=True)

    x0, y, dt, rids = build_diffusion_tensors(cohort, regional_tau, alignment)

    gkf = GroupKFold(n_splits=n_folds)
    maes, rmses, alphas, betas = [], [], [], []
    for train_idx, test_idx in gkf.split(x0, y, rids):
        alpha, beta = fit_growth_diffusion(x0[train_idx], y[train_idx], dt[train_idx], L)
        preds = predict_growth_diffusion(x0[test_idx], dt[test_idx], L, alpha, beta)
        maes.append(mean_absolute_error(y[test_idx].ravel(), preds.ravel()))
        rmses.append(np.sqrt(mean_squared_error(y[test_idx].ravel(), preds.ravel())))
        alphas.append(alpha)
        betas.append(beta)

    return {
        "model": "network_diffusion_growth_v2", "MAE": np.mean(maes), "MAE_std": np.std(maes),
        "RMSE": np.mean(rmses), "RMSE_std": np.std(rmses),
        "alpha_mean": np.mean(alphas), "beta_mean": np.mean(betas),
        "n_subjects": len(rids), "per_fold_mae": maes,
    }


if __name__ == "__main__":
    result = evaluate_growth_diffusion_v2()
    print(result)
