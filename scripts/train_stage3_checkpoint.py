"""
scripts/train_stage3_checkpoint_v2.py

Trains the Stage 3 GNN ONCE on the full cohort (no held-out split --
appropriate here since Stage 2 already answered "does this generalize",
and Stage 3/5 is asking "does attribution correspond to structurally
sensible paths", per the docstring in attribution_v2.py).

Persists BOTH:
  1. model state_dict -> results/checkpoints/stage3_gnn_v3.pt
  2. the fitted StandardScaler (on non-baseline features) -> 
     results/checkpoints/stage3_scaler.pkl

This fixes two bugs found by comparing gnn_v3.py and attribution_v2.py:
  - hidden_dim mismatch: train_one_fold() uses the TauDigitalTwinGNN
    default hidden_dim=16, but load_trained_model_for_subject_set()
    defaulted to hidden_dim=32 -- fixed here by training explicitly
    at hidden_dim=16 and the checkpoint loader must match.
  - scaling mismatch: gnn_v3.py's training pipeline ALWAYS scales
    amyloid/thickness/dt_years via scale_non_baseline_features()
    before fitting, but attribution_v2.py fed raw, unscaled X_lookup
    into the model at inference time. The saved scaler here must be
    applied to X before every attribution call.
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.model.gnn_v3 import (
    build_normalized_adjacency,
    build_individualized_tensors,
    TauDigitalTwinGNN,
    DEVICE,
)
from src.data.loaders_v2 import build_subject_dataset

SEED = 42
HIDDEN_DIM = 16  # must match train_one_fold()'s default in gnn_v3.py
N_EPOCHS = 200
LR = 0.01
WEIGHT_DECAY = 1e-4
BASELINE_FEATURE_IDX = 0

CHECKPOINT_DIR = "results/checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "stage3_gnn_v3.pt")
SCALER_PATH = os.path.join(CHECKPOINT_DIR, "stage3_scaler.pkl")


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_scaler_on_full_cohort(X: np.ndarray, baseline_feature_idx: int = 0) -> StandardScaler:
    """
    Fits a StandardScaler on the non-baseline features (amyloid,
    thickness, dt_years) across the FULL cohort -- mirrors
    scale_non_baseline_features() in gnn_v3.py but fit once on all
    subjects rather than per-fold, since this checkpoint is trained
    on the full cohort with no train/test split.
    """
    n_subj, n_nodes, n_feat = X.shape
    other_idx = [i for i in range(n_feat) if i != baseline_feature_idx]
    scaler = StandardScaler()
    scaler.fit(X[:, :, other_idx].reshape(-1, len(other_idx)))
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler, baseline_feature_idx: int = 0) -> np.ndarray:
    n_subj, n_nodes, n_feat = X.shape
    other_idx = [i for i in range(n_feat) if i != baseline_feature_idx]
    X_scaled = X.copy()
    X_scaled[:, :, other_idx] = scaler.transform(
        X[:, :, other_idx].reshape(-1, len(other_idx))
    ).reshape(n_subj, n_nodes, len(other_idx))
    return X_scaled


def train_full_cohort(X_scaled: np.ndarray, y: np.ndarray, A_norm: torch.Tensor) -> TauDigitalTwinGNN:
    model = TauDigitalTwinGNN(
        in_dim=X_scaled.shape[-1],
        hidden_dim=HIDDEN_DIM,
        baseline_feature_idx=BASELINE_FEATURE_IDX,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    A_norm = A_norm.to(DEVICE)

    model.train()
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        preds = model(X_t, A_norm)
        loss = loss_fn(preds, y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"epoch {epoch + 1}/{N_EPOCHS}  loss={loss.item():.6f}")

    return model


def main() -> None:
    set_all_seeds(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment = build_subject_dataset()
    A_norm = build_normalized_adjacency(adj)

    X, y, dt, rids = build_individualized_tensors(
        cohort, regional_tau, amyloid_df, thickness_df, alignment, region_labels
    )
    print(f"Training on {len(rids)} subjects (full cohort, no held-out split)")

    scaler = fit_scaler_on_full_cohort(X, baseline_feature_idx=BASELINE_FEATURE_IDX)
    X_scaled = apply_scaler(X, scaler, baseline_feature_idx=BASELINE_FEATURE_IDX)

    model = train_full_cohort(X_scaled, y, A_norm)

    torch.save(model.state_dict(), MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saved model checkpoint to {MODEL_PATH}")
    print(f"Saved fitted scaler to {SCALER_PATH}")


if __name__ == "__main__":
    main()
