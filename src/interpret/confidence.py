"""
src/interpret/confidence_v1.py

Stage 3 (per the thesis plan): "confidence.py -- per-region prediction
uncertainty." Missing from the repo until now -- attribution_v2.py
covered the "why" (integrated gradients), this covers the "how sure
is the model" (regional uncertainty), which Stage 6 also explicitly
requires as a "toggleable confidence-heatmap layer."

METHOD: Monte Carlo Dropout.
gnn_v3.TauDigitalTwinGNN already has dropout=0.3 built into its two
GCN layers. Normally dropout is disabled at inference (model.eval()).
MC Dropout instead keeps dropout ACTIVE during inference and runs many
stochastic forward passes -- the spread (std) of predictions across
those passes approximates the model's epistemic uncertainty for that
input, with zero architecture changes and zero retraining required.

Reference: Gal & Ghahramani (2016), "Dropout as a Bayesian
Approximation: Representing Model Uncertainty in Deep Learning."

This deliberately reuses load_trained_model_for_subject_set from
attribution_v2.py so the SAME model (full-cohort fit, no held-out
split, per that module's own documented rationale) is used for both
attribution and uncertainty -- keeping the two Stage 6 interpretability
overlays consistent with each other for a given subject.
"""

import numpy as np
import pandas as pd
import torch

from src.data.loaders_v2 import build_subject_dataset
from src.model.gnn_v3 import build_normalized_adjacency, build_individualized_tensors, DEVICE
from src.interpret.attribution_v2 import load_trained_model_for_subject_set

N_MC_SAMPLES = 100


def enable_mc_dropout(model):
    """Sets ONLY dropout layers to train mode, keeping BatchNorm/etc (none present
    here, but future-proofed) in eval mode. For gnn_v3's architecture this is
    equivalent to model.train(), since its only stochastic layers are nn.Dropout."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_predict(model, x, A_norm, n_samples=N_MC_SAMPLES):
    """
    x: [n_nodes, n_features] tensor for ONE subject.
    Returns (mean_pred, std_pred): both [n_nodes] arrays.
    std_pred is the per-region epistemic uncertainty estimate.
    """
    model.eval()
    enable_mc_dropout(model)

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x.unsqueeze(0), A_norm)[0]
            samples.append(pred.cpu().numpy())

    samples = np.stack(samples, axis=0)  # [n_samples, n_nodes]
    return samples.mean(axis=0), samples.std(axis=0)


def compute_subject_confidence(rid, model, A_norm, region_labels, X_lookup, rid_to_idx, n_samples=N_MC_SAMPLES):
    if rid not in rid_to_idx:
        raise IndexError(f"Subject {rid} dropped by build_individualized_tensors (no valid follow-up pair or unresolvable NaN).")

    x = torch.tensor(X_lookup[rid_to_idx[rid]], dtype=torch.float32, device=DEVICE)
    mean_pred, std_pred = mc_dropout_predict(model, x, A_norm, n_samples=n_samples)

    df = pd.DataFrame({
        "subject_id": rid,
        "region_label": region_labels,
        "mean_predicted_suvr": mean_pred,
        "uncertainty_std": std_pred,
    })
    # Normalized 0-1 uncertainty within THIS subject, for consistent heatmap scaling
    # across subjects whose absolute SUVR ranges differ substantially (e.g. 4168 vs 6952).
    denom = df["uncertainty_std"].max() - df["uncertainty_std"].min()
    df["uncertainty_normalized"] = (
        (df["uncertainty_std"] - df["uncertainty_std"].min()) / denom if denom > 0 else 0.0
    )
    return df


def run_confidence_for_cohort(subject_ids, model_path=None, n_samples=N_MC_SAMPLES, verbose=True):
    adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment = build_subject_dataset()
    A_norm = build_normalized_adjacency(adj).to(DEVICE)
    model = load_trained_model_for_subject_set(model_path=model_path)

    X, y, dt, rids = build_individualized_tensors(cohort, regional_tau, amyloid_df, thickness_df, alignment, region_labels)
    rid_to_idx = {rid: i for i, rid in enumerate(rids)}

    all_rows, dropped_subjects = [], []
    for rid in subject_ids:
        try:
            df = compute_subject_confidence(rid, model, A_norm, region_labels, X, rid_to_idx, n_samples=n_samples)
            all_rows.append(df)
        except IndexError as e:
            dropped_subjects.append((rid, str(e)))
            continue

    if verbose and dropped_subjects:
        print(f"Dropped {len(dropped_subjects)} subject(s) during confidence estimation:")
        for rid, reason in dropped_subjects:
            print(f"  subject_id={rid}: {reason}")

    result = pd.concat(all_rows, ignore_index=True)
    return result, dropped_subjects


if __name__ == "__main__":
    adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment = build_subject_dataset()
    sample_subjects = cohort["subject_id"].values[:10]  # small pilot run first, matches attribution_v2 pattern

    result, dropped = run_confidence_for_cohort(sample_subjects, model_path=None, n_samples=N_MC_SAMPLES)
    result.to_csv("results/figures/stage3_confidence_pilot_v1.csv", index=False)
    print(f"Confidence result shape: {result.shape}")
    print(f"Dropped subjects: {len(dropped)}")
    print(result.head(10))
    print(f"Mean uncertainty_std across all regions/subjects: {result['uncertainty_std'].mean():.4f}")
