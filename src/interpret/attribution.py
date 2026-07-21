"""
src/interpret/attribution_v3.py
====================================
Stage 3: Interpretability layer, per the thesis plan:
"Apply integrated gradients or GNNExplainer to the trained model, per
subject, to extract which connectome edges/regions drove each specific
prediction. Store these attribution weights alongside predictions --
they are both a scientific output (Stage 5) and a visualization input
(Stage 6)."

Uses gnn_v3 (the final chosen Stage 2 model: simpler than v4's ranking-loss
variant, statistically equivalent on hit-rate, and the one with the
strongest, cleanest significance result vs. persistence/diffusion/
growth-diffusion). No external interpretability library (e.g. captum) is
assumed available, so Integrated Gradients is implemented directly in
pure PyTorch -- this keeps the method fully transparent and avoids adding
an unverified new dependency.

TWO ATTRIBUTION TARGETS, both requested by the plan:
1. NODE/FEATURE attribution: which region + which input feature
(baseline_tau, amyloid, thickness, dt_years) drove the prediction
for a GIVEN target region.
2. EDGE attribution: which connectome edges (in the normalized
adjacency A_norm) drove the prediction for a given target region --
computed by treating A_norm as a continuous input and integrating
gradients from an all-zero adjacency baseline to the real one.

Both use the standard Integrated Gradients formula:
IG_i(x) = (x_i - x'_i) * (1/m) * sum_{k=1}^{m} d/dx_i F(x' + k/m * (x - x'))
where x' is the baseline (all-zero) input and F is the model's scalar
output for the target region of interest.

--- v2 -> v2-fixed CHANGES ---
Two bugs found when reconciling this file against gnn_v3.py's actual
training pipeline (train_one_fold / evaluate_gnn_per_subject):

1. hidden_dim mismatch: load_trained_model_for_subject_set() defaulted
   to hidden_dim=32, but train_one_fold() in gnn_v3.py builds
   TauDigitalTwinGNN with the default hidden_dim=16. Loading a
   hidden_dim=16 checkpoint into a hidden_dim=32 model would raise a
   state_dict shape-mismatch error, or worse -- silently produce a
   differently-shaped, effectively untrained model if defaults ever
   happened to align by accident. Fixed: default is now 16, and
   run_attribution_for_cohort/load_trained_model_for_subject_set both
   accept an explicit scaler_path so the SAME preprocessing used in
   training is guaranteed at inference time.

2. scaling mismatch: gnn_v3.py's training pipeline ALWAYS calls
   scale_non_baseline_features() (StandardScaler on amyloid/thickness/
   dt_years; baseline_tau left raw for the residual connection) before
   fitting. But compute_subject_attribution() fed the raw, unscaled
   X_lookup straight into the model. A model trained on standardized
   inputs but evaluated on raw-scale inputs produces meaningless
   gradients -- this was the most likely root cause of the inflated
   completeness error (0.168) and the collapsed structural correlation
   (r=-0.119) seen in stage3_attribution_diagnostic_v3. Fixed: X is now
   scaled with the SAME fitted scaler used at training time
   (results/checkpoints/stage3_scaler.pkl) before being handed to
   compute_subject_attribution.
"""

import pickle

import numpy as np
import pandas as pd
import torch

from src.data.loaders_v2 import build_subject_dataset, get_subject_tau_vector, get_subject_baseline_covariates
from src.model.gnn_v3 import build_normalized_adjacency, build_individualized_tensors, TauDigitalTwinGNN, DEVICE

FEATURE_NAMES = ["baseline_tau", "amyloid_suvr", "cortical_thickness", "dt_years"]

DEFAULT_HIDDEN_DIM = 16  # matches train_one_fold()'s default in gnn_v3.py
DEFAULT_SCALER_PATH = "results/checkpoints/stage3_scaler.pkl"


def apply_saved_scaler(X: np.ndarray, scaler_path: str, baseline_feature_idx: int = 0) -> np.ndarray:
    """
    Applies the StandardScaler fitted at training time (on non-baseline
    features only) to X before it is fed to the model for attribution.
    This mirrors scale_non_baseline_features() in gnn_v3.py exactly --
    baseline_tau (feature 0) is left unscaled because the model's
    residual connection expects it in raw SUVR units.
    """
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    n_subj, n_nodes, n_feat = X.shape
    other_idx = [i for i in range(n_feat) if i != baseline_feature_idx]
    X_scaled = X.copy()
    X_scaled[:, :, other_idx] = scaler.transform(
        X[:, :, other_idx].reshape(-1, len(other_idx))
    ).reshape(n_subj, n_nodes, len(other_idx))
    return X_scaled


def integrated_gradients_node_features(model, x, A_norm, target_region_idx, m_steps=50):
    """
    x: [n_nodes, n_features] tensor for ONE subject (no batch dim).
    Returns attribution: [n_nodes, n_features] array -- how much each
    region's each feature contributed to the model's prediction for
    target_region_idx.
    """
    x_baseline = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, m_steps + 1, device=x.device)[1:]

    total_grad = torch.zeros_like(x)
    for alpha in alphas:
        x_interp = (x_baseline + alpha * (x - x_baseline)).clone().detach().requires_grad_(True)
        pred = model(x_interp.unsqueeze(0), A_norm)[0, target_region_idx]
        grad = torch.autograd.grad(pred, x_interp)[0]
        total_grad += grad

    avg_grad = total_grad / m_steps
    attribution = (x - x_baseline) * avg_grad
    return attribution.detach().cpu().numpy()


def integrated_gradients_edges(model, x, A_norm, target_region_idx, m_steps=50):
    """
    A_norm: [n_nodes, n_nodes] normalized adjacency for the WHOLE cohort
    (shared connectome). Returns attribution: [n_nodes, n_nodes] array --
    how much each edge contributed to the prediction for target_region_idx,
    for this specific subject's feature vector x.
    """
    A_baseline = torch.zeros_like(A_norm)
    alphas = torch.linspace(0, 1, m_steps + 1, device=A_norm.device)[1:]

    total_grad = torch.zeros_like(A_norm)
    for alpha in alphas:
        A_interp = (A_baseline + alpha * (A_norm - A_baseline)).clone().detach().requires_grad_(True)
        pred = model(x.unsqueeze(0), A_interp)[0, target_region_idx]
        grad = torch.autograd.grad(pred, A_interp)[0]
        total_grad += grad

    avg_grad = total_grad / m_steps
    attribution = (A_norm - A_baseline) * avg_grad
    return attribution.detach().cpu().numpy()


def load_trained_model_for_subject_set(model_path=None, hidden_dim=DEFAULT_HIDDEN_DIM, in_dim=4):
    """
    Loads a trained TauDigitalTwinGNN checkpoint. hidden_dim now
    defaults to 16 to match train_one_fold()'s actual default in
    gnn_v3.py -- previously defaulted to 32, which would either error
    on load_state_dict or silently mismatch.

    Expects model_path to point to a saved state_dict
    (torch.save(model.state_dict(), path)). If model_path is None,
    trains a fresh model on the FULL cohort (no held-out split) purely
    for attribution purposes. Prefer passing model_path explicitly --
    see scripts/train_stage3_checkpoint.py, which trains ONE model on
    the full cohort with a fixed seed and persists it, so the pilot
    attribution run and the diagnostic run use the SAME model.
    """
    model = TauDigitalTwinGNN(in_dim=in_dim, hidden_dim=hidden_dim, baseline_feature_idx=0).to(DEVICE)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model


def compute_subject_attribution(rid, model, A_norm, region_labels, cohort, regional_tau,
                                 amyloid_df, thickness_df, alignment, target_region_idx=None,
                                 m_steps=100, X_lookup=None, rid_to_idx=None,
                                 top_k_edges=20, save_full_edges=False):
    """
    Computes both node-feature and edge attribution for ONE subject.
    If target_region_idx is None, uses the region with the HIGHEST
    predicted follow-up SUVR as the target (i.e. "explain the model's
    top prediction for this subject").

    IMPORTANT: X_lookup must already be SCALED with the same scaler
    used at training time (see apply_saved_scaler) before being passed
    in here -- this function no longer scales internally, to keep
    scaling an explicit, auditable step in run_attribution_for_cohort.

    --- v2 -> v3 CHANGE (edge attribution truncation bias) ---
    diagnose_edge_correlation_v2.py found that keeping only the
    top_k_edges=20 highest-|attribution| edges per subject introduces a
    SELECTION ARTIFACT: on the full, untruncated edge set the
    attribution-vs-structural-weight correlation is strongly positive
    (Pearson r=0.192, p<0.0001; Spearman r=0.457, p<0.0001), but on the
    top-20-per-subject subset it flips to negative and largely loses
    significance (Pearson r=-0.138, p=0.0518). This is because
    structurally strong edges already carry signal through the graph
    convolution and need smaller marginal gradient corrections, while
    weak edges sometimes need disproportionately large corrections --
    so filtering by |attribution| magnitude alone oversamples weak-
    structural/high-gradient edges.

    Fix: this function now returns the FULL off-diagonal edge
    attribution set by default when save_full_edges=True, so downstream
    correlation analysis is computed on the true (unbiased) edge
    population. The top_k_edges truncation is now ONLY used for
    lightweight per-subject visualization exports, and is clearly
    labeled as such -- it should never be used to compute or report a
    structural-plausibility correlation statistic.
    """
    if X_lookup is not None and rid_to_idx is not None:
        if rid not in rid_to_idx:
            raise IndexError(f"Subject {rid} dropped by build_individualized_tensors (no valid follow-up pair or unresolvable NaN).")
        x = torch.tensor(X_lookup[rid_to_idx[rid]], dtype=torch.float32, device=DEVICE)
    else:
        x0 = get_subject_tau_vector(regional_tau, rid, 0, alignment)
        row = cohort[cohort["subject_id"] == rid].iloc[0]
        dt_years = row["inter_visit_interval_days"] / 365.25
        amy_vec, thick_vec = get_subject_baseline_covariates(amyloid_df, thickness_df, rid, alignment)
        dt_feature = np.full_like(x0, dt_years)
        x = torch.tensor(np.stack([x0, amy_vec, thick_vec, dt_feature], axis=-1),
                          dtype=torch.float32, device=DEVICE)

    model.eval()
    with torch.no_grad():
        preds = model(x.unsqueeze(0), A_norm)[0].cpu().numpy()

    if target_region_idx is None:
        target_region_idx = int(np.argmax(preds))

    node_attr = integrated_gradients_node_features(model, x, A_norm, target_region_idx, m_steps=m_steps)
    edge_attr = integrated_gradients_edges(model, x, A_norm, target_region_idx, m_steps=m_steps)

    node_attr_df = pd.DataFrame(node_attr, columns=FEATURE_NAMES)
    node_attr_df.insert(0, "region_label", region_labels)
    node_attr_df.insert(0, "subject_id", rid)
    node_attr_df.insert(2, "target_region", region_labels[target_region_idx])
    node_attr_df.insert(3, "predicted_suvr_target", preds[target_region_idx])

    A_norm_np = A_norm.detach().cpu().numpy()

    if save_full_edges:
        n_nodes = edge_attr.shape[0]
        iu, ju = np.triu_indices(n_nodes, k=1)  # off-diagonal, no self-loops, no duplicate (i,j)/(j,i)
        edge_rows = [{
            "subject_id": rid, "target_region": region_labels[target_region_idx],
            "source_region": region_labels[i], "dest_region": region_labels[j],
            "edge_attribution": edge_attr[i, j], "edge_weight_in_A_norm": float(A_norm_np[i, j]),
        } for i, j in zip(iu, ju)]
    else:
        # legacy top-K export -- for lightweight visualization ONLY.
        # DO NOT use this subset to compute structural-plausibility
        # correlation statistics; it is a biased sample by construction.
        top_edges_idx = np.dstack(
            np.unravel_index(np.argsort(-np.abs(edge_attr).ravel()), edge_attr.shape)
        )[0][:top_k_edges]
        edge_rows = [{
            "subject_id": rid, "target_region": region_labels[target_region_idx],
            "source_region": region_labels[i], "dest_region": region_labels[j],
            "edge_attribution": edge_attr[i, j], "edge_weight_in_A_norm": float(A_norm_np[i, j]),
        } for i, j in top_edges_idx]

    edge_attr_df = pd.DataFrame(edge_rows)

    return node_attr_df, edge_attr_df


def run_attribution_for_cohort(subject_ids, model_path=None, scaler_path=DEFAULT_SCALER_PATH,
                                m_steps=100, verbose=True, top_k_edges=20, save_full_edges=False):
    adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment = build_subject_dataset()
    A_norm = build_normalized_adjacency(adj).to(DEVICE)
    model = load_trained_model_for_subject_set(model_path=model_path)

    X, y, dt, rids = build_individualized_tensors(cohort, regional_tau, amyloid_df, thickness_df, alignment, region_labels)

    if scaler_path is not None:
        X = apply_saved_scaler(X, scaler_path, baseline_feature_idx=0)

    rid_to_idx = {rid: i for i, rid in enumerate(rids)}

    all_node_rows, all_edge_rows, dropped_subjects = [], [], []
    for rid in subject_ids:
        try:
            node_df, edge_df = compute_subject_attribution(
                rid, model, A_norm, region_labels, cohort, regional_tau,
                amyloid_df, thickness_df, alignment, m_steps=m_steps,
                X_lookup=X, rid_to_idx=rid_to_idx,
                top_k_edges=top_k_edges, save_full_edges=save_full_edges
            )
            all_node_rows.append(node_df)
            all_edge_rows.append(edge_df)
        except IndexError as e:
            dropped_subjects.append((rid, str(e)))
            continue

    node_result = pd.concat(all_node_rows, ignore_index=True) if all_node_rows else pd.DataFrame()
    edge_result = pd.concat(all_edge_rows, ignore_index=True) if all_edge_rows else pd.DataFrame()

    if verbose:
        print(f"Node attribution shape: {node_result.shape}")
        print(f"Edge attribution shape: {edge_result.shape}")
        print(f"Dropped subjects: {len(dropped_subjects)}")

    return node_result, edge_result, dropped_subjects
