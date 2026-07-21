"""
scripts/run_attribution_full_cohort_v1.py

Scales Stage 3 attribution from the 10-subject pilot to the FULL 517-
subject cohort, now that all three diagnostic checks pass on the fixed
pipeline (checkpoint + scaler + full-edge-set correlation):

  - Mean completeness error: 0.000258 (negligible vs. SUVR scale 0.05-1.5)
  - baseline_tau share of attribution: 95.1% (expected given residual
    model design)
  - Edge attribution vs. structural connectome weight: Pearson r=0.192
    (p<0.0001), Spearman r=0.457 (p<0.0001) -- strong, significant,
    positive -- good sign for Stage 5.

STORAGE STRATEGY:
Saving every off-diagonal edge row for all 517 subjects at ~2,278 edges/
subject would produce ~1.18M rows -- too large and mostly uninformative
for Stage 5/6 consumption. Instead, this script saves THREE outputs:

  1. Node/feature attribution -- full detail, one row per
     (subject, region), saved in full (this is already small: 680 rows
     for 10 subjects -> ~35K rows for 517 subjects, entirely manageable).

  2. Per-subject top-K edges (K=20) by |attribution| -- for
     visualization/exploration in Stage 6, clearly labeled as a
     magnitude-biased subset, NOT to be used for correlation stats
     (per attribution_v3.py's docstring).

  3. Per-subject summary statistics computed on the FULL edge set
     (Pearson r, Spearman r, mean/max |attribution|) -- this preserves
     the unbiased structural-plausibility signal per subject without
     storing every raw edge row, and lets you check whether the
     positive correlation holds consistently across subjects or is
     driven by a handful of outliers.
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from src.data.loaders_v2 import build_subject_dataset
from src.model.gnn_v3 import build_normalized_adjacency, build_individualized_tensors, DEVICE
from src.interpret.attribution_v3 import (
    load_trained_model_for_subject_set,
    apply_saved_scaler,
    integrated_gradients_node_features,
    integrated_gradients_edges,
    FEATURE_NAMES,
)

MODEL_PATH = "results/checkpoints/stage3_gnn_v3.pt"
SCALER_PATH = "results/checkpoints/stage3_scaler.pkl"
M_STEPS = 100
TOP_K_EDGES = 20

NODE_OUT_PATH = "results/figures/stage3_node_attribution_full_cohort_v1.csv"
EDGE_TOPK_OUT_PATH = "results/figures/stage3_edge_attribution_topk_full_cohort_v1.csv"
EDGE_SUMMARY_OUT_PATH = "results/figures/stage3_edge_correlation_summary_full_cohort_v1.csv"


def main():
    adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment = build_subject_dataset()
    A_norm = build_normalized_adjacency(adj).to(DEVICE)
    A_norm_np = A_norm.detach().cpu().numpy()
    model = load_trained_model_for_subject_set(model_path=MODEL_PATH)

    X, y, dt, rids = build_individualized_tensors(cohort, regional_tau, amyloid_df, thickness_df, alignment, region_labels)
    X = apply_saved_scaler(X, SCALER_PATH, baseline_feature_idx=0)
    rid_to_idx = {rid: i for i, rid in enumerate(rids)}

    n_nodes = A_norm_np.shape[0]
    iu, ju = np.triu_indices(n_nodes, k=1)
    struct_weights_flat = A_norm_np[iu, ju]

    all_node_rows, all_topk_edge_rows, summary_rows, dropped_subjects = [], [], [], []

    for i, rid in enumerate(rids):
        x = torch.tensor(X[rid_to_idx[rid]], dtype=torch.float32, device=DEVICE)
        model.eval()
        with torch.no_grad():
            preds = model(x.unsqueeze(0), A_norm)[0].cpu().numpy()
        target_region_idx = int(np.argmax(preds))

        node_attr = integrated_gradients_node_features(model, x, A_norm, target_region_idx, m_steps=M_STEPS)
        node_df = pd.DataFrame(node_attr, columns=FEATURE_NAMES)
        node_df.insert(0, "region_label", region_labels)
        node_df.insert(0, "subject_id", rid)
        node_df.insert(2, "target_region", region_labels[target_region_idx])
        node_df.insert(3, "predicted_suvr_target", preds[target_region_idx])
        all_node_rows.append(node_df)

        edge_attr = integrated_gradients_edges(model, x, A_norm, target_region_idx, m_steps=M_STEPS)
        edge_attr_flat = edge_attr[iu, ju]

        r_p, p_p = pearsonr(np.abs(edge_attr_flat), struct_weights_flat)
        r_s, p_s = spearmanr(np.abs(edge_attr_flat), struct_weights_flat)
        summary_rows.append({
            "subject_id": rid,
            "target_region": region_labels[target_region_idx],
            "pearson_r": r_p, "pearson_p": p_p,
            "spearman_r": r_s, "spearman_p": p_s,
            "mean_abs_edge_attribution": float(np.mean(np.abs(edge_attr_flat))),
            "max_abs_edge_attribution": float(np.max(np.abs(edge_attr_flat))),
        })

        top_idx = np.argsort(-np.abs(edge_attr_flat))[:TOP_K_EDGES]
        for k in top_idx:
            ii, jj = iu[k], ju[k]
            all_topk_edge_rows.append({
                "subject_id": rid, "target_region": region_labels[target_region_idx],
                "source_region": region_labels[ii], "dest_region": region_labels[jj],
                "edge_attribution": float(edge_attr[ii, jj]),
                "edge_weight_in_A_norm": float(A_norm_np[ii, jj]),
            })

        if (i + 1) % 50 == 0:
            print(f"processed {i + 1}/{len(rids)} subjects")

    node_result = pd.concat(all_node_rows, ignore_index=True)
    topk_edge_result = pd.DataFrame(all_topk_edge_rows)
    summary_result = pd.DataFrame(summary_rows)

    node_result.to_csv(NODE_OUT_PATH, index=False)
    topk_edge_result.to_csv(EDGE_TOPK_OUT_PATH, index=False)
    summary_result.to_csv(EDGE_SUMMARY_OUT_PATH, index=False)

    print(f"Node attribution shape: {node_result.shape} -> {NODE_OUT_PATH}")
    print(f"Top-K edge attribution shape: {topk_edge_result.shape} -> {EDGE_TOPK_OUT_PATH}")
    print(f"Per-subject correlation summary shape: {summary_result.shape} -> {EDGE_SUMMARY_OUT_PATH}")
    mean_pearson = summary_result["pearson_r"].mean()
    mean_spearman = summary_result["spearman_r"].mean()
    print(f"Mean per-subject Pearson r (full edge set): {mean_pearson:.3f}")
    print(f"Mean per-subject Spearman r (full edge set): {mean_spearman:.3f}")
    frac_positive = (summary_result["pearson_r"] > 0).mean()
    print(f"Fraction of subjects with positive Pearson r: {frac_positive:.2%}")


if __name__ == "__main__":
    main()
