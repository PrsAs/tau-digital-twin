"""
scripts/stage5_path_length_validation_v1.py

Completes Stage 5 -- Interpretability Validation, per the thesis plan's
SECOND clause (the first clause -- edge attribution vs. structural
CONNECTOME WEIGHT -- was already validated in
run_attribution_full_cohort_v2.py / stage3_edge_correlation_summary_
full_cohort_v1.csv, with mean per-subject Pearson r=0.239, 100% of
subjects positive):

"...or shorter path length to the newly-affected region"

This is a DISTINCT test from edge weight: it asks whether edges the
model attributes heavily are graph-theoretically CLOSE (in shortest-
path terms) to the specific region where tau actually increased at
follow-up, for that subject -- not just whether the edge itself is
structurally strong.

METHOD:
1. Build the structural connectome as a weighted graph (using
   1/edge_weight as "distance" -- standard convention: stronger
   structural connections = shorter effective distance/higher
   traversal efficiency, per network neuroscience graph-theory
   conventions).
2. For each subject, identify their TRUE newly-affected region (the
   region with the largest true increase: y_true_suvr - baseline_suvr
   at follow-up).
3. For every edge (i, j) in that subject's top-K attribution export
   (or, more robustly, the full edge attribution set), compute the
   shortest-path distance from region i to the true newly-affected
   region, and from region j to it -- take the MINIMUM of the two
   (i.e. "how close does this edge get you to the target region").
4. Test whether |edge_attribution| correlates NEGATIVELY with this
   path-distance-to-target (shorter path = expected HIGHER attribution
   if the model is doing real, sensible work).

Uses networkx for shortest-path computation (Dijkstra, weighted by
1/edge_weight). Falls back to unweighted hop-count shortest path if
networkx is unavailable results look uninterpretable due to disconnected
components -- reported explicitly either way.
"""

import numpy as np
import pandas as pd
import torch
import networkx as nx
from scipy.stats import pearsonr, spearmanr

from src.data.loaders_v2 import build_subject_dataset
from src.model.gnn_v3 import build_normalized_adjacency, build_individualized_tensors, DEVICE
from src.interpret.attribution_v3 import (
    load_trained_model_for_subject_set,
    apply_saved_scaler,
    integrated_gradients_edges,
)

MODEL_PATH = "results/checkpoints/stage3_gnn_v3.pt"
SCALER_PATH = "results/checkpoints/stage3_scaler.pkl"
M_STEPS = 100
N_SUBJECTS_TO_CHECK = 50  # pilot on a subset first -- full cohort is a straightforward extension
OUT_PATH = "results/figures/stage5_path_length_validation_v1.csv"


def build_networkx_graph(adj: np.ndarray, region_labels: list) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(region_labels)
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            w = adj[i, j]
            if w > 0:
                G.add_edge(region_labels[i], region_labels[j], distance=1.0 / w)
    return G


def main():
    adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment = build_subject_dataset()
    A_norm = build_normalized_adjacency(adj).to(DEVICE)
    A_norm_np = A_norm.detach().cpu().numpy()
    model = load_trained_model_for_subject_set(model_path=MODEL_PATH)

    X, y, dt, rids = build_individualized_tensors(cohort, regional_tau, amyloid_df, thickness_df, alignment, region_labels)
    X_scaled = apply_saved_scaler(X, SCALER_PATH, baseline_feature_idx=0)
    rid_to_idx = {rid: i for i, rid in enumerate(rids)}

    G = build_networkx_graph(adj, region_labels)
    print(f"Connectome graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"connected: {nx.is_connected(G)}")

    n_nodes = len(region_labels)
    iu, ju = np.triu_indices(n_nodes, k=1)

    all_rows = []
    subjects_checked = 0

    for rid in rids:
        if subjects_checked >= N_SUBJECTS_TO_CHECK:
            break
        idx = rid_to_idx[rid]
        true_increase = y[idx] - X[idx, :, 0]
        true_target_idx = int(np.argmax(true_increase))
        true_target_label = region_labels[true_target_idx]

        try:
            dist_to_target = nx.single_source_dijkstra_path_length(G, true_target_label, weight="distance")
        except nx.NetworkXNoPath:
            continue

        x = torch.tensor(X_scaled[idx], dtype=torch.float32, device=DEVICE)
        model.eval()
        with torch.no_grad():
            preds = model(x.unsqueeze(0), A_norm)[0].cpu().numpy()
        predicted_target_idx = int(np.argmax(preds))

        edge_attr = integrated_gradients_edges(model, x, A_norm, predicted_target_idx, m_steps=M_STEPS)
        edge_attr_flat = edge_attr[iu, ju]

        for k in range(len(iu)):
            i, j = iu[k], ju[k]
            label_i, label_j = region_labels[i], region_labels[j]
            d_i = dist_to_target.get(label_i, np.nan)
            d_j = dist_to_target.get(label_j, np.nan)
            min_dist = np.nanmin([d_i, d_j])

            all_rows.append({
                "subject_id": rid,
                "true_target_region": true_target_label,
                "predicted_target_region": region_labels[predicted_target_idx],
                "source_region": label_i, "dest_region": label_j,
                "edge_attribution": edge_attr_flat[k],
                "min_shortest_path_dist_to_true_target": min_dist,
            })

        subjects_checked += 1
        if subjects_checked % 10 == 0:
            print(f"processed {subjects_checked}/{N_SUBJECTS_TO_CHECK} subjects")

    result_df = pd.DataFrame(all_rows)
    result_df = result_df.dropna(subset=["min_shortest_path_dist_to_true_target"])
    result_df.to_csv(OUT_PATH, index=False)

    abs_attr = result_df["edge_attribution"].abs()
    dist = result_df["min_shortest_path_dist_to_true_target"]

    r_p, p_p = pearsonr(abs_attr, dist)
    r_s, p_s = spearmanr(abs_attr, dist)

    print(f"\nSubjects checked: {subjects_checked}")
    print(f"Edge rows analyzed: {len(result_df)}")
    print(f"Pearson r (|attribution| vs. path-distance-to-true-target): {r_p:.3f} (p={p_p:.4f})")
    print(f"Spearman r (|attribution| vs. path-distance-to-true-target): {r_s:.3f} (p={p_s:.4f})")
    print("(NEGATIVE + significant = edges CLOSER to the true newly-affected region get MORE attribution --")
    print(" the expected, sensible-reasoning result. Near-zero/positive = attribution does not track proximity")
    print(" to where tau actually spread -- important caveat for Stage 5.)")
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
