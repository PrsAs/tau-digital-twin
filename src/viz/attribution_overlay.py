"""
scripts/stage6_attribution_glow_overlay_v3.py

FIX from v1: subject 6952's top-attributed edges are between
RIGHT-hemisphere regions (R_entorhinal -> R_insula), but v1 only
built a LEFT-hemisphere mesh + LEFT-hemisphere region centroids.
Since `centroids` only contained "L_*" keys, build_glow_traces()
silently skipped every edge for this subject (both endpoints failed
the `if src not in centroids` check) -- nothing was a coincidence,
subject 4168 just happened to have its top region on the left.

FIX: build centroids AND mesh for BOTH hemispheres, and render BOTH
hemisphere surfaces together so glow-lines draw correctly regardless
of which hemisphere the model's top-attributed edges fall in.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import templateflow.api as tflow
import nibabel as nib
from nilearn.datasets import load_fsaverage

from src.data.loaders_v2 import build_subject_dataset
from src.model.gnn_v3 import build_normalized_adjacency, build_individualized_tensors, DEVICE
from src.interpret.attribution_v2 import compute_subject_attribution, load_trained_model_for_subject_set

CASE_STUDY_PATH = "results/figures/stage6_case_study_subjects_v1.csv"
BASELINE_FRAME_TEMPLATE = "results/figures/case_studies/{case_type}_{subject_id}_baseline_v1.csv"
OUT_DIR = "results/figures/case_studies/attribution"
DENSITY = "10k"
FS_MESH_KEY = "fsaverage5"
HEMIS = ["left", "right"]
HEMI_TF_MAP = {"left": "L", "right": "R"}
HEMI_PREFIX_MAP = {"left": "L_", "right": "R_"}
N_GLOW_LAYERS = 3
TOP_N_EDGES_TO_DRAW = 12


def load_dk_surface_and_lut(hemi_tf):
    label_gii = tflow.get("fsaverage", atlas="Desikan2006", density=DENSITY, hemi=hemi_tf, extension="label.gii")
    gii = nib.load(str(label_gii))
    roi_data = gii.darrays[0].data

    lut_path = tflow.get("fsaverage", atlas="Desikan2006", suffix="dseg", extension="tsv")
    lut = pd.read_csv(lut_path, sep="\t")
    lut_hemi = lut[lut["hemi"] == hemi_tf].reset_index(drop=True)
    name_to_id = {str(row["name"]).lower().strip(): pos for pos, row in lut_hemi.iterrows()}
    return roi_data, name_to_id


def region_values_to_vertex_map(region_value_df, roi_data, name_to_id, hemi_prefix):
    vertex_values = np.full(roi_data.shape[0], np.nan)
    for _, row in region_value_df.iterrows():
        region_label = row["region_label"]
        if not region_label.startswith(hemi_prefix):
            continue
        bare_name = region_label[len(hemi_prefix):].lower()
        region_id = name_to_id.get(bare_name)
        if region_id is None:
            continue
        vertex_values[roi_data == region_id] = row["value"]
    return vertex_values


def compute_region_centroids(coords, roi_data, name_to_id, hemi_prefix):
    centroids = {}
    for bare_name, region_id in name_to_id.items():
        mask = roi_data == region_id
        if not mask.any():
            continue
        centroids[f"{hemi_prefix}{bare_name}"] = coords[mask].mean(axis=0)
    return centroids


def build_glow_traces(edge_df, centroids, n_layers=N_GLOW_LAYERS, top_n=TOP_N_EDGES_TO_DRAW):
    edge_df = edge_df.copy()
    edge_df["abs_attr"] = edge_df["edge_attribution"].abs()
    edge_df = edge_df.sort_values("abs_attr", ascending=False).head(top_n)

    max_attr = edge_df["abs_attr"].max() if len(edge_df) else 1.0
    traces = []
    dropped = []

    for _, row in edge_df.iterrows():
        src, dst = row["source_region"], row["dest_region"]
        if src not in centroids or dst not in centroids:
            dropped.append((src, dst))
            continue
        p0, p1 = centroids[src], centroids[dst]
        strength = row["abs_attr"] / max_attr if max_attr > 0 else 0.0

        for layer in range(n_layers):
            width = 2 + strength * 10 * (n_layers - layer) / n_layers
            opacity = 0.15 + 0.5 * strength if layer == n_layers - 1 else 0.08 * (layer + 1) / n_layers
            traces.append(go.Scatter3d(
                x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
                mode="lines",
                line=dict(color="cyan", width=width),
                opacity=opacity,
                hoverinfo="text",
                text=f"{src} -> {dst}, attribution={row['edge_attribution']:.4f}",
                showlegend=False,
            ))
    if dropped:
        print(f"    WARNING: {len(dropped)} edges dropped (centroid not found): {dropped}")
    return traces


def build_overlay_figure(hemi_meshes, hemi_vertex_values, edge_df, centroids, vmin, vmax, title):
    mesh_traces = []
    for hemi in HEMIS:
        mesh = hemi_meshes[hemi]
        coords, faces = mesh.coordinates, mesh.faces
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        mesh_traces.append(go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            intensity=hemi_vertex_values[hemi],
            colorscale="Hot", cmin=vmin, cmax=vmax,
            opacity=0.55,
            colorbar=dict(title="Tau SUVR") if hemi == "left" else None,
            showscale=(hemi == "left"),
            lighting=dict(ambient=0.7, diffuse=0.6, specular=0.05),
            name=f"{hemi} baseline tau",
        ))

    glow_traces = build_glow_traces(edge_df, centroids)

    fig = go.Figure(data=mesh_traces + glow_traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = pd.read_csv(CASE_STUDY_PATH)

    print("Loading cohort + connectome + trained model for attribution...")
    adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment = build_subject_dataset()
    A_norm = build_normalized_adjacency(adj).to(DEVICE)
    model = load_trained_model_for_subject_set(model_path=None)
    X, y, dt, rids = build_individualized_tensors(cohort, regional_tau, amyloid_df, thickness_df, alignment, region_labels)
    rid_to_idx = {rid: idx for idx, rid in enumerate(rids)}

    print("Loading BOTH hemisphere surfaces + DK atlases...")
    fsaverage_meshes = load_fsaverage(FS_MESH_KEY)
    hemi_meshes, hemi_roi_data, hemi_name_to_id, centroids = {}, {}, {}, {}
    for hemi in HEMIS:
        hemi_tf, hemi_prefix = HEMI_TF_MAP[hemi], HEMI_PREFIX_MAP[hemi]
        hemi_meshes[hemi] = fsaverage_meshes["pial"].parts[hemi]
        roi_data, name_to_id = load_dk_surface_and_lut(hemi_tf)
        hemi_roi_data[hemi] = roi_data
        hemi_name_to_id[hemi] = name_to_id
        centroids.update(compute_region_centroids(hemi_meshes[hemi].coordinates, roi_data, name_to_id, hemi_prefix))
    print(f"Computed centroids for {len(centroids)} regions total (both hemispheres).")

    for _, row in cases.iterrows():
        subject_id, case_type = row["subject_id"], row["case_type"]
        print(f"[{case_type}] subject {subject_id}: computing edge attribution...")

        try:
            node_df, edge_df = compute_subject_attribution(
                subject_id, model, A_norm, region_labels, cohort, regional_tau,
                amyloid_df, thickness_df, alignment, target_region_idx=None,
                m_steps=100, X_lookup=X, rid_to_idx=rid_to_idx,
            )
        except IndexError as e:
            print(f"    SKIPPED: {e}")
            continue

        target_region = edge_df["target_region"].iloc[0]
        print(f"    target region (highest predicted SUVR): {target_region}")
        top_row = edge_df.iloc[0]
        print(f"    top edge: {top_row['source_region']} -> {top_row['dest_region']} (attribution={top_row['edge_attribution']:.4f})")

        baseline_csv = BASELINE_FRAME_TEMPLATE.format(case_type=case_type, subject_id=subject_id)
        baseline_df = pd.read_csv(baseline_csv)
        vmin, vmax = baseline_df["value"].min(), baseline_df["value"].max()

        hemi_vertex_values = {
            hemi: region_values_to_vertex_map(baseline_df, hemi_roi_data[hemi], hemi_name_to_id[hemi], HEMI_PREFIX_MAP[hemi])
            for hemi in HEMIS
        }

        fig = build_overlay_figure(
            hemi_meshes, hemi_vertex_values, edge_df, centroids, vmin, vmax,
            title=f"Subject {subject_id} ({case_type}) -- top {TOP_N_EDGES_TO_DRAW} attributed "
                  f"connectome edges for predicting {target_region}",
        )

        out_path = f"{OUT_DIR}/{case_type}_{subject_id}_attribution_overlay_v2.html"
        fig.write_html(out_path)
        edge_df.to_csv(f"{OUT_DIR}/{case_type}_{subject_id}_edge_attribution_v1.csv", index=False)
        print(f"    saved -> {out_path}")

    print("\nDone. Both hemispheres now rendered -- glow-lines will appear")
    print("regardless of which hemisphere a subjects top-attributed edges fall in.")


if __name__ == "__main__":
    main()
