"""
scripts/stage6_confidence_overlay_v1.py

Builds the LAST missing Stage 6 deliverable: "Toggleable confidence-
heatmap layer showing regional prediction uncertainty."

Reuses confidence_v1.py (MC Dropout) to get per-region uncertainty for
each case-study subject, then renders a Plotly figure with TWO Mesh3d
surfaces on the SAME geometry -- one colored by predicted tau (the
familiar "Hot" scale), one colored by normalized uncertainty (a
perceptually distinct "Viridis" scale so it's never confused with the
tau scale) -- and a button pair to toggle between them. Only one is
visible at a time; clicking the other button swaps which is shown.

This is deliberately a SEPARATE script/output from the glow-line
attribution overlay (stage6_attribution_glow_overlay_v3.py) rather than
merging all three layers (tau, glow-lines, uncertainty) into one dense
figure -- keeping each interpretability layer as its own clean,
readable HTML matches the plan's framing of these as a "small set of
interactive HTML renders," not one maximal all-in-one dashboard.
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
from src.interpret.attribution_v2 import load_trained_model_for_subject_set
from src.interpret.confidence_v1 import compute_subject_confidence, N_MC_SAMPLES

CASE_STUDY_PATH = "results/figures/stage6_case_study_subjects_v1.csv"
BASELINE_FRAME_TEMPLATE = "results/figures/case_studies/{case_type}_{subject_id}_baseline_v1.csv"
OUT_DIR = "results/figures/case_studies/confidence"
DENSITY = "10k"
FS_MESH_KEY = "fsaverage5"
HEMIS = ["left", "right"]
HEMI_TF_MAP = {"left": "L", "right": "R"}
HEMI_PREFIX_MAP = {"left": "L_", "right": "R_"}


def load_dk_surface_and_lut(hemi_tf):
    label_gii = tflow.get("fsaverage", atlas="Desikan2006", density=DENSITY, hemi=hemi_tf, extension="label.gii")
    gii = nib.load(str(label_gii))
    roi_data = gii.darrays[0].data

    lut_path = tflow.get("fsaverage", atlas="Desikan2006", suffix="dseg", extension="tsv")
    lut = pd.read_csv(lut_path, sep="\t")
    lut_hemi = lut[lut["hemi"] == hemi_tf].reset_index(drop=True)
    name_to_id = {str(row["name"]).lower().strip(): pos for pos, row in lut_hemi.iterrows()}
    return roi_data, name_to_id


def region_col_to_vertex_map(region_df, value_col, roi_data, name_to_id, hemi_prefix):
    vertex_values = np.full(roi_data.shape[0], np.nan)
    for _, row in region_df.iterrows():
        region_label = row["region_label"]
        if not region_label.startswith(hemi_prefix):
            continue
        bare_name = region_label[len(hemi_prefix):].lower()
        region_id = name_to_id.get(bare_name)
        if region_id is None:
            continue
        vertex_values[roi_data == region_id] = row[value_col]
    return vertex_values


def build_toggle_figure(hemi_meshes, hemi_tau_values, hemi_uncertainty_values, tau_vmin, tau_vmax, title):
    mesh_traces = []
    n_hemis = len(HEMIS)

    # Traces 0..n_hemis-1: tau layer (visible by default). Traces n_hemis..2n_hemis-1: uncertainty layer (hidden).
    for hemi in HEMIS:
        mesh = hemi_meshes[hemi]
        coords, faces = mesh.coordinates, mesh.faces
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        mesh_traces.append(go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            intensity=hemi_tau_values[hemi],
            colorscale="Hot", cmin=tau_vmin, cmax=tau_vmax,
            colorbar=dict(title="Predicted Tau SUVR", x=1.0) if hemi == "left" else None,
            showscale=(hemi == "left"),
            lighting=dict(ambient=0.7, diffuse=0.6, specular=0.05),
            visible=True,
            name=f"{hemi} tau",
        ))

    for hemi in HEMIS:
        mesh = hemi_meshes[hemi]
        coords, faces = mesh.coordinates, mesh.faces
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        mesh_traces.append(go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            intensity=hemi_uncertainty_values[hemi],
            colorscale="Viridis", cmin=0.0, cmax=1.0,
            colorbar=dict(title="Normalized Uncertainty (MC Dropout std)", x=1.0) if hemi == "left" else None,
            showscale=(hemi == "left"),
            lighting=dict(ambient=0.7, diffuse=0.6, specular=0.05),
            visible=False,
            name=f"{hemi} uncertainty",
        ))

    tau_visibility = [True] * n_hemis + [False] * n_hemis
    uncertainty_visibility = [False] * n_hemis + [True] * n_hemis

    fig = go.Figure(data=mesh_traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        updatemenus=[dict(
            type="buttons", direction="right", showactive=True,
            x=0.02, y=0.02, xanchor="left", yanchor="bottom",
            buttons=[
                dict(label="Predicted Tau", method="update", args=[{"visible": tau_visibility}]),
                dict(label="Prediction Uncertainty", method="update", args=[{"visible": uncertainty_visibility}]),
            ],
        )],
    )
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = pd.read_csv(CASE_STUDY_PATH)

    print("Loading cohort + connectome + trained model...")
    adj, region_labels, cohort, regional_tau, amyloid_df, thickness_df, alignment = build_subject_dataset()
    A_norm = build_normalized_adjacency(adj).to(DEVICE)
    model = load_trained_model_for_subject_set(model_path=None)
    X, y, dt, rids = build_individualized_tensors(cohort, regional_tau, amyloid_df, thickness_df, alignment, region_labels)
    rid_to_idx = {rid: idx for idx, rid in enumerate(rids)}

    print("Loading both hemisphere surfaces + DK atlases...")
    fsaverage_meshes = load_fsaverage(FS_MESH_KEY)
    hemi_meshes, hemi_roi_data, hemi_name_to_id = {}, {}, {}
    for hemi in HEMIS:
        hemi_tf = HEMI_TF_MAP[hemi]
        hemi_meshes[hemi] = fsaverage_meshes["pial"].parts[hemi]
        roi_data, name_to_id = load_dk_surface_and_lut(hemi_tf)
        hemi_roi_data[hemi] = roi_data
        hemi_name_to_id[hemi] = name_to_id

    for _, row in cases.iterrows():
        subject_id, case_type = row["subject_id"], row["case_type"]
        print(f"[{case_type}] subject {subject_id}: running MC Dropout ({N_MC_SAMPLES} samples)...")

        try:
            conf_df = compute_subject_confidence(subject_id, model, A_norm, region_labels, X, rid_to_idx)
        except IndexError as e:
            print(f"    SKIPPED: {e}")
            continue

        top_uncertain = conf_df.sort_values("uncertainty_std", ascending=False).iloc[0]
        print(f"    highest-uncertainty region: {top_uncertain['region_label']} "
              f"(std={top_uncertain['uncertainty_std']:.4f}, "
              f"mean_pred={top_uncertain['mean_predicted_suvr']:.3f})")

        baseline_csv = BASELINE_FRAME_TEMPLATE.format(case_type=case_type, subject_id=subject_id)
        baseline_df = pd.read_csv(baseline_csv)
        tau_vmin, tau_vmax = baseline_df["value"].min(), baseline_df["value"].max()

        hemi_tau_values, hemi_uncertainty_values = {}, {}
        for hemi in HEMIS:
            hemi_prefix = HEMI_PREFIX_MAP[hemi]
            tau_df = conf_df.rename(columns={"mean_predicted_suvr": "value"})
            hemi_tau_values[hemi] = region_col_to_vertex_map(
                tau_df, "value", hemi_roi_data[hemi], hemi_name_to_id[hemi], hemi_prefix)
            hemi_uncertainty_values[hemi] = region_col_to_vertex_map(
                conf_df, "uncertainty_normalized", hemi_roi_data[hemi], hemi_name_to_id[hemi], hemi_prefix)

        fig = build_toggle_figure(
            hemi_meshes, hemi_tau_values, hemi_uncertainty_values, tau_vmin, tau_vmax,
            title=f"Subject {subject_id} ({case_type}) -- toggle predicted tau vs. prediction uncertainty",
        )

        out_path = f"{OUT_DIR}/{case_type}_{subject_id}_confidence_overlay_v1.html"
        fig.write_html(out_path)
        conf_df.to_csv(f"{OUT_DIR}/{case_type}_{subject_id}_confidence_v1.csv", index=False)
        print(f"    saved -> {out_path}")

    print("\nDone. Open the .html files -- click \'Prediction Uncertainty\' to toggle")
    print("from the tau heatmap to the MC-Dropout uncertainty heatmap for each subject.")


if __name__ == "__main__":
    main()
