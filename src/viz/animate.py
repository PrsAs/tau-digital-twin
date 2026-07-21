"""
scripts/stage6_animate_case_study_v1.py

Builds the SCRUBBABLE baseline -> predicted -> follow-up animation
requested by the thesis plan ("Per-subject animation: baseline scan,
model's predicted intermediate/final state, real follow-up overlay,
scrubbable in time... Output as interactive HTML render").

Uses Plotly's Mesh3d + frames/slider mechanism instead of matplotlib,
since matplotlib PNGs (used for the static Stage 6 panels) cannot be
scrubbed -- Plotly HTML is the right tool for this specific deliverable.

Reuses the SAME region-name matching + shared (vmin, vmax) logic
validated in stage6_render_case_study_surfaces_v7.py, just swapping
the matplotlib backend for a Plotly Mesh3d figure so the surface is
draggable/rotatable AND has a time slider across the 3 frames.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import templateflow.api as tflow
import nibabel as nib
from nilearn.datasets import load_fsaverage

MANIFEST_PATH = "results/figures/case_studies/case_study_frame_manifest_v1.csv"
OUT_DIR = "results/figures/case_studies/animations"
HEMI = "left"
HEMI_TF = "L"
HEMI_PREFIX = "L_"
DENSITY = "10k"
FS_MESH_KEY = "fsaverage5"
FRAME_ORDER = ["baseline", "predicted", "followup"]
FRAME_TITLES = {"baseline": "Baseline", "predicted": "GNN Predicted", "followup": "Actual Follow-up"}


def load_dk_surface_and_lut():
    label_gii = tflow.get("fsaverage", atlas="Desikan2006", density=DENSITY, hemi=HEMI_TF, extension="label.gii")
    gii = nib.load(str(label_gii))
    roi_data = gii.darrays[0].data

    lut_path = tflow.get("fsaverage", atlas="Desikan2006", suffix="dseg", extension="tsv")
    lut = pd.read_csv(lut_path, sep="\t")
    lut_hemi = lut[lut["hemi"] == HEMI_TF].reset_index(drop=True)
    name_to_id = {str(row["name"]).lower().strip(): pos for pos, row in lut_hemi.iterrows()}
    return roi_data, name_to_id


def region_values_to_vertex_map(region_value_df, roi_data, name_to_id, hemi_prefix):
    vertex_values = np.full(roi_data.shape[0], np.nan)
    unmatched = []
    for _, row in region_value_df.iterrows():
        region_label = row["region_label"]
        if not region_label.startswith(hemi_prefix):
            continue
        bare_name = region_label[len(hemi_prefix):].lower()
        region_id = name_to_id.get(bare_name)
        if region_id is None:
            unmatched.append(region_label)
            continue
        vertex_values[roi_data == region_id] = row["value"]
    if unmatched:
        print(f"    WARNING: {len(unmatched)} unmatched regions: {unmatched}")
    return vertex_values


def build_animated_figure(mesh, roi_data, name_to_id, manifest_row):
    coords, faces = mesh.coordinates, mesh.faces
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    vmin, vmax = manifest_row["vmin"], manifest_row["vmax"]
    frame_vertex_values = {}
    for frame_name in FRAME_ORDER:
        df = pd.read_csv(manifest_row[f"{frame_name}_path"])
        frame_vertex_values[frame_name] = region_values_to_vertex_map(df, roi_data, name_to_id, HEMI_PREFIX)

    base_frame = FRAME_ORDER[0]
    mesh3d = go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        intensity=frame_vertex_values[base_frame],
        colorscale="Hot", cmin=vmin, cmax=vmax,
        colorbar=dict(title="Tau SUVR"),
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.1),
        name=FRAME_TITLES[base_frame],
    )

    frames = [
        go.Frame(
            data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                             intensity=frame_vertex_values[frame_name],
                             colorscale="Hot", cmin=vmin, cmax=vmax)],
            name=frame_name,
        )
        for frame_name in FRAME_ORDER
    ]

    fig = go.Figure(data=[mesh3d], frames=frames)

    slider_steps = [
        dict(
            method="animate",
            args=[[frame_name], dict(mode="immediate",
                                      frame=dict(duration=0, redraw=True),
                                      transition=dict(duration=300))],
            label=FRAME_TITLES[frame_name],
        )
        for frame_name in FRAME_ORDER
    ]

    fig.update_layout(
        title=f"Subject {manifest_row['subject_id']} ({manifest_row['case_type']} case) -- "
              f"scrub baseline -> predicted -> follow-up",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data",
        ),
        sliders=[dict(
            active=0, currentvalue=dict(prefix="Frame: "),
            steps=slider_steps, x=0.15, len=0.7, y=0,
        )],
        updatemenus=[dict(
            type="buttons", showactive=False, y=0, x=0.02,
            buttons=[dict(label="Play", method="animate",
                          args=[None, dict(frame=dict(duration=800, redraw=True),
                                           fromcurrent=True, transition=dict(duration=300))])],
        )],
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    roi_data, name_to_id = load_dk_surface_and_lut()

    fsaverage_meshes = load_fsaverage(FS_MESH_KEY)
    mesh = fsaverage_meshes["pial"].parts[HEMI]

    for _, row in manifest.iterrows():
        subject_id, case_type = row["subject_id"], row["case_type"]
        print(f"[{case_type}] subject {subject_id}: building animation...")

        fig = build_animated_figure(mesh, roi_data, name_to_id, row)

        out_path = f"{OUT_DIR}/{case_type}_{subject_id}_animation_v1.html"
        fig.write_html(out_path, auto_play=False)
        print(f"    saved -> {out_path}")

    print("\nDone. Open the .html files in a browser -- drag the slider or hit")
    print("Play to scrub baseline -> predicted -> follow-up. Mesh is also")
    print("draggable/rotatable in 3D, unlike the static matplotlib panels.")


if __name__ == "__main__":
    main()
