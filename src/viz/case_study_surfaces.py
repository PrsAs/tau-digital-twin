"""
scripts/stage6_render_case_study_surfaces_v7.py

FIX from v3: load_fsaverage_data() returns a SurfaceImage object.
Its per-hemisphere array lives at `.data.parts[hemi]`, NOT `.parts[hemi]`
directly -- that shortcut only applies to mesh objects (e.g.
load_fsaverage(...)['pial'].parts[hemi], used correctly below), not to
data objects returned by load_fsaverage_data().

FIX from v2: nilearn's fetch_surf_fsaverage() does NOT ship
lh.aparc.annot / rh.aparc.annot -- that raw FreeSurfer file simply
isn't part of nilearn's bundled fsaverage download, which is why v2
failed with FileNotFoundError even after fetch_surf_fsaverage() ran
successfully.

FIX: use TemplateFlow's Desikan-Killiany (Desikan2006) surface atlas
instead, which nilearn's own docs recommend for exactly this use case
(surface-based DK parcellation, no local FreeSurfer install needed).
TemplateFlow auto-downloads the .label.gii parcellation + matching
.tsv lookup table on first run -- no manual annot file placement.

Requires: pip install templateflow
"""

import os
import numpy as np
import pandas as pd
import templateflow.api as tflow
from nilearn.surface import load_surf_data
from nilearn.datasets import load_fsaverage_data, load_fsaverage
from nilearn.plotting import plot_surf_stat_map

MANIFEST_PATH = "results/figures/case_studies/case_study_frame_manifest_v1.csv"
LABELS_PATH = "data/enigma_hcp_dk_labels.csv"
OUT_DIR = "results/figures/case_studies/renders"
HEMI = "left"
HEMI_TF = "L"  # TemplateFlow hemi code
HEMI_PREFIX = "L_"  # prefix used in our region_label column
DENSITY = "10k"  # matches fsaverage5, good balance of speed/resolution
FS_DENSITY_MAP = {"3k": "fsaverage4", "10k": "fsaverage5", "41k": "fsaverage6", "164k": "fsaverage"}
FRAME_ORDER = ["baseline", "predicted", "followup"]


def load_dk_region_names(labels_path: str) -> list:
    with open(labels_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def fetch_dk_surface_atlas(hemi_tf: str, density: str):
    label_gii = tflow.get(
        "fsaverage", atlas="Desikan2006", density=density,
        hemi=hemi_tf, extension="label.gii",
    )
    roi_data = load_surf_data(label_gii)

    lut_path = tflow.get(
        "fsaverage", atlas="Desikan2006", suffix="dseg", extension="tsv",
    )
    lut = pd.read_csv(lut_path, sep="\t")
    print("LUT columns:", lut.columns.tolist())
    print(lut.head(10))
    return roi_data, lut


def build_name_to_id_from_lut(lut: pd.DataFrame, hemi_tf: str) -> dict:
    # CRITICAL FIX from v6: the integer values baked into the .label.gii
    # ROI array are NOT the LUT's 'index' column (that is a huge encoded
    # RGBA-derived integer, unrelated to region identity) and NOT
    # 'fs_index' (1000-1035, FreeSurfer's own convention) either.
    # Empirically, the label.gii encodes each vertex's region as the
    # *row position* (0, 1, 2, ...) within the hemisphere-filtered LUT,
    # in the LUT's on-disk row order (0=unknown, 1=bankssts, 2=
    # caudalanteriorcingulate, ...). So we must filter to this hemi and
    # use positional index via reset_index(), not any LUT column value.
    lut_hemi = lut[lut["hemi"] == hemi_tf].reset_index(drop=True)
    name_to_id = {}
    for pos, row in lut_hemi.iterrows():
        bare = str(row["name"]).lower().strip()
        name_to_id[bare] = pos
    return name_to_id


def region_values_to_vertex_map(region_value_df: pd.DataFrame, roi_data: np.ndarray,
                                 name_to_id: dict, hemi_prefix: str) -> np.ndarray:
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
        print(f"    WARNING: {len(unmatched)} regions unmatched: {unmatched}")
    return vertex_values


def render_one_frame(mesh, bg_map, vertex_values, vmin, vmax, out_path, title):
    # FIX from v4: `darkness` kwarg was removed/deprecated in the currently
    # installed nilearn version -- current nilearn's plot_surf_stat_map no
    # longer accepts it directly (it is now handled internally per backend).
    # Also force engine="matplotlib" explicitly since nilearn's default
    # backend can vary by version/install, and matplotlib is what gives us
    # a static PNG via fig.savefig() -- the plotly backend returns a
    # different figure object that does not have .savefig().
    fig = plot_surf_stat_map(
        mesh, vertex_values, hemi=HEMI, view="lateral", colorbar=True,
        vmin=vmin, vmax=vmax, cmap="hot", bg_map=bg_map, bg_on_data=True,
        title=title, engine="matplotlib",
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"    saved -> {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    region_names = load_dk_region_names(LABELS_PATH)
    print(f"Loaded {len(region_names)} DK region names (expect 68).")

    roi_data, lut = fetch_dk_surface_atlas(HEMI_TF, DENSITY)
    name_to_id = build_name_to_id_from_lut(lut, HEMI_TF)
    print(f"Built {len(name_to_id)}-entry name->id lookup from TemplateFlow LUT.")

    fs_mesh_key = FS_DENSITY_MAP[DENSITY]
    fsaverage_meshes = load_fsaverage(fs_mesh_key)
    mesh = fsaverage_meshes["pial"].parts[HEMI]
    sulcal = load_fsaverage_data(mesh=fs_mesh_key, data_type="sulcal").data.parts[HEMI]

    for _, row in manifest.iterrows():
        subject_id, case_type = row["subject_id"], row["case_type"]
        vmin, vmax = row["vmin"], row["vmax"]
        print(f"[{case_type}] subject {subject_id} (vmin={vmin:.3f}, vmax={vmax:.3f})")

        for frame_name in FRAME_ORDER:
            region_value_df = pd.read_csv(row[f"{frame_name}_path"])
            vertex_values = region_values_to_vertex_map(
                region_value_df, roi_data, name_to_id, HEMI_PREFIX
            )
            n_valid = np.sum(~np.isnan(vertex_values))
            print(f"    {frame_name}: {n_valid}/{vertex_values.shape[0]} vertices assigned a value")
            if n_valid == 0:
                raise RuntimeError(
                    f"All-NaN vertex map for subject={subject_id}, frame={frame_name}. "
                    "No DK region names matched the TemplateFlow LUT -- check the "
                    "'unmatched' warning above and confirm HEMI_PREFIX/bare-name "
                    "stripping actually matches the LUT's fs_name convention "
                    "(e.g. 'ctx-lh-bankssts' vs your region_label 'L_bankssts')."
                )
            out_path = f"{OUT_DIR}/{case_type}_{subject_id}_{frame_name}_v1.png"
            render_one_frame(mesh, sulcal, vertex_values, vmin, vmax, out_path,
                              title=f"Subject {subject_id} -- {frame_name}")

    print("\nDone. Six PNGs on shared per-subject color scales saved to", OUT_DIR)


if __name__ == "__main__":
    main()
