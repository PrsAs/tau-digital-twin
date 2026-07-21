"""
scripts/stage6_static_renderer_v1.py

First Stage 6 build step: a static single-subject glass-brain render of
baseline tau burden, using the region_label -> vertex_indices lookup
built in stage6_build_region_vertex_lookup_v1.py. This validates the
full pipeline (lookup -> per-vertex color mapping -> 3D render) before
adding animation and attribution-glow-line complexity on top.

Uses mne.viz.Brain since it already operates on the same fsaverage
subject/label set used to build the lookup -- no need for a separate
pyvista mesh-loading step.

OUTPUT: saves a static screenshot PNG (lateral + medial views) for one
chosen subject's baseline tau map. Once this looks right, the next
scripts add: (1) the predicted/follow-up animation frames, (2)
attribution glow-lines, (3) the confidence overlay.
"""

import pickle
import numpy as np
import pandas as pd
import mne

REGION_VERTEX_LOOKUP_PATH = "results/figures/stage6_region_vertex_lookup_v1.pkl"
PER_SUBJECT_PREDICTIONS_PATH = "results/figures/stage4_per_subject_predictions_v1.csv"
SUBJECT_ID = None  # set to a specific RID string, or leave None to auto-pick the first available
OUT_SCREENSHOT_PATH = "results/figures/stage6_static_baseline_tau_v2.png"


def build_vertex_value_lists(region_values: dict, region_vertex_lookup: dict):
    """
    Maps a dict of {region_label: value} onto (vertices, values) pairs
    for each hemisphere -- NO NaN-filled full-length array. MNE's
    Brain.add_data() has a known issue where NaN values break its
    internal fmin/fmid/fmax monotonicity assertion, even when explicit
    clim is passed. Passing only the actual matched vertices (via the
    `vertices=` argument to add_data) avoids this entirely.
    """
    verts_lh, vals_lh = [], []
    verts_rh, vals_rh = [], []

    for region_label, value in region_values.items():
        if region_label not in region_vertex_lookup:
            continue
        entry = region_vertex_lookup[region_label]
        verts = entry["vertices"]
        if entry["hemi"] == "lh":
            verts_lh.extend(verts)
            vals_lh.extend([value] * len(verts))
        else:
            verts_rh.extend(verts)
            vals_rh.extend([value] * len(verts))

    return (np.array(verts_lh), np.array(vals_lh)), (np.array(verts_rh), np.array(vals_rh))


def main():
    with open(REGION_VERTEX_LOOKUP_PATH, "rb") as f:
        region_vertex_lookup = pickle.load(f)

    node_df = pd.read_csv(PER_SUBJECT_PREDICTIONS_PATH)

    global SUBJECT_ID
    if SUBJECT_ID is None:
        SUBJECT_ID = node_df["subject_id"].iloc[0]
    print(f"Rendering baseline tau for subject: {SUBJECT_ID}")

    subject_rows = node_df[node_df["subject_id"] == SUBJECT_ID]
    if subject_rows.empty:
        raise ValueError(f"No rows found for subject_id={SUBJECT_ID}")

    region_values = dict(zip(subject_rows["region_label"], subject_rows["baseline_suvr"]))
    print(f"Baseline tau range for this subject: {min(region_values.values()):.3f} - {max(region_values.values()):.3f}")

    subjects_dir = mne.datasets.fetch_fsaverage(verbose=False).parent
    brain = mne.viz.Brain(
        "fsaverage", hemi="both", surf="pial", subjects_dir=subjects_dir,
        background="white", cortex="low_contrast", size=(1200, 900),
    )

    (verts_lh, vals_lh), (verts_rh, vals_rh) = build_vertex_value_lists(region_values, region_vertex_lookup)

    all_vals = np.concatenate([vals_lh, vals_rh])
    fmin, fmax = float(all_vals.min()), float(all_vals.max())
    fmid = (fmin + fmax) / 2
    print(f"Color scale range: {fmin:.3f} to {fmax:.3f}")

    brain.add_data(vals_lh, vertices=verts_lh, hemi="lh", colormap="hot", colorbar=True,
                   clim=dict(kind="value", lims=[fmin, fmid, fmax]))
    brain.add_data(vals_rh, vertices=verts_rh, hemi="rh", colormap="hot", colorbar=False,
                   clim=dict(kind="value", lims=[fmin, fmid, fmax]))

    brain.show_view("lateral")
    brain.save_image(OUT_SCREENSHOT_PATH)
    print(f"\nSaved static render -> {OUT_SCREENSHOT_PATH}")
    print("If this looks correct (tau burden colored per-region on a real cortical "
          "surface, no gaps or misaligned regions), Stage 6 is ready to extend to "
          "animation frames and attribution glow-lines.")


if __name__ == "__main__":
    main()
