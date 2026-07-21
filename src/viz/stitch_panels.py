"""
scripts/stage6_stitch_case_study_panels_v1.py

Stitches each subject's 3 separately-rendered surface PNGs
(baseline / predicted / followup) into a single 1x3 comparison panel
with ONE shared colorbar, for direct visual comparison in the Stage 7
write-up.

Reads the same manifest used by the renderer so file paths and the
per-subject (vmin, vmax) stay perfectly consistent -- no re-deriving
scale info, no risk of the panel using a different range than the
individual PNGs were rendered with.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd

MANIFEST_PATH = "results/figures/case_studies/case_study_frame_manifest_v1.csv"
RENDER_DIR = "results/figures/case_studies/renders"
OUT_DIR = "results/figures/case_studies/panels"
FRAME_ORDER = ["baseline", "predicted", "followup"]
FRAME_TITLES = {"baseline": "Baseline", "predicted": "GNN Predicted", "followup": "Actual Follow-up"}


def stitch_one_subject(case_type: str, subject_id, vmin: float, vmax: float):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2),
                              gridspec_kw={"width_ratios": [1, 1, 1, 0.06]})

    for ax, frame_name in zip(axes[:3], FRAME_ORDER):
        img_path = f"{RENDER_DIR}/{case_type}_{subject_id}_{frame_name}_v1.png"
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(FRAME_TITLES[frame_name], fontsize=13)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap="hot")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=axes[3])
    cbar.set_label("Tau SUVR", fontsize=11)

    subject_label = "Success case" if case_type == "success" else "Ambiguous case"
    fig.suptitle(f"Subject {subject_id} -- {subject_label}", fontsize=15, y=1.02)
    fig.tight_layout()

    out_path = f"{OUT_DIR}/{case_type}_{subject_id}_panel_v1.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved -> {out_path}")
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)

    panel_paths = []
    for _, row in manifest.iterrows():
        subject_id, case_type = row["subject_id"], row["case_type"]
        vmin, vmax = row["vmin"], row["vmax"]
        print(f"[{case_type}] subject {subject_id}")
        panel_paths.append(stitch_one_subject(case_type, subject_id, vmin, vmax))

    print("\nDone. Panels saved:")
    for p in panel_paths:
        print(f"  {p}")
    print("\nNEXT: these two panels are the qualitative centerpiece for Stage 7 --")
    print("drop them directly into the write-up alongside the Stage 4/5 quantitative")
    print("tables. Optional follow-on: feed the same 3 frames per subject into")
    print("animate.py for a scrubbable baseline->predicted->followup HTML version.")


if __name__ == "__main__":
    main()
