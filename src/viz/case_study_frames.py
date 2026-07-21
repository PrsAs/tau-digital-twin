"""
scripts/stage6_render_case_study_frames_v2.py

Extends the validated static single-frame renderer (stage6_static_baseline_tau_v2)
to the two Stage 6 case-study subjects chosen from Stage 4 results:

    subject 4168  (success case, low severity)
    subject 6952  (ambiguous case, high severity)

For each subject, produces THREE region-value frames on a SHARED color
scale so baseline -> predicted -> actual-followup can be compared
visually without the colorbar shifting between panels:

    {subject}_baseline_v1.csv    -- baseline_suvr per region
    {subject}_predicted_v1.csv   -- y_pred_suvr per region (GNN forecast)
    {subject}_followup_v1.csv    -- y_true_suvr per region (actual scan)

These feed directly into the same surface-mesh coloring step used for
stage6_static_baseline_tau_v2.jpg -- just swap the per-region value
column and reuse the same vmin/vmax so panels are visually comparable.
"""

import numpy as np
import pandas as pd

PREDICTIONS_PATH = "results/figures/stage4_per_subject_predictions_v1.csv"
CASE_STUDY_PATH = "results/figures/stage6_case_study_subjects_v1.csv"
OUT_DIR = "results/figures/case_studies"

FRAME_COLUMNS = {
    "baseline": "baseline_suvr",
    "predicted": "y_pred_suvr",
    "followup": "y_true_suvr",
}


def build_subject_frames(df: pd.DataFrame, subject_id, case_type: str):
    sub = df[df["subject_id"] == subject_id].copy()
    if sub.empty:
        raise ValueError(f"No rows found for subject_id={subject_id}")

    vmin = sub[list(FRAME_COLUMNS.values())].min().min()
    vmax = sub[list(FRAME_COLUMNS.values())].max().max()

    frame_paths = {}
    for frame_name, col in FRAME_COLUMNS.items():
        out = sub[["region_label", col]].rename(columns={col: "value"})
        path = f"{OUT_DIR}/{case_type}_{subject_id}_{frame_name}_v1.csv"
        out.to_csv(path, index=False)
        frame_paths[frame_name] = path

    return frame_paths, (vmin, vmax)


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(PREDICTIONS_PATH)
    cases = pd.read_csv(CASE_STUDY_PATH)

    manifest_rows = []
    for _, row in cases.iterrows():
        subject_id = row["subject_id"]
        case_type = row["case_type"]

        frame_paths, (vmin, vmax) = build_subject_frames(df, subject_id, case_type)

        print(f"[{case_type}] subject {subject_id}: shared color scale "
              f"vmin={vmin:.3f}, vmax={vmax:.3f}")
        for frame_name, path in frame_paths.items():
            print(f"    {frame_name:>9} -> {path}")

        manifest_rows.append({
            "case_type": case_type,
            "subject_id": subject_id,
            "vmin": vmin,
            "vmax": vmax,
            **{f"{k}_path": v for k, v in frame_paths.items()},
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = f"{OUT_DIR}/case_study_frame_manifest_v1.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest saved -> {manifest_path}")
    print("\nNEXT: feed each *_v1.csv through the same surface-mesh coloring")
    print("step used for stage6_static_baseline_tau_v2.jpg, using the")
    print("shared (vmin, vmax) per subject so baseline/predicted/followup")
    print("panels share one colorbar for direct visual comparison.")


if __name__ == "__main__":
    main()
