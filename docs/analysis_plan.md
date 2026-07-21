# Individualized Tau Progression Digital Twin — Analysis Plan (v2)

## Research Question
Can an interpretable spatiotemporal GNN predict an individual patient's
*specific* future pattern of tau spread (not just aggregate regional
accuracy), validate against their real follow-up scan, and be communicated
through an interactive per-subject 3D visualization?

## Locked Design Decisions (updated)
- **Connectome**: group-level normative structural connectome (e.g.
  HCP-derived, Desikan-Killiany or Schaefer-200 parcellation) -- NOT
  per-subject DTI. ADNI diffusion coverage is smaller/later than tau-PET
  coverage and would shrink the eligible longitudinal cohort further.
- **Tau input**: UC Berkeley AV-1451 (flortaucipir) tau-PET, **PVC (partial
  volume corrected), 6mm-resolution** regional SUVR. PVC preferred over
  non-PVC specifically for AV-1451 because RBV-style PVC has been shown to
  increase longitudinal effect sizes for this tracer (Costoya-Sanchez et
  al. 2024) -- important since the model must detect subtle within-subject
  change, not just cross-sectional differences. (This is tracer-specific;
  evidence for other tracers e.g. GTP1 doesn't show the same benefit.)
- **Metadata/labels source**: ADNIMERGE.csv (single merged table) instead
  of separate MRI/diagnosis exports -- has_baseline_mri is now derived from
  presence of FreeSurfer volumetric columns at the baseline visit rather
  than a dedicated flag.
- **Time handling**: inter-visit interval stored explicitly per subject,
  passed to the model as a continuous covariate (Stage 2).

## Raw Data Files Needed (LONI/IDA download)
1. `UC Berkeley - Tau PET PVC 6mm Res analysis [ADNI2,3,4]` -> saved as
   `data/raw/UCBERKELEYAV1451_PVC.csv`
2. `ADNIMERGE.csv` (Study Data -> Study Info package) -> saved as
   `data/raw/ADNIMERGE.csv`

## Repo Layout

```
tau-digital-twin/
├── data/
│   ├── raw/
│   │   ├── UCBERKELEYAV1451_PVC.csv   # AV-1451 tau-PET, PVC, regional SUVR
│   │   └── ADNIMERGE.csv              # merged demographics/dx/FreeSurfer flags
│   └── processed/
│       └── cohort_manifest.csv        # Stage 1 output
├── src/
│   ├── data/
│   │   ├── build_cohort.py            # Stage 1 (this file)
│   │   └── loaders.py                 # reuse/extend from tau-gnn-progression
│   ├── model/
│   │   ├── gnn.py
│   │   └── train.py
│   ├── interpret/
│   │   ├── attribution.py
│   │   └── confidence.py
│   ├── validate/
│   │   └── followup_eval.py
│   └── viz/
│       ├── glass_brain.py
│       ├── animate.py
│       └── attribution_overlay.py
├── results/
│   ├── figures/
│   └── subject_renders/
├── docs/
│   └── analysis_plan.md               # this file
└── requirements.txt
```

## Stage 1 — Cohort Construction (current stage)
- Eligibility: >=2 AV-1451 tau-PET visits (PVC) + evidence of baseline MRI
  processing (via ADNIMERGE FreeSurfer columns).
- Report cohort size, visit-count distribution, inter-visit interval stats,
  and baseline diagnosis breakdown *before* proceeding to Stage 2 -- this
  determines whether the individualized design is statistically viable or
  whether the Stage 5 fallback (interpretability-first) should be adopted
  early.

## Stages 2-7
(unchanged from v1 -- individualized GNN head, interpretability layer,
follow-up validation, interpretability validation, glass-brain viz,
write-up. See prior plan version for full detail.)
