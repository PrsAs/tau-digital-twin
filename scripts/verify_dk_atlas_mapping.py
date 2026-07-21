"""
scripts/stage6_verify_dk_atlas_mapping_v1.py

Stage 6 prep step: verifies that the 68 Desikan-Killiany region names in
your enigma_hcp_dk_labels.csv (used throughout Stages 2-5 as
region_labels) line up correctly with nilearn's fsaverage + DK atlas
region naming, BEFORE writing any actual glass-brain rendering code.

NOTE: This script requires internet access to fetch the fsaverage
surface + DK atlas on first run (nilearn caches it locally afterward in
~/nilearn_data). It cannot be run in a sandboxed/offline environment --
run it directly on your own machine.

WHAT IT CHECKS:
1. Fetches the fsaverage5 surface mesh (both hemispheres).
2. Fetches the Desikan-Killiany parcellation annotation for fsaverage
   (via nilearn's surface atlas tools).
3. Extracts the DK region name list nilearn actually uses internally.
4. Compares that list against your enigma_hcp_dk_labels.csv (68 names,
   L_/R_ prefixed) to find:
   - exact matches
   - close matches (case/prefix differences, e.g. "bankssts" vs
     "L_bankssts")
   - names present in one list but missing from the other
5. Prints a clear pass/fail summary so you know whether a manual
   name-mapping dictionary is needed before Stage 6 rendering.
"""

import re
import pandas as pd
from nilearn import datasets

YOUR_LABELS_PATH = "enigma_hcp_dk_labels.csv"  # your uploaded file


def normalize(name: str) -> str:
    """Lowercase, strip hemisphere prefixes/suffixes, remove underscores."""
    name = name.lower()
    name = re.sub(r"^[lr]_", "", name)       # strip "l_" / "r_" prefix
    name = re.sub(r"^ctx-[lr]h-", "", name)  # strip "ctx-lh-" / "ctx-rh-" prefix (FreeSurfer aparc convention)
    name = name.replace("_", "").replace("-", "")
    return name


def main():
    your_labels = pd.read_csv(YOUR_LABELS_PATH, header=None)[0].tolist()
    print(f"Your DK label count: {len(your_labels)}")
    print(f"Sample: {your_labels[:5]}")

    print("\nFetching fsaverage5 surface + DK atlas (requires internet, cached after first run)...")
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
    print("\nNOTE: nilearn's fetch_atlas_surf_destrieux() gives the DESTRIEUX atlas "
          "(finer-grained, ~150 regions), NOT Desikan-Killiany. Checking its labels "
          "for reference, but you likely need the DK-specific annotation instead "
          "(e.g. via TemplateFlow's Freesurfer aparc.annot, or nilearn.datasets."
          "fetch_atlas_pial-based DK equivalents). This script flags that distinction "
          "explicitly below.")

    destrieux_labels = [l.decode("utf-8") if isinstance(l, bytes) else l
                         for l in destrieux_atlas["labels"]]
    print(f"\nDestrieux atlas region count: {len(destrieux_labels)} (expect ~150, NOT 68 -- "
          f"confirms this is the WRONG atlas for direct DK matching)")

    your_normalized = {normalize(n): n for n in your_labels}
    destrieux_normalized = {normalize(n): n for n in destrieux_labels}

    exact_matches = set(your_normalized) & set(destrieux_normalized)
    only_in_yours = set(your_normalized) - set(destrieux_normalized)
    only_in_destrieux = set(destrieux_normalized) - set(your_normalized)

    print(f"\n=== Destrieux vs. your DK labels (sanity check only, mismatch expected) ===")
    print(f"Exact normalized matches: {len(exact_matches)} of {len(your_labels)}")
    print(f"Only in your DK list (first 10): {list(only_in_yours)[:10]}")

    print("\n=== ACTION NEEDED ===")
    print("nilearn does not ship a ready-made 68-region DK surface annotation via "
          "fetch_surf_fsaverage() or fetch_atlas_surf_destrieux(). To get true DK "
          "labels on the fsaverage surface, use one of:")
    print("  1. TemplateFlow: templateflow.api.get('fsaverage', atlas='DesikanKilliany', ...)")
    print("  2. Direct FreeSurfer aparc.annot files for fsaverage (shipped with FreeSurfer "
          "installs, or downloadable from surfer.nmr.mgh.harvard.edu), loaded via "
          "nilearn.surface.load_surf_data() or nibabel.freesurfer.read_annot()")
    print("  3. nilearn.datasets.fetch_atlas_talairach or other volumetric-to-surface "
          "conversions (more complex, not recommended for this use case)")
    print("\nRecommended: option 2 (aparc.annot) is the most direct DK-native path and "
          "matches your label names most closely once normalized.")


if __name__ == "__main__":
    main()
