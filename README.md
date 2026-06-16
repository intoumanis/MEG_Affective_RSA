# MEG Affective RSA

Analysis code for:

**Ntoumanis, I. & Papadelis, C. (2026). Distinct neural patterns of valence and arousal in the developing brain: A representational similarity analysis of MEG.** *(under review, NeuroImage)*

This repository contains the full analysis pipeline, from MEG source signal extraction to representational network analysis, accompanying the manuscript above. This study is a follow-up of our prior work:

> Ntoumanis, I., Townsend, M., Cooper, C. M., & Papadelis, C. (2026). Rapid engagement of salience and prefrontal systems during emotional processing in children: An MEG study. *NeuroImage*, 328, 121806. https://doi.org/10.1016/j.neuroimage.2026.121806

---

## Overview

We applied partial representational similarity analysis (RSA) to source-localized MEG data from 57 typically developing children and adolescents (ages 5–18) viewing 420 affective pictures from the Nencki Affective Picture System (NAPS). The pipeline dissociates the unique contributions of valence and arousal to neural representational geometry, across ROIs and time windows from 50 to 1000 ms post-stimulus.

---

## Repository Structure

```
MEG_Affective_RSA/
│
├── 1__Extract_source_signal.m          # MATLAB: extract & time-window source data from Brainstorm
├── 2__Group_average_sources.py         # Build group-averaged source data across participants
├── 3__Model_RDMs.py                    # Construct valence, arousal, and nuisance model RDMs
├── 4__RSA_early_effects.py             # Partial RSA — hypothesis-driven analysis (50–100 ms)
├── 5__RSA_late_effects.py              # Partial RSA — exploratory analysis (100–1000 ms)
├── 6__Representational_networks.py     # Beta-product representational network analysis
└── 7__Jaccard_similarity.py            # Signed Jaccard comparison of valence vs arousal networks
```

Scripts are numbered in execution order.

---

## Pipeline

### Step 1 — `1__Extract_source_signal.m` (MATLAB)
Extracts trial-level source estimates from the Brainstorm database and averages them within 18 time windows (50–1000 ms). Output per participant: a matrix of size 420 trials × 13,267 grid points × 18 time windows. Requires access to a Brainstorm project with dSPM source estimates already computed.

### Step 2 — `2__Group_average_sources.py`
Remaps presentation-order trial indices to canonical stimulus indices using each participant's E-Prime log file and a ratings spreadsheet (`Ratings.xlsx`). Accumulates and averages source data across participants, yielding a group-level array (420 stimuli × 13,267 grid points × 18 time windows).

### Step 3 — `3__Model_RDMs.py`
Constructs pairwise representational dissimilarity matrices (RDMs) from normative stimulus ratings (valence, arousal, and four low-level visual features: luminance, contrast, complexity, entropy) using absolute differences. Also runs the Lind–Mehlum U-shape test on the valence–arousal relationship and reports pairwise Mantel correlations among all model RDMs.

### Step 4 — `4__RSA_early_effects.py`
Hypothesis-driven partial RSA restricted to the 50–100 ms time window and seven bilateral ROIs (IFG, OFC, MFG, Insula, MCC, PCC, STG), selected based on Ntoumanis et al. (2026). At each ROI, the neural RDM is regressed on both the valence and arousal model RDMs simultaneously. Statistical significance is assessed via 10,000 stimulus-label permutations; p-values are FDR-corrected across ROIs (Benjamini–Hochberg, q < .05).

### Step 5 — `5__RSA_late_effects.py`
Exploratory partial RSA over 17 time windows (100–1000 ms) and 15 bilateral ROIs spanning temporal, ventral visual, medial/cingulate, subcortical, and frontal regions. FDR correction is applied across all ROI × time window combinations (255 tests).

### Step 6 — `6__Representational_networks.py`
Computes the representational network for each affective dimension by testing, at each pair of ROIs and time window, whether the two ROIs encode the same dimension. Significance is assessed via permutation testing with FDR correction. Outputs time-course plots, glass brain visualizations, and summary tables.

### Step 7 — `7__Jaccard_similarity.py`
Quantifies the overlap between the valence and arousal representational networks across time using a signed Jaccard similarity index. Produces time-series plots and edge-count comparisons.

---

## Requirements

### MATLAB
- MATLAB (tested on R2023b)
- [Brainstorm](https://neuroimage.usc.edu/brainstorm/) with preprocessed MEG data and dSPM source estimates

### Python
- Python 3.9+

Install all Python dependencies with:

```bash
pip install numpy pandas scipy matplotlib seaborn nibabel nilearn h5py openpyxl python-docx statannotations
```

---

## Configuration

Set `BASE_DIR` (Python scripts) or `db_path` / `output_dir` (MATLAB script) to match your local directory structure.

The expected folder structure within `BASE_DIR` is:

```
BASE_DIR/
├── Source data/          # Output of Step 1; input to Steps 2, 4, 5, 6
├── Results/
│   ├── Model RDMs/       # Output of Step 3; input to Steps 4, 5, 6
│   │   └── Ratings.xlsx  # Stimulus ratings (valence, arousal, visual features)
│   ├── Early effects/    # Output of Step 4
│   ├── Late effects/     # Output of Step 5
│   └── Connectivity/     # Output of Steps 6 and 7
└── Code/
    └── AAL3v1.nii        # AAL3 atlas (Rolls et al., 2020)
```

The AAL3 atlas can be downloaded from https://www.gin.cnrs.fr/en/tools/aal/.

---

## Citation

If you use this code, please cite:

Ntoumanis, I. & Papadelis, C. (2026). Distinct neural patterns of valence and arousal in the developing brain: A representational similarity analysis of MEG. *(under review, NeuroImage)*

---

## License

MIT License. See `LICENSE` for details.
