# -*- coding: utf-8 -*-
"""
===================================
ROI-based partial RSA — HYPOTHESIS-DRIVEN ANALYSIS ONLY (50-100 ms window)

Based on a priori hypothesis from Ntoumanis et al. (2026) NeuroImage.

Author: Ioannis Ntoumanis
Created: 2026
"""

import numpy as np
import nibabel as nib
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from nilearn.plotting import plot_glass_brain
from nilearn.image import smooth_img  # Add this import at top
import matplotlib.pyplot as plt
import h5py
import os
import pickle
import time

# ===========================================================================
# CONFIGURATION
# ===========================================================================

SOURCE_AVG_FILE = r"Source data/source_avg.npy"
VALID_MASK_FILE = r"Source data/valid_mask.npy"
GRID_LOC_FILE   = r"Source data/grid_loc.npy"
MNI_LOC_FILE    = r"Source data/grid_loc_mni.mat"
TIME_WIN_FILE   = r"Source data/time_windows.npy"
MODEL_RDM_DIR   = r"Results/Model RDMs"
AAL_ATLAS_FILE  = r"Code/AAL3v1.nii"
OUTPUT_DIR      = r"Results/Early effects"

MODEL_NAMES  = ["rdm_valence", "rdm_arousal"]
MODEL_LABELS = ["Valence", "Arousal"]

N_PERM       = 10000
RANDOM_SEED  = 25
FDR_Q        = 0.05
MIN_SUBJECTS = 15
ASSIGNMENT_RADIUS = 10  # mm

# Time window (50-100 ms)
CONFIRM_TIME_IDX = 0

ROIS = {
    'Insula':   ['Insula_L', 'Insula_R'],
    'OFC':      ['Frontal_Med_Orb_L', 'Frontal_Med_Orb_R'],
    'PCC':      ['Cingulate_Post_L', 'Cingulate_Post_R'],
    'IFG':      ['Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R'],
    'MFG':      ['Frontal_Mid_2_L', 'Frontal_Mid_2_R'],
    'STG':      ['Temporal_Sup_L', 'Temporal_Sup_R'],
    'MCC':      ['Cingulate_Mid_L', 'Cingulate_Mid_R']
}

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def rank_vector(v):
    return rankdata(v).astype(np.float64)

def fdr_bh(p_values, q=0.05):
    """Benjamini-Hochberg FDR on a flat array of p-values."""
    p_flat = p_values.flatten()
    valid = ~np.isnan(p_flat)
    p_valid = p_flat[valid]
    n = len(p_valid)
    if n == 0:
        return np.zeros_like(p_values, dtype=bool).reshape(p_values.shape), 0.0

    sorted_idx = np.argsort(p_valid)
    sorted_p = p_valid[sorted_idx]
    bh_critical = np.arange(1, n + 1) / n * q
    below = sorted_p <= bh_critical

    if not np.any(below):
        return np.zeros_like(p_values, dtype=bool).reshape(p_values.shape), 0.0

    threshold = sorted_p[np.max(np.where(below)[0])]
    significant = np.zeros_like(p_flat, dtype=bool)
    significant[valid] = p_flat[valid] <= threshold
    return significant.reshape(p_values.shape), threshold

def assign_grid_to_aal(grid_mm, aal_img, aal_labels, aal_indices, roi_dict,
                        radius_mm=7.5):
    """Assign each grid point to an AAL region based on nearest labeled voxel."""
    aal_data = aal_img.get_fdata()
    aal_affine = aal_img.affine
    inv_affine = np.linalg.inv(aal_affine)

    index_to_roi = {}
    for roi_name, label_substrings in roi_dict.items():
        for substr in label_substrings:
            for i, label in enumerate(aal_labels):
                if label == substr or substr in label:
                    idx = int(aal_indices[i])
                    index_to_roi[idx] = roi_name

    print(f"  AAL indices mapped to ROIs: {len(index_to_roi)}")

    roi_assignments = {name: [] for name in roi_dict.keys()}

    for g in range(len(grid_mm)):
        mni_coord = np.array([grid_mm[g, 0], grid_mm[g, 1], grid_mm[g, 2], 1.0])
        vox = inv_affine @ mni_coord
        vi, vj, vk = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))

        if (0 <= vi < aal_data.shape[0] and
            0 <= vj < aal_data.shape[1] and
            0 <= vk < aal_data.shape[2]):
            aal_idx = int(aal_data[vi, vj, vk])
            if aal_idx in index_to_roi:
                roi_assignments[index_to_roi[aal_idx]].append(g)

        if not assigned:
            search_range = int(np.ceil(radius_mm / np.abs(aal_affine[0, 0])))
            found = False
            for di in range(-search_range, search_range + 1):
                if found: break
                for dj in range(-search_range, search_range + 1):
                    if found: break
                    for dk in range(-search_range, search_range + 1):
                        ni, nj, nk = vi + di, vj + dj, vk + dk
                        if (0 <= ni < aal_data.shape[0] and
                            0 <= nj < aal_data.shape[1] and
                            0 <= nk < aal_data.shape[2]):
                            aal_idx = int(aal_data[ni, nj, nk])
                            if aal_idx in index_to_roi:
                                vox_mm = aal_affine @ np.array([ni, nj, nk, 1.0])
                                dist   = np.sqrt(np.sum((grid_mm[g] - vox_mm[:3])**2))
                                if dist <= radius_mm:
                                    roi_assignments[index_to_roi[aal_idx]].append(g)
                                    found = True
                                    break

    for name in roi_assignments:
        roi_assignments[name] = np.array(roi_assignments[name], dtype=int)
        print(f"    {name}: {len(roi_assignments[name])} grid points")

    return roi_assignments

def make_nifti(grid_mm, values, resolution=5):
    """Create a NIfTI volume from scattered MNI points."""
    x_range = np.arange(-80, 85, resolution)
    y_range = np.arange(-115, 80, resolution)
    z_range = np.arange(-65, 90, resolution)

    vol = np.zeros((len(x_range), len(y_range), len(z_range)))

    for i in range(len(grid_mm)):
        if values[i] == 0: continue
        xi = np.argmin(np.abs(x_range - grid_mm[i, 0]))
        yi = np.argmin(np.abs(y_range - grid_mm[i, 1]))
        zi = np.argmin(np.abs(z_range - grid_mm[i, 2]))
        vol[xi, yi, zi] = values[i]

    affine = np.eye(4)
    affine[0, 3] = x_range[0]
    affine[1, 3] = y_range[0]
    affine[2, 3] = z_range[0]
    affine[0, 0] = resolution
    affine[1, 1] = resolution
    affine[2, 2] = resolution

    return nib.Nifti1Image(vol, affine)

def make_nifti_smooth(grid_mm, values, resolution=3, fwhm=2):  # Lower res = smoother
    """Create a NIfTI volume from scattered MNI points + GAUSSIAN SMOOTHING."""
    x_range = np.arange(-80, 85, resolution)
    y_range = np.arange(-115, 80, resolution)
    z_range = np.arange(-65, 90, resolution)

    vol = np.zeros((len(x_range), len(y_range), len(z_range)))

    for i in range(len(grid_mm)):
        if values[i] == 0: continue
        xi = np.argmin(np.abs(x_range - grid_mm[i, 0]))
        yi = np.argmin(np.abs(y_range - grid_mm[i, 1]))
        zi = np.argmin(np.abs(z_range - grid_mm[i, 2]))
        vol[xi, yi, zi] = values[i]

    affine = np.eye(4)
    affine[0, 3] = x_range[0]
    affine[1, 3] = y_range[0]
    affine[2, 3] = z_range[0]
    affine[0, 0] = resolution
    affine[1, 1] = resolution
    affine[2, 2] = resolution

    img = nib.Nifti1Image(vol, affine)
    
    # SMOOTH EDGES (FWHM=6mm → beautiful ROI boundaries)
    img_smooth = smooth_img(img, fwhm=fwhm)
    
    return img_smooth

# ===========================================================================
# MAIN
# ===========================================================================
t_start = time.time()
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("ROI PARTIAL RSA — (50-100 ms only)")
print(f"  Permutations: {N_PERM}")
print("=" * 60)

# ----- 1. Load data -----
print("\nStep 1: Loading data")
source_avg   = np.load(SOURCE_AVG_FILE)
valid_mask   = np.load(VALID_MASK_FILE)
time_windows = np.load(TIME_WIN_FILE)
grid_coords = np.load(GRID_LOC_FILE)

n_stimuli, n_grid, n_time = source_avg.shape
valid_idx = np.where(valid_mask)[0]
n_valid = len(valid_idx)

print(f"  Source: {source_avg.shape}")
print(f"  Valid stimuli: {n_valid}")
print(f"  Hypothesis-driven window: {CONFIRM_TIME_IDX}")

with h5py.File(MNI_LOC_FILE, 'r') as f:
    grid_mni = np.array(f['grid_mni']).T
grid_mm = grid_mni * 1000

n_models = len(MODEL_NAMES)
model_rdms_trimmed = []
for name in MODEL_NAMES:
    rdm = np.load(os.path.join(MODEL_RDM_DIR, f"{name}.npy"))
    rdm_trim = rdm[np.ix_(valid_idx, valid_idx)]
    model_rdms_trimmed.append(rdm_trim)

n_pairs = n_valid * (n_valid - 1) // 2

# ----- Sort stimuli by valence (for RDM visualizations) -----
import pandas as pd
ratings = pd.read_excel("Results/Model RDMs/Ratings.xlsx")
valence_all = ratings['valence'].values
valence_valid = valence_all[valid_idx]
sort_idx = np.argsort(valence_valid)

# ----- 2. Build design matrices (REUSED for all analyses) -----
print("\nStep 2: Building design matrices")
obs_model_vecs = []
for m_idx in range(n_models):
    vec = squareform(model_rdms_trimmed[m_idx], checks=False)
    rv = rank_vector(vec)
    rv -= rv.mean()
    obs_model_vecs.append(rv)

X_obs = np.column_stack(obs_model_vecs + [np.ones(n_pairs)])
Xt_obs = X_obs.T
XtX_inv_obs = np.linalg.inv(Xt_obs @ X_obs)

print(f"  Design matrix: {X_obs.shape}")

print(f"  Building {N_PERM} permuted design matrices...", end=" ", flush=True)
rng = np.random.default_rng(RANDOM_SEED)
perm_orders = np.zeros((N_PERM, n_valid), dtype=np.int64)
for p in range(N_PERM):
    perm_orders[p] = rng.permutation(n_valid)

Xt_stack = np.zeros((N_PERM, n_models + 1, n_pairs), dtype=np.float64)
XtX_inv_stack = np.zeros((N_PERM, n_models + 1, n_models + 1), dtype=np.float64)

for p in range(N_PERM):
    cols = []
    for m_idx in range(n_models):
        rdm_perm = model_rdms_trimmed[m_idx][np.ix_(perm_orders[p], perm_orders[p])]
        vec = rank_vector(squareform(rdm_perm, checks=False))
        vec -= vec.mean()
        cols.append(vec)
    cols.append(np.ones(n_pairs))
    X_perm = np.column_stack(cols)
    Xt_stack[p] = X_perm.T
    XtX_inv_stack[p] = np.linalg.inv(X_perm.T @ X_perm)
print("done")

# ----- 3. Assign grid points to ROIs -----
print("\nStep 3: Assigning grid points to AAL3 ROIs")
aal_img = nib.load(AAL_ATLAS_FILE)

aal_dir = os.path.dirname(AAL_ATLAS_FILE)
labels = []; indices = []
for label_file in ['AAL3v1.nii.txt', 'AAL3v1_1mm.nii.txt', 'ROI_MNI_V7.txt',
                   'AAL3v1.xml', 'AAL3_labels.txt']:
    lf = os.path.join(aal_dir, label_file)
    if os.path.exists(lf):
        print(f"  Found label file: {lf}")
        with open(lf, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        idx = int(parts[0])
                        name = parts[1]
                        indices.append(idx)
                        labels.append(name)
                    except ValueError:
                        continue
        break

roi_assignments = assign_grid_to_aal(grid_mm, aal_img, labels, indices, ROIS,
                                      radius_mm=ASSIGNMENT_RADIUS)

active_rois = {name: idx for name, idx in roi_assignments.items() if len(idx) >= 5}
roi_names = sorted(active_rois.keys())
n_rois = len(roi_names)
print(f"\n  Active ROIs (>= 5 grid points): {n_rois}")

# ----- 4. HYPOTHESIS-DRIVEN ROI PARTIAL RSA (50-100 ms only) -----
print(f"\nStep 4: Hypothesis-driven ROI analysis ({n_rois} ROIs)")
print("=" * 60)

t_confirm = CONFIRM_TIME_IDX
source_valid = source_avg[valid_idx, :, t_confirm]

roi_observed = np.zeros((n_rois, n_models), dtype=np.float64)
roi_pvalues  = np.ones((n_rois, n_models), dtype=np.float64)
perm_betas_confirm = np.zeros((n_rois, N_PERM, n_models), dtype=np.float64)  # For CIs

for r_idx, roi_name in enumerate(roi_names):
    grid_indices = active_rois[roi_name]
    n_roi_points = len(grid_indices)
    print(f"  {roi_name} ({n_roi_points} grid points)")

    patterns = source_valid[:, grid_indices]
    patterns_zm = patterns - patterns.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(patterns_zm ** 2, axis=1))

    if np.any(norms < 1e-12):
        roi_observed[r_idx] = np.nan
        roi_pvalues[r_idx] = np.nan
        continue

    patterns_normed = patterns_zm / norms[:, np.newaxis]
    corr_matrix = patterns_normed @ patterns_normed.T
    np.clip(corr_matrix, -1.0, 1.0, out=corr_matrix)
    neural_rdm = 1.0 - corr_matrix
    neural_vec = squareform(neural_rdm, checks=False)
    neural_ranked = rank_vector(neural_vec)
    neural_ranked -= neural_ranked.mean()

    # Observed betas
    obs_betas = XtX_inv_obs @ (Xt_obs @ neural_ranked)
    roi_observed[r_idx] = obs_betas[:n_models]

    # Permutation p-values AND CIs (single computation!)
    XtY_all = np.einsum('ijk,k->ij', Xt_stack, neural_ranked)
    perm_betas_all = np.einsum('ijk,ik->ij', XtX_inv_stack, XtY_all)
    perm_betas_confirm[r_idx, :, :] = perm_betas_all[:, :n_models]

    for m_idx in range(n_models):
        roi_pvalues[r_idx, m_idx] = np.mean(perm_betas_all[:, m_idx] >= obs_betas[m_idx])

# ----- 5. FDR CORRECTION -----
print("\nStep 5: FDR correction across ROIs")
print("=" * 60)

confirm_fdr_sig = np.zeros((n_rois, n_models), dtype=bool)
tw_s_confirm = time_windows[t_confirm, 0] * 1000
tw_e_confirm = time_windows[t_confirm, 1] * 1000
print(f"  Time window: {tw_s_confirm:.0f}–{tw_e_confirm:.0f} ms")

for m_idx in range(n_models):
    p_confirm = roi_pvalues[:, m_idx]
    sig, threshold = fdr_bh(p_confirm, q=FDR_Q)
    confirm_fdr_sig[:, m_idx] = sig
    n_sig = sig.sum()

    print(f"\n  {MODEL_LABELS[m_idx]}:")
    print(f"    FDR threshold: {threshold:.4f}")
    print(f"    Significant ROIs: {n_sig}/{n_rois}")

    for r_idx in range(n_rois):
        beta = roi_observed[r_idx, m_idx]
        p = roi_pvalues[r_idx, m_idx]
        sig_str = "  ***" if confirm_fdr_sig[r_idx, m_idx] else ""
        uncorr = " (p<.05 uncorr)" if p < 0.05 and not confirm_fdr_sig[r_idx, m_idx] else ""
        print(f"      {roi_names[r_idx]:>12s}  beta={beta:>8.4f}  "
              f"p={p:.4f}{sig_str}{uncorr}")

# ----- 6. Save results -----
print(f"\nStep 6: Saving results to {OUTPUT_DIR}/")
np.save(os.path.join(OUTPUT_DIR, "roi_observed_confirm.npy"), roi_observed)
np.save(os.path.join(OUTPUT_DIR, "roi_pvalues_confirm.npy"), roi_pvalues)
np.save(os.path.join(OUTPUT_DIR, "confirm_fdr_sig.npy"), confirm_fdr_sig)
np.save(os.path.join(OUTPUT_DIR, "perm_betas_confirm.npy"), perm_betas_confirm)
np.save(os.path.join(OUTPUT_DIR, "time_windows.npy"), time_windows)
np.save(os.path.join(OUTPUT_DIR, "perm_orders.npy"), perm_orders)

with open(os.path.join(OUTPUT_DIR, "roi_info.pkl"), 'wb') as f:
    pickle.dump({
        'roi_names': roi_names,
        'roi_assignments': active_rois,
        'model_names': MODEL_NAMES,
        'model_labels': MODEL_LABELS,
        'n_perm': N_PERM,
        'fdr_q': FDR_Q,
        'confirm_time_idx': CONFIRM_TIME_IDX,
    }, f)


# ===========================================================================
# NEURAL RDM VISUALIZATION
# ===========================================================================
# Which ROI and time window to visualize
PLOT_ROI = 'Insula'       # change to any ROI name
PLOT_TIME = 0          # index into time_windows (0 = 50-100 ms)

print(f"\n  Saving neural RDM for {PLOT_ROI} at window {PLOT_TIME}")

import pandas as pd

# Load ratings to get valence for sorting
ratings = pd.read_excel("Results/Model RDMs/Ratings.xlsx")
valence_all = ratings['valence'].values
valence_valid = valence_all[valid_idx]

# Sort stimuli by valence
sort_idx = np.argsort(valence_valid)

# Recompute neural RDM for this ROI
r_idx_plot = roi_names.index(PLOT_ROI)
grid_indices = active_rois[PLOT_ROI]

if PLOT_TIME == 0 and CONFIRM_TIME_IDX == 0:
    patterns = source_avg[valid_idx][:, grid_indices, CONFIRM_TIME_IDX]
else:
    patterns = source_avg[valid_idx][:, grid_indices, PLOT_TIME]

patterns_zm = patterns - patterns.mean(axis=1, keepdims=True)
norms = np.sqrt(np.sum(patterns_zm ** 2, axis=1))
patterns_normed = patterns_zm / norms[:, np.newaxis]
corr_matrix = patterns_normed @ patterns_normed.T
np.clip(corr_matrix, -1.0, 1.0, out=corr_matrix)
neural_rdm = 1.0 - corr_matrix

# Sort by valence
neural_rdm_sorted = neural_rdm[np.ix_(sort_idx, sort_idx)]

# ---- Full RDM sorted by valence ----
fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(neural_rdm_sorted, cmap='viridis', interpolation='nearest')
ax.set_xlabel('Stimuli (sorted by valence)', fontsize=10)
ax.set_ylabel('Stimuli (sorted by valence)', fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation distance (1 − r)')

plt.tight_layout()
outfile = f"{OUTPUT_DIR}/neural_rdm_{PLOT_ROI}_full.svg"
fig.savefig(outfile, format='svg', dpi=300, bbox_inches='tight')
print(f"  Saved: {outfile}")
plt.close()

# ---- Upper triangle only ----
neural_rdm_upper = neural_rdm_sorted.copy()
# Mask lower triangle and diagonal
mask_lower = np.tril(np.ones_like(neural_rdm_upper, dtype=bool), k=0)
neural_rdm_upper[mask_lower] = np.nan

fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(neural_rdm_upper, cmap='viridis', interpolation='nearest')
ax.set_xlabel('Stimuli (sorted by valence)', fontsize=10)
ax.set_ylabel('Stimuli (sorted by valence)', fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
# Set background color for masked region
ax.set_facecolor('white')
plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation distance (1 − r)')

plt.tight_layout()
outfile = f"{OUTPUT_DIR}/neural_rdm_{PLOT_ROI}_upper.svg"
fig.savefig(outfile, format='svg', dpi=300, bbox_inches='tight')
print(f"  Saved: {outfile}")
plt.close()

print(f"  RDM shape: {neural_rdm.shape}")
print(f"  Dissimilarity range: [{neural_rdm_sorted[~np.eye(len(neural_rdm_sorted), dtype=bool)].min():.4f}, "
      f"{neural_rdm_sorted[~np.eye(len(neural_rdm_sorted), dtype=bool)].max():.4f}]")

# ----- 7. Visualization -----
print("\nStep 7: Creating figures")

# Glass brain figures
print("  Creating glass brain figures...")
for m_idx in range(n_models):
    model_label = MODEL_LABELS[m_idx]
    brain_sig = np.zeros(n_grid, dtype=np.float64)

    sig_roi_labels = []
    for r_idx, roi_name in enumerate(roi_names):
        if confirm_fdr_sig[r_idx, m_idx]:
            grid_idx = active_rois[roi_name]
            beta = roi_observed[r_idx, m_idx]
            brain_sig[grid_idx] = beta
            sig_roi_labels.append(f"{roi_name} (β={beta:.3f})")

    if np.any(brain_sig != 0):
        img_sig = make_nifti(grid_mm, brain_sig)
                
        if m_idx == 0:  # Valence: Red positive (normal RdBu_r)
            cmap = 'Reds'
            vmin_val = 0
        else:  # Arousal: Blue positive (RdBu_r reversed → white→blue)
            cmap = 'Blues'  # Reversed RdBu_r
            vmin_val = 0

        g = plot_glass_brain(
            img_sig,
            display_mode="lzr",
            symmetric_cbar=False,
            colorbar=True,
            cmap=cmap,
            cbar_tick_format='%.3f',
            alpha=0.85, plot_abs=False,
            threshold=0,
            vmin=0,
            vmax=0.06,
        )

        outfile = f"{OUTPUT_DIR}/glass_brain_confirmatory_{model_label.lower()}.svg"
        g.savefig(outfile, dpi=300)
        plt.close('all')
        print(f"    Significant: {', '.join(sig_roi_labels)}")
    else:
        print(f"  {model_label}: no significant ROIs")

# Forest plots
print("  Creating forest plots...")

# COMPUTE GLOBAL X-LIMITS (same for both models)
global_ci_low = np.inf
global_ci_high = -np.inf

for m_idx in range(n_models):
    for r_idx in range(n_rois):
        obs_b = roi_observed[r_idx, m_idx]
        if np.isnan(obs_b): continue
        
        perm_b = perm_betas_confirm[r_idx, :, m_idx]
        perm_se = np.std(perm_b)
        ci_low = obs_b - 1.96 * perm_se
        ci_high = obs_b + 1.96 * perm_se
        
        global_ci_low = min(global_ci_low, ci_low)
        global_ci_high = max(global_ci_high, ci_high)

# Add 5% padding
x_padding = 0.05 * (global_ci_high - global_ci_low)
xlim_left = global_ci_low - x_padding
xlim_right = global_ci_high + x_padding

print(f"  Global x-limits: [{xlim_left:.3f}, {xlim_right:.3f}]")

for m_idx in range(n_models):
    model_label = MODEL_LABELS[m_idx]

    fig, ax = plt.subplots(figsize=(4.5, 0.45 * n_rois + 0.8))

    betas = roi_observed[:, m_idx]
    y_positions = np.arange(n_rois)

    for r_idx in range(n_rois):
        obs_b = betas[r_idx]
        perm_b = perm_betas_confirm[r_idx, :, m_idx]

        if np.isnan(obs_b):
            continue

        # CI from permutation distribution (FULL symmetric range)
        perm_se = np.std(perm_b)
        ci_low = obs_b - 1.96 * perm_se
        ci_high = obs_b + 1.96 * perm_se

        # MODEL-SPECIFIC COLORS
        if confirm_fdr_sig[r_idx, m_idx]:
            color = 'firebrick' if m_idx == 0 else 'steelblue'
            marker = 'D'
            markersize = 7
            linewidth = 2.5
            zorder = 5
        else:
            color = 'lightgray'
            marker = 'D'
            markersize = 5
            linewidth = 1.5
            zorder = 3

        # Plot CI line
        ax.plot([ci_low, ci_high], [y_positions[r_idx], y_positions[r_idx]],
                color=color, linewidth=linewidth, zorder=zorder)
        # Plot point estimate
        ax.plot(obs_b, y_positions[r_idx], marker=marker, color=color,
                markersize=markersize, markeredgecolor='black',
                markeredgewidth=0.5, zorder=zorder + 1)

    # Zero line
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-', zorder=1)

    # GLOBAL X-LIMITS + CONSISTENT XTICKS
    ax.set_xlim([xlim_left, xlim_right])
    xticks = np.arange(np.ceil(xlim_left*10)/10, np.floor(xlim_right*10)/10 + 0.01, 0.02)
    ax.set_xticks(xticks)
    
    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(roi_names, fontsize=9)
    ax.set_xticks(np.arange(-0.08, 0.09, 0.04))
    ax.set_xlabel('Beta coefficient', fontsize=10)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    plt.tight_layout()
    outfile = f"{OUTPUT_DIR}/forest_confirmatory_{model_label.lower()}.svg"
    fig.savefig(outfile, format='svg', dpi=300, bbox_inches='tight')
    print(f"  Saved: {outfile}")
    plt.close()

# ----- 8. Summary table -----
total_time = time.time() - t_start
print(f"\n{'=' * 60}")
print(f"DONE. Runtime: {total_time/60:.1f} min")
print(f"HYPOTHESIS-DRIVEN RESULTS ({tw_s_confirm:.0f}–{tw_e_confirm:.0f} ms)")
print(f"FDR across {n_rois} ROIs per model, q = {FDR_Q}")
print(f"{'=' * 60}")
print(f"{'ROI':>12s}  {'Val beta':>9s}  {'Val p':>8s}  {'Val FDR':>7s}  "
      f"{'Aro beta':>9s}  {'Aro p':>8s}  {'Aro FDR':>7s}")
print(f"{'-'*12}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*8}  {'-'*7}")

for r_idx, roi_name in enumerate(roi_names):
    vb = roi_observed[r_idx, 0]; vp = roi_pvalues[r_idx, 0]; vf = "YES" if confirm_fdr_sig[r_idx, 0] else "no"
    ab = roi_observed[r_idx, 1]; ap = roi_pvalues[r_idx, 1]; af = "YES" if confirm_fdr_sig[r_idx, 1] else "no"
    print(f"{roi_name:>12s}  {vb:>9.4f}  {vp:>8.4f}  {vf:>7s}  "
          f"{ab:>9.4f}  {ap:>8.4f}  {af:>7s}")