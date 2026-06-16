# -*- coding: utf-8 -*-
"""
Construct Representational Dissimilarity Matrices (RDMs) from stimulus ratings.

This script reads an Excel file with columns: picture, valence, arousal,
luminance, contrast, complexity, entropy.

It computes the following RDMs (all as pairwise absolute differences):
  1. Valence RDM
  2. Arousal RDM
  3. Luminance RDM
  4. Contrast RDM
  5. Complexity RDM
  6. Entropy RDM

It also runs the Lind & Mehlum (2010) U-shape test on the valence-arousal
relationship across the stimulus set (Section 1b), the Python equivalent of the
R utest/uslopes procedure.

All RDMs are saved in canonical stimulus order (the row order of Ratings.xlsx,
matching the ordering of the neural RDMs) as .npy files. Heatmaps (full and
upper-triangle), sorted by valence for display purposes only, are saved as SVGs.
Pairwise Mantel correlations are reported for all model pairs.

Author: Ioannis Ntoumanis
Created: 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr, t as t_dist
from itertools import combinations

# Set BASE_DIR to the root of your RSA project folder.
BASE_DIR = '/path/to/RSA'
os.chdir(BASE_DIR)


# ── 1. Load ratings ────────────────────────────────────────────────────────────
df = pd.read_excel('Ratings.xlsx')

stimulus_ids = df['picture'].values
valence    = df['valence'].values.astype(float)
arousal    = df['arousal'].values.astype(float)
luminance  = df['luminance'].values.astype(float)
contrast   = df['contrast'].values.astype(float)
complexity = df['complexity'].values.astype(float)
entropy    = df['entropy'].values.astype(float)

# ── 1b. Lind–Mehlum U-shape test on the valence–arousal relationship ──────────
# OLS fit: arousal ~ 1 + valence + valence^2
X = np.column_stack([np.ones_like(valence), valence, valence ** 2])
y = arousal
XtX_inv = np.linalg.inv(X.T @ X)
beta = XtX_inv @ (X.T @ y)                         # [b0, b1, b2]
resid = y - X @ beta
n_obs = len(y)
k = X.shape[1]                                     # 3 parameters
df_resid = n_obs - k                               # = 417 for 420 stimuli
mse = (resid @ resid) / df_resid
cov_beta = mse * XtX_inv                            # covariance of [b0, b1, b2]

b1, b2 = beta[1], beta[2]
x_lo, x_hi = valence.min(), valence.max()

def _slope_and_t(x0):
    """Fitted slope b1 + 2*b2*x0 (a linear combination c·beta) with its t-stat."""
    c = np.array([0.0, 1.0, 2.0 * x0])
    slope = c @ beta
    se = np.sqrt(c @ cov_beta @ c)
    return slope, se, slope / se

slope_lo, se_lo, t_lo = _slope_and_t(x_lo)
slope_hi, se_hi, t_hi = _slope_and_t(x_hi)

# Shape implied by the quadratic term: b2 > 0 → U-shaped, b2 < 0 → inverted U.
if b2 > 0:                                          # U: slope<0 at xlo, slope>0 at xhi
    shape = 'U-shaped'
    p_lo = t_dist.cdf(t_lo, df_resid)               # H1: slope at min < 0
    p_hi = t_dist.sf(t_hi, df_resid)                # H1: slope at max > 0
else:                                               # inverted-U: slope>0 at xlo, slope<0 at xhi
    shape = 'inverted-U'
    p_lo = t_dist.sf(t_lo, df_resid)                # H1: slope at min > 0
    p_hi = t_dist.cdf(t_hi, df_resid)               # H1: slope at max < 0

# Intersection–union test
t_overall = min(abs(t_lo), abs(t_hi))
p_overall = max(p_lo, p_hi)
extremum = -b1 / (2.0 * b2)                          # turning-point location

print('\n── Lind–Mehlum U-shape test (arousal ~ valence + valence^2) ──────────')
print(f'  Quadratic coefficient b2 = {b2:+.4f}  →  {shape} relationship')
print(f'  Slope at valence min ({x_lo:.3f}): {slope_lo:+.4f}  '
      f'(t = {t_lo:+.2f}, one-sided p = {p_lo:.4g})')
print(f'  Slope at valence max ({x_hi:.3f}): {slope_hi:+.4f}  '
      f'(t = {t_hi:+.2f}, one-sided p = {p_hi:.4g})')
print(f'  Turning point at valence = {extremum:.3f} '
      f'({"inside" if x_lo < extremum < x_hi else "OUTSIDE"} the data range)')
print(f'  Overall U-test: t({df_resid}) = {t_overall:.2f}, p = {p_overall:.4g}')

# ── 2. Sort index by valence (used for visualization only) ─────────────────────
valence_sort_idx = np.argsort(valence)

# ── 3. Helper: compute RDM as pairwise absolute differences ───────────────────
def make_rdm(values):
    """Return a symmetric N×N RDM of pairwise absolute differences."""
    return squareform(pdist(values.reshape(-1, 1), metric='euclidean'))

# ── 4. Compute all RDMs ────────────────────────────────────────────────────────
rdm_valence    = make_rdm(valence)
rdm_arousal    = make_rdm(arousal)
rdm_luminance  = make_rdm(luminance)
rdm_contrast   = make_rdm(contrast)
rdm_complexity = make_rdm(complexity)
rdm_entropy    = make_rdm(entropy)

# Collect for loop-based operations
models = {
    'valence':    rdm_valence,
    'arousal':    rdm_arousal,
    'luminance':  rdm_luminance,
    'contrast':   rdm_contrast,
    'complexity': rdm_complexity,
    'entropy':    rdm_entropy,
}

# ── 5. Save .npy files (canonical stimulus order, matching the neural RDMs) ────
for name, rdm in models.items():
    np.save(f'rdm_{name}.npy', rdm)
    print(f'Saved rdm_{name}.npy  (shape {rdm.shape}, canonical order)')

# ── 6. Plot full heatmaps (sorted by valence) ─────────────────────────────────
for name, rdm in models.items():
    rdm_sorted = rdm[np.ix_(valence_sort_idx, valence_sort_idx)]
    plt.figure(figsize=(8, 7))
    plt.imshow(rdm_sorted, cmap='viridis')
    plt.title(f'{name.capitalize()} RDM (sorted by valence)')
    plt.colorbar(label='Dissimilarity')
    plt.xlabel('Stimulus')
    plt.ylabel('Stimulus')
    plt.tight_layout()
    plt.savefig(f'rdm_{name}_sorted.svg', dpi=300, bbox_inches='tight')
    plt.show()

# ── 7. Plot upper-triangle heatmaps (sorted by valence) ───────────────────────
cmap_ut = plt.cm.viridis.copy()
cmap_ut.set_bad(color='white')

for name, rdm in models.items():
    rdm_sorted = rdm[np.ix_(valence_sort_idx, valence_sort_idx)]
    mask = np.tril(np.ones_like(rdm_sorted, dtype=bool), k=-1)
    masked_rdm = np.ma.array(rdm_sorted, mask=mask)
    plt.figure(figsize=(8, 7))
    plt.imshow(masked_rdm, cmap=cmap_ut, interpolation='nearest')
    plt.title(f'{name.capitalize()} RDM (sorted by valence, upper triangle)')
    plt.colorbar(label='Dissimilarity')
    plt.xlabel('Stimulus')
    plt.ylabel('Stimulus')
    plt.tight_layout()
    plt.savefig(f'rdm_{name}_sorted_upper_triangle.svg', dpi=300, bbox_inches='tight')
    plt.show()

# ── 8. Pairwise Mantel correlations (all model pairs) ─────────────────────────
N_PERM_MANTEL = 10000
rng = np.random.default_rng(0)
n_stim = len(valence)
tri = np.triu_indices(n_stim, k=1)

print('\n── Pairwise Mantel correlations (Spearman r, two-tailed) ──────────────')
print(f'{"Model pair":<30}  {"r":>7}  {"p (Mantel)":>12}')
print('-' * 55)

for name_a, name_b in combinations(models.keys(), 2):
    rdm_a = models[name_a]
    rdm_b = models[name_b]

    obs_r, _ = spearmanr(rdm_a[tri], rdm_b[tri])

    # Mantel permutation test
    perm_r = np.empty(N_PERM_MANTEL)
    for k in range(N_PERM_MANTEL):
        perm = rng.permutation(n_stim)
        rdm_b_perm = rdm_b[np.ix_(perm, perm)]
        perm_r[k], _ = spearmanr(rdm_a[tri], rdm_b_perm[tri])

    p_mantel = (np.sum(np.abs(perm_r) >= np.abs(obs_r)) + 1) / (N_PERM_MANTEL + 1)
    pair_label = f'{name_a} vs {name_b}'
    print(f'{pair_label:<30}  {obs_r:>7.4f}  {p_mantel:>12.4g}')