# -*- coding: utf-8 -*-
"""
Signed Jaccard Comparison of Valence vs Arousal Representational Networks
==========================================================================
Polished version using seaborn for publication-quality figures.

Colour scheme (consistent with the rest of the manuscript):
  - Valence  → red       (#c0392b)
  - Arousal  → blue      (#2c5aa0)
  - Shared   → purple    (#7b3294)   — blend of red and blue
  - Jaccard  → dark teal (#1a6e6e)   — distinct from valence/arousal/shared

Outputs:
  jaccard_timeseries.{png,svg}
  edge_counts_timeseries.{png,svg}
  jaccard_and_counts_combined.{png,svg}
  signed_jaccard.npy, edge_counts.npy, edge_counts.csv

Author: Ioannis Ntoumanis
Created: 2026
"""

import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import seaborn as sns

# ── paths ────────────────────────────────────────────────────────────────────

CONN_DIR   = r"Results/Connectivity"
OUTPUT_DIR = os.path.join(CONN_DIR, "valence_vs_arousal_jaccard")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── colour palette ───────────────────────────────────────────────────────────
COL_VAL     = "#c0392b"   # valence red
COL_ARO     = "#2c5aa0"   # arousal blue
COL_SHARED  = "#7b3294"   # purple (valence ∩ arousal)
COL_JAC     = "#1a6e6e"   # dark teal — Jaccard line
COL_FILL    = "#1a6e6e22" # light teal — under-Jaccard fill (alpha hex)
COL_GRID    = "#dddddd"

# ── seaborn / matplotlib styling ─────────────────────────────────────────────
sns.set_theme(style="white", context="paper")
plt.rcParams.update({
    "font.family":          "DejaVu Sans",
    "font.size":            11,
    "axes.titlesize":       13,
    "axes.titleweight":     "bold",
    "axes.labelsize":       11,
    "axes.labelweight":     "regular",
    "axes.edgecolor":       "#333333",
    "axes.linewidth":       1.0,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "xtick.color":          "#333333",
    "ytick.color":          "#333333",
    "legend.fontsize":      9,
    "legend.frameon":       False,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "figure.dpi":           120,
})

# ── load saved outputs ───────────────────────────────────────────────────────
print("Loading network results …")
obs_conn     = np.load(os.path.join(CONN_DIR, "obs_conn.npy"))
fdr_sig      = np.load(os.path.join(CONN_DIR, "fdr_sig.npy"))
time_windows = np.load(os.path.join(CONN_DIR, "time_windows.npy"))

with open(os.path.join(CONN_DIR, "info.pkl"), 'rb') as f:
    info = pickle.load(f)
roi_names    = info['roi_names']
model_labels = info['model_labels']
n_rois       = len(roi_names)

n_models, _, _, n_time = obs_conn.shape
assert n_models == 2

M_VAL = model_labels.index("Valence")
M_ARO = model_labels.index("Arousal")

time_ms_center = ((time_windows[:, 0] + time_windows[:, 1]) / 2 * 1000).astype(int)
time_labels    = [f"{int(round(time_windows[t,0]*1000))}–"
                  f"{int(round(time_windows[t,1]*1000))} ms"
                  for t in range(n_time)]

triu_r1, triu_r2 = np.triu_indices(n_rois, k=1)

# ── per-window edge classification ───────────────────────────────────────────
# Columns of edge_counts:
#   0 = valence-only
#   1 = arousal-only
#   2 = shared (since no opposite-sign edges exist, this is all shared)
edge_counts    = np.zeros((n_time, 3), dtype=int)
signed_jaccard = np.full(n_time, np.nan)

for t in range(n_time):
    val_sig = fdr_sig[M_VAL, triu_r1, triu_r2, t]
    aro_sig = fdr_sig[M_ARO, triu_r1, triu_r2, t]

    val_only = val_sig & ~aro_sig
    aro_only = aro_sig & ~val_sig
    shared   = val_sig & aro_sig

    edge_counts[t] = [int(val_only.sum()),
                      int(aro_only.sum()),
                      int(shared.sum())]

    union = edge_counts[t].sum()
    if union > 0:
        signed_jaccard[t] = edge_counts[t, 2] / union
    else:
        signed_jaccard[t] = np.nan

# ── save arrays ──────────────────────────────────────────────────────────────
np.save(os.path.join(OUTPUT_DIR, "signed_jaccard.npy"), signed_jaccard)
np.save(os.path.join(OUTPUT_DIR, "edge_counts.npy"),    edge_counts)

with open(os.path.join(OUTPUT_DIR, "edge_counts.csv"), 'w') as f:
    f.write("time_window,val_only,aro_only,shared,signed_jaccard\n")
    for t in range(n_time):
        jstr = f"{signed_jaccard[t]:.4f}" if not np.isnan(signed_jaccard[t]) else "NA"
        f.write(f"{time_labels[t]},{edge_counts[t,0]},{edge_counts[t,1]},"
                f"{edge_counts[t,2]},{jstr}\n")

print("\nPer time-window summary:")
print(f"  {'Time':<14s}{'V-only':>8s}{'A-only':>8s}{'Shared':>8s}{'J':>8s}")
for t in range(n_time):
    jstr = f"{signed_jaccard[t]:.3f}" if not np.isnan(signed_jaccard[t]) else "—"
    print(f"  {time_labels[t]:<14s}{edge_counts[t,0]:>8d}"
          f"{edge_counts[t,1]:>8d}{edge_counts[t,2]:>8d}{jstr:>8s}")

# ── helper: clean spines + grid ───────────────────────────────────────────────
def style_axis(ax, ylabel=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.tick_params(axis='both', which='major', length=4, color='#333333')
    ax.yaxis.grid(True, color=COL_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)

# ── figure 1: Jaccard time series ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 2.5))

# Soft fill under the curve
mask = ~np.isnan(signed_jaccard)
ax.fill_between(time_ms_center[mask], 0, signed_jaccard[mask],
                color=COL_JAC, alpha=0.12, zorder=2)

# Line + markers
ax.plot(time_ms_center, signed_jaccard,
        color=COL_JAC, linewidth=2.2, zorder=3)
ax.scatter(time_ms_center, signed_jaccard,
           color=COL_JAC, s=55, edgecolor='white', linewidth=1.5, zorder=4)

# Zero baseline
ax.axhline(0, color='#999999', linewidth=0.7, linestyle='--', zorder=1)

style_axis(ax, ylabel='Signed Jaccard index')
ax.set_xlabel('Time (ms, window centre)')
#ax.set_title('Similarity between valence and arousal representational networks')

y_max = 1#max(0.5, np.nanmax(signed_jaccard) * 1.15) if mask.any() else 0.5
ax.set_ylim(-0.02, y_max)
ax.set_xlim(time_ms_center[0] - 30, time_ms_center[-1] + 30)

# Mark empty windows
empty = (edge_counts.sum(axis=1) == 0)
for t in np.where(empty)[0]:
    ax.text(time_ms_center[t], 0.005, '×', ha='center', va='bottom',
            fontsize=10, color='#888888', fontweight='bold')

plt.tight_layout()
for ext in ['png', 'svg']:
    fig.savefig(os.path.join(OUTPUT_DIR, f"jaccard_timeseries.{ext}"))
plt.close(fig)
print("\nSaved jaccard_timeseries.png/.svg")

# ── figure 2: edge counts — one figure per model ─────────────────────────────
# Use the manuscript RGB colors directly
COL_VAL_RGB = (178/255, 34/255, 34/255)   # firebrick — valence
COL_ARO_RGB = (70/255, 130/255, 180/255)  # steelblue — arousal

# Total significant edges per time window per model
# (counts both model-specific edges and shared edges, since both indicate
# the model's network had a significant edge there)
val_total = edge_counts[:, 0]# + edge_counts[:, 2]   # val_only + shared
aro_total = edge_counts[:, 1]# + edge_counts[:, 2]   # aro_only + shared

def plot_model_counts(counts, color, model_label, fname_base):
    fig, ax = plt.subplots(figsize=(4.5, 2.5))

    sns.barplot(
        x=time_ms_center,
        y=counts,
        color=color,
        edgecolor='white',
        linewidth=0.8,
        ax=ax,
    )

    style_axis(ax, ylabel='Number of significant edges')
    ax.set_xlabel('Time (ms, window centre)')
    ax.set_title(f'{model_label} representational network: '
                 f'significant edges over time')

    # Rotate x-tick labels slightly for readability
    ax.set_xticklabels([str(t) for t in time_ms_center],
                       rotation=0, fontsize=9)

    # Tight y-limit with a little headroom
    y_top = 20#max(1, counts.max()) * 1.18
    ax.set_ylim(0, y_top)

    plt.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{fname_base}.{ext}"))
    plt.close(fig)
    print(f"Saved {fname_base}.png/.svg")

plot_model_counts(val_total, COL_VAL_RGB, 'Valence',
                  'edge_counts_valence')
plot_model_counts(aro_total, COL_ARO_RGB, 'Arousal',
                  'edge_counts_arousal')

# ===========================================================================
# CONSISTENCY MATRICES (PANEL C — one per model)
# ===========================================================================
# Per-pair coefficient of variation (CV) across time windows:
#   CV_pair = std(S_t) / mean(|S_t|)
# Higher CV = more variable (less temporally stable) co-representation;
# lower CV = more stable. Normalizing by mean amplitude lets pairs with
# different absolute S be compared on a common stability scale.
# ===========================================================================

print("\nComputing per-pair consistency matrices …")

variation = np.full((n_models, n_rois, n_rois), np.nan)

for m_idx in range(n_models):
    gm_3d        = obs_conn[m_idx]                              # (n_rois, n_rois, n_time)
    std_gm       = np.nanstd(gm_3d, axis=2)
    mean_abs_gm  = np.nanmean(np.abs(gm_3d), axis=2)
    cv           = std_gm / (mean_abs_gm)
    np.fill_diagonal(cv, np.nan)
    variation[m_idx] = cv

np.save(os.path.join(OUTPUT_DIR, "variation_matrix.npy"), variation)

# Summary
for m_idx, lab in enumerate(model_labels):
    vals = variation[m_idx][np.triu_indices(n_rois, k=1)]
    vals = vals[~np.isnan(vals)]
    print(f"  {lab}: median consistency = {np.median(vals):.3f}, "
          f"frac > 0.8 = {np.mean(vals > 0.8):.2f}, "
          f"frac > 0.5 = {np.mean(vals > 0.5):.2f}")

# ── Plot one upper-triangle figure per model ─────────────────────────────────
PER_MODEL_CMAP   = 'viridis'    # sequential, 0 (light) → 1 (dark)
TRIANGLE_TO_SHOW = 'lower'   # set to 'lower' to flip orientation

for m_idx, model_label in enumerate(model_labels):
    mat = variation[m_idx].copy()

    # Mask the unused triangle and diagonal
    if TRIANGLE_TO_SHOW == 'upper':
        mask = np.tril(np.ones_like(mat, dtype=bool), k=0)   # mask lower + diag
    else:
        mask = np.triu(np.ones_like(mat, dtype=bool), k=0)   # mask upper + diag
    masked = np.ma.array(mat, mask=mask)

    cmap_c = plt.get_cmap(PER_MODEL_CMAP).copy()
    cmap_c.set_bad(color='white')

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(masked, cmap=cmap_c, vmin=0, vmax=2,
                   interpolation='nearest', aspect='equal')

    ax.set_xticks(range(n_rois))
    ax.set_yticks(range(n_rois))
    ax.set_xticklabels(roi_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(roi_names, fontsize=9)
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(1.0)

    ax.set_title(f'{model_label} co-representation: '
                 f'temporal variation',
                 fontsize=12, fontweight='bold', pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Coefficient of variation', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks([0, 0.5, 1, 1.5, 2])

    plt.tight_layout()
    fname_base = f"variation_matrix_{model_label.lower()}"
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{fname_base}.{ext}"))
    plt.close(fig)
    print(f"  Saved {fname_base}.png/.svg")

# ===========================================================================
# PAIRED VIOLIN + STRIP PLOT: CV comparison Valence vs Arousal
# ===========================================================================
# Statistical test: paired Wilcoxon signed-rank (each ROI pair contributes
# one valence-CV and one arousal-CV value).
# Effect size: Cohen's d_z for paired differences and median difference.
# Visualisation modelled on the user's template — half-violins with strip dots
# on the inner side, no connecting lines (commented out for future use).
# ===========================================================================

from scipy.stats import wilcoxon
import seaborn as sns

print("\nPaired CV comparison: Valence vs Arousal …")

# ── Gather per-pair CVs (upper triangle, both models) ────────────────────────
iu = np.triu_indices(n_rois, k=1)
cv_val = variation[M_VAL][iu] if 'cv_matrix' in dir() else \
         (np.nanstd(obs_conn[M_VAL], axis=2) /
          (np.nanmean(np.abs(obs_conn[M_VAL]), axis=2) + 1e-12))[iu]
cv_aro = variation[M_ARO][iu] if 'cv_matrix' in dir() else \
         (np.nanstd(obs_conn[M_ARO], axis=2) /
          (np.nanmean(np.abs(obs_conn[M_ARO]), axis=2) + 1e-12))[iu]

# Drop pairs with NaN in either
valid_pairs = ~(np.isnan(cv_val) | np.isnan(cv_aro))
cv_val_v = cv_val[valid_pairs]
cv_aro_v = cv_aro[valid_pairs]
n_valid_pairs = len(cv_val_v)
print(f"  Valid pairs for comparison: {n_valid_pairs}")

# ── Wilcoxon signed-rank test ────────────────────────────────────────────────
diff = cv_aro_v - cv_val_v
stat, p_val = wilcoxon(cv_val_v, cv_aro_v, alternative='two-sided')

# Effect size: Cohen's d_z for paired data
mean_diff = np.mean(diff)
sd_diff   = np.std(diff, ddof=1)
d_z = mean_diff / sd_diff if sd_diff > 0 else np.nan
median_diff = np.median(diff)

print(f"\n  Wilcoxon signed-rank test (paired):")
print(f"    W = {stat:.2f}, p = {p_val:.4f}")
print(f"    Median CV valence: {np.median(cv_val_v):.3f}")
print(f"    Median CV arousal: {np.median(cv_aro_v):.3f}")
print(f"    Median difference (arousal - valence): {median_diff:+.3f}")
print(f"    Cohen's d_z: {d_z:+.3f}")

# Stars for significance
def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'

stars = sig_stars(p_val)

# ── Plot ─────────────────────────────────────────────────────────────────────
import pandas as pd
from statannotations.Annotator import Annotator

COL_VAL_RGB = (178/255, 34/255, 34/255)   # firebrick
COL_ARO_RGB = (70/255, 130/255, 180/255)  # steelblue

# Build long-format dataframe
df_violin = pd.DataFrame({
    'Model': (['Valence'] * n_valid_pairs) + (['Arousal'] * n_valid_pairs),
    'CV':    np.concatenate([cv_val_v, cv_aro_v]),
})

fig, ax = plt.subplots(figsize=(5, 4))

# Strip plot (data points)
sns.stripplot(
    data=df_violin, x='Model', y='CV',
    hue='Model', palette=[COL_VAL_RGB, COL_ARO_RGB],
    hue_order=['Valence', 'Arousal'],
    order=['Valence', 'Arousal'],
    ax=ax, jitter=0.03, size=6, alpha=0.7,
    dodge=False, legend=False, linewidth=0,
)

# Split violin (outline only, no fill)
sns.violinplot(
    data=df_violin, x='Model', y='CV',
    hue='Model', hue_order=['Valence', 'Arousal'],
    width=0.3, legend=False, fill=False,
    order=['Valence', 'Arousal'],
    split=True, cut=0, density_norm='count',
    dodge=False, ax=ax,
    palette=[COL_VAL_RGB, COL_ARO_RGB],
    inner_kws={'box_width': 7, 'whis_width': 2},
)

# Paired Wilcoxon via statannotations
pairs = [('Valence', 'Arousal')]
annotator = Annotator(
    ax, pairs, data=df_violin, x='Model', y='CV',
    order=['Valence', 'Arousal'],
)
annotator.configure(test='Wilcoxon', text_format='star', loc='inside')
annotator.apply_and_annotate()

# ── OPTIONAL: connecting lines between paired observations ───────────────────
# Uncomment to enable. With 100+ pairs this will look crowded.
# for i in range(n_valid_pairs):
#     ax.plot([0, 1], [cv_val_v[i], cv_aro_v[i]],
#             color='gray', linewidth=0.3, alpha=0.25, zorder=0)

ax.set_xlabel('')
ax.set_ylabel('Coefficient of variation')

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, f"cv_violin_valence_vs_arousal.svg"), dpi=300)
plt.close(fig)
print(f"\nSaved cv_violin_valence_vs_arousal.png/.svg")

# ===========================================================================
# REPORTING BLOCK — everything needed to write the CV paragraph (Section 3.3)
# ===========================================================================
print("\n" + "=" * 70)
print("CV REPORTING BLOCK (copy values into manuscript)")
print("=" * 70)

# ── 1. Sample size, medians, IQRs, ranges ───────────────────────────────────
def iqr(x):
    q1, q3 = np.percentile(x, [25, 75])
    return q1, q3

for lab, cv in [("Valence", cv_val_v), ("Arousal", cv_aro_v)]:
    med = np.median(cv)
    q1, q3 = iqr(cv)
    print(f"\n  {lab} network CV:")
    print(f"    n pairs        = {len(cv)}")
    print(f"    median         = {med:.3f}")
    print(f"    IQR            = {q1:.3f} – {q3:.3f}")
    print(f"    range          = {cv.min():.3f} – {cv.max():.3f}")

print(f"\n  Matching units (paired ROI pairs): n = {n_valid_pairs}")

# ── 2. Wilcoxon signed-rank + rank-biserial effect size ─────────────────────
# Rank-biserial r for a Wilcoxon signed-rank test:
#   r_rb = W+ - W-  /  (W+ + W-)  =  (sum of positive ranks - negative) / total
diff_nz = diff[diff != 0]                       # drop zero differences
ranks   = np.argsort(np.argsort(np.abs(diff_nz))) + 1
W_pos   = ranks[diff_nz > 0].sum()
W_neg   = ranks[diff_nz < 0].sum()
r_rb    = (W_pos - W_neg) / (W_pos + W_neg)

print(f"\n  Wilcoxon signed-rank (paired, two-sided):")
print(f"    W              = {stat:.2f}")
print(f"    p              = {p_val:.4f}")
print(f"    n (nonzero diff) = {len(diff_nz)}")
print(f"    rank-biserial r = {r_rb:+.3f}")
print(f"    median diff (arousal - valence) = {median_diff:+.3f}")
print(f"    direction       : "
      f"{'valence more stable (lower CV)' if np.median(cv_val_v) < np.median(cv_aro_v) else 'arousal more stable (lower CV)'}")

# ── 3. Qualitative pattern: which significant network edges are most/least
#       stable? Restrict to edges that were FDR-significant in >=1 window for
#       each model, so the pattern ties back to the networks in 3.3.
# ===========================================================================
def pair_name(r1, r2):
    return f"{roi_names[r1]}–{roi_names[r2]}"

for m_idx, lab in [(M_VAL, "Valence"), (M_ARO, "Arousal")]:
    # Edges significant in at least one time window for this model
    sig_any = fdr_sig[m_idx][:, :, :].any(axis=2)        # (n_rois, n_rois)
    rows = []
    for r1, r2 in zip(triu_r1, triu_r2):
        if sig_any[r1, r2] and not np.isnan(variation[m_idx, r1, r2]):
            rows.append((variation[m_idx, r1, r2], pair_name(r1, r2)))
    rows.sort(key=lambda x: x[0])
    print(f"\n  {lab} network — significant edges ranked by CV "
          f"(low = most stable), n = {len(rows)}:")
    print("    Most stable (lowest CV):")
    for cv_v, nm in rows[:5]:
        print(f"      {nm:<22s} CV = {cv_v:.3f}")
    print("    Least stable (highest CV):")
    for cv_v, nm in rows[-5:][::-1]:
        print(f"      {nm:<22s} CV = {cv_v:.3f}")

print("\n" + "=" * 70)


print(f"\nAll outputs saved to: {OUTPUT_DIR}/")