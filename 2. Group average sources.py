# -*- coding: utf-8 -*-
"""
Build group-averaged source data from subject-level .mat files.
Remaps presentation-order trial indices to canonical stimulus indices
using each subject's E-Prime Excel file and Ratings.xlsx.

Saves:
  source_avg.npy    (420, 13267, n_time)  float32
  source_count.npy  (420,)                int
  valid_mask.npy    (420,)                bool, count >= MIN_SUBJECTS
  time_windows.npy  (n_time, 2)
"""

import numpy as np
import pandas as pd
import h5py
import os

# Set BASE_DIR to the root of your RSA project folder
BASE_DIR     = '/path/to/RSA'
source_dir   = os.path.join(BASE_DIR, 'Source data')
eprime_base  = os.path.join(BASE_DIR, 'data_raw')
ratings_file = os.path.join(BASE_DIR, 'Results', 'Model RDMs', 'Ratings.xlsx')
output_dir   = os.path.join(BASE_DIR, 'Source data')

MIN_SUBJECTS = 15

os.makedirs(output_dir, exist_ok=True)

# Subject IDs
subject_ids = [f"{i:02}" for i in range(1, 59) if i != 32]

# =============================================================================
# Load ratings
# =============================================================================
ratings     = pd.read_excel(ratings_file)
picture_ids = ratings['picture'].values   # canonical order, 420 filenames
n_stimuli   = len(picture_ids)
print(f'Loaded ratings for {n_stimuli} stimuli')

# =============================================================================
# Load time windows
# =============================================================================
with h5py.File(os.path.join(source_dir, 'time_windows.mat'), 'r') as f:
    time_windows = np.array(f['time_windows']).T   # (n_time, 2)
n_windows = time_windows.shape[0]
print(f'Time windows: {n_windows}')

# =============================================================================
# Accumulate source data across subjects
# =============================================================================
source_sum   = np.zeros((n_stimuli, 13267, n_windows), dtype=np.float64)
source_count = np.zeros(n_stimuli, dtype=int)

for subj_id in subject_ids:
    print(f'\nProcessing sub-{subj_id}...')

    # ── load source data ────────────────────────────────────────────────────
    src_file = os.path.join(source_dir, f'sub-{subj_id}_source_data.mat')
    if not os.path.exists(src_file):
        print(f'  WARNING: source file not found — skipping')
        continue

    with h5py.File(src_file, 'r') as f:
        source_data   = np.array(f['source_data'])
        trial_numbers = np.array(f['trial_numbers']).flatten().astype(int)

    # h5py always reverses MATLAB dims:
    # MATLAB saved (n_trials, n_grid, n_time) → h5py reads (n_time, n_grid, n_trials)
    # Always transpose to (n_trials, n_grid, n_time)
    source_data = source_data.transpose(2, 1, 0)
    print(f'  Source data shape: {source_data.shape}')
    print(f'  Valid trials: {len(trial_numbers)}')

    # ── load E-Prime log ────────────────────────────────────────────────────
    eprime_file = os.path.join(eprime_base, f'sub-{subj_id}', 'emotion.xlsx')
    if not os.path.exists(eprime_file):
        print(f'  WARNING: E-Prime file not found — skipping')
        continue
    eprime         = pd.read_excel(eprime_file, skiprows=1)
    eprime_objects = eprime['Object'].values   # presentation order

    # ── remap and accumulate ────────────────────────────────────────────────
    n_matched = 0
    for trial_num in trial_numbers:
        # trial_num is 1-based presentation index
        if trial_num < 1 or trial_num > len(eprime_objects):
            continue

        pic_name = eprime_objects[trial_num - 1]

        # Look up canonical index in Ratings.xlsx
        pic_idx = np.where(picture_ids == pic_name)[0]
        if len(pic_idx) == 0:
            # Try without extension
            pic_name_stripped = os.path.splitext(pic_name)[0]
            pic_idx = np.where(
                [os.path.splitext(p)[0] == pic_name_stripped for p in picture_ids]
            )[0]
        if len(pic_idx) == 0:
            continue
        pic_idx = pic_idx[0]

        # Get source data row for this trial (1-based → 0-based)
        trial_data = source_data[trial_num - 1, :, :]   # (n_grid, n_time)
        if np.isnan(trial_data).any():
            continue

        source_sum[pic_idx, :, :]  += trial_data
        source_count[pic_idx]      += 1
        n_matched += 1

    print(f'  Trials matched and accumulated: {n_matched}')

# =============================================================================
# Compute average
# =============================================================================
print('\nComputing group average...')
source_avg = np.full((n_stimuli, 13267, n_windows), np.nan, dtype=np.float64)
nonzero = source_count > 0
source_avg[nonzero] = source_sum[nonzero] / source_count[nonzero, None, None]
source_avg = source_avg.astype(np.float32)

valid_mask = source_count >= MIN_SUBJECTS

print(f'Stimuli with any data:        {nonzero.sum()}/{n_stimuli}')
print(f'Stimuli >= {MIN_SUBJECTS} subjects:       {valid_mask.sum()}/{n_stimuli}')
print(f'Subjects per stimulus: min={source_count[nonzero].min()}, '
      f'max={source_count[nonzero].max()}, '
      f'mean={source_count[nonzero].mean():.1f}')

# =============================================================================
# Save
# =============================================================================
np.save(os.path.join(output_dir, 'source_avg.npy'),   source_avg)
np.save(os.path.join(output_dir, 'source_count.npy'), source_count)
np.save(os.path.join(output_dir, 'valid_mask.npy'),   valid_mask)
np.save(os.path.join(output_dir, 'time_windows.npy'), time_windows)

print(f'\nSaved to {output_dir}/:')
print(f'  source_avg.npy   {source_avg.shape}  {source_avg.dtype}')
print(f'  source_count.npy {source_count.shape}')
print(f'  valid_mask.npy   {valid_mask.shape}')
print(f'  time_windows.npy {time_windows.shape}')
print('\n=== Done ===')