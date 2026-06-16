%% Extract projected source data from Brainstorm for RSA
% Averages source data within time windows matching the NeuroImage paper.
% Output per subject: 420 x 13267 x 18 (trials x grid_points x time_windows)
% Rejected trials are NaN.

clear; clc;

%% Parameters
% Set these to match your local Brainstorm database and output locations.
db_path    = '/path/to/brainstorm_db/Autism_RSA/data/Group_analysis';
output_dir = '/path/to/output/Neural_RDMs';

subjects = {
    'sub-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-02_task-emotion_tsss_meg_bl_notch_band'
    'sub-03_task-emotion_tsss_meg_bl_notch_band'
    'sub-04_task-emotion_tsss_meg_bl_notch_band'
    'sub-05_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-06_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-07_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-08_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-09_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-10_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-11_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-12_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-13_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-14_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-15_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-16_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-17_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-18_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-19_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-20_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-21_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-22_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-23_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-24_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-25_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-26_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-27_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-28_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-29_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-30_ses-01_task-emotion_proc-tsss_meg_bl_notch_band'
    'sub-31_ses-01_task-emotion_proc-tsss_meg_bl_notch_band'
    'sub-33_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-34_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-35_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-36_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-37_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-38_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-39_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-40_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-41_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-42_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-43_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-44_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-45_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-46_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-47_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-48_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-49_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-50_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-51_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-52_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-53_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-54_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-55_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-56_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-57_ses-01_task-emotion_tsss_meg_bl_notch_band'
    'sub-58_ses-01_task-emotion_tsss_meg_bl_notch_band'
};

n_trials_total = 420;
n_grid = 13267;

% Time windows (in seconds) matching description: 50-100 ms, then 100-ms overlapping shifted 50 ms to 900-1000 ms
starts_ms = [50, 100:50:900];  % start times in ms
ends_ms = [100, 200:50:1000];  % end times in ms (first +50 ms, rest +100 ms)
time_windows = [starts_ms(:)/1000, ends_ms(:)/1000];  % [start, end] in seconds
n_windows = size(time_windows, 1);

if ~exist(output_dir, 'dir'), mkdir(output_dir); end

%% Loop over subjects
for s = 1:length(subjects)
    fprintf('\n=== Processing %s (%d/%d) ===\n', subjects{s}, s, length(subjects));
    
    subj_dir = fullfile(db_path, subjects{s});
    files = dir(fullfile(subj_dir, 'results_dSPM*.mat'));
    
    if isempty(files)
        warning('No source files found for %s', subjects{s});
    end
    
    % Load first file to get time vector
    tmp = load(fullfile(subj_dir, files(1).name), 'Time');
    time_vec = tmp.Time;
    
    % Find sample indices for each time window
    win_idx = zeros(n_windows, 2);
    for w = 1:n_windows
        [~, win_idx(w,1)] = min(abs(time_vec - time_windows(w,1)));
        [~, win_idx(w,2)] = min(abs(time_vec - time_windows(w,2)));
    end
    
    fprintf('  Found %d trial files\n', length(files));
    
    % Initialize output matrix with NaN
    source_data = NaN(n_trials_total, n_grid, n_windows, 'single');
    trial_numbers = [];
    
    % Loop over trial files
    for f = 1:length(files)
        data = load(fullfile(subj_dir, files(f).name), 'ImageGridAmp', 'Comment');
        
        % Extract trial number from Comment field
        token = regexp(data.Comment, '#(\d+)', 'tokens');
        if isempty(token)
            warning('  Could not parse trial number from: %s', data.Comment);
        end
        trial_num = str2double(token{1}{1});
        trial_numbers(end+1) = trial_num;
        
        % Average within each time window
        for w = 1:n_windows
            source_data(trial_num, :, w) = single(mean(data.ImageGridAmp(:, win_idx(w,1):win_idx(w,2)), 2))';
        end
        
        if mod(f, 100) == 0
            fprintf('  Loaded %d/%d trials\n', f, length(files));
        end
    end
    
    fprintf('  Trial numbers range: %d to %d\n', min(trial_numbers), max(trial_numbers));
    fprintf('  Total valid trials: %d out of %d\n', length(trial_numbers), n_trials_total);
    
    % Save
    subj_id = regexp(subjects{s}, 'sub-(\d+)', 'tokens');
    subj_id = subj_id{1}{1};
    
    out_file = fullfile(output_dir, sprintf('sub-%s_source_data.mat', subj_id));
    save(out_file, 'source_data', 'trial_numbers', 'time_windows', '-v7.3');
    fprintf('  Saved: %s (%.1f MB)\n', out_file, dir(out_file).bytes / 1e6);
end

%% Save grid coordinates (once)
%tmp = load(fullfile(db_path, subjects{1}, files(1).name), 'GridLoc');
%grid_loc = tmp.GridLoc;
%save(fullfile(output_dir, 'grid_loc.mat'), 'grid_loc', '-v7.3');
%fprintf('\nGrid coordinates saved: %d points\n', size(grid_loc, 1));

%% Save time window info
save(fullfile(output_dir, 'time_windows.mat'), 'time_windows', '-v7.3');
fprintf('Time windows saved: %d windows\n', n_windows);
fprintf('\n=== Done! ===\n');
fprintf('Each file: %d trials x %d grid points x %d time windows\n', n_trials_total, n_grid, n_windows);
