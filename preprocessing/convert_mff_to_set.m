clear; clc;
eeglab nogui;

% Set paths
bids_root = 'D:\python\mtcode\data2raweeg';  % adjust if different
output_root = 'D:\python\mtcode\data2mousenew';

% Tasks you actually have
tasks = {'imagechoice', 'wordchoice'};

for subj = 1:31
    for t = 1:length(tasks)
        task = tasks{t};
        
        subj_str = sprintf('sub-%02d', subj);
        mff_dir = fullfile(bids_root, sprintf('%s_task-%s_eeg.mff', subj_str, task));

        % Confirm the .mff folder exists
        if ~isfolder(mff_dir)
            fprintf('SKIPPED: %s (not found)\n', mff_dir);
            continue;
        end

        fprintf('Converting %s...\n', mff_dir);
        EEG = pop_mffimport(mff_dir, 'code', 0, 0);

        % Set output filename/path
        output_dir = fullfile(output_root, subj_str, 'eeg');
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end

        outname = sprintf('%s_task-%s_eeg.set', subj_str, task);
        pop_saveset(EEG, 'filename', outname, 'filepath', output_dir, 'savemode', 'onefile');
    end
end

fprintf('All available .mff files converted to .set.\n');
