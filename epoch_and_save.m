clear; clc;
eeglab nogui;

% 根目录
root_dir = 'D:\python\mtcode\data2mousenew';
tasks = {'imagechoice', 'wordchoice'};

% 事件码（刺激）
stim_events = {'0400', '0401', '0402', '0403'};
epoch_window = [-0.2 0.8];  % 单位秒
baseline_window = [-200 0];  % 单位毫秒

for subj = 1:31
    subj_str = sprintf('sub-%02d', subj);
    
    for t = 1:length(tasks)
        task = tasks{t};
        
        % .set 文件路径
        set_path = fullfile(root_dir, subj_str, 'eeg', ...
            sprintf('%s_task-%s_eeg.set', subj_str, task));
        
        if ~isfile(set_path)
            fprintf('SKIPPED (not found): %s\n', set_path);
            continue;
        end
        
        fprintf('Processing: %s\n', set_path);
        EEG = pop_loadset('filename', set_path);

        % epoching
        EEG = pop_epoch(EEG, stim_events, epoch_window);
        EEG = pop_rmbase(EEG, baseline_window);

        % 保存到 eeg2 文件夹
        outdir = fullfile(root_dir, subj_str, 'eeg2');
        if ~exist(outdir, 'dir')
            mkdir(outdir);
        end
        outname = sprintf('%s_task-%s_epoched.set', subj_str, task);
        pop_saveset(EEG, 'filename', outname, 'filepath', outdir, 'savemode', 'onefile');
    end
end

fprintf('✅ 所有数据均已 epoch 并保存到 eeg2 文件夹。\n');
