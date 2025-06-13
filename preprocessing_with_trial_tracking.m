
% ref: https://sccn.ucsd.edu/wiki/Makoto%27s_useful_EEGLAB_code
clear;clc;
eeglab nogui;
setenv('CUDA_CACHE_MAXSIZE', '536870912');

%% set path
bids_path = 'D:\python\mtcode\data2mousenew';
output_path = fullfile('..', 'data');
eeg_preproc_path = fullfile(output_path, 'EGI_PROC');
log_path = fullfile(output_path, 'eeg_log');

if exist(log_path, 'dir')
    rmdir(log_path, 's');
end
mkdir(log_path);

montage_path = 'D:\MATLAB\eeglab2024.0\functions\supportfiles\channel_location_files\philips_neuro\GSN-HydroCel-129.sfp';


wordtask = "wordchoice";
imagetask = "imagechoice";

fileID = fopen(fullfile(output_path, 'exp_log.txt'), 'w');

for task = [wordtask imagetask]
    fprintf(fileID, '\n-------------------------------- %s --------------------------------\n', task);

    for i = 1:1
        fprintf(fileID, 'sub-%02d, ', i);
        subj_dir = fullfile(bids_path, sprintf('sub-%02d', i));
        input_path = fullfile(subj_dir, 'eeg');
        f_path = fullfile(input_path, sprintf('sub-%02d_task-%s_eeg.set', i, task));
        outname = sprintf('sub-%02d_task-%s_eeg.set', i, task);
        sprintf('------------------- Processing %s -------------------', f_path);

        %% import data
        EEG = pop_loadset(f_path);

        %% load channel location
        EEG = pop_chanedit(EEG, 'load', {montage_path, 'filetype', 'autodetect'}, 'changefield', {132, 'labels', 'E129'});

        %% resample
        EEG = pop_resample(EEG, 100);

        %% filter
        EEG = pop_eegfiltnew(EEG, 'locutoff', 0.1);
        EEG = pop_eegfiltnew(EEG, 'hicutoff', 30);

        %% detect 20Hz noise
        fig = figure;
        [spectra, freq] = pop_spectopo(EEG, 1, [0 EEG.xmax * 1000], 'EEG', 'percent', 50, 'freq', [6 10 22], 'freqrange', [0.1 30], 'electrodes', 'off');
        saveas(fig, fullfile(log_path, sprintf('task-%s_sub-%02d_eeg', task, i)), 'png');
        close;
        [~, locs] = findpeaks(mean(spectra), freq, 'MinPeakProminence', 2);
        if any(locs == 20)
            fprintf(fileID, '20Hz noise, ');
        end

        %% remove useless channels
        EEG = pop_select(EEG, 'nochannel', {'E125', 'E128', 'E43', 'E48', 'E49', 'E56', 'E63', ...
            'E68', 'E73', 'E81', 'E88', 'E94', 'E99', 'E107', 'E113', 'E120', 'E119', ...
            'E1', 'E8', 'E14', 'E17', 'E21', 'E25', 'E32', 'E38', 'E121', 'E126', 'E127'});

        %% select valid stimulus-response pairs
        stim_mark = ["0400", "0401", "0402", "0403"];
        if task == "wordchoice"
            resp_mark = ["0500", "0503", "0505", "0506"];
        else
            resp_mark = ["0500", "0501"];
        end


        select_list = [];
        for index = 1:length(EEG.event) - 1
            if any(find(stim_mark == EEG.event(index).type)) && any(find(resp_mark == EEG.event(index + 1).type))
                select_list(end + 1) = index;
                select_list(end + 1) = index + 1;
            end
        end

        EEG = pop_selectevent(EEG, 'event', select_list, 'deleteevents', 'on');
        EEG = eeg_checkset(EEG, 'eventconsistency');

        %% extract preserved trial indices
        preserved_stim_indices = [];
        for e = 1:length(EEG.event)
            if ismember(EEG.event(e).type, stim_mark)
                preserved_stim_indices(end + 1) = EEG.event(e).urevent;
            end
        end

        %% export preserved indices as CSV
        index_out_path = fullfile(subj_dir, 'eeg2');
        if ~exist(index_out_path, 'dir')
            mkdir(index_out_path);
        end
        csvwrite(fullfile(index_out_path, sprintf('sub-%02d_task-%s_preserved_trials.csv', i, task)), preserved_stim_indices');

        %% remove artifacts
        ch_E129 = EEG.chanlocs(end);
        EEG = pop_select(EEG, 'nochannel', {'E129'});
        originalEEG = EEG;
        EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion', 5, 'ChannelCriterion', 0.8, 'LineNoiseCriterion', 4, 'Highpass', 'off', ...
            'BurstCriterion', 'off', 'WindowCriterion', 'off', 'BurstRejection', 'off', 'Distance', 'Euclidian');
        fprintf(fileID, 'reserved bad channel: %d/%d, ', EEG.nbchan, originalEEG.nbchan);

        %% interpolate and re-reference
        EEG = pop_interp(EEG, originalEEG.chanlocs, 'spherical');
        EEG.chanlocs(end + 1) = ch_E129;
        EEG.nbchan = EEG.nbchan + 1;
        EEG.data(end + 1, :) = zeros(1, EEG.pnts);
        EEG = pop_reref(EEG, []);

        %% reject bad segments by ASR
        EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion', 'off', 'ChannelCriterion', 'off', 'LineNoiseCriterion', 'off', ...
            'Highpass', 'off', 'BurstCriterion', 100, 'WindowCriterion', 'off', 'BurstRejection', 'on', 'Distance', 'Euclidian');

        %% ICA and artifact removal
        % EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1);
        % ica_components = size(EEG.icaweights, 1);
        % EEG = pop_iclabel(EEG, 'default');
        % EEG = pop_icflag(EEG, [NaN NaN; 0.7 1; 0.7 1; 0.7 1; 0.7 1; 0.7 1; NaN NaN]);
        % EEG = pop_subcomp(EEG, [], 0, 0);
        % fprintf(fileID, 'reserved components: %d/%d, ', size(EEG.icaweights, 1), ica_components);
        fprintf(fileID, 'ICA skipped\n');


        %% save continuous data and epoch
        EEG = pop_saveset(EEG, 'filename', outname, 'filepath', index_out_path, 'savemode', 'onefile');
        EEG = pop_epoch(EEG, cellstr(stim_mark), [-0.2 0.8], 'newname', outname(1:end - 4), 'epochinfo', 'yes');
        EEG = pop_rmbase(EEG, [-200 0], []);
        fprintf(fileID, 'reserved trials: %d\n', EEG.trials);
    end
end

fclose(fileID);
