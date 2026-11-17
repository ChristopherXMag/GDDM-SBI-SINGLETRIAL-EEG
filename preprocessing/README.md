# EEG preprocessing

The original MATLAB scripts support parts of the EGI image-choice and word-choice dataset workflow. They require EEGLAB, relevant import/cleaning plugins, dataset-specific event codes, and configured local paths.

| Script | Role |
| --- | --- |
| [`convert_mff_to_set.m`](convert_mff_to_set.m) | MFF-to-SET conversion |
| [`preprocessing_with_trial_tracking.m`](preprocessing_with_trial_tracking.m) | Filtering, channel/artifact processing, and preserved stimulus-event indices |
| [`epoch_and_save.m`](epoch_and_save.m) | Stimulus-locked epochs and baseline correction |

Each script has its own input/output settings. Check these before execution and preserve trial identities when joining EEG with behavior. See [data preparation](../docs/data.md#preprocessing-coverage) for coverage and current subject-loop settings.
