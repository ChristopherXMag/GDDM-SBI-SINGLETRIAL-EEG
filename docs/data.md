# Data format and preparation

## Datasets used in the thesis

| Dataset | Task context | Source |
| --- | --- | --- |
| A | Value-based binary choices with simultaneous EEG-fMRI | Pisauro et al. (2017), [Neural correlates of evidence accumulation during value-based decisions revealed via simultaneous EEG-fMRI](https://doi.org/10.1038/ncomms15808) |
| B | Perceptual binary choices with EEG and mouse tracking; image-choice and word-choice tasks | Chen et al. (2022), [A resource for assessing dynamic binary choices in the adult brain using EEG and mouse-tracking](https://doi.org/10.1038/s41597-022-01538-5) |

Obtain recordings from the sources associated with those studies, following their terms and attribution requirements. This repository includes neither raw recordings nor the processed subject CSVs referenced by the notebooks.

## Feature tables

The published Python analysis code reads these case-sensitive column names:

| Column | Meaning | Representation |
| --- | --- | --- |
| `RT` | Reaction time | Seconds |
| `Choice_Correct` | Binary behavioral outcome | 0/1, interpreted consistently with the chosen model and task |
| `N200_Latency` | Single-trial N200 latency | Seconds |
| `CPP_Slope` | Single-trial CPP slope | Preserve the extraction method's scale and sign convention |

The CSV header can therefore be:

```csv
RT,Choice_Correct,N200_Latency,CPP_Slope
```

Each row represents one aligned trial. Keep EEG features and behavioral outcomes linked by trial identity when removing artifacts or missing values. The MATLAB preprocessing script exports preserved stimulus-event indices to support this alignment.

`Choice_Correct` should not automatically be interpreted as a left/right response. Check how a task's correctness or preference coding maps to the PyDDM simulator's upper/lower boundary coding.

## Feature order

The final joint simulator's adapter concatenates:

```text
[RT, choice, N200 latency, CPP slope]
```

Several empirical notebooks and the multi-subject helper select:

```text
[RT, Choice_Correct, CPP_Slope, N200_Latency]
```

Those last two columns are reversed relative to the training script. Review the selected checkpoint's training schema before reusing these cells. The behavior-only model expects two features, `[RT, choice]`; the initial loading cell in `recovery_companion_behavior.ipynb` still selects four.

## Preprocessing coverage

[`preprocessing/`](../preprocessing/) contains three MATLAB/EEGLAB scripts for the EGI dataset:

1. `convert_mff_to_set.m`: convert MFF recordings to EEGLAB SET files.
2. `preprocessing_with_trial_tracking.m`: filtering, channel handling, artifact processing, and preserved-event export. Its current subject loop processes only subject 1; ICA is commented out.
3. `epoch_and_save.m`: epoch SET files and apply baseline correction, using its own configured input directory.

These are individually configured research scripts, not a single coordinated pipeline. Check input/output paths and whether you are epoching raw or preprocessed SET files. The scripts do not themselves export the full N200/CPP feature tables consumed by Python.
