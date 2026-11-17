# Original-to-current file map

The repository was reorganized on November 17, 2026. Every original scientific file listed below retains its original contents, including notebook outputs and trained model bytes.

The original flat layout is available at [commit `d0f241f`](https://github.com/ChristopherXMag/GDDM-SBI-SINGLETRIAL-EEG/tree/d0f241f0de70e2ef2279cb4498aeced05315fcc4). The root README has been rewritten; the new guides and dependency inventory were added during the cleanup.

| Original root path | Current path |
| --- | --- |
| `EEGGDDM_Behavonly1.py` | [`models/EEGGDDM_Behavonly1.py`](../models/EEGGDDM_Behavonly1.py) |
| `EEGGDDM_Behavonly2.py` | [`models/EEGGDDM_Behavonly2.py`](../models/EEGGDDM_Behavonly2.py) |
| `EEGGDDMfinal.py` | [`models/EEGGDDMfinal.py`](../models/EEGGDDMfinal.py) |
| `EEGGDDMfinal2.py` | [`models/EEGGDDMfinal2.py`](../models/EEGGDDMfinal2.py) |
| `analyze_multiple_subjects2.py` | [`analysis/analyze_multiple_subjects2.py`](../analysis/analyze_multiple_subjects2.py) |
| `convert_mff_to_set.m` | [`preprocessing/convert_mff_to_set.m`](../preprocessing/convert_mff_to_set.m) |
| `epoch_and_save.m` | [`preprocessing/epoch_and_save.m`](../preprocessing/epoch_and_save.m) |
| `posterior_pre_companion.ipynb` | [`notebooks/posterior_pre_companion.ipynb`](../notebooks/posterior_pre_companion.ipynb) |
| `posterior_pre_companion_fin1.ipynb` | [`notebooks/posterior_pre_companion_fin1.ipynb`](../notebooks/posterior_pre_companion_fin1.ipynb) |
| `posterior_predictive_check.py` | [`analysis/posterior_predictive_check.py`](../analysis/posterior_predictive_check.py) |
| `posterior_visual_companion.ipynb` | [`notebooks/posterior_visual_companion.ipynb`](../notebooks/posterior_visual_companion.ipynb) |
| `precovery_and_multiple_subjects.ipynb` | [`notebooks/precovery_and_multiple_subjects.ipynb`](../notebooks/precovery_and_multiple_subjects.ipynb) |
| `preprocessing_with_trial_tracking.m` | [`preprocessing/preprocessing_with_trial_tracking.m`](../preprocessing/preprocessing_with_trial_tracking.m) |
| `recovery_companion_behavior.ipynb` | [`notebooks/recovery_companion_behavior.ipynb`](../notebooks/recovery_companion_behavior.ipynb) |
| `run_parameter_recovery_analysis.py` | [`analysis/run_parameter_recovery_analysis.py`](../analysis/run_parameter_recovery_analysis.py) |
| `trial_dynamics_visualization.py` | [`analysis/trial_dynamics_visualization.py`](../analysis/trial_dynamics_visualization.py) |
| `unified_joint_model_behavior_final.keras` | [`checkpoints_unified_joint/unified_joint_model_behavior_final.keras`](../checkpoints_unified_joint/unified_joint_model_behavior_final.keras) |
| `unified_joint_model_behavior_final2.keras` | [`checkpoints_unified_joint/unified_joint_model_behavior_final2.keras`](../checkpoints_unified_joint/unified_joint_model_behavior_final2.keras) |
| `unified_joint_model_final.keras` | [`checkpoints_unified_joint/unified_joint_model_final.keras`](../checkpoints_unified_joint/unified_joint_model_final.keras) |
| `unified_joint_model_final2.keras` | [`checkpoints_unified_joint/unified_joint_model_final2.keras`](../checkpoints_unified_joint/unified_joint_model_final2.keras) |

## Running files after the move

Run model scripts from the repository root, for example `python models/EEGGDDMfinal2.py`. Use the notebook import/working-directory setup in [reproduction notes](reproduction.md#opening-the-original-notebooks) when running the preserved notebooks. Old links to files at the repository root can be replaced using this map.
