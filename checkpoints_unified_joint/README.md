# Original trained checkpoints

All four uploaded `.keras` archives are preserved byte for byte from the original public repository. Their new directory matches the relative checkpoint paths used by most analysis notebooks.

| Archive | Family indicated by filename | Keras metadata: saved date | Keras version |
| --- | --- | --- | --- |
| [`unified_joint_model_final2.keras`](unified_joint_model_final2.keras) | Joint EEG + behavior | 2025-05-29 | 3.10.0 |
| [`unified_joint_model_final.keras`](unified_joint_model_final.keras) | Joint EEG + behavior | 2025-05-30 | 3.10.0 |
| [`unified_joint_model_behavior_final2.keras`](unified_joint_model_behavior_final2.keras) | Behavior only | 2025-05-29 | 3.10.0 |
| [`unified_joint_model_behavior_final.keras`](unified_joint_model_behavior_final.keras) | Behavior only | 2025-05-30 | 3.10.0 |

Dates above are read from `metadata.json` inside each archive; they are not new training or validation dates. The archive configuration identifies BayesFlow `ContinuousApproximator` models.

A full per-run record of the simulator revision, prior, dependency versions, and random seeds is not bundled. In particular, the higher filename suffix does not indicate a later saved date. Verify model input/output schema and simulator compatibility before sampling or performing posterior predictive checks.

The current training scripts write to `unified_joint_model.keras` and `unified_joint_model_behavior.keras`, rather than overwriting these four historical files. See [setup and reproduction](../docs/reproduction.md).
