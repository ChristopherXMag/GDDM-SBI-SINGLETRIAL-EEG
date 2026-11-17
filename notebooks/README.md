# Analysis notebooks

These are the original thesis notebooks, including their saved outputs. Begin with the [setup instructions](../docs/reproduction.md#opening-the-original-notebooks) before running cells in the new directory layout.

| Notebook | Main contents |
| --- | --- |
| [`precovery_and_multiple_subjects.ipynb`](precovery_and_multiple_subjects.ipynb) | Joint-model recovery, illustrative accumulator figures, and multi-subject analysis cells |
| [`posterior_pre_companion.ipynb`](posterior_pre_companion.ipynb) | Empirical posterior inference and predictive-check calls using `unified_joint_model_final2.keras` |
| [`posterior_pre_companion_fin1.ipynb`](posterior_pre_companion_fin1.ipynb) | Parallel posterior/predictive analysis using `unified_joint_model_final.keras` |
| [`posterior_visual_companion.ipynb`](posterior_visual_companion.ipynb) | Posterior estimates and illustrative trial-dynamics visualizations |
| [`recovery_companion_behavior.ipynb`](recovery_companion_behavior.ipynb) | Behavior-only model setup and recovery analysis |

The names `fin1` and `final2` are historical labels. Match a checkpoint to its training schema and simulator rather than inferring provenance from its filename. The notebooks contain absolute data paths, some unpublished-module imports, and feature-selection differences documented in the [reproduction notes](../docs/reproduction.md).
