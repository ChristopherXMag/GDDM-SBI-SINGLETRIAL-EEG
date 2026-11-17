# Model guide

## Final joint model

[`EEGGDDMfinal2.py`](../models/EEGGDDMfinal2.py) defines a subject-level model with 200 simulated trials per dataset instance, a 5-second behavioral horizon, and fixed diffusion noise of 1.0. The decision process combines state-dependent leaky drift with exponentially collapsing symmetric bounds.

The simulator returns two arrays, each of shape `(n_trials, 2)`:

- `behavioral_data`: `[RT, choice]`
- `eeg_data`: `[N200 latency, CPP slope]`

The adapter concatenates them in that order. The summary network is a `DeepSet` with summary dimension 128 and four layers; the inference network is a `CouplingFlow` with depth 10. The model infers eight parameters in the following order:

| Parameter | Meaning in the implementation | Uniform prior |
| --- | --- | --- |
| `v_base` | Baseline drift | [-3.0, 3.0] |
| `a_init` | Initial bound magnitude | [0.7, 2.0] |
| `t_nd` | Non-decision time, seconds | [0.25, 0.45] |
| `leak` | State-dependent drift coefficient | [0.0, 2.0] |
| `collapse_rate` | Exponential bound-collapse rate | [0.0, 1.0] |
| `gamma` | CPP feature scaling | [-2.0, 2.0] |
| `sigma_cpp` | Log-normal CPP noise scale | [3.0, 7.0] |
| `sigma_n200` | N200 Gaussian noise scale | [0.01, 0.04] |

These are the executable arrays in `PRIOR_LOWER` and `PRIOR_UPPER`; some historical inline comments list an extra `z` that is not inferred by this version.

N200 latency uses a saturating transform of non-decision time with Gaussian noise, clipped to `[0.15, 0.35]` seconds. CPP slope uses the absolute difference between late and midpoint trajectory averages, multiplied by `gamma` and log-normal noise, then clipped to `[-5, 5]`. Read the executable functions for the precise definitions; some old comments and console messages describe earlier settings.

Behavioral RT/choice samples come from a PyDDM solution, while EEG trajectories are generated in a separate simulation. The code does not derive each trial's RT and EEG features from one shared first-passage trajectory. This organization update retains that original construction.

## Behavior-only comparison

[`EEGGDDM_Behavonly2.py`](../models/EEGGDDM_Behavonly2.py) uses the same five behavioral parameter names and prior ranges as the final joint script, without the three EEG-specific parameters. It uses `[RT, choice]` as its summary variables and also simulates 200 trials per instance.

## Other variants

| Script | Parameter order | Main distinctions |
| --- | --- | --- |
| [`EEGGDDMfinal.py`](../models/EEGGDDMfinal.py) | `v_base, a_init, t_nd, leak, collapse_rate, gamma, sigma_cpp, sigma_n200` | `t_nd` prior is [0.10, 0.35]; retained as the alternative-prior joint model |
| [`EEGGDDM_Behavonly1.py`](../models/EEGGDDM_Behavonly1.py) | `v_base, a_init, z, t_nd, leak, collapse_rate` | 300 trials, wider/different priors, inferred starting bias, and an earlier network/training configuration |

The earlier behavior-only script starts training at module scope. Importing it can launch training; use it as a historical experiment rather than a utility module.

## Training outputs and saved checkpoints

Run training scripts with the repository root as the working directory. The final scripts write new models to `checkpoints_unified_joint/unified_joint_model.keras` and `checkpoints_unified_joint/unified_joint_model_behavior.keras`. The uploaded `*_final*.keras` files are separate saved artifacts; filenames alone do not establish the exact script, prior, or training run that produced them. See the [checkpoint inventory](../checkpoints_unified_joint/README.md).

Training and recovery settings are defined in the scripts and their `__main__` blocks. The commands launch simulation, training, and diagnostics; they are not quick inference demos.
