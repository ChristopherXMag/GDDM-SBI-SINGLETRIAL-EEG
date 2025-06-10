# %% Updated Joint Modeling of EEG and Behavioral Data in GDDM with BayesFlow

import os
import numpy as np
import matplotlib.pyplot as plt
import keras
import torch
import bayesflow as bf

from bayesflow.simulators import make_simulator
from bayesflow.adapters import Adapter
from bayesflow.networks import DeepSet, CouplingFlow
from bayesflow.workflows import BasicWorkflow
from bayesflow.diagnostics.plots import recovery

from pyddm import Model, ICPoint
from pyddm.models import Drift, Noise, Bound, Overlay
from pyddm.models.overlay import OverlayNonDecision

# %% Global Configuration
PARAM_NAMES = ['v_base', 'a_init', 'z', 't_nd', 'leak', 'collapse_rate']
N_PARAMS = len(PARAM_NAMES)
N_TRIALS = 300
DT = 0.01
T_DUR = 5.0
NOISE_CONST = 1.0

PRIOR_LOWER = np.array([-5.0,  0.5, 0.3, 0.1, 0.0, 0.0])
PRIOR_UPPER = np.array([ 5.0,  3.0, 0.7, 0.8, 2.0, 1.0])

CHECKPOINT_DIR = './checkpoints_gddm_joint'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'model1behavior.keras')

# %% Prior
def prior():
    theta = np.random.uniform(PRIOR_LOWER, PRIOR_UPPER).astype(np.float32)
    return dict(zip(PARAM_NAMES, theta))

# %% Custom DDM Components
class DriftLeaky(Drift):
    name = "Leaky Drift"
    required_parameters = ["v_base", "leak"]
    def get_drift(self, x, t, conditions, **kwargs):
        return np.clip(self.v_base - self.leak * x, -5.0, 5.0)

class BoundExpCollapse(Bound):
    name = "Exponential Collapse"
    required_parameters = ["a_init", "collapse_rate"]
    def get_bound(self, t, conditions, **kwargs):
        return np.clip(self.a_init * np.exp(-self.collapse_rate * t), 0.01, 3.0)

class NoiseFixed(Noise):
    name = "Fixed Noise"
    required_parameters = ["noise"]
    def get_noise(self, x, t, conditions, **kwargs):
        return self.noise

# %% Behavioral Data Wrapper
def fixed_behavioral_data(sol, sample, n):
    rt_up = np.array(sample.choice_upper)
    rt_lo = np.array(sample.choice_lower)
    rt = np.concatenate([rt_up, rt_lo])
    choice = np.concatenate([np.ones(len(rt_up)), np.zeros(len(rt_lo))])
    total = len(rt)
    while total < n:
        additional_sample = sol.sample(n - total)
        rt_add_up = np.array(additional_sample.choice_upper)
        rt_add_lo = np.array(additional_sample.choice_lower)
        rt_add = np.concatenate([rt_add_up, rt_add_lo])
        choice_add = np.concatenate([np.ones(len(rt_add_up)), np.zeros(len(rt_add_lo))])
        rt = np.concatenate([rt, rt_add])
        choice = np.concatenate([choice, choice_add])
        total = len(rt)
    rt = rt[:n]
    choice = choice[:n]
    return np.stack([rt, choice], axis=1).astype(np.float32)

# %% Likelihood Function with Cross-Participant EEG Noise

def likelihood(v_base, a_init, z, t_nd, leak, collapse_rate, n_trials=N_TRIALS):
    model = Model(
        drift=DriftLeaky(v_base=v_base, leak=leak),
        noise=NoiseFixed(noise=NOISE_CONST),
        bound=BoundExpCollapse(a_init=a_init, collapse_rate=collapse_rate),
        IC=ICPoint(x0=z * a_init - a_init / 2.0),
        overlay=OverlayNonDecision(nondectime=t_nd),
        dx=0.01, dt=DT, T_dur=T_DUR
    )
    sol = model.solve()
    samp = sol.sample(n_trials)
    behav = fixed_behavioral_data(sol, samp, n_trials)

    return {'behavioral_data': behav}

# %% Simulator
simulator = make_simulator([prior, likelihood])

# %% Adapter
adapter = Adapter() \
    .to_array() \
    .convert_dtype('float64', 'float32') \
    .rename('behavioral_data', 'summary_variables') \
    .concatenate(PARAM_NAMES, into='inference_variables', axis=-1) \
    .standardize('summary_variables') \
    .keep(['summary_variables', 'inference_variables'])

# %% Networks
summary_net = DeepSet(summary_dim=128)
inference_net = CouplingFlow(
    depth=6,
    input_dim=N_PARAMS,
    subnet='mlp',
    permutation='random',
    use_actnorm=True,
    base_distribution='normal'
)

# %% Workflow
workflow = BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    summary_network=summary_net,
    inference_network=inference_net,
    checkpoint_path=CHECKPOINT_PATH,
    max_to_keep=3
)

# %% Training
optimizer = keras.optimizers.Adam(learning_rate=1e-4)
history = workflow.fit_online(
    epochs=10,
    simulations_per_epoch=10000,
    batch_size=32,
    optimizer=optimizer,
    show_progress=True
)
workflow.approximator.save(filepath=CHECKPOINT_PATH, overwrite=True)

# %% Parameter Recovery
print("\nRunning Parameter Recovery...")
recovery_data = workflow.simulator.sample(batch_size=500)
adapted = adapter(recovery_data)
valid_idx = [i for i, x in enumerate(adapted['summary_variables']) if not np.isnan(x).any()]

true_params = adapted['inference_variables'][valid_idx]
conditions = {k: v[valid_idx] for k, v in recovery_data.items() if k not in PARAM_NAMES}

if len(valid_idx):
    posteriors = workflow.sample(conditions=conditions, num_samples=1000)
    posterior_array = np.stack([posteriors[k] for k in PARAM_NAMES], axis=-1)
    if posterior_array.ndim == 4 and posterior_array.shape[2] == 1:
        posterior_array = np.squeeze(posterior_array, axis=2)

    estimates_mean = posterior_array.mean(axis=1)
    for i, name in enumerate(PARAM_NAMES):
        true_vals = true_params[:, i]
        est_vals = estimates_mean[:, i]
        r = np.corrcoef(true_vals, est_vals)[0, 1]
        print(f"{name:>15}: r = {r:.3f}")

    recovery(
        estimates=posterior_array,
        targets=true_params,
        variable_names=PARAM_NAMES,
        point_agg=np.median
    )
    plt.tight_layout()
    plt.show()
else:
    print("No valid simulations.")
