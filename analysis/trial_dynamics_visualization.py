import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def visualize_trial_dynamics(params, n_trials=10, n_timepoints=200, save_path=None, z=0.5):
    """
    Visualize detailed trial-level dynamics based on inferred model parameters.

    Parameters:
    -----------
    params : dict
        Dictionary of model parameters (from posterior mean/median)
    n_trials : int
        Number of trials to visualize
    n_timepoints : int
        Number of timepoints per trial
    save_path : str or None
        Path to save the figure, if None the figure is displayed
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Extract parameters
    v_base = params['v_base']
    a_init = params['a_init']
    t_nd = params['t_nd']
    leak = params['leak']
    collapse_rate = params['collapse_rate']
    gamma = params['gamma']
    sigma_cpp = params['sigma_cpp']
    sigma_n200 = params['sigma_n200']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Simulate decision trajectories
    timepoints = np.linspace(0, 1, n_timepoints)
    trajectories = np.zeros((n_trials, n_timepoints))
    v_trial = np.random.normal(v_base, 0.2, size=n_trials)
    start_points = (z - 0.5) * a_init * 2
    current_pos = np.ones(n_trials) * start_points
    upper_bounds = np.zeros((n_timepoints, n_trials))
    lower_bounds = np.zeros((n_timepoints, n_trials))

    for t_idx, t in enumerate(timepoints):
        current_bound = a_init * np.exp(-collapse_rate * t)
        upper_bounds[t_idx] = current_bound
        lower_bounds[t_idx] = -current_bound
        if t_idx == 0:
            trajectories[:, t_idx] = current_pos
            continue
        drift = v_trial - leak * current_pos
        noise = np.random.normal(0, 1.0 * np.sqrt(1 / n_timepoints), size=n_trials)
        current_pos += drift / n_timepoints + noise
        current_pos = np.clip(current_pos, -current_bound, current_bound)
        trajectories[:, t_idx] = current_pos

    ax = axes[0, 0]
    for i in range(n_trials):
        color = 'green' if trajectories[i, -1] > 0 else 'red'
        ax.plot(timepoints, trajectories[i], color=color, alpha=0.7)
    for t_idx in range(n_timepoints):
        ax.plot(timepoints[t_idx] * np.ones(n_trials), upper_bounds[t_idx], 'k.', markersize=1)
        ax.plot(timepoints[t_idx] * np.ones(n_trials), lower_bounds[t_idx], 'k.', markersize=1)
    ax.set_title('Decision Trajectories with Collapsing Bounds')
    ax.set_xlabel('Time (normalized)')
    ax.set_ylabel('Decision Variable')
    ax.grid(True, alpha=0.3)

    # RT distributions by choice
    ax = axes[0, 1]
    n_dist_trials = 200
    v_trial_dist = np.random.normal(v_base, 0.2, size=n_dist_trials)
    rts = np.random.gamma(shape=2, scale=0.3, size=n_dist_trials)
    rts = rts * (1 + 0.1 * (a_init - 1.5)) + t_nd
    choices = np.random.binomial(1, 1 / (1 + np.exp(-v_trial_dist)), size=n_dist_trials)
    sns.histplot(x=rts[choices == 1], color='green', alpha=0.5, label='Upper Bound', ax=ax)
    sns.histplot(x=rts[choices == 0], color='red', alpha=0.5, label='Lower Bound', ax=ax)
    ax.set_title('RT Distributions by Choice')
    ax.set_xlabel('Response Time (s)')
    ax.set_ylabel('Count')
    ax.legend()

    # N200 and CPP Relationships
    ax = axes[1, 0]
    n200_latencies = t_nd * 0.6 + np.random.normal(0, sigma_n200, size=n_dist_trials)
    n200_latencies = np.clip(n200_latencies, 0.05, 0.8)
    cpp_slopes = v_trial_dist * gamma * np.random.lognormal(0, sigma_cpp, size=n_dist_trials)
    cpp_slopes = np.clip(cpp_slopes, -5.0, 5.0)
    ax.scatter(rts, cpp_slopes, c=choices, cmap='coolwarm', alpha=0.7)
    ax.set_title('Relationship: RT vs CPP Slope')
    ax.set_xlabel('Response Time (s)')
    ax.set_ylabel('CPP Slope')
    ax.grid(True, alpha=0.3)

    # N200 Latency vs RT
    ax = axes[1, 1]
    ax.scatter(n200_latencies, rts, c=choices, cmap='coolwarm', alpha=0.7)
    ax.set_title('Relationship: N200 Latency vs RT')
    ax.set_xlabel('N200 Latency (s)')
    ax.set_ylabel('Response Time (s)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def visualize_parameter_dependencies(posterior_samples, param_names):
    """
    Visualize how the inferred parameters depend on each other.
    
    Parameters:
    -----------
    posterior_samples : np.ndarray
        Array of posterior samples, shape (n_samples, n_params)
    param_names : list
        List of parameter names
    """
    n_params = len(param_names)
    
    # Create grid of plots
    fig, axes = plt.subplots(n_params, n_params, figsize=(20, 20))
    
    # For each parameter pair
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:  # Diagonal: show marginal distributions
                sns.histplot(posterior_samples[:, i], kde=True, ax=ax)
                ax.set_title(param_names[i])
                ax.set_xlabel('')
                if i < n_params - 1:
                    ax.set_xticklabels([])
            else:  # Off-diagonal: show joint distributions
                sns.scatterplot(
                    x=posterior_samples[:, j], 
                    y=posterior_samples[:, i],
                    alpha=0.3,
                    ax=ax
                )
                
                # Add correlation coefficient
                corr = np.corrcoef(posterior_samples[:, j], posterior_samples[:, i])[0, 1]
                ax.text(0.05, 0.95, f'r = {corr:.2f}', 
                        transform=ax.transAxes, 
                        verticalalignment='top',
                        fontsize=10)
                
                if i < n_params - 1:
                    ax.set_xticklabels([])
                if j > 0:
                    ax.set_yticklabels([])
            
            if i == n_params - 1:
                ax.set_xlabel(param_names[j])
            if j == 0:
                ax.set_ylabel(param_names[i])
    
    plt.tight_layout()
    plt.savefig('parameter_dependencies.png', dpi=300)
    plt.show()

# Usage example:
# params = dict(zip(PARAM_NAMES, mean_estimates))
# visualize_trial_dynamics(params, save_path='trial_dynamics.png')
# visualize_parameter_dependencies(posterior_array, PARAM_NAMES)
