import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def posterior_predictive_check(posterior_samples, param_names, likelihood_fn, real_data):
    """
    Perform posterior predictive checks by simulating data from posterior samples
    and comparing with observed data.
    
    Parameters:
    -----------
    posterior_samples : np.ndarray
        Array of posterior samples, shape (n_samples, n_params)
    param_names : list
        List of parameter names
    likelihood_fn : function
        Function that generates data given parameters
    real_data : dict
        Dictionary containing real behavioral and EEG data
    """
    n_samples = min(100, posterior_samples.shape[0])  # Use 100 posterior samples for efficiency
    selected_samples = posterior_samples[np.random.choice(posterior_samples.shape[0], n_samples, replace=False)]
    
    # Extract real data for comparison
    real_behavior = real_data['behavioral_data']
    real_eeg = real_data['eeg_data']
    
    # Storage for simulated data
    all_sim_rts = []
    all_sim_choices = []
    all_sim_n200 = []
    all_sim_cpp = []
    
    # Generate predicted data from each posterior sample
    for i in range(n_samples):
        params = dict(zip(param_names, selected_samples[i]))
        # Ensure all σ values are positive

        # Call likelihood function with unpacked parameters
        FIXED_Z = 0.5
        sim_data = likelihood_fn(
            params['v_base'], params['a_init'], params['t_nd'],
            params['leak'], params['collapse_rate'], params['gamma'], 
            params['sigma_cpp'], params['sigma_n200']
        )
        
        # Extract and store simulated data
        sim_behavior = sim_data['behavioral_data']
        sim_eeg = sim_data['eeg_data']

        all_sim_rts.append(sim_behavior[:, 0])
        all_sim_choices.append(sim_behavior[:, 1])
        all_sim_n200.append(sim_eeg[:, 0])
        all_sim_cpp.append(sim_eeg[:, 1])
    
    # Setup figure for PPC plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. RT distributions
    ax = axes[0, 0]
    for i in range(min(20, n_samples)):  # Plot first 20 for clarity
        sns.kdeplot(all_sim_rts[i], ax=ax, color='blue', alpha=0.1)
    sns.kdeplot(real_behavior[:, 0], ax=ax, color='red', linewidth=2, label='Observed')
    ax.set_title('Reaction Time Distributions')
    ax.set_xlabel('RT (s)')
    ax.legend()
    
    # 2. Choice proportions
    ax = axes[0, 1]
    sim_choice_props = [np.mean(choices) for choices in all_sim_choices]
    ax.hist(sim_choice_props, bins=20, alpha=0.7)
    ax.axvline(np.mean(real_behavior[:, 1]), color='red', linewidth=2, 
               label=f'Observed: {np.mean(real_behavior[:, 1]):.2f}')
    ax.set_title('Choice Proportions')
    ax.set_xlabel('Proportion of Choice 1')
    ax.legend()
    
    # 3. N200 Latency
    ax = axes[1, 0]
    for i in range(min(20, n_samples)):
        sns.kdeplot(all_sim_n200[i], ax=ax, color='blue', alpha=0.1)
    sns.kdeplot(real_eeg[:, 0], ax=ax, color='red', linewidth=2, label='Observed')
    ax.set_title('N200 Latency Distributions')
    ax.set_xlabel('Latency (s)')
    ax.legend()
    
    # 4. CPP Slope
    ax = axes[1, 1]
    for i in range(min(20, n_samples)):
        sns.kdeplot(all_sim_cpp[i], ax=ax, color='blue', alpha=0.1,warn_singular=False)
    sns.kdeplot(real_eeg[:, 1], ax=ax, color='red', linewidth=2, label='Observed')
    ax.set_title('CPP Slope Distributions')
    ax.set_xlabel('Slope')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('posterior_predictive_checks.png', dpi=300)
    plt.show()
    
    # Calculate summary statistics
    print("\nSummary Statistics Comparison:")
    print(f"{'Measure':<15} {'Observed':>10} {'Predicted (Mean)':>18} {'95% CI':>25}")
    print("-" * 70)
    
    # RT mean
    real_rt_mean = np.mean(real_behavior[:, 0])
    sim_rt_means = [np.mean(rts) for rts in all_sim_rts]
    sim_rt_mean = np.mean(sim_rt_means)
    sim_rt_ci = np.percentile(sim_rt_means, [2.5, 97.5])
    print(f"{'RT Mean':<15} {real_rt_mean:>10.3f} {sim_rt_mean:>18.3f} "
          f"{f'[{sim_rt_ci[0]:.3f}, {sim_rt_ci[1]:.3f}]':>25}")
    
    # Choice proportion
    real_choice = np.mean(real_behavior[:, 1])
    sim_choice = np.mean(sim_choice_props)
    sim_choice_ci = np.percentile(sim_choice_props, [2.5, 97.5])
    print(f"{'Choice Prop':<15} {real_choice:>10.3f} {sim_choice:>18.3f} "
          f"{f'[{sim_choice_ci[0]:.3f}, {sim_choice_ci[1]:.3f}]':>25}")
    
    # N200 Latency
    real_n200 = np.mean(real_eeg[:, 0])
    sim_n200_means = [np.mean(n200) for n200 in all_sim_n200]
    sim_n200_mean = np.mean(sim_n200_means)
    sim_n200_ci = np.percentile(sim_n200_means, [2.5, 97.5])
    print(f"{'N200 Latency':<15} {real_n200:>10.3f} {sim_n200_mean:>18.3f} "
          f"{f'[{sim_n200_ci[0]:.3f}, {sim_n200_ci[1]:.3f}]':>25}")
    
    # CPP Slope
    real_cpp = np.mean(real_eeg[:, 1])
    sim_cpp_means = [np.mean(cpp) for cpp in all_sim_cpp]
    sim_cpp_mean = np.mean(sim_cpp_means)
    sim_cpp_ci = np.percentile(sim_cpp_means, [2.5, 97.5])
    print(f"{'CPP Slope':<15} {real_cpp:>10.3f} {sim_cpp_mean:>18.3f} "
          f"{f'[{sim_cpp_ci[0]:.3f}, {sim_cpp_ci[1]:.3f}]':>25}")

# Usage in your script:
# posterior_predictive_check(posterior_array, PARAM_NAMES, likelihood, real_data_dict)
