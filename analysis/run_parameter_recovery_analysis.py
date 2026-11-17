import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def run_parameter_recovery_analysis(approximator, simulator, adapter, param_names, n_sims=100):
    """
    Comprehensive parameter recovery analysis to validate the model's ability
    to recover known parameters.
    
    Parameters:
    -----------
    workflow : BayesFlow workflow object
        The trained workflow
    simulator : function
        The simulator function
    adapter : BayesFlow adapter
        The data adapter
    param_names : list
        List of parameter names
    n_sims : int
        Number of simulations to run
    """
    print(f"Running parameter recovery analysis with {n_sims} simulations...")
    
    # Generate simulations with known parameters
    recovery_data = simulator.sample(batch_size=n_sims)
    adapted = adapter(recovery_data)
    
    # Find valid simulations (no NaNs)
    valid_idx = [i for i, x in enumerate(adapted['summary_variables']) if not np.isnan(x).any()]
    
    if len(valid_idx) < 5:
        print(f"ERROR: Only {len(valid_idx)} valid simulations out of {n_sims}.")
        print("Check your simulator for stability issues.")
        return
    
    print(f"Found {len(valid_idx)} valid simulations out of {n_sims}.")
    
    # Extract true parameters
    true_params = adapted['inference_variables'][valid_idx]
    conditions = {k: v[valid_idx] for k, v in recovery_data.items() if k not in param_names}
    
    # Run inference
    posteriors = approximator.sample(conditions=conditions, num_samples=1000)
    posterior_array = np.stack([posteriors[k] for k in param_names], axis=-1)
    
    # Reshape if needed
    if posterior_array.ndim == 4 and posterior_array.shape[2] == 1:
        posterior_array = np.squeeze(posterior_array, axis=2)
    
    # Calculate point estimates
    estimates_mean = posterior_array.mean(axis=1)
    estimates_median = np.median(posterior_array, axis=1)
    
    # Calculate recovery metrics for each parameter
    metrics = []
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, name in enumerate(param_names):
        true_vals = true_params[:, i]
        est_vals_mean = estimates_mean[:, i]
        est_vals_median = estimates_median[:, i]
        
        # Calculate correlation and error metrics
        r_mean = np.corrcoef(true_vals, est_vals_mean)[0, 1]
        r_median = np.corrcoef(true_vals, est_vals_median)[0, 1]
        rmse_mean = np.sqrt(np.mean((true_vals - est_vals_mean)**2))
        rmse_median = np.sqrt(np.mean((true_vals - est_vals_median)**2))
        
        # Calculate calibration error
        calibration_err = calculate_calibration_error(posterior_array[:, :, i], true_vals)
        
        metrics.append({
            'param': name,
            'r_mean': r_mean,
            'r_median': r_median,
            'rmse_mean': rmse_mean,
            'rmse_median': rmse_median,
            'calibration_err': calibration_err
        })
        
        # Create recovery plot
        if i < len(axes):
            ax = axes[i]
            ax.scatter(true_vals, est_vals_median, alpha=0.7)
            
            # Add error bars for uncertainty
            for j in range(min(50, len(true_vals))):  # Plot first 50 points with error bars
                posterior_samples = posterior_array[j, :, i]
                ci = np.percentile(posterior_samples, [2.5, 97.5])
                ax.plot([true_vals[j], true_vals[j]], ci, color='gray', alpha=0.3)
            
            # Plot identity line
            min_val = min(np.min(true_vals), np.min(est_vals_median))
            max_val = max(np.max(true_vals), np.max(est_vals_median))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--')
            
            ax.set_title(f'{name}: r = {r_median:.3f}, RMSE = {rmse_median:.3f}')
            ax.set_xlabel('True Parameter')
            ax.set_ylabel('Estimated Parameter (Median)')
    
    plt.tight_layout()
    plt.savefig('parameter_recovery.png', dpi=300)
    plt.show()
    
    # Print metrics table
    print("\nParameter Recovery Metrics:")
    print(f"{'Parameter':<15} {'r (Mean)':>10} {'r (Median)':>10} {'RMSE (Mean)':>12} {'RMSE (Median)':>12} {'Calib. Err':>10}")
    print("-" * 75)
    for m in metrics:
        print(f"{m['param']:<15} {m['r_mean']:>10.3f} {m['r_median']:>10.3f} {m['rmse_mean']:>12.3f} {m['rmse_median']:>12.3f} {m['calibration_err']:>10.3f}")
    
    # Calculate and plot overall recovery metrics
    print("\nOverall Recovery Performance:")
    mean_r = np.mean([m['r_median'] for m in metrics])
    mean_rmse = np.mean([m['rmse_median'] for m in metrics])
    mean_calib = np.mean([m['calibration_err'] for m in metrics])
    print(f"Average correlation (r): {mean_r:.3f}")
    print(f"Average RMSE: {mean_rmse:.3f}")
    print(f"Average calibration error: {mean_calib:.3f}")
    
    return metrics

def calculate_calibration_error(posterior_samples, true_values):
    """
    Calculate calibration error for a parameter.
    Lower values indicate better calibrated posteriors.
    """
    # Calculate empirical coverage probabilities for different credible intervals
    alphas = np.linspace(0.1, 0.9, 9)
    empirical_coverage = []
    
    for alpha in alphas:
        lower = (1 - alpha) / 2
        upper = 1 - lower
        
        # Calculate credible intervals for each simulation
        cis = np.array([np.percentile(posterior_samples[i], [lower*100, upper*100]) 
                        for i in range(len(true_values))])
        
        # Count how many true values fall within their respective CI
        covered = np.sum((true_values >= cis[:, 0]) & (true_values <= cis[:, 1]))
        empirical_coverage.append(covered / len(true_values))
    
    # Calibration error is RMSE between empirical and nominal coverage
    calibration_error = np.sqrt(np.mean((empirical_coverage - alphas) ** 2))
    return calibration_error

# Usage in your script:
# run_parameter_recovery_analysis(workflow, simulator, adapter, PARAM_NAMES)
