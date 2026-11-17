import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy import stats
import keras
from bayesflow.adapters import Adapter

def analyze_multiple_subjects(data_dir, model_path, param_names, output_dir='./group_results2'):

    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} subject/condition files")
    
    # Load model
    print(f"Loading model from {model_path}")
    approximator = keras.saving.load_model(model_path)
    
    # Create adapter
    adapter = Adapter() \
        .to_array() \
        .convert_dtype('float64', 'float32') \
        .standardize('summary_variables') \
        .keep(['summary_variables'])
    
    # Storage for results
    all_subject_results = []
    all_posteriors = {}
    
    # Process each subject/condition
    for csv_file in csv_files:
        subject_id = os.path.basename(csv_file).split('_')[0]
        print(f"\nProcessing {subject_id}...")
        
        # Load data
        df = pd.read_csv(csv_file)
        
        # Extract features
        summary_vars = df[['RT', 'Choice_Correct', 'CPP_Slope', 'N200_Latency']].to_numpy(dtype=np.float32)
        
        # Remove NaNs
        valid_rows = ~np.isnan(summary_vars).any(axis=1)
        valid_summary = summary_vars[valid_rows]
        
        if len(valid_summary) < 150:
            print(f"  Warning: Only {len(valid_summary)} valid trials for {subject_id}, skipping")
            continue
            
        print(f"  Using {len(valid_summary)} valid trials for inference")
        
        # Reshape for BayesFlow
        valid_summary = valid_summary.reshape(1, *valid_summary.shape)
        
        # Adapt data
        conditions = adapter({'summary_variables': valid_summary})
        
        # Run inference
        posteriors = approximator.sample(conditions=conditions, num_samples=1000)
        
        # Stack posteriors
        posterior_array = np.stack([posteriors[k] for k in param_names], axis=-1)
        posterior_array = np.squeeze(posterior_array)
        
        # Calculate statistics
        mean_estimates = posterior_array.mean(axis=0)
        median_estimates = np.median(posterior_array, axis=0)
        std_estimates = np.std(posterior_array, axis=0)
        
        # Calculate HDIs
        hdis = [compute_hdi(posterior_array[:, i]) for i in range(len(param_names))]
        hdi_lows = [hdi[0] for hdi in hdis]
        hdi_highs = [hdi[1] for hdi in hdis]
        
        # Store results
        for i, param in enumerate(param_names):
            all_subject_results.append({
                'Subject': subject_id,
                'Parameter': param,
                'Mean': mean_estimates[i],
                'Median': median_estimates[i],
                'Std': std_estimates[i],
                'HDI_Low': hdi_lows[i],
                'HDI_High': hdi_highs[i]
            })
        
        # Store full posteriors
        all_posteriors[subject_id] = posterior_array
    
    if not all_subject_results:
        print("No valid results to analyze")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_subject_results)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'all_subject_parameters.csv'), index=False)
    
    # Plot parameter comparisons across subjects
    plot_parameter_comparisons(results_df, output_dir)
    
    # Perform statistical analyses
    perform_statistical_analyses(results_df, all_posteriors, output_dir)
    
    return results_df, all_posteriors

def compute_hdi(samples, credible_mass=0.95):
    """
    Compute Highest Density Interval from posterior samples
    """
    sorted_samples = np.sort(samples)
    ci_index = int(credible_mass * len(sorted_samples))
    n_intervals = len(sorted_samples) - ci_index
    interval_width = sorted_samples[ci_index:] - sorted_samples[:n_intervals]
    
    min_idx = np.argmin(interval_width)
    hdi_low = sorted_samples[min_idx]
    hdi_high = sorted_samples[min_idx + ci_index]
    
    return hdi_low, hdi_high

def plot_parameter_comparisons(results_df, output_dir):
    """
    Create visualizations comparing parameters across subjects/conditions
    """
    # Get unique parameters and subjects
    params = results_df['Parameter'].unique()
    subjects = results_df['Subject'].unique()
    
    # Create directory for parameter plots
    param_dir = os.path.join(output_dir, 'parameter_plots')
    os.makedirs(param_dir, exist_ok=True)
    
    # Plot each parameter across subjects
    for param in params:
        param_data = results_df[results_df['Parameter'] == param]
        
        plt.figure(figsize=(12, 6))
        
        # Create error bars
        plt.errorbar(
            x=param_data['Subject'],
            y=param_data['Median'],
            yerr=[
                param_data['Median'] - param_data['HDI_Low'],
                param_data['HDI_High'] - param_data['Median']
            ],
            fmt='o',
            capsize=5,
            elinewidth=2,
            markeredgewidth=2
        )
        
        plt.title(f'Parameter Comparison: {param}')
        plt.ylabel('Parameter Value')
        plt.xlabel('Subject/Condition')
        plt.grid(True, alpha=0.3)
        
        # Add overall mean as horizontal line
        plt.axhline(y=param_data['Median'].mean(), color='r', linestyle='--', 
                    label=f'Group Mean: {param_data["Median"].mean():.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(param_dir, f'{param}_comparison.png'), dpi=300)
        plt.close()
    
    # Create overview plot with all parameters
    plt.figure(figsize=(15, 10))
    
    # Prepare data for seaborn
    plot_data = pd.pivot_table(
        results_df, 
        values='Median', 
        index='Subject', 
        columns='Parameter'
    )
    
    # Standardize values for better visualization
    plot_data_std = (plot_data - plot_data.mean()) / plot_data.std()
    
    # Create heatmap
    sns.heatmap(plot_data_std, cmap='coolwarm', annot=True, fmt='.2f')
    plt.title('Standardized Parameter Values Across Subjects/Conditions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_heatmap.png'), dpi=300)
    plt.close()

def perform_statistical_analyses(results_df, all_posteriors, output_dir):
    """
    Perform statistical analyses on parameter distributions across subjects/conditions
    """
    # Create report file
    with open(os.path.join(output_dir, 'statistical_report.txt'), 'w') as f:
        f.write("STATISTICAL ANALYSIS REPORT\n")
        f.write("==========================\n\n")
        
        # 1. Basic descriptive statistics
        f.write("1. DESCRIPTIVE STATISTICS\n")
        f.write("------------------------\n")
        
        params = results_df['Parameter'].unique()
        for param in params:
            param_data = results_df[results_df['Parameter'] == param]
            f.write(f"\nParameter: {param}\n")
            f.write(f"  Mean across subjects: {param_data['Mean'].mean():.4f}\n")
            f.write(f"  Median across subjects: {param_data['Median'].mean():.4f}\n")
            f.write(f"  Min: {param_data['Median'].min():.4f}, Max: {param_data['Median'].max():.4f}\n")
            f.write(f"  Standard deviation across subjects: {param_data['Median'].std():.4f}\n")
            f.write(f"  Coefficient of variation: {param_data['Median'].std() / param_data['Median'].mean():.4f}\n")
        
        # 2. Parameter correlations across subjects
        f.write("\n\n2. PARAMETER CORRELATIONS\n")
        f.write("------------------------\n")
        # Aggregate if multiple entries exist per (Subject, Parameter)
        results_df = results_df.groupby(['Subject', 'Parameter'], as_index=False).mean()

        
        # Reshape data to wide format for correlation analysis
        wide_data = results_df.pivot(index='Subject', columns='Parameter', values='Median')
        
        # Calculate correlations
        correlations = wide_data.corr()
        
        # Save correlation matrix as CSV
        correlations.to_csv(os.path.join(output_dir, 'parameter_correlations.csv'))
        
        # Write important correlations to report
        f.write("\nStrong parameter correlations (|r| > 0.5):\n")
        for param1 in params:
            for param2 in params:
                if param1 >= param2:  # Avoid duplicates
                    continue
                corr = correlations.loc[param1, param2]
                if abs(corr) > 0.5:
                    f.write(f"  {param1} & {param2}: r = {corr:.4f}\n")
        
        # 3. Cluster analysis (simplified)
        f.write("\n\n3. SUBJECT CLUSTERING\n")
        f.write("--------------------\n")
        f.write("A clustering analysis could be performed here to identify groups of subjects\n")
        f.write("with similar parameter profiles.\n")
    
    print(f"Statistical analyses completed. Report saved to {output_dir}/statistical_report.txt")

# Usage example:
# analyze_multiple_subjects(
#     data_dir='path/to/data_files',
#     model_path='./checkpoints_gddm_joint/model1.keras',
#     param_names=PARAM_NAMES,
#     output_dir='./group_results'
# )
