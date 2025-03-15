import pandas as pd
import numpy as np
from scipy import stats
from numba import jit
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import statsmodels
from statsmodels.stats.multitest import multipletests
import time

def calculate_indval_metrics(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_permutations: int = 999,
) -> pd.DataFrame:
    """
    Calculate IndVal (APCF) metrics for each species across habitat types.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Species presence matrix (sites × species)
    y : pd.Series
        Habitat type labels
    n_permutations : int
        Number of permutations for significance testing
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with IndVal results for each species
    """
    print(f"Calculating IndVal metrics with {n_permutations} permutations...")
    start_time = time.time()
    
    # Convert to numpy arrays for faster computation
    X_np = X.values.T  # Transpose to species × sites
    y_np = y.values
    
    # Get unique habitat types (clusters)
    unique_clusters = np.unique(y_np)
    g = len(unique_clusters)
    
    # Map cluster values to sequential indices
    cluster_map = {val: idx for idx, val in enumerate(unique_clusters)}
    clusters_idx = np.array([cluster_map[val] for val in y_np])
    
    # Get IndVal metrics using APCF formula
    species_count, sites_count = X_np.shape
    
    # Initialize matrices
    A = np.zeros((species_count, g))  # Specificity
    B = np.zeros((species_count, g))  # Fidelity
    indval_matrix = np.zeros((species_count, g))  # IndVal scores
    
    # Calculate metrics for each cluster
    for j in range(g):
        cluster_sites = np.where(clusters_idx == j)[0]
        
        # Skip empty clusters
        if len(cluster_sites) == 0:
            continue
        
        # Mean abundance (or presence in this case) of each species in cluster j
        x_ij = np.mean(X_np[:, cluster_sites], axis=1)
        
        # Sum of mean abundances across all clusters
        sum_x_ih = np.zeros(species_count)
        for h in range(g):
            h_sites = np.where(clusters_idx == h)[0]
            if len(h_sites) > 0:
                sum_x_ih += np.mean(X_np[:, h_sites], axis=1)
        
        # Concentration (Specificity)
        with np.errstate(divide='ignore', invalid='ignore'):
            A[:, j] = np.where(sum_x_ih > 0, x_ij / sum_x_ih, 0)
        
        # Fidelity (proportion of sites in cluster j where species occurs)
        presence_absence = (X_np[:, cluster_sites] > 0).astype(int)
        B[:, j] = np.mean(presence_absence, axis=1)
        
        # Calculate IndVal
        indval_matrix[:, j] = 100 * A[:, j] * B[:, j]
    
    # Find maximum IndVal and corresponding group for each species
    indval_max = np.max(indval_matrix, axis=1)
    indval_max_group = np.argmax(indval_matrix, axis=1)
    
    # Map back to original cluster values
    reverse_map = {idx: val for val, idx in cluster_map.items()}
    indval_max_cluster = np.array([reverse_map[idx] for idx in indval_max_group])
    
    # Permutation test for significance
    print(f"Running {n_permutations} permutations for significance testing...")
    pvalues = np.ones(species_count)
    
    # For each permutation
    for perm in range(n_permutations):
        if perm % 100 == 0 and perm > 0:
            print(f"  Completed {perm} permutations...")
        
        # Shuffle cluster assignments
        y_shuffled = np.random.permutation(clusters_idx)
        
        # Initialize matrices for permutation
        A_perm = np.zeros((species_count, g))
        B_perm = np.zeros((species_count, g))
        indval_perm = np.zeros((species_count, g))
        
        # Calculate metrics for each cluster in permutation
        for j in range(g):
            perm_cluster_sites = np.where(y_shuffled == j)[0]
            
            if len(perm_cluster_sites) == 0:
                continue
            
            # Mean abundance in permuted cluster
            x_ij_perm = np.mean(X_np[:, perm_cluster_sites], axis=1)
            
            # Sum of mean abundances across all permuted clusters
            sum_x_ih_perm = np.zeros(species_count)
            for h in range(g):
                h_sites = np.where(y_shuffled == h)[0]
                if len(h_sites) > 0:
                    sum_x_ih_perm += np.mean(X_np[:, h_sites], axis=1)
            
            # Permuted Specificity
            with np.errstate(divide='ignore', invalid='ignore'):
                A_perm[:, j] = np.where(sum_x_ih_perm > 0, x_ij_perm / sum_x_ih_perm, 0)
            
            # Permuted Fidelity
            presence_absence_perm = (X_np[:, perm_cluster_sites] > 0).astype(int)
            B_perm[:, j] = np.mean(presence_absence_perm, axis=1)
            
            # Calculate permuted IndVal
            indval_perm[:, j] = 100 * A_perm[:, j] * B_perm[:, j]
        
        # Find maximum IndVal for each species in permutation
        indval_perm_max = np.max(indval_perm, axis=1)
        
        # Update p-values: count permutations with IndVal >= observed
        pvalues += (indval_perm_max >= indval_max).astype(int)
    
    # Calculate final p-values
    pvalues = pvalues / (n_permutations + 1)
    
    # Prepare results dataframe
    results = []
    
    for i in range(species_count):
        species_name = X.columns[i]
        cluster = indval_max_cluster[i]
        
        # Get original specificity and fidelity for this species/cluster
        cluster_idx = cluster_map[cluster]
        specificity = A[i, cluster_idx]
        fidelity = B[i, cluster_idx]
        
        results.append({
            'Species': species_name,
            'Cluster': cluster,
            'IndVal': indval_max[i],
            'Specificity': specificity,
            'Fidelity': fidelity,
            'pvalue': pvalues[i]
        })
    
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction for multiple testing
    results_df['pvalue_adj'] = statsmodels.stats.multitest.fdrcorrection(
        results_df['pvalue'], alpha=0.05, method='indep'
    )[1]
    
    # Calculate relative strength metrics
    indval_mean = results_df['IndVal'].mean()
    indval_std = results_df['IndVal'].std()
    
    results_df['indval_zscore'] = (results_df['IndVal'] - indval_mean) / indval_std
    
    # Classify strength
    results_df['relative_strength'] = pd.cut(
        results_df['indval_zscore'],
        bins=[-float('inf'), -1, 0, 1, float('inf')],
        labels=['weak', 'below_average', 'above_average', 'strong']
    )
    
    elapsed = time.time() - start_time
    print(f"IndVal calculation completed in {elapsed:.2f} seconds")
    
    return results_df

# ... rest of the script remains unchanged ...

def interpret_indval_scores(indval_results, alpha=0.05):
    """
    Interpret IndVal scores with validated calculations.
    """
    # Validate input ranges
    assert np.all(indval_results['Specificity'] >= 0) and np.all(indval_results['Specificity'] <= 1)
    assert np.all(indval_results['Fidelity'] >= 0) and np.all(indval_results['Fidelity'] <= 1)
    assert np.all(indval_results['IndVal'] >= 0) and np.all(indval_results['IndVal'] <= 100)
    
    # Filter significant species first
    significant_mask = indval_results['pvalue'] < alpha
    significant_results = indval_results[significant_mask].copy()
    
    # Calculate z-scores for IndVal (standardized relative strength)
    indval_mean = significant_results['IndVal'].mean()
    indval_std = significant_results['IndVal'].std()
    significant_results['indval_zscore'] = (significant_results['IndVal'] - indval_mean) / indval_std
    
    # Relative strength categories based on standard deviations
    significant_results['relative_strength'] = pd.cut(
        significant_results['indval_zscore'],
        bins=[-np.inf, -1, 0, 1, 2, np.inf],
        labels=['very_weak', 'weak', 'average', 'strong', 'very_strong']
    )
    
    # Calculate percentile ranks within significant species
    for metric in ['IndVal', 'Specificity', 'Fidelity']:
        significant_results[f'{metric.lower()}_percentile'] = (
            significant_results[metric].rank(pct=True)
        )
    
    # Add ranks
    significant_results['rank'] = significant_results['IndVal'].rank(ascending=False)
    total_significant = len(significant_results)
    
    def create_relative_interpretation(row):
        zscore = row['indval_zscore']
        strength = row['relative_strength']
        
        # Base interpretation on standard deviations from mean
        if zscore > 2:
            strength_desc = "Exceptionally strong"
        elif zscore > 1:
            strength_desc = "Very strong"
        elif zscore > 0:
            strength_desc = "Above average"
        elif zscore > -1:
            strength_desc = "Below average"
        else:
            strength_desc = "Weak"
            
        return (
            f"{strength_desc} indicator (z-score: {zscore:.1f}, "
            f"ranked #{int(row['rank'])} of {total_significant} significant species)\n"
            f"IndVal: {row['IndVal']:.1f} "
            f"(stronger than {row['indval_percentile']:.1%} of significant species)\n"
            f"Specificity: {row['Specificity']:.1%} "
            f"(better than {row['specificity_percentile']:.1%} of significant species)\n"
            f"Fidelity: {row['Fidelity']:.1%} "
            f"(better than {row['fidelity_percentile']:.1%} of significant species)\n"
            f"p={row['pvalue']:.1e}"
        )
    
    # Apply interpretation only to significant species
    indval_results['interpretation'] = "Not a significant indicator species"
    indval_results.loc[significant_mask, 'interpretation'] = (
        significant_results.apply(create_relative_interpretation, axis=1)
    )
    
    # Copy other columns back to main results
    for col in significant_results.columns:
        if col not in indval_results.columns:
            indval_results[col] = np.nan
            indval_results.loc[significant_mask, col] = significant_results[col]
    
    return indval_results

def visualize_indval_results(results_df, output_prefix=""):
    """
    Create visualizations of top indicator species and their characteristics.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with IndVal results
    output_prefix : str
        Prefix for output files (default: "")
    """
    # Filter to significant species and sort by IndVal
    significant = results_df[results_df['pvalue'] < 0.05].copy()
    top_indicators = significant.nlargest(15, 'IndVal')
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Set figure aesthetics
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 11,
        'axes.titlesize': 12
    })
    
    # 1. Bar plot of IndVal scores for top species
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(top_indicators)), 
                   top_indicators['IndVal'])
    plt.xticks(range(len(top_indicators)), 
               [s.replace('species_', '') for s in top_indicators['Species']], 
               rotation=45, ha='right')
    plt.ylabel('IndVal Score')
    plt.title('Top 10 Species by IndVal Score')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}top_indval_scores.png")
    plt.close()
    
    # 2. Scatter plot of specificity vs. fidelity
    plt.figure(figsize=(10, 6))
    plt.scatter(significant['Specificity'], 
               significant['Fidelity'],
               alpha=0.5)
    
    # Highlight top indicators
    plt.scatter(top_indicators['Specificity'],
               top_indicators['Fidelity'],
               color='red', label='Top 15 indicators')
    
    # Add labels for top indicators
    for _, row in top_indicators.iterrows():
        plt.annotate(row['Species'].replace('species_', ''),
                    (row['Specificity'], row['Fidelity']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)
    
    plt.xlabel("Specificity (A)")
    plt.ylabel('Fidelity (B)')
    plt.title('Specificity vs Fidelity for IndVal Indicator Species')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}specificity_vs_fidelity.png")
    plt.close()
    
    # 3. Heatmap of top indicators' characteristics
    plt.figure(figsize=(12, 8))
    heatmap_data = top_indicators[['IndVal', 'Specificity', 
                                  'Fidelity', 'indval_zscore']].copy()
    heatmap_data.index = [s.replace('species_', '') for s in top_indicators['Species']]
    
    # Normalize the data for better visualization
    heatmap_norm = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()
    
    sns.heatmap(heatmap_norm, cmap='RdYlBu_r', center=0,
                xticklabels=['IndVal', 'Specificity', 'Fidelity', 'Z-score'],
                yticklabels=True, annot=heatmap_data.round(2),
                fmt='.2f', annot_kws={'size': 8})
    plt.title('Characteristics of Top 15 Indicator Species')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}indval_characteristics_heatmap.png")
    plt.close()
    
    # 4. Community-level analysis
    if 'community' in results_df.columns:
        # Boxplot of IndVal scores by community
        plt.figure(figsize=(14, 8))
        
        # Filter communities with enough species
        community_counts = results_df['community'].value_counts()
        valid_communities = community_counts[community_counts >= 3].index
        community_data = results_df[results_df['community'].isin(valid_communities)]
        
        # Create boxplot
        sns.boxplot(
            data=community_data,
            x='community',
            y='IndVal',
            palette='tab20'
        )
        
        # Add swarmplot
        sns.swarmplot(
            data=community_data,
            x='community',
            y='IndVal',
            color='black',
            alpha=0.5,
            size=4
        )
        
        plt.xlabel('Community')
        plt.ylabel('IndVal Score')
        plt.title('Distribution of IndVal Scores by Community')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}community_indval_distribution.png")
        plt.close()
        
        # Community-level statistics
        community_stats = results_df.groupby('community').agg({
            'IndVal': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'Specificity': ['mean'],
            'Fidelity': ['mean'],
            'pvalue': ['mean', lambda x: (x < 0.05).mean()]
        })
        
        # Flatten column names
        community_stats.columns = ['_'.join(col).strip() for col in community_stats.columns.values]
        community_stats = community_stats.rename(columns={
            'pvalue_<lambda_0>': 'proportion_significant'
        })
        
        # Save community statistics
        community_stats.to_csv(f"{output_prefix}community_indval_stats.csv")
    
    # 5. Summary statistics table
    summary_stats = pd.DataFrame({
        'Species': [s.replace('species_', '') for s in top_indicators['Species']],
        'IndVal': top_indicators['IndVal'],
        'Specificity': top_indicators['Specificity'],
        'Fidelity': top_indicators['Fidelity'],
        'P-value': top_indicators['pvalue'].apply(lambda x: f'{x:.1e}'),
        'Z-score': top_indicators['indval_zscore']
    })
    
    # Save summary to CSV
    summary_stats.to_csv(f"{output_prefix}top_indval_summary.csv", index=False)
    
    print("\nTop 15 Indicator Species Summary:")
    print("=" * 80)
    print(summary_stats.to_string(index=False))
    print("\nVisualization files saved:")
    print(f"- {output_prefix}top_indval_scores.png")
    print(f"- {output_prefix}specificity_vs_fidelity.png")
    print(f"- {output_prefix}indval_characteristics_heatmap.png")
    if 'community' in results_df.columns:
        print(f"- {output_prefix}community_indval_distribution.png")
        print(f"- {output_prefix}community_indval_stats.csv")
    print(f"- {output_prefix}top_indval_summary.csv")

def save_and_summarize_results(results_df, output_prefix=""):
    """
    Save and summarize IndVal results with focus on relative strength.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with IndVal results
    output_prefix : str
        Prefix for output files (default: "")
    """
    significant_indicators = results_df[results_df['pvalue'] < 0.05].copy()
    
    print("\nIndVal Analysis Summary:")
    print(f"Total species analyzed: {len(results_df)}")
    print(f"Significant indicators found: {len(significant_indicators)}")
    
    print("\nDistribution of Indicator Strength (significant species):")
    print("Relative to mean IndVal score:")
    if 'relative_strength' in significant_indicators.columns:
        print(significant_indicators['relative_strength'].value_counts().sort_index())
    
    print("\nIndVal Score Distribution:")
    desc = significant_indicators['IndVal'].describe()
    print(f"Mean: {desc['mean']:.1f}")
    print(f"Std Dev: {desc['std']:.1f}")
    print(f"Min: {desc['min']:.1f}")
    print(f"25%: {desc['25%']:.1f}")
    print(f"Median: {desc['50%']:.1f}")
    print(f"75%: {desc['75%']:.1f}")
    print(f"Max: {desc['max']:.1f}")
    
    print("\nTop 10 Indicator Species (>1 standard deviation above mean):")
    if 'indval_zscore' in significant_indicators.columns:
        top_indicators = significant_indicators[
            significant_indicators['indval_zscore'] > 1
        ].sort_values('IndVal', ascending=False).head(10)
        
        for _, row in top_indicators.iterrows():
            print(f"\n{row['Species']}:")
            if 'interpretation' in row:
                print(row['interpretation'])
            else:
                print(f"IndVal: {row['IndVal']:.1f}, Specificity: {row['Specificity']:.3f}, Fidelity: {row['Fidelity']:.3f}, p={row['pvalue']:.1e}")
    
    # Save results
    results_df.to_csv(f"{output_prefix}indval_full_results.csv", index=False)
    significant_indicators.to_csv(f"{output_prefix}significant_indicators.csv", index=False)
    if 'indval_zscore' in significant_indicators.columns:
        strong_indicators = significant_indicators[significant_indicators['indval_zscore'] > 1].sort_values('IndVal', ascending=False)
        strong_indicators.to_csv(f"{output_prefix}strong_indicators.csv", index=False)

def main():
    """Main analysis pipeline with command line arguments."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze species indicator values for VCM sites.')
    parser.add_argument('--input', type=str, default='inat-data-matrix-latlong.csv',
                        help='Path to input CSV file with species presence and VCM data')
    parser.add_argument('--output-prefix', type=str, default='outputs/indval_analysis/',
                        help='Prefix for output files (can include directory path)')
    parser.add_argument('--permutations', type=int, default=4999,
                        help='Number of permutations for significance testing')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    if args.output_prefix and '/' in args.output_prefix:
        os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {args.input}...")
    data = pd.read_csv(args.input)
    print(f"The columns are {data.columns}")
    
    # Extract species columns
    species_columns = [col for col in data.columns if col.startswith('species_')]
    X = data[species_columns]
    
    # Extract VCM labels
    y = data['vcm_label'] if 'vcm_label' in data.columns else None
    if y is None:
        raise ValueError("VCM column not found in dataset")
    
    # Perform IndVal analysis
    print(f"\nCalculating IndVal metrics with {args.permutations} permutations...")
    results_df = calculate_indval_metrics(X, y, n_permutations=args.permutations)
    
    # Interpret results
    print("\nInterpreting IndVal results...")
    results_df = interpret_indval_scores(results_df)
    
    # Save and summarize results
    save_and_summarize_results(results_df, output_prefix=args.output_prefix)
    
    # Add visualization
    visualize_indval_results(results_df, output_prefix=args.output_prefix)
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
