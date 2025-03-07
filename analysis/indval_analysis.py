import pandas as pd
import numpy as np
from scipy import stats
from numba import jit
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_indval(species_data, group_labels, target_group):
    """
    Calculate IndVal (Indicator Value) for a species in a target group.
    
    Parameters:
    -----------
    species_data : pd.Series
        Binary presence/absence data for a single species
    group_labels : pd.Series
        Group labels for each site
    target_group : any
        The group for which to calculate IndVal
        
    Returns:
    --------
    float, float, float
        IndVal score, specificity (A), and fidelity (B)
    """
    # Sites belonging to target group
    target_sites = group_labels == target_group
    
    # Specificity (A): relative abundance in target group vs all groups
    abundance_in_target = species_data[target_sites].mean()
    abundance_in_all = species_data.mean()
    specificity = abundance_in_target / abundance_in_all if abundance_in_all > 0 else 0
    
    # Fidelity (B): proportion of target sites where species occurs
    fidelity = species_data[target_sites].mean()
    
    # IndVal = A * B * 100
    indval = specificity * fidelity * 100
    
    return indval, specificity, fidelity

@jit(nopython=True)
def fast_permutation_test(species_data, group_labels, n_permutations):
    """
    Optimized permutation test using Numba.
    
    Parameters:
    -----------
    species_data : np.array
        Binary presence/absence data for a single species
    group_labels : np.array
        Group labels
    n_permutations : int
        Number of permutations
        
    Returns:
    --------
    float
        p-value
    """
    observed_indval = calculate_indval_fast(species_data, group_labels)
    random_indvals = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        random_labels = np.random.permutation(group_labels)
        random_indvals[i] = calculate_indval_fast(species_data, random_labels)
    
    return (np.sum(random_indvals >= observed_indval) + 1) / (n_permutations + 1)

@jit(nopython=True)
def calculate_indval_fast(species_data, group_labels):
    """
    Optimized IndVal calculation using Numba.
    Ensures all intermediate calculations are valid proportions.
    """
    target_sites = group_labels == 1
    species_present = species_data > 0
    
    # Count actual occurrences
    n_occurrences_vcm = np.sum(species_present & target_sites)
    n_occurrences_total = np.sum(species_present)
    n_vcm_sites = np.sum(target_sites)
    
    # Specificity: proportion of species' occurrences in VCM
    if n_occurrences_total > 0:
        specificity = n_occurrences_vcm / n_occurrences_total
    else:
        specificity = 0
    
    # Fidelity: proportion of VCM sites where species occurs
    if n_vcm_sites > 0:
        fidelity = n_occurrences_vcm / n_vcm_sites
    else:
        fidelity = 0
    
    # Sanity checks (Numba doesn't support assertions)
    if specificity > 1.0:
        specificity = 1.0
    if fidelity > 1.0:
        fidelity = 1.0
    
    indval = specificity * fidelity * 100
    return indval

def calculate_indval_metrics(X, y, n_permutations=999):
    """
    Vectorized calculation of IndVal metrics for all species.
    Includes validation of intermediate results.
    """
    X_array = X.to_numpy()
    y_array = y.to_numpy()
    n_species = X_array.shape[1]
    
    # Pre-allocate arrays
    indvals = np.zeros(n_species)
    specificities = np.zeros(n_species)
    fidelities = np.zeros(n_species)
    pvalues = np.zeros(n_species)
    
    # Calculate metrics for all species
    print("Calculating IndVal metrics...")
    target_sites = y_array == 1
    n_vcm_sites = np.sum(target_sites)
    
    for i in tqdm(range(n_species)):
        species_data = X_array[:, i]
        species_present = species_data > 0
        
        # Calculate raw counts
        n_occurrences_vcm = np.sum(species_present & target_sites)
        n_occurrences_total = np.sum(species_present)
        
        # Calculate components with bounds checking
        if n_occurrences_total > 0:
            specificities[i] = min(n_occurrences_vcm / n_occurrences_total, 1.0)
        else:
            specificities[i] = 0
            
        if n_vcm_sites > 0:
            fidelities[i] = min(n_occurrences_vcm / n_vcm_sites, 1.0)
        else:
            fidelities[i] = 0
        
        # Calculate IndVal
        indvals[i] = specificities[i] * fidelities[i] * 100
        
        # Permutation test
        pvalues[i] = fast_permutation_test(species_data, y_array, n_permutations)
    
    # Final validation
    assert np.all(specificities >= 0) and np.all(specificities <= 1), "Specificity out of bounds"
    assert np.all(fidelities >= 0) and np.all(fidelities <= 1), "Fidelity out of bounds"
    assert np.all(indvals >= 0) and np.all(indvals <= 100), "IndVal out of bounds"
    
    # Add diagnostic information
    results = pd.DataFrame({
        'Species': X.columns,
        'IndVal': indvals,
        'Specificity': specificities,
        'Fidelity': fidelities,
        'pvalue': pvalues,
        'n_occurrences_total': [np.sum(X_array[:, i] > 0) for i in range(n_species)],
        'n_occurrences_vcm': [np.sum((X_array[:, i] > 0) & target_sites) for i in range(n_species)]
    })
    
    # Add proportion in VCM for validation
    results['proportion_in_vcm'] = results['n_occurrences_vcm'] / results['n_occurrences_total']
    assert np.allclose(results['proportion_in_vcm'], results['Specificity']), "Specificity calculation mismatch"
    
    return results

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

def summarize_indval_results(results_df, output_prefix=""):
    """
    Summarize IndVal results with focus on relative strength and occurrence counts.
    """
    significant_indicators = results_df[results_df['pvalue'] < 0.05].copy()
    
    print("\nIndVal Analysis Summary:")
    print(f"Total species analyzed: {len(results_df)}")
    print(f"Significant indicators found: {len(significant_indicators)}")
    
    print("\nDistribution of Indicator Strength (significant species):")
    print("Relative to mean IndVal score:")
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
    top_indicators = significant_indicators[
        significant_indicators['indval_zscore'] > 1
    ].sort_values('IndVal', ascending=False).head(10)
    
    for _, row in top_indicators.iterrows():
        print(f"\n{row['Species']}:")
        print(f"Total observations: {row['n_occurrences_total']} "
              f"(VCM: {row['n_occurrences_vcm']}, "
              f"non-VCM: {row['n_occurrences_total'] - row['n_occurrences_vcm']})")
        print(f"Specificity: {row['Specificity']:.1%} of occurrences in VCM areas")
        print(f"Fidelity: Present in {row['Fidelity']:.1%} of VCM areas")
        print(f"IndVal: {row['IndVal']:.1f}")
        print(row['interpretation'])
    
    # Save results
    results_df.to_csv(f"{output_prefix}indval_full_results.csv", index=False)
    significant_indicators.to_csv(f"{output_prefix}significant_indicators.csv", index=False)
    top_indicators.to_csv(f"{output_prefix}strong_indicators.csv", index=False)

def visualize_indval_results(results_df, output_prefix=""):
    """
    Create visualizations of top indicator species and their characteristics.
    """
    # Filter to significant species and get top indicators
    significant = results_df[results_df['pvalue'] < 0.05].copy()
    top_indicators = significant.nlargest(15, 'IndVal')
    
    # Set up plotting style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 11,
        'axes.titlesize': 12
    })
    
    # 1. Bar plot of IndVal scores
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(top_indicators)), 
                   top_indicators['IndVal'])
    plt.xticks(range(len(top_indicators)), 
               [s.replace('species_', '') for s in top_indicators['Species']], 
               rotation=45, ha='right')
    plt.ylabel('IndVal Score')
    plt.title('Top 15 Species by IndVal Score')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}top_indicators_indval.png")
    plt.close()
    
    # 2. Scatter plot of specificity vs fidelity
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
    
    plt.xlabel('Specificity')
    plt.ylabel('Fidelity')
    plt.title('Specificity vs Fidelity for VCM Indicators')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}specificity_vs_fidelity.png")
    plt.close()
    
    # 3. Heatmap of top indicators' characteristics
    plt.figure(figsize=(12, 8))
    heatmap_data = top_indicators[['IndVal', 'Specificity', 'Fidelity', 
                                 'indval_zscore']].copy()
    heatmap_data.index = [s.replace('species_', '') for s in top_indicators['Species']]
    
    # Normalize the data for better visualization
    heatmap_norm = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()
    
    sns.heatmap(heatmap_norm, cmap='RdYlBu_r', center=0,
                xticklabels=['IndVal', 'Specificity', 'Fidelity', 'Z-score'],
                yticklabels=True, annot=heatmap_data.round(2),
                fmt='.2f', annot_kws={'size': 8})
    plt.title('Characteristics of Top 15 Indicator Species')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}indicator_characteristics_heatmap.png")
    plt.close()
    
    # 4. Summary statistics table
    summary_stats = pd.DataFrame({
        'Species': [s.replace('species_', '') for s in top_indicators['Species']],
        'IndVal': top_indicators['IndVal'].round(2),
        'Specificity (%)': (top_indicators['Specificity'] * 100).round(1),
        'Fidelity (%)': (top_indicators['Fidelity'] * 100).round(1),
        'P-value': top_indicators['pvalue'].apply(lambda x: f'{x:.1e}'),
        'Z-score': top_indicators['indval_zscore'].round(2),
        'VCM Occurrences': top_indicators['n_occurrences_vcm'],
        'Total Occurrences': top_indicators['n_occurrences_total']
    })
    
    # Save summary to CSV
    summary_stats.to_csv(f"{output_prefix}top_indicators_summary.csv", index=False)
    
    print("\nTop 15 VCM Indicator Species Summary:")
    print("=" * 80)
    print(summary_stats.to_string(index=False))
    print("\nVisualization files saved:")
    print(f"- {output_prefix}top_indicators_indval.png")
    print(f"- {output_prefix}specificity_vs_fidelity.png")
    print(f"- {output_prefix}indicator_characteristics_heatmap.png")
    print(f"- {output_prefix}top_indicators_summary.csv")

def main():
    """Optimized main analysis pipeline."""
    print("Reading data...")
    data = pd.read_csv("inat-data-matrix-gdf.csv")
    
    # Efficient data preparation
    species_columns = data.columns[data.columns.str.startswith('species_')]
    X_species = data[species_columns]
    y_vcm = data['vcm_label']
    
    # Calculate IndVal metrics
    results = calculate_indval_metrics(X_species, y_vcm, n_permutations=999)
    
    # Interpret results
    interpreted_results = interpret_indval_scores(results)
    
    # Summarize and save
    summarize_indval_results(interpreted_results)
    
    # Add visualization
    visualize_indval_results(interpreted_results, output_prefix="")

if __name__ == "__main__":
    main()
