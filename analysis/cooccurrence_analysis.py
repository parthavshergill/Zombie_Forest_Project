import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
import argparse

def load_species_data(file_path):
    """
    Load and prepare species data from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing species data
        
    Returns:
    --------
    pd.DataFrame, list, pd.Series
        Processed data, species columns, and VCM labels
    """
    data = pd.read_csv(file_path)
    species_columns = [col for col in data.columns if col.startswith('species_')]
    vcm_label = data['vcm_label']
    
    print(f"Number of grid boxes: {len(data)}")
    print(f"Number of species analyzed: {len(species_columns)}")
    
    return data, species_columns, vcm_label

def calculate_contingency_table(species_data, vcm_data):
    """
    Create a 2x2 contingency table for species presence and VCM labels.
    
    Parameters:
    -----------
    species_data : pd.Series
        Binary presence/absence data for a single species
    vcm_data : pd.Series
        VCM labels
        
    Returns:
    --------
    pd.DataFrame
        2x2 contingency table
    """
    return pd.crosstab(species_data > 0, vcm_data)

def calculate_effect_sizes(contingency_table, chi2_stat):
    """
    Calculate various effect size measures.
    
    Parameters:
    -----------
    contingency_table : pd.DataFrame
        2x2 contingency table
    chi2_stat : float
        Chi-square statistic
        
    Returns:
    --------
    dict
        Dictionary containing effect size measures
    """
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    
    # Cramer's V
    cramer_v = np.sqrt(chi2_stat / (n * min_dim)) if n > 0 and min_dim > 0 else 0
    
    # Odds ratio
    try:
        odds_ratio = (contingency_table.loc[True, 1] * contingency_table.loc[False, 0]) / \
                    (contingency_table.loc[True, 0] * contingency_table.loc[False, 1])
    except (KeyError, ZeroDivisionError):
        odds_ratio = np.nan
    
    # Calculate co-occurrence ratio
    try:
        total_species_occurrences = contingency_table.loc[True].sum()
        occurrences_in_vcm = contingency_table.loc[True, 1]
        co_occurrence_ratio = occurrences_in_vcm / total_species_occurrences
    except (KeyError, ZeroDivisionError):
        co_occurrence_ratio = 0
    
    return {
        "cramer_v": cramer_v,
        "odds_ratio": odds_ratio,
        "co_occurrence_ratio": co_occurrence_ratio
    }

def calculate_cooccurrence(species_data, vcm_data):
    """
    Calculate co-occurrence statistics for a single species.
    
    Parameters:
    -----------
    species_data : pd.Series
        Binary presence/absence data for a single species
    vcm_data : pd.Series
        VCM labels
        
    Returns:
    --------
    dict
        Co-occurrence statistics including ratio, p-value, and effect sizes
    """
    # Create contingency table
    contingency_table = calculate_contingency_table(species_data, vcm_data)
    
    # Calculate chi-square test
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    
    # Calculate effect sizes
    effect_sizes = calculate_effect_sizes(contingency_table, chi2)
    
    return {
        **effect_sizes,
        "p_value": p_value,
        "chi2_statistic": chi2
    }

def analyze_cooccurrence(data, species_columns, vcm_label):
    """
    Analyze co-occurrence patterns for all species.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Full dataset
    species_columns : list
        List of species column names
    vcm_label : pd.Series
        VCM labels
        
    Returns:
    --------
    pd.DataFrame
        Results for all species with adjusted p-values
    """
    results = []
    
    for species in species_columns:
        stats = calculate_cooccurrence(data[species], vcm_label)
        results.append({
            "species": species,
            **stats
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply multiple testing correction
    _, adjusted_pvalues, _, _ = multipletests(
        results_df['p_value'], 
        method='fdr_bh'  # Benjamini-Hochberg FDR correction
    )
    results_df['adjusted_p_value'] = adjusted_pvalues
    
    # Calculate preference index and relative risk here
    n = len(vcm_label)
    vcm_proportion = vcm_label.mean()  # proportion of all sites that are VCM
    
    for species in species_columns:
        species_data = data[species]
        vcm_data = vcm_label
        
        # Calculate preference index
        species_in_vcm = (species_data & vcm_data).sum() / species_data.sum()
        preference_index = (species_in_vcm - vcm_proportion) / max(vcm_proportion, 1 - vcm_proportion)
        results_df.loc[results_df['species'] == species, 'preference_index'] = preference_index
        
        # Calculate relative risk
        prob_in_vcm = (species_data & vcm_data).sum() / vcm_data.sum()
        prob_in_nonvcm = (species_data & ~vcm_data).sum() / (~vcm_data).sum()
        relative_risk = prob_in_vcm / prob_in_nonvcm if prob_in_nonvcm > 0 else np.inf
        results_df.loc[results_df['species'] == species, 'relative_risk'] = relative_risk
    
    return results_df

def interpret_cooccurrence_results(results_df, alpha=0.05):
    """
    Interpret co-occurrence results focusing on relative strength between species.
    """
    # Filter significant species first
    significant_mask = results_df['adjusted_p_value'] < alpha
    significant_results = results_df[significant_mask].copy()
    
    # Calculate z-scores for co-occurrence ratio
    ratio_mean = significant_results['co_occurrence_ratio'].mean()
    ratio_std = significant_results['co_occurrence_ratio'].std()
    significant_results['cooccurrence_zscore'] = (
        (significant_results['co_occurrence_ratio'] - ratio_mean) / ratio_std
    )
    
    # Relative strength categories based on standard deviations
    significant_results['relative_strength'] = pd.cut(
        significant_results['cooccurrence_zscore'],
        bins=[-np.inf, -1, 0, 1, 2, np.inf],
        labels=['very_weak', 'weak', 'average', 'strong', 'very_strong']
    )
    
    # Calculate percentile ranks
    for metric in ['co_occurrence_ratio', 'cramer_v', 'odds_ratio']:
        significant_results[f'{metric}_percentile'] = (
            significant_results[metric].rank(pct=True)
        )
    
    # Add ranks
    significant_results['rank'] = significant_results['co_occurrence_ratio'].rank(ascending=False)
    total_significant = len(significant_results)
    
    def create_relative_interpretation(row):
        zscore = row['cooccurrence_zscore']
        
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
            f"{strength_desc} association with VCM (z-score: {zscore:.1f}, "
            f"ranked #{int(row['rank'])} of {total_significant} significant species)\n"
            f"Co-occurrence ratio: {row['co_occurrence_ratio']:.1%} "
            f"(higher than {row['co_occurrence_ratio_percentile']:.1%} of significant species)\n"
            f"Effect size (Cramer's V): {row['cramer_v']:.3f} "
            f"(stronger than {row['cramer_v_percentile']:.1%} of significant species)\n"
            f"Odds ratio: {row['odds_ratio']:.1f} "
            f"(higher than {row['odds_ratio_percentile']:.1%} of significant species)\n"
            f"p={row['adjusted_p_value']:.1e}"
        )
    
    # Apply interpretation
    results_df['interpretation'] = "Not a significant association"
    results_df.loc[significant_mask, 'interpretation'] = (
        significant_results.apply(create_relative_interpretation, axis=1)
    )
    
    # Copy other columns back
    for col in significant_results.columns:
        if col not in results_df.columns:
            results_df[col] = np.nan
            results_df.loc[significant_mask, col] = significant_results[col]
    
    return results_df

def save_and_summarize_results(results_df, output_prefix=""):
    """
    Save and summarize co-occurrence results with focus on relative strength.
    """
    significant_indicators = results_df[results_df['adjusted_p_value'] < 0.05].copy()
    
    print("\nCo-occurrence Analysis Summary:")
    print(f"Total species analyzed: {len(results_df)}")
    print(f"Significant associations found: {len(significant_indicators)}")
    
    print("\nDistribution of Association Strength (significant species):")
    print("Relative to mean co-occurrence ratio:")
    print(significant_indicators['relative_strength'].value_counts().sort_index())
    
    print("\nCo-occurrence Ratio Distribution:")
    desc = significant_indicators['co_occurrence_ratio'].describe()
    print(f"Mean: {desc['mean']:.1%}")
    print(f"Std Dev: {desc['std']:.1%}")
    print(f"Min: {desc['min']:.1%}")
    print(f"25%: {desc['25%']:.1%}")
    print(f"Median: {desc['50%']:.1%}")
    print(f"75%: {desc['75%']:.1%}")
    print(f"Max: {desc['max']:.1%}")
    
    print("\nTop 10 VCM-Associated Species (>1 standard deviation above mean):")
    top_indicators = significant_indicators[
        significant_indicators['cooccurrence_zscore'] > 1
    ].sort_values('co_occurrence_ratio', ascending=False).head(10)
    
    for _, row in top_indicators.iterrows():
        print(f"\n{row['species']}:")
        print(row['interpretation'])
    
    # Save results
    results_df.to_csv(f"{output_prefix}cooccurrence_full_results.csv", index=False)
    significant_indicators.to_csv(f"{output_prefix}significant_associations.csv", index=False)
    top_indicators.to_csv(f"{output_prefix}strong_associations.csv", index=False)

def visualize_top_indicators(results_df, output_prefix=""):
    """
    Create visualizations of top indicator species and their characteristics.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Filter to significant species and sort by co-occurrence ratio
    significant = results_df[results_df['adjusted_p_value'] < 0.05].copy()
    top_indicators = significant.nlargest(15, 'co_occurrence_ratio')
    
    # Set up the plotting style - use a built-in style instead of seaborn
    plt.style.use('default')  # Changed from 'seaborn' to 'default'
    
    # Set figure aesthetics that would have come from seaborn
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 11,
        'axes.titlesize': 12
    })
    
    # 1. Bar plot of co-occurrence ratios for top species
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(top_indicators)), 
                   top_indicators['co_occurrence_ratio'] * 100)
    plt.xticks(range(len(top_indicators)), 
               [s.replace('species_', '') for s in top_indicators['species']], 
               rotation=45, ha='right')
    plt.ylabel('Co-occurrence with VCM (%)')
    plt.title('Top 15 Species by VCM Co-occurrence')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}top_indicators_cooccurrence.png")
    plt.close()
    
    # 2. Scatter plot of effect sizes
    plt.figure(figsize=(10, 8))
    plt.scatter(significant['cramer_v'], 
               significant['co_occurrence_ratio'] * 100,
               alpha=0.5)
    
    # Highlight top indicators
    plt.scatter(top_indicators['cramer_v'],
               top_indicators['co_occurrence_ratio'] * 100,
               color='red', label='Top 15 indicators')
    
    # Add labels for top indicators
    for _, row in top_indicators.iterrows():
        plt.annotate(row['species'].replace('species_', ''),
                    (row['cramer_v'], row['co_occurrence_ratio'] * 100),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)
    
    plt.xlabel("Cramer's V (Effect Size)")
    plt.ylabel('Co-occurrence with VCM (%)')
    plt.title('Effect Size vs Co-occurrence Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}effect_size_vs_cooccurrence.png")
    plt.close()
    
    # 3. Heatmap of top indicators' characteristics
    plt.figure(figsize=(12, 8))
    heatmap_data = top_indicators[['co_occurrence_ratio', 'cramer_v', 
                                 'odds_ratio', 'preference_index', 
                                 'relative_risk']].copy()
    heatmap_data.index = [s.replace('species_', '') for s in top_indicators['species']]
    
    # Normalize the data for better visualization
    heatmap_norm = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()
    
    sns.heatmap(heatmap_norm, cmap='RdYlBu_r', center=0,
                xticklabels=['Co-occurrence', "Cramer's V", 'Odds Ratio',
                            'Preference', 'Relative Risk'],
                yticklabels=True, annot=heatmap_data.round(2),
                fmt='.2f', annot_kws={'size': 8})
    plt.title('Characteristics of Top 15 Indicator Species')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}indicator_characteristics_heatmap.png")
    plt.close()
    
    # 4. Summary statistics table
    summary_stats = pd.DataFrame({
        'Species': [s.replace('species_', '') for s in top_indicators['species']],
        'Co-occurrence (%)': (top_indicators['co_occurrence_ratio'] * 100).round(1),
        "Cramer's V": top_indicators['cramer_v'].round(3),
        'Odds Ratio': top_indicators['odds_ratio'].round(2),
        'P-value': top_indicators['adjusted_p_value'].apply(lambda x: f'{x:.1e}'),
        'Z-score': top_indicators['cooccurrence_zscore'].round(2)
    })
    
    # Save summary to CSV
    summary_stats.to_csv(f"{output_prefix}top_indicators_summary.csv", index=False)
    
    print("\nTop 15 VCM Indicator Species Summary:")
    print("=" * 80)
    print(summary_stats.to_string(index=False))
    print("\nVisualization files saved:")
    print(f"- {output_prefix}top_indicators_cooccurrence.png")
    print(f"- {output_prefix}effect_size_vs_cooccurrence.png")
    print(f"- {output_prefix}indicator_characteristics_heatmap.png")
    print(f"- {output_prefix}top_indicators_summary.csv")

def main():
    """Main analysis pipeline with command line arguments."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze species co-occurrence with VCM sites.')
    parser.add_argument('--input', type=str, default='inat-data-matrix-latlong.csv',
                        help='Path to input CSV file with species presence and VCM data')
    parser.add_argument('--output-prefix', type=str, default='outputs/cooccurrence_analysis/',
                        help='Prefix for output files (can include directory path)')
    
    args = parser.parse_args()
    
    # Load data
    file_path = args.input
    data, species_columns, vcm_label = load_species_data(file_path)
    
    # Perform co-occurrence analysis
    print(f"\nAnalyzing species co-occurrence patterns from {file_path}...")
    results_df = analyze_cooccurrence(data, species_columns, vcm_label)
    
    # Interpret results
    results_df = interpret_cooccurrence_results(results_df)
    
    # Save and summarize results
    save_and_summarize_results(results_df, output_prefix=args.output_prefix)
    
    # Add visualization
    visualize_top_indicators(results_df, output_prefix=args.output_prefix)
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
