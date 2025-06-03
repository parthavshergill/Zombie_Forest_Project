import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from indval_analysis import calculate_indval_metrics
import warnings
warnings.filterwarnings('ignore')

# Grid constants from processing.py
LATITUDE_INCREMENT = 0.008998
LONGITUDE_INCREMENT = 0.008998

def preprocess_data_to_matrix(df, min_species_count=100, n_top_species=300, use_median=True):
    """
    Convert observation data to site × species matrix format.
    Based on processing.py logic.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw observation data
    min_species_count : int
        Minimum observations required per species
    n_top_species : int
        Number of top species to include
    use_median : bool
        Use median instead of mean for habitat_suitability aggregation
    
    Returns:
    --------
    pd.DataFrame
        Site × species matrix with environmental variables
    """
    print("Preprocessing observation data to site × species matrix...")
    
    # Show initial habitat_suitability distribution
    hab_suit = df['habitat_suitability'].dropna()
    print(f"Habitat suitability range: {hab_suit.min():.3f} to {hab_suit.max():.3f}")
    print(f"Habitat suitability mean: {hab_suit.mean():.3f}, median: {hab_suit.median():.3f}")
    
    # Show initial VCM distribution based on existing column if available
    if 'composite_zf_class' in df.columns:
        vcm_obs = df['composite_zf_class'].str.contains('VCM', na=False).sum()
        total_obs = len(df[df['composite_zf_class'].notna()])
        print(f"Original VCM proportion: {vcm_obs}/{total_obs} = {vcm_obs/total_obs:.3f}")
    
    # Filter species with sufficient observations
    species_counts = df['species'].value_counts()
    common_species = species_counts[species_counts >= min_species_count].index
    df_filtered = df[df['species'].isin(common_species)].copy()
    
    print(f"Filtered to {len(common_species)} species with ≥{min_species_count} observations")
    print(f"Remaining observations: {len(df_filtered)}")
    
    # Assign grid locations
    def calculate_grid_location(lat, lon):
        lat_grid = int(lat // LATITUDE_INCREMENT)
        lon_grid = int(lon // LONGITUDE_INCREMENT)
        return f"{lat_grid}_{lon_grid}"
    
    df_filtered['grid_location'] = df_filtered.apply(
        lambda row: calculate_grid_location(row['decimallatitude'], row['decimallongitude']), 
        axis=1
    )
    
    print(f"Created {df_filtered['grid_location'].nunique()} unique grid locations")
    
    # Create species presence matrix for top species
    species_occurrence_counts = df_filtered['species'].value_counts()
    top_species = species_occurrence_counts.head(n_top_species).index.tolist()
    
    print(f"Selected top {len(top_species)} species by occurrence")
    print(f"Top 5 species: {top_species[:5]}")
    
    # Filter for top species
    data_top_species = df_filtered[df_filtered['species'].isin(top_species)]
    
    # Create presence/absence matrix
    species_presence = data_top_species.groupby(['grid_location', 'species']).size().unstack(fill_value=0)
    species_presence = (species_presence > 0).astype(int)
    
    # Ensure all grid locations are included
    full_grid_index = pd.Index(df_filtered['grid_location'].unique())
    species_presence = species_presence.reindex(full_grid_index, fill_value=0)
    species_presence.columns = [f"species_{col}" for col in species_presence.columns]
    
    # Aggregate habitat_suitability by grid location (median or mean)
    if use_median:
        habitat_suit_by_grid = df_filtered.groupby('grid_location')['habitat_suitability'].median().reindex(full_grid_index)
        print("Using median habitat_suitability aggregation")
    else:
        habitat_suit_by_grid = df_filtered.groupby('grid_location')['habitat_suitability'].mean().reindex(full_grid_index)
        print("Using mean habitat_suitability aggregation")
    
    # Show aggregated habitat_suitability distribution
    hab_suit_agg = habitat_suit_by_grid.dropna()
    print(f"Aggregated habitat suitability range: {hab_suit_agg.min():.3f} to {hab_suit_agg.max():.3f}")
    print(f"Aggregated habitat suitability mean: {hab_suit_agg.mean():.3f}, median: {hab_suit_agg.median():.3f}")
    
    # Combine species matrix with habitat suitability
    final_matrix = species_presence.copy()
    final_matrix['habitat_suitability'] = habitat_suit_by_grid
    
    # Remove any sites with missing habitat_suitability
    final_matrix = final_matrix.dropna(subset=['habitat_suitability'])
    
    print(f"Final matrix shape: {final_matrix.shape}")
    print(f"Sites with valid habitat_suitability: {len(final_matrix)}")
    print(f"Species columns: {len([col for col in final_matrix.columns if col.startswith('species_')])}")
    
    return final_matrix

def habitat_suitability_sensitivity_analysis(df, cutoffs=None, n_permutations=999):
    """
    Analyze how IndVal scores change with different habitat_suitability cutoffs.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Site × species matrix with habitat_suitability column
    cutoffs : list
        List of habitat_suitability cutoffs to test
    n_permutations : int
        Number of permutations for IndVal significance testing
    
    Returns:
    --------
    pd.DataFrame
        Results showing cutoff values and corresponding IndVal metrics
    """
    if cutoffs is None:
        cutoffs = np.linspace(0.6, 0.9, 5)
    
    print("Starting Habitat Suitability Sensitivity Analysis...")
    print(f"Testing {len(cutoffs)} cutoffs: {cutoffs}")
    
    # Extract species columns
    species_columns = [col for col in df.columns if col.startswith('species_')]
    print(f"Found {len(species_columns)} species columns")
    
    results = []
    cutoff_details = []
    
    for i, cutoff in enumerate(cutoffs):
        print(f"\n--- Analyzing cutoff {i+1}/{len(cutoffs)}: {cutoff:.3f} ---")
        
        # Create vcm_label based on cutoff
        # habitat_suitability <= cutoff means VCM (less suitable for stable conifer)
        # habitat_suitability > cutoff means Non-VCM (more suitable for stable conifer)
        df_cutoff = df.copy()
        df_cutoff['vcm_label'] = (df_cutoff['habitat_suitability'] <= cutoff).astype(int)
        
        # Check group sizes
        vcm_counts = df_cutoff['vcm_label'].value_counts().sort_index()
        print(f"Group sizes - VCM (≤{cutoff:.3f}): {vcm_counts.get(1, 0)}, Non-VCM (>{cutoff:.3f}): {vcm_counts.get(0, 0)}")
        
        # Check if we have both groups with sufficient size
        if len(vcm_counts) < 2:
            print(f"Warning: Only one group at cutoff {cutoff}, skipping...")
            continue
        
        min_group_size = min(vcm_counts.values)
        if min_group_size < 10:  # Minimum group size threshold
            print(f"Warning: Smallest group has only {min_group_size} sites, results may be unreliable")
        
        # Extract species data
        X = df_cutoff[species_columns]
        y = df_cutoff['vcm_label']
        
        # Run IndVal analysis
        print(f"Running IndVal analysis with {n_permutations} permutations...")
        try:
            indval_results = calculate_indval_metrics(X, y, n_permutations=n_permutations)
            
            # Filter for VCM indicators (cluster 1)
            vcm_indicators = indval_results[indval_results['Cluster'] == 1].copy()
            
            if len(vcm_indicators) == 0:
                print(f"Warning: No VCM indicators found at cutoff {cutoff}")
                continue
            
            # Sort by IndVal score and get top 10
            vcm_indicators_sorted = vcm_indicators.sort_values('IndVal', ascending=False)
            top_10 = vcm_indicators_sorted.head(10)
            
            # Calculate metrics
            sum_top_10_indval = top_10['IndVal'].sum()
            mean_top_10_indval = top_10['IndVal'].mean()
            max_indval = top_10['IndVal'].max() if len(top_10) > 0 else 0
            
            # Count significant indicators
            significant_vcm = vcm_indicators[vcm_indicators['pvalue_adj'] < 0.05]
            n_significant = len(significant_vcm)
            
            print(f"Results: Sum top 10 IndVal = {sum_top_10_indval:.1f}, Mean = {mean_top_10_indval:.1f}, Max = {max_indval:.1f}")
            print(f"Significant VCM indicators (FDR < 0.05): {n_significant}")
            
            # Store results
            results.append({
                'cutoff': cutoff,
                'sum_top_10_indval': sum_top_10_indval,
                'mean_top_10_indval': mean_top_10_indval,
                'max_indval': max_indval,
                'n_significant_vcm': n_significant,
                'n_vcm_sites': vcm_counts.get(1, 0),
                'n_nonvcm_sites': vcm_counts.get(0, 0),
                'total_sites': len(df_cutoff)
            })
            
            # Store detailed species results for this cutoff
            top_10_with_cutoff = top_10.copy()
            top_10_with_cutoff['cutoff'] = cutoff
            cutoff_details.append(top_10_with_cutoff)
            
        except Exception as e:
            print(f"Error processing cutoff {cutoff}: {str(e)}")
            continue
    
    results_df = pd.DataFrame(results)
    details_df = pd.concat(cutoff_details, ignore_index=True) if cutoff_details else pd.DataFrame()
    
    return results_df, details_df

def plot_sensitivity_results(results_df, output_dir=""):
    """
    Create comprehensive plots of sensitivity analysis results.
    """
    if len(results_df) == 0:
        print("No results to plot")
        return None
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a 2x2 subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Sum of top 10 IndVal scores
    ax1.plot(results_df['cutoff'], results_df['sum_top_10_indval'], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Habitat Suitability Cutoff')
    ax1.set_ylabel('Sum of Top 10 IndVal Scores')
    ax1.set_title('IndVal Sensitivity to Habitat Suitability Cutoff')
    ax1.grid(True, alpha=0.3)
    
    # Highlight maximum
    max_idx = results_df['sum_top_10_indval'].idxmax()
    max_cutoff = results_df.loc[max_idx, 'cutoff']
    max_score = results_df.loc[max_idx, 'sum_top_10_indval']
    ax1.plot(max_cutoff, max_score, 'r*', markersize=15, label=f'Max: {max_score:.1f} at {max_cutoff:.3f}')
    ax1.legend()
    
    # Plot 2: Sample sizes
    ax2.plot(results_df['cutoff'], results_df['n_vcm_sites'], 'o-', label='VCM sites', linewidth=2)
    ax2.plot(results_df['cutoff'], results_df['n_nonvcm_sites'], 's-', label='Non-VCM sites', linewidth=2)
    ax2.set_xlabel('Habitat Suitability Cutoff')
    ax2.set_ylabel('Number of Sites')
    ax2.set_title('Sample Sizes by Cutoff')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mean IndVal of top 10
    ax3.plot(results_df['cutoff'], results_df['mean_top_10_indval'], 'o-', color='green', linewidth=2, markersize=8)
    ax3.set_xlabel('Habitat Suitability Cutoff')
    ax3.set_ylabel('Mean IndVal Score (Top 10)')
    ax3.set_title('Mean IndVal Score of Top 10 Species')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Number of significant indicators
    ax4.bar(results_df['cutoff'], results_df['n_significant_vcm'], alpha=0.7, color='orange')
    ax4.set_xlabel('Habitat Suitability Cutoff')
    ax4.set_ylabel('Number of Significant VCM Indicators')
    ax4.set_title('Significant VCM Indicators (FDR < 0.05)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        plot_path = os.path.join(output_dir, 'habitat_suitability_sensitivity_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Sensitivity plot saved to: {plot_path}")
    
    return fig

def save_results(results_df, details_df, output_dir=""):
    """
    Save analysis results to CSV files.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary results
        summary_path = os.path.join(output_dir, 'sensitivity_summary.csv')
        results_df.to_csv(summary_path, index=False)
        print(f"Summary results saved to: {summary_path}")
        
        # Save detailed species results
        if not details_df.empty:
            details_path = os.path.join(output_dir, 'sensitivity_species_details.csv')
            details_df.to_csv(details_path, index=False)
            print(f"Detailed species results saved to: {details_path}")
        
        # Print summary statistics
        print("\n=== SENSITIVITY ANALYSIS SUMMARY ===")
        max_idx = results_df['sum_top_10_indval'].idxmax()
        optimal_cutoff = results_df.loc[max_idx, 'cutoff']
        max_score = results_df.loc[max_idx, 'sum_top_10_indval']
        
        print(f"Optimal cutoff: {optimal_cutoff:.3f}")
        print(f"Maximum sum of top 10 IndVal scores: {max_score:.1f}")
        print(f"VCM sites at optimal cutoff: {results_df.loc[max_idx, 'n_vcm_sites']}")
        print(f"Non-VCM sites at optimal cutoff: {results_df.loc[max_idx, 'n_nonvcm_sites']}")
        print(f"Significant VCM indicators at optimal cutoff: {results_df.loc[max_idx, 'n_significant_vcm']}")

def main():
    """Main execution function with command-line arguments."""
    parser = argparse.ArgumentParser(description='Habitat Suitability Sensitivity Analysis for IndVal scores')
    parser.add_argument('--input', type=str, default='data_sources/inat-table-for-parthav-alt-lat.csv',
                        help='Path to input CSV file (observation format)')
    parser.add_argument('--output-dir', type=str, default='outputs/sensitivity_analysis',
                        help='Directory to save output files')
    parser.add_argument('--cutoffs', type=str, default='0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95',
                        help='Comma-separated list of cutoff values to test')
    parser.add_argument('--permutations', type=int, default=999,
                        help='Number of permutations for IndVal significance testing')
    parser.add_argument('--min-species-count', type=int, default=100,
                        help='Minimum observations required per species')
    parser.add_argument('--top-species', type=int, default=300,
                        help='Number of top species to include in analysis')
    parser.add_argument('--use-mean', action='store_true',
                        help='Use mean instead of median for habitat_suitability aggregation')
    
    args = parser.parse_args()
    
    # Parse cutoffs
    cutoffs = [float(x.strip()) for x in args.cutoffs.split(',')]
    print(f"Using cutoffs: {cutoffs}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load raw observation data
    print(f"Loading observation data from {args.input}...")
    raw_data = pd.read_csv(args.input)
    print(f"Loaded {len(raw_data)} observations")
    
    # Check for required columns
    required_columns = ['species', 'decimallatitude', 'decimallongitude', 'habitat_suitability']
    missing_columns = [col for col in required_columns if col not in raw_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove rows with missing habitat_suitability
    clean_data = raw_data[raw_data['habitat_suitability'].notna()].copy()
    print(f"Observations with valid habitat_suitability: {len(clean_data)}")
    
    # Preprocess to site × species matrix
    site_species_matrix = preprocess_data_to_matrix(
        clean_data, 
        min_species_count=args.min_species_count,
        n_top_species=args.top_species,
        use_median=not args.use_mean
    )
    
    # Save the preprocessed matrix for reference
    matrix_path = os.path.join(args.output_dir, 'site_species_matrix.csv')
    site_species_matrix.to_csv(matrix_path)
    print(f"Site × species matrix saved to: {matrix_path}")
    
    # Run sensitivity analysis
    results_df, details_df = habitat_suitability_sensitivity_analysis(
        site_species_matrix, cutoffs=cutoffs, n_permutations=args.permutations
    )
    
    if len(results_df) == 0:
        print("No valid results generated. Check your data and cutoff values.")
        return
    
    # Create plots
    fig = plot_sensitivity_results(results_df, args.output_dir)
    
    # Save results
    save_results(results_df, details_df, args.output_dir)
    
    # Show plot
    plt.show()
    
    print(f"\nAnalysis complete! Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main() 