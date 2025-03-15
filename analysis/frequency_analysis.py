import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

def analyze_species_frequencies(input_file, output_prefix, min_occurrences=10):
    """
    Analyze frequency of species occurrences in VCM and non-VCM areas.
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file with species presence and VCM data
    output_prefix : str
        Prefix for output files (can include directory path)
    min_occurrences : int
        Minimum number of species occurrences required for analysis
    """
    print(f"\nLoading data from {input_file}...")
    try:
        data = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Extract species columns
    species_cols = [col for col in data.columns if col.startswith('species_')]
    print(f"Found {len(species_cols)} species columns")
    
    # Check for VCM column
    if 'vcm_label' not in data.columns and 'vcm' not in data.columns:
        print("Error: VCM column not found in dataset")
        return None
    
    # Set VCM column
    vcm_col = 'vcm_label' if 'vcm_label' in data.columns else 'vcm'
    
    # Count occurrences
    vcm_sites = data[vcm_col] == 1
    non_vcm_sites = data[vcm_col] == 0
    
    # Count total sites
    total_sites = len(data)
    vcm_site_count = vcm_sites.sum()
    non_vcm_site_count = non_vcm_sites.sum()
    
    print(f"Total sites: {total_sites}")
    print(f"VCM sites: {vcm_site_count} ({vcm_site_count/total_sites*100:.1f}%)")
    print(f"Non-VCM sites: {non_vcm_site_count} ({non_vcm_site_count/total_sites*100:.1f}%)")
    
    # Calculate occurrence statistics for each species
    results = []
    skipped_species = 0
    invalid_p_values = 0
    
    # Process all species columns
    for species_col in species_cols:
        species_name = species_col
        
        # Calculate occurrences
        total_occurrences = data[species_col].sum()
        
        # Skip very rare species (optional threshold)
        if total_occurrences < min_occurrences:
            skipped_species += 1
            continue
            
        vcm_occurrences = data.loc[vcm_sites, species_col].sum()
        non_vcm_occurrences = data.loc[non_vcm_sites, species_col].sum()
        
        # Calculate percentages
        vcm_percentage = (vcm_occurrences / total_occurrences * 100) if total_occurrences > 0 else 0
        
        # Calculate site-level frequencies
        vcm_frequency = (vcm_occurrences / vcm_site_count * 100) if vcm_site_count > 0 else 0
        non_vcm_frequency = (non_vcm_occurrences / non_vcm_site_count * 100) if non_vcm_site_count > 0 else 0
        
        # Calculate frequency ratio (as indicator of preference)
        if non_vcm_frequency > 0:
            frequency_ratio = vcm_frequency / non_vcm_frequency
        else:
            frequency_ratio = float('inf') if vcm_frequency > 0 else 0
            
        # Calculate binomial test for difference between VCM and non-VCM frequencies
        try:
            success = int(vcm_occurrences)  # Ensure integer
            trials = int(total_occurrences)  # Ensure integer
            expected = vcm_site_count / total_sites  # Expected proportion based on VCM site ratio
            
            # Handle edge cases
            if trials == 0:
                p_value = 1.0  # No occurrences, no significance
            elif success > trials:
                # This shouldn't happen, but just in case of data issues
                print(f"Warning: Success > trials for {species_name}, capping success at trials value")
                success = trials
                p_value = 1.0
            elif expected <= 0 or expected >= 1:
                # Edge case where expected proportion is at the boundary
                p_value = 1.0
            else:
                # Normal case: run binomial test
                p_value_result = stats.binomtest(success, trials, expected)
                # Handle different return types across scipy versions
                if hasattr(p_value_result, 'pvalue'):
                    p_value = p_value_result.pvalue
                else:
                    p_value = p_value_result
                
                # Validate p-value
                if not (0 <= p_value <= 1):
                    print(f"Warning: Invalid p-value {p_value} for {species_name}. Using 1.0 instead.")
                    p_value = 1.0
                    invalid_p_values += 1
                    
        except Exception as e:
            print(f"Error calculating p-value for {species_name}: {e}")
            p_value = 1.0
            invalid_p_values += 1
        
        # Determine if species has preference for VCM or non-VCM
        preference = "VCM" if vcm_percentage > (vcm_site_count/total_sites*100) else "non-VCM"
        
        # Store results
        results.append({
            'Species': species_name,
            'Total_Occurrences': total_occurrences,
            'VCM_Occurrences': vcm_occurrences,
            'Non_VCM_Occurrences': non_vcm_occurrences,
            'Percent_in_VCM': vcm_percentage,
            'VCM_Frequency': vcm_frequency,
            'Non_VCM_Frequency': non_vcm_frequency,
            'Frequency_Ratio': frequency_ratio,
            'p_value': p_value,
            'Preference': preference
        })
    
    print(f"Skipped {skipped_species} species with fewer than {min_occurrences} occurrences")
    print(f"Encountered {invalid_p_values} issues with p-value calculations")
    
    if not results:
        print("\nERROR: No results were generated. Check species data format.")
        return None
    
    # Convert to DataFrame and sort by total occurrences
    results_df = pd.DataFrame(results)
    
    # Print p-value diagnostics
    print(f"\nP-value diagnostics:")
    print(f"Range: {results_df['p_value'].min():.2e} to {results_df['p_value'].max():.2e}")
    print(f"Mean: {results_df['p_value'].mean():.4f}")
    print(f"Median: {results_df['p_value'].median():.4f}")
    print(f"NaN values: {results_df['p_value'].isna().sum()}")
    print(f"Values = 0: {(results_df['p_value'] == 0).sum()}")
    print(f"Values = 1: {(results_df['p_value'] == 1).sum()}")
    
    # Fix any remaining invalid p-values
    results_df['p_value'] = results_df['p_value'].fillna(1.0)
    
    # Apply FDR correction for multiple testing
    try:
        # Use statsmodels implementation
        reject, qvals = fdrcorrection(results_df['p_value'].values)
        results_df['adjusted_p_value'] = qvals
        print(f"Applied FDR correction successfully")
    except Exception as e:
        print(f"Warning: Error applying FDR correction: {e}")
        print("Using raw p-values instead.")
        results_df['adjusted_p_value'] = results_df['p_value']
    
    # Create a significance flag
    results_df['significant'] = results_df['adjusted_p_value'] < 0.05
    sig_count = results_df['significant'].sum()
    print(f"Found {sig_count} species with significant habitat preference (FDR-corrected p < 0.05)")
    
    # Create VCM preference strength indicator
    # Compute the expected VCM percentage based on site distribution
    expected_vcm_pct = vcm_site_count / total_sites * 100
    
    # Calculate deviation from expectation
    results_df['VCM_Preference_Strength'] = results_df['Percent_in_VCM'] - expected_vcm_pct
    
    # Sort by preference strength (absolute value for top deviators in either direction)
    results_df['abs_preference'] = abs(results_df['VCM_Preference_Strength'])
    results_df_sorted = results_df.sort_values('abs_preference', ascending=False)
    
    # Save full results
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    results_df.to_csv(f"{output_prefix}species_frequency_analysis.csv", index=False)
    
    # Save significant results
    sig_results = results_df[results_df['significant']].sort_values('VCM_Preference_Strength', ascending=False)
    sig_results.to_csv(f"{output_prefix}significant_vcm_preferences.csv", index=False)
    
    # Summary statistics
    print("\nFrequency Analysis Summary:")
    print(f"Total species analyzed: {len(results_df)}")
    print(f"Species with significant VCM preference: {len(sig_results[sig_results['VCM_Preference_Strength'] > 0])}")
    print(f"Species with significant non-VCM preference: {len(sig_results[sig_results['VCM_Preference_Strength'] < 0])}")
    
    # Visualizations
    create_visualizations(results_df, output_prefix, expected_vcm_pct)
    
    # Print top VCM and non-VCM species
    top_vcm = sig_results[sig_results['VCM_Preference_Strength'] > 0].head(10)
    top_non_vcm = sig_results[sig_results['VCM_Preference_Strength'] < 0].sort_values('VCM_Preference_Strength').head(10)
    
    print("\nTop 10 VCM-preferring species:")
    for i, (_, species) in enumerate(top_vcm.iterrows(), 1):
        print(f"{i}. {species['Species']}: {species['Percent_in_VCM']:.1f}% in VCM (vs expected {expected_vcm_pct:.1f}%), p={species['adjusted_p_value']:.3e}")
    
    print("\nTop 10 non-VCM-preferring species:")
    for i, (_, species) in enumerate(top_non_vcm.iterrows(), 1):
        print(f"{i}. {species['Species']}: {species['Percent_in_VCM']:.1f}% in VCM (vs expected {expected_vcm_pct:.1f}%), p={species['adjusted_p_value']:.3e}")
    
    return results_df

def create_visualizations(results_df, output_prefix, expected_vcm_pct):
    """
    Create visualizations for frequency analysis results.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with frequency analysis results
    output_prefix : str
        Prefix for output files
    expected_vcm_pct : float
        Expected percentage in VCM based on site distribution
    """
    # Set style
    plt.style.use('default')
    sns.set_palette("colorblind")
    
    # 1. Distribution of VCM percentages
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['Percent_in_VCM'], bins=30, kde=True)
    plt.axvline(expected_vcm_pct, color='red', linestyle='--', 
                label=f'Expected ({expected_vcm_pct:.1f}%)')
    plt.xlabel('Percentage of occurrences in VCM')
    plt.ylabel('Number of species')
    plt.title('Distribution of species occurrences in VCM areas')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}vcm_percentage_distribution.png")
    plt.close()
    
    # 2. Scatterplot of occurrence frequency by habitat
    plt.figure(figsize=(10, 8))
    
    # Plot points
    scatter = plt.scatter(
        results_df['Non_VCM_Frequency'], 
        results_df['VCM_Frequency'],
        c=results_df['significant'],
        alpha=0.6,
        cmap='coolwarm',
        s=results_df['Total_Occurrences']/10 + 10
    )
    
    # Add diagonal line (equal frequency)
    max_val = max(results_df['Non_VCM_Frequency'].max(), results_df['VCM_Frequency'].max()) * 1.1
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal frequency')
    
    # Add expected VCM proportion line
    slope = expected_vcm_pct / (100 - expected_vcm_pct) 
    plt.plot([0, max_val], [0, max_val * slope], 'r--', alpha=0.5, 
             label=f'Expected ratio ({expected_vcm_pct:.1f}%)')
    
    # Label top species (significant with extreme preferences)
    sig_species = results_df[results_df['significant']].sort_values('abs_preference', ascending=False).head(15)
    for _, row in sig_species.iterrows():
        plt.annotate(
            row['Species'],
            (row['Non_VCM_Frequency'], row['VCM_Frequency']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    plt.xlabel('Frequency in non-VCM areas (%)')
    plt.ylabel('Frequency in VCM areas (%)')
    plt.title('Species occurrence frequencies by habitat type')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}frequency_comparison.png")
    plt.close()
    
    # 3. Top species by VCM preference (both directions)
    plt.figure(figsize=(12, 8))
    
    # Get top 15 species in each direction
    top_vcm = results_df[results_df['significant'] & (results_df['VCM_Preference_Strength'] > 0)].nlargest(15, 'VCM_Preference_Strength')
    top_non_vcm = results_df[results_df['significant'] & (results_df['VCM_Preference_Strength'] < 0)].nsmallest(15, 'VCM_Preference_Strength')
    
    # Combine and sort
    top_species = pd.concat([top_vcm, top_non_vcm])
    
    # Create plot
    bar_colors = ['forestgreen' if x > 0 else 'royalblue' for x in top_species['VCM_Preference_Strength']]
    ax = top_species.sort_values('VCM_Preference_Strength').plot.barh(
        x='Species', 
        y='VCM_Preference_Strength',
        figsize=(12, 10),
        color=bar_colors
    )
    
    # Add vertical line at 0
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add expected value line
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Expected')
    
    plt.xlabel('VCM preference strength (difference from expected %)')
    plt.title('Top species by habitat preference')
    plt.grid(axis='x', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}top_habitat_preferences.png")
    plt.close()
    
    # 4. Relationship between occurrence and preference strength
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot of total occurrences vs preference strength
    plt.scatter(
        results_df['Total_Occurrences'], 
        results_df['VCM_Preference_Strength'],
        c=results_df['significant'],
        cmap='coolwarm',
        alpha=0.6
    )
    
    # Highlight top species
    top_diverse = results_df.nlargest(10, 'Total_Occurrences')
    for _, row in top_diverse.iterrows():
        plt.annotate(
            row['Species'],
            (row['Total_Occurrences'], row['VCM_Preference_Strength']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            color='purple'
        )
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Total occurrences')
    plt.ylabel('VCM preference strength')
    plt.title('Relationship between species abundance and habitat preference')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}occurrence_vs_preference.png")
    plt.close()

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze frequency of species occurrences in VCM and non-VCM areas.'
    )
    parser.add_argument('--input', type=str, default='inat-data-matrix-latlong.csv',
                      help='Path to input CSV file (default: inat-data-matrix-latlong.csv)')
    parser.add_argument('--output-prefix', type=str, default='outputs/frequency_analysis/',
                      help='Prefix for output files (default: outputs/frequency_analysis/)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output_prefix:
        os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    
    # Run analysis
    results = analyze_species_frequencies(args.input, args.output_prefix)
    
    if results is not None:
        print(f"\nAnalysis complete. Results saved to {args.output_prefix}")
        print(f"CSV outputs:")
        print(f"- {args.output_prefix}species_frequency_analysis.csv")
        print(f"- {args.output_prefix}significant_vcm_preferences.csv")
        print(f"\nVisualizations:")
        print(f"- {args.output_prefix}vcm_percentage_distribution.png")
        print(f"- {args.output_prefix}frequency_comparison.png")
        print(f"- {args.output_prefix}top_habitat_preferences.png")
        print(f"- {args.output_prefix}occurrence_vs_preference.png")

if __name__ == "__main__":
    main()