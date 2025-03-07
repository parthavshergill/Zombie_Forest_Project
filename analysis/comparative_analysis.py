import pandas as pd
import numpy as np

def analyze_top_species_occurrences():
    """Analyze occurrence patterns of top indicator species from both analyses."""
    
    # Load the raw data
    data = pd.read_csv("inat-data-matrix-gdf.csv")
    
    # Load results from both analyses - fix the paths to handle spaces
    indval_results = pd.read_csv("outputs_rerun/indval results/significant_indicators.csv")
    cooccurrence_results = pd.read_csv("outputs_rerun/cooccurrence results/significant_associations.csv")
    
    # Get top 10 species from each analysis
    top_indval = indval_results.nlargest(10, 'IndVal')
    top_cooccurrence = cooccurrence_results.nlargest(10, 'co_occurrence_ratio')
    
    # Combine unique species from both analyses
    all_top_species = pd.concat([
        top_indval['Species'],
        top_cooccurrence['species']
    ]).unique()
    print("All top species:", all_top_species)
    
    # Print debug info
    print("\nAvailable columns in data:")
    print(data.columns.tolist())
    
    # Calculate occurrence statistics for each species
    results = []
    for species in all_top_species:
        # Get the column name from the species name
        species_col = species if species.startswith('species_') else f'species_{species}'
        
        print(f"\nProcessing species: {species}")
        print(f"Looking for column: {species_col}")
        
        if species_col in data.columns:
            print(f"Found column for: {species}")
            # Calculate occurrences
            vcm_occurrences = data[data['vcm_label'] == 1][species_col].sum()
            non_vcm_occurrences = data[data['vcm_label'] == 0][species_col].sum()
            total_occurrences = vcm_occurrences + non_vcm_occurrences
            
            # Calculate percentages
            vcm_percentage = (vcm_occurrences / total_occurrences * 100) if total_occurrences > 0 else 0
            
            results.append({
                'Species': species,
                'Total_Occurrences': total_occurrences,
                'VCM_Occurrences': vcm_occurrences,
                'Non_VCM_Occurrences': non_vcm_occurrences,
                'Percent_in_VCM': vcm_percentage,
                'In_IndVal_Top10': species in top_indval['Species'].values,
                'In_Cooccurrence_Top10': species in top_cooccurrence['species'].values
            })
        else:
            print(f"WARNING: Could not find column for species: {species}")
    
    if not results:
        print("\nERROR: No results were generated. Check species names and column names.")
        return None
    
    # Convert to DataFrame and sort by total occurrences
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Total_Occurrences', ascending=False)
    
    # Print results
    print("\nOccurrence Patterns for Top Indicator Species:")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"\n{row['Species']}:")
        print(f"Total observations: {row['Total_Occurrences']}")
        print(f"VCM areas: {row['VCM_Occurrences']} ({row['Percent_in_VCM']:.1f}%)")
        print(f"Non-VCM areas: {row['Non_VCM_Occurrences']}")
        print(f"Identified by: " + 
              ("Both analyses" if row['In_IndVal_Top10'] and row['In_Cooccurrence_Top10']
               else "IndVal only" if row['In_IndVal_Top10']
               else "Co-occurrence only" if row['In_Cooccurrence_Top10']
               else "Neither analysis"))
    
    # Save to CSV
    results_df.to_csv("top_species_occurrences.csv", index=False)
    return results_df

if __name__ == "__main__":
    results = analyze_top_species_occurrences()