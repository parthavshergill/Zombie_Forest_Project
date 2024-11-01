from scipy.stats import fisher_exact, chi2_contingency
import numpy as np
import pandas as pd

# Step 1: Calculate species co-occurrence with VCM
def species_vcm_cooccurrence_stats(data):
    # Drop rows where 'conifer_vcm_class' or 'species' is missing
    df = data.dropna(subset=['conifer_vcm_class', 'species'])
    
    # Encode VCM presence as binary (1 for VCM, 0 for other classes)
    df['is_vcm'] = df['conifer_vcm_class'].apply(lambda x: 1 if 'VCM' in x else 0)
    
    # Step 2: Calculate co-occurrence statistics
    species_stats = []
    for species, group in df.groupby('species'):
        vcm_count = group['is_vcm'].sum()
        non_vcm_count = len(group) - vcm_count
        obs_count = len(group)
        total_vcm = df['is_vcm'].sum()
        total_non_vcm = len(df) - total_vcm

        # Contingency table for statistical tests
        contingency_table = np.array([
            [vcm_count, total_vcm - vcm_count],
            [non_vcm_count, total_non_vcm - non_vcm_count]
        ])
        
        # Step 3: Calculate Fisher's Exact Test and Chi-square Test p-values
        try:
            fisher_p = fisher_exact(contingency_table)[1]
            chi2_p = chi2_contingency(contingency_table)[1]
        except ValueError:
            fisher_p, chi2_p = np.nan, np.nan  # Handle cases where the test fails
        
        species_stats.append({
            'species': species,
            'vcm_count': vcm_count,
            'non_vcm_count': non_vcm_count,
            'obs_count': obs_count,
            'fisher_p': fisher_p,
            'chi2_p': chi2_p
        })
    
    # Convert results to DataFrame for easy viewing
    species_stats_df = pd.DataFrame(species_stats)
    return species_stats_df

def raw_proportion_counts(data):
    # Load the dataset

    # Ensure 'vcm_count' and 'obs_count' are present in the data
    # Calculate the proportion of zombie forest observations for each species
    data['zombie_forest_proportion'] = data['vcm_count'] / data['obs_count']

    # Display the species and their corresponding zombie forest proportions
    zombie_forest_proportions = data[['species', 'zombie_forest_proportion']]
    return zombie_forest_proportions

# Calculate co-occurrence statistics
data = pd.read_csv('outputs/species_stats.csv')
#species_stats_df = species_vcm_cooccurrence_stats(data)
proportions = raw_proportion_counts(data[data['obs_count'] > 100])
proportions.to_csv('outputs/zombie_forest_cooccurrence_proportion_frequencies.csv')
#species_stats_df.to_csv('outputs/species_stats.csv', index=False)