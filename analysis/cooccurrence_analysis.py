import pandas as pd
from scipy.stats import chi2_contingency

# Reload the uploaded file path
file_path = 'data_sources/inat-data-matrix.csv'
data = pd.read_csv(file_path)

print(f"Number of grid boxes: {len(data)}")

# Identify species columns by excluding non-species columns
species_columns = [col for col in data.columns if col.startswith('species_')]

# Extract the VCM label column
vcm_label = data['vcm_label']

# Initialize a dictionary to store co-occurrence ratios and significance results
results = []

# Calculate co-occurrence ratio and perform significance test for each species
for species in species_columns:
    # Create a contingency table
    contingency_table = pd.crosstab(data[species] > 0, vcm_label)
    
    # Perform Chi-squared test of independence
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    
    # Calculate co-occurrence ratio
    co_occurrence = contingency_table.loc[True, 1] / contingency_table.loc[True].sum() if 1 in contingency_table.columns else 0
    
    # Append results
    results.append({
        "species": species,
        "co_occurrence_ratio": co_occurrence,
        "p_value": p_value
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Flag statistically significant results (p-value < 0.05)
results_df['significant'] = results_df['p_value'] < 0.05

# Filter significant results and sort by co-occurrence ratio in descending order
significant_results = results_df[results_df['significant']].sort_values(by='co_occurrence_ratio', ascending=False)

# Save the full results and the significant results to CSV files
results_df.to_csv("cooccurrence_ratio_sig_test.csv", index=False)
significant_results.to_csv("significant_cooccurrence_ratios.csv", index=False)

# Display a summary of the top significant results
print(f"Number of statistically significant species: {len(significant_results)}")
print("Top significant species with highest co-occurrence ratios:")
print(significant_results.head())

print("Analysis complete. Results saved to 'cooccurrence_ratio_sig_test.csv' and 'significant_cooccurrence_ratios.csv'.")
