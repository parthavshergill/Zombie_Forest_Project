import pandas as pd
import matplotlib.pyplot as plt

# Load the uploaded CSV file with test statistics for species
species_stats_path = 'outputs/species_stats.csv'
species_stats_df = pd.read_csv(species_stats_path)

# Filter species with more than 50 observations
filtered_species_df = species_stats_df[species_stats_df['obs_count'] > 100]

# Select the top 40 most significant species based on the Chi-square p-value
top_significant_df = filtered_species_df.nsmallest(50, 'chi2_p')

# Select the top 50 most significant species based on chi2_p and fisher_p
top_50_chi2 = set(filtered_species_df.nsmallest(10, 'chi2_p')['species'])
top_50_fisher = set(filtered_species_df.nsmallest(10, 'fisher_p')['species'])

# Calculate the intersection of the two sets to find species in both lists
common_species = top_50_chi2.intersection(top_50_fisher)
common_species_count = len(common_species)

print(len(filtered_species_df))

print('Common species are:', common_species_count, common_species)


# Plotting the Fisher p-values and Chi-square p-values for the top significant species
plt.figure(figsize=(20, 12))
plt.barh(top_significant_df['species'], top_significant_df['chi2_p'], color='blue', label='Chi-squared p-value')
#plt.barh(top_significant_df['species'], top_significant_df['chi2_p'], color='red', label='Chi-square p-value')

# Formatting the plot
plt.xscale('log')  # Log scale for better visibility of small p-values
plt.xlabel('P-Value (log scale)')
plt.ylabel('Species')
plt.title('Top 50 Most Significant Species by P-Values')
plt.legend()
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.tight_layout()
plt.show()