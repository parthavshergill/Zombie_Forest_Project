import pandas as pd
from scipy.stats import pearsonr

# Load the dataset
input_file = "data_sources/inat-data-matrix.csv"
data = pd.read_csv(input_file)

print(f"There are {len(data)} grid boxes in the data matrix.")

# Filter species columns and vcm_label
species_columns = [col for col in data.columns if col.startswith("species_")]
features = species_columns + ["vcm_label"]

# Initialize a dictionary to store correlations and p-values
correlation_results = {"species": [], "pearson_correlation": [], "p_value": [], "significant": []}

# Compute Pearson correlation and p-values for each species with vcm_label
for species in species_columns:
    try:
        corr, p_value = pearsonr(data[species], data["vcm_label"])
        correlation_results["species"].append(species)
        correlation_results["pearson_correlation"].append(corr)
        correlation_results["p_value"].append(p_value)
        # Flag significance based on a p-value threshold (e.g., 0.05)
        correlation_results["significant"].append(p_value < 0.05)
    except Exception as e:
        # If there's an error (e.g., constant data), append NaN
        correlation_results["species"].append(species)
        correlation_results["pearson_correlation"].append(float('nan'))
        correlation_results["p_value"].append(float('nan'))
        correlation_results["significant"].append(False)

# Convert the results to a DataFrame
correlation_df = pd.DataFrame(correlation_results)

# Save the full correlation results to a CSV file
correlation_df.to_csv("pearson_vcm_correlations.csv", index=False)

# Print a summary of significant correlations
significant_correlations = correlation_df[correlation_df["significant"]]
print(f"Number of significant correlations: {len(significant_correlations)}")
print("Significant correlations:")
print(significant_correlations)

# Save the significant correlations to a separate CSV file
significant_correlations.to_csv("significant_pearson_vcm_correlations.csv", index=False)

print("Pearson correlation analysis complete, with significant correlations flagged.")
