import pandas as pd
import numpy as np

# Function to calculate IndVal for a binary group
def indval_binary(X, y, group=1):
    """
    Compute IndVal for each species in X, for the specified group.
    X : pd.DataFrame, species presence/absence or abundance data (samples x species)
    y : pd.Series, binary group labels (e.g., 0 or 1)
    group : target group label (e.g., 1 for VCM presence)
    Returns: pd.DataFrame with IndVal results ['species', 'A', 'B', 'IndVal']
    """
    results = []
    for species in X.columns:
        x_all = X[species].values
        x_group = X.loc[y == group, species]
        
        # Specificity (A): mean presence in the group / mean presence overall
        mean_in_group = x_group.mean()
        mean_all = x_all.mean()
        A = mean_in_group / mean_all if mean_all > 0 else 0
        
        # Fidelity (B): proportion of sites in the group with species present
        B = mean_in_group
        
        # IndVal = A * B * 100
        IndVal = A * B * 100
        results.append({'species': species, 'A': A, 'B': B, 'IndVal': IndVal})
    return pd.DataFrame(results).sort_values('IndVal', ascending=False)

# Function to run permutation test
def indval_permutation_test(X, y, group=1, n_permutations=99):
    """
    Perform permutation test for IndVal significance.
    X : pd.DataFrame, species data (samples x species)
    y : pd.Series, binary group labels (e.g., 0 or 1)
    group : target group label (e.g., 1 for VCM presence)
    n_permutations : number of permutations
    Returns: pd.DataFrame with IndVal results and p-values.
    """
    # Precompute necessary group data for efficiency
    group_mask = (y == group)
    group_size = group_mask.sum()
    total_size = len(y)

    # Observed IndVal
    observed = indval_binary(X, y, group)
    observed_vals = observed.set_index('species')['IndVal']

    # Pre-allocate p-values
    pvals = pd.Series(0, index=observed_vals.index, dtype=float)

    # Run permutation test
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        group_mask_perm = (y_perm == group)
        for species in X.columns:
            # Compute permutation IndVal for this species
            x_group_perm = X.loc[group_mask_perm, species].mean()
            x_all_perm = X[species].mean()
            A_perm = x_group_perm / x_all_perm if x_all_perm > 0 else 0
            B_perm = x_group_perm
            indval_perm = A_perm * B_perm * 100
            if indval_perm >= observed_vals[species]:
                pvals[species] += 1

    # Calculate final p-values
    pvals = (pvals + 1) / (n_permutations + 1)  # Conservative estimate

    # Combine results
    results = observed.copy()
    results['pvalue'] = results['species'].map(pvals)
    return results.sort_values('IndVal', ascending=False)

# Load your dataset
file_path = 'data_sources/inat-data-matrix.csv'  # Change this to your file path
data = pd.read_csv(file_path)

# Prepare the species data and target labels
species_columns = [col for col in data.columns if col.startswith('species_')]
X_species = data[species_columns]
y_vcm = data['vcm_label']

# Calculate IndVal and run permutation test
indval_results = indval_permutation_test(X_species, y_vcm, group=1, n_permutations=99)

# Flag species with high IndVal and low p-value
flagged_species = indval_results[(indval_results['IndVal'] > 50) & (indval_results['pvalue'] < 0.05)]

# Save results to a CSV file
indval_results.to_csv("indval_results.csv", index=False)
flagged_species.to_csv("flagged_species.csv", index=False)

# Display results
print("Top 10 Indicator Species by IndVal:")
print(indval_results.head(10))
print("\nFlagged Species with High IndVal and Low P-Value:")
print(flagged_species)
