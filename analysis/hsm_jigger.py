import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data_sources/inat-table-for-parthav-alt-lat.csv')

# Remove rows with missing conifer_vcm_class or habitat_suitability
df = df[df['conifer_vcm_class'].notna() & df['habitat_suitability'].notna()]

# Print columns for debugging
print(df.columns)
print(df['conifer_vcm_class'].unique())
print(f"number of vcm obs are: {len(df[df['conifer_vcm_class'] != 'stable conifer'])}")
print(f"number of non-vcm obs are: {len(df[df['conifer_vcm_class'] == 'stable conifer'])}")

# Define VCM: not stable_conifer, Non-VCM: stable_conifer
df['is_vcm'] = (df['conifer_vcm_class'] != 'stable conifer').astype(int)

# Define cutoffs
cutoffs = np.linspace(df['habitat_suitability'].min(), df['habitat_suitability'].max(), 50)
vcm_counts = []
non_vcm_counts = []

for cutoff in cutoffs:
    vcm = ((df['habitat_suitability'] <= cutoff)).sum()
    non_vcm = ((df['habitat_suitability'] > cutoff)).sum()
    vcm_counts.append(vcm)
    non_vcm_counts.append(non_vcm)

plt.figure(figsize=(10, 6))
plt.plot(cutoffs, vcm_counts, label='VCM (HSM < cutoff)')
plt.plot(cutoffs, non_vcm_counts, label='Non-VCM (HSM >= cutoff)')
plt.xlabel('Habitat Suitability Cutoff')
plt.ylabel('Count above cutoff')
plt.title('VCM vs. Non-VCM as Habitat Suitability Cutoff Changes')
plt.legend()
plt.tight_layout()
plt.savefig('vcm_vs_nonvcm_cutoff.png')
plt.show()