import pandas as pd
from analysis import k_means_analysis
import matplotlib.pyplot as plt

file_path = 'large_dataset.csv'  # Replace with your actual file path
sample_size = 10000
print("Reading data...")
data = pd.read_csv('data_sources/data_format_sample.csv')
print("Data read.")

# Select relevant columns
relevant_data = data[['species', 'decimallatitude', 'decimallongitude', 'conifer_class']]

# Drop rows with missing species or conifer_class
relevant_data = relevant_data.dropna(subset=['species', 'conifer_class'])

k_means_analysis.identify_best_k(relevant_data) # k = 6 minimizes silhouette score 

relevant_data['cluster'] = k_means_analysis.k_means(relevant_data, k=6)

k_means_analysis.visualize_clusters(relevant_data)

# Convert categorical variables like species and conifer_class to 'category' dtype for memory efficiency
relevant_data['species'] = relevant_data['species'].astype('category')
relevant_data['conifer_class'] = relevant_data['conifer_class'].astype('category')





