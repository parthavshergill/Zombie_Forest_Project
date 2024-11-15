import pandas as pd
from analysis import k_means_analysis
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'data_sources/inat-table-for-parthav.csv'  # Replace with your actual file path
sample_size = 10000

print("Reading data...")
chunks = pd.read_csv(file_path, chunksize=sample_size)
print("Data read.")

data = next(chunks)

# Select relevant columns
relevant_data = data[['species', 'decimallatitude', 'decimallongitude', 'conifer_class']]
print(relevant_data)

# Drop rows with missing species or conifer_class
relevant_data = relevant_data.dropna(subset=['species', 'conifer_class'])

# k_means_analysis.identify_best_k(relevant_data) # k = 4 minimizes silhouette score 

# relevant_data['cluster'] = k_means_analysis.k_means(relevant_data, k=4)

# k_means_analysis.visualize_clusters(relevant_data)

# Convert categorical variables like species and conifer_class to 'category' dtype for memory efficiency
relevant_data['species'] = relevant_data['species'].astype('category')
relevant_data['conifer_class'] = relevant_data['conifer_class'].astype('category')

# a. Geographic distribution of observations (latitude and longitude)
plt.figure(figsize=(10, 6))
plt.scatter(relevant_data['decimallongitude'], relevant_data['decimallatitude'], c='blue', alpha=0.5)
plt.title('Geographic Distribution of Observations (Random Sample of 10,000)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# b. Distribution of conifer_class
plt.figure(figsize=(10, 6))
sns.countplot(x='conifer_class', data=relevant_data)
plt.title('Distribution of Conifer Class')
plt.show()

# c. Most common species in the sample
plt.figure(figsize=(10, 6))
relevant_data['species'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Most Common Species in the Sample')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()

# 4. Identify correlations with the conifer_class column

# Checking correlations only for numerical columns
numerical_columns = relevant_data.select_dtypes(include=['float64', 'int64']).columns

# Plot a heatmap for correlations between numerical columns and conifer_class
# Encoding conifer_class as numbers for correlation calculation
relevant_data['conifer_class_encoded'] = relevant_data['conifer_class'].cat.codes

# Calculate the correlation matrix
corr_matrix = relevant_data[numerical_columns].corrwith(relevant_data['conifer_class_encoded'])

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix.to_frame(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation of Numerical Features with Conifer Class')
plt.show()





