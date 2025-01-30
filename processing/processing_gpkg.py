import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset from a GeoPackage file
input_file = "data_sources/inat-table-w-clim.gpkg"
data = gpd.read_file(input_file)

print(f"Number of observations is: {len(data)}")

# Retain observations with missing coordinate uncertainty or coordinate uncertainty < 1000
data = data[(data['coordinateuncertaintyinmeters'].isna()) | (data['coordinateuncertaintyinmeters'] < 1000)]

# Define latitude and longitude increments for a 1 km x 1 km grid
LATITUDE_INCREMENT = 0.008998
LONGITUDE_INCREMENT = 0.008998

def calculate_grid_location(lat, lon):
    lat_grid = int(lat // LATITUDE_INCREMENT)
    lon_grid = int(lon // LONGITUDE_INCREMENT)
    return f"{lat_grid}_{lon_grid}"

data['grid_location'] = data.apply(lambda row: calculate_grid_location(row['geometry'].y, row['geometry'].x), axis=1)

print(f"Remaining observations: {len(data)}")

print(f"There are {data['species'].nunique()} unique species in the dataset.")

# Add a VCM column
data['vcm_label'] = data['composite_zf_class'].apply(lambda x: 1 if 'VCM' in str(x) else 0)

# Calculate the majority VCM label for each grid location
grid_vcm_majority = data.groupby('grid_location')['vcm_label'].mean().apply(lambda x: 1 if x >= 0.5 else 0)

data = data.merge(grid_vcm_majority.rename('grid_vcm_label'), on='grid_location')
data['vcm_label'] = data['grid_vcm_label']

# Calculate the majority veg_class for each grid location
def calculate_majority(series):
    return series.mode()[0]  # Use the mode (most frequent value)

grid_veg_class_majority = data.groupby('grid_location')['veg_class'].apply(calculate_majority)

# Calculate the total occurrence count for each species across the dataset
species_occurrence_counts = data['species'].value_counts()

# Get the top 300 species by occurrence
top_species = species_occurrence_counts.head(300).index.tolist()

total_top_species_count = species_occurrence_counts.head(300).sum()
print(f"Total occurrence count of the top 300 species: {total_top_species_count}")

print("\nTop 300 species and their occurrence counts:")
print(species_occurrence_counts.head(300))

# Filter the data to include only the top 300 species
data_top_species = data[data['species'].isin(top_species)]

# Plot the distribution of occurrences for the top 300 species
plt.figure(figsize=(24, 12))
species_occurrence_counts.head(300).plot(kind='bar', color='blue', alpha=0.7)
plt.title('Distribution of Occurrences for the Top 300 Species')
plt.xlabel('Species')
plt.ylabel('Occurrence Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Calculate the average of bio variables for each grid location
bio_columns = [col for col in data_top_species.columns if col.startswith('bio')]
bio_averages = data_top_species.groupby('grid_location')[bio_columns].mean()

# Pivot table to create categorical columns for each species
species_counts = data_top_species.groupby(['grid_location', 'species']).size().reset_index(name='count')
species_matrix = species_counts.pivot_table(index='grid_location', columns='species', values='count', fill_value=0)

# Prefix each species column with 'species_'
species_matrix.columns = [f"species_{col}" for col in species_matrix.columns]

# Add the majority veg_class, bio averages, and grid-level VCM label
data_matrix = pd.concat([species_matrix, bio_averages], axis=1)
data_matrix['veg_class'] = grid_veg_class_majority
data_matrix['vcm_label'] = grid_vcm_majority

# Save the data matrix to a CSV file
output_file_matrix = "inat-data-matrix.csv"
data_matrix.to_csv(output_file_matrix)
print(f"Data matrix with species, bio variable averages, majority veg_class, and VCM labels saved to {output_file_matrix}.")

# Save the cleaned and processed data to a new CSV file
output_file = "processed-inat-data-complete.csv"
data.to_csv(output_file, index=False)
print(f"Data processing complete. Cleaned data with VCM labels saved to {output_file}.")
