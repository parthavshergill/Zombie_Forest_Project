import pandas as pd
import rasterio
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt

# File paths
data_csv = 'data_sources/processed-inat-data-complete.csv'
raster_path = 'data_sources/Gymno800mGAM_ecoSN401520.tif'

# Load point data
data_large = pd.read_csv(data_csv)
geometry = [Point(xy) for xy in zip(data_large['decimallongitude'], data_large['decimallatitude'])]
data_gdf = gpd.GeoDataFrame(data_large, geometry=geometry, crs='EPSG:4326')

# Open raster dataset
with rasterio.open(raster_path) as src:
    transform = src.transform  # Affine transformation
    raster_array = src.read(1)  # Read first band
    
    # Use rasterio sample function for raster extraction
    coords = [(point.x, point.y) for point in data_gdf.geometry]
    sampled_values = [val[0] for val in src.sample(coords)]
    
    # Assign extracted values to the DataFrame
    data_gdf['grid_location'] = sampled_values

# Drop rows with no grid assigned
data_gdf = data_gdf.dropna(subset=['grid_location'])

# Count unique grid boxes
unique_grid_boxes = data_gdf['grid_location'].nunique()
print(f"Unique grid boxes: {unique_grid_boxes}")

# Count observations per grid box
grid_box_counts = data_gdf['grid_location'].value_counts().reset_index()
grid_box_counts.columns = ['grid_location', 'observation_count']

# Summary statistics
print(f"Mean observations per grid box: {grid_box_counts['observation_count'].mean()}")
print(f"Median observations per grid box: {grid_box_counts['observation_count'].median()}")
print(f"Max observations in a grid box: {grid_box_counts['observation_count'].max()}")
print(f"Min observations in a grid box: {grid_box_counts['observation_count'].min()}")

# Plot distribution of observations per grid box
plt.figure(figsize=(10, 6))
plt.hist(grid_box_counts['observation_count'], bins=30, edgecolor='k')
plt.xlabel('Number of Observations per Grid Box')
plt.ylabel('Frequency')
plt.title('Distribution of Observations per Grid Box')
plt.show()

# Show top grid boxes with most observations
top_grid_boxes = grid_box_counts.head(10)
print("Top grid boxes with the most observations:")
print(top_grid_boxes)

# Select top 300 species based on occurrence counts
species_occurrence_counts = data_gdf['species'].value_counts()
top_species = species_occurrence_counts.head(300).index.tolist()

# Filter data to include only the top 300 species
data_gdf_top_species = data_gdf[data_gdf['species'].isin(top_species)]

# Convert species presence into binary (1 if present in a grid, 0 otherwise)
species_presence = data_gdf_top_species.groupby(['grid_location', 'species']).size().unstack(fill_value=0)
species_presence = (species_presence > 0).astype(int)

# Prefix species column names with 'species_'
species_presence.columns = [f"species_{col}" for col in species_presence.columns]

# Calculate the average of bio variables for each grid location
bio_columns = [col for col in data_gdf.columns if col.startswith("bio")]
bio_averages = data_gdf.groupby(['grid_location'])[bio_columns].mean()

# Aggregate veg_class and VCM label by grid location
data_matrix_gdf = pd.concat([species_presence, bio_averages], axis=1)
data_matrix_gdf['veg_class'] = data_gdf.groupby('grid_location')['veg_class'].agg(lambda x: x.mode()[0])
data_matrix_gdf['vcm_label'] = data_gdf.groupby('grid_location')['vcm_label'].mean().apply(lambda x: 1 if x >= 0.5 else 0)

# Save the new inat-data-matrix-gdf
output_file_gdf = "inat-data-matrix-gdf.csv"
data_matrix_gdf.to_csv(output_file_gdf)
print(f"Data matrix with species presence/absence, bio variable averages, majority veg_class, and VCM labels saved to {output_file_gdf}.")
