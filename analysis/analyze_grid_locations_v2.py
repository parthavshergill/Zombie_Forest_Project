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
    
    # Assign unique values to each raster cell
    unique_values = np.zeros_like(raster_array, dtype=int)
    
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
