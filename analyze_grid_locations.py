import pandas as pd 

# Load the larger dataset with grid locations
file_path = 'data_sources/updated_data_with_grid_location.csv'
data_large = pd.read_csv(file_path)

print(len(data_large))

LATITUDE_INCREMENT = 0.008998
LONGITUDE_INCREMENT = 0.008998

# Count the unique grid boxes and calculate corresponding square kilometers
unique_grid_boxes = data_large['grid_location'].nunique()
square_kilometers = unique_grid_boxes  # Each grid box corresponds to 1 square kilometer

print(unique_grid_boxes, square_kilometers)

# Calculate the geographic spread based on the latitude and longitude ranges

# Find min and max latitude and longitude to determine the coverage area
min_latitude = data_large['decimallatitude'].min()
max_latitude = data_large['decimallatitude'].max()
min_longitude = data_large['decimallongitude'].min()
max_longitude = data_large['decimallongitude'].max()

# Calculate the latitudinal and longitudinal spread in kilometers
latitudinal_spread_km = (max_latitude - min_latitude) / LATITUDE_INCREMENT  # 1 km per 0.008998 latitude degrees
longitudinal_spread_km = (max_longitude - min_longitude) / LONGITUDE_INCREMENT  # 1 km per 0.008998 longitude degrees

print(latitudinal_spread_km*longitudinal_spread_km)

import pandas as pd
import matplotlib.pyplot as plt

data = data_large

# Count observations per grid box
grid_box_counts = data['grid_location'].value_counts().reset_index()
grid_box_counts.columns = ['grid_location', 'observation_count']

# Summary statistics
mean_observations = grid_box_counts['observation_count'].mean()
median_observations = grid_box_counts['observation_count'].median()
max_observations = grid_box_counts['observation_count'].max()

print(f"Mean observations per grid box: {mean_observations}")
print(f"Median observations per grid box: {median_observations}")
print(f"Max observations in a grid box: {max_observations}")

# Plot distribution of observations per grid box
plt.figure(figsize=(10, 6))
plt.hist(grid_box_counts['observation_count'], bins=30, edgecolor='k')
plt.xlabel('Number of Observations per Grid Box')
plt.ylabel('Frequency')
plt.title('Distribution of Observations per Grid Box')
plt.show()

# Show top grid boxes with the most observations
top_grid_boxes = grid_box_counts.head(10)
print("Top grid boxes with the most observations:")
print(top_grid_boxes)