'''
This file generates a new csv file for the datasetw ith the addition of the grid location codes
'''


import pandas as pd

# Load the dataset
file_path = 'data_sources/inat-table-for-parthav-with-lat-long.csv'
data = pd.read_csv(file_path)

# Drop observations with coordinate uncertainty greater than 1000 meters
data = data[data['coordinateuncertaintyinmeters'] <= 1000]

# Define latitude and longitude increments for a 1 km x 1 km grid
# At the equator, 1 km is approximately 0.008998 degrees latitude and longitude.
LATITUDE_INCREMENT = 0.008998
LONGITUDE_INCREMENT = 0.008998

# Define a function to calculate grid box identifier
def calculate_grid_location(lat, lon):
    lat_grid = int(lat // LATITUDE_INCREMENT)
    lon_grid = int(lon // LONGITUDE_INCREMENT)
    return f"{lat_grid}_{lon_grid}"

# Apply the function to each observation
data['grid_location'] = data.apply(lambda row: calculate_grid_location(row['decimallatitude'], row['decimallongitude']), axis=1)

# Display the updated dataset to the user (or save to file if needed)
#data.to_csv('outputs/updated_data_with_grid_location.csv', index=False)