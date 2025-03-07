import pandas as pd
import rasterio
import geopandas as gpd
import numpy as np
from shapely import wkt
import matplotlib.pyplot as plt
from typing import List, Tuple

def load_and_filter_data(data_csv: str) -> pd.DataFrame:
    """
    Load data from CSV and filter out unwanted taxonomic entries.
    
    Args:
        data_csv: Path to the input CSV file
    Returns:
        Filtered DataFrame
    """
    data_large = pd.read_csv(data_csv)
    print(f"Data before filtering: {len(data_large)}")
    
    unwanted_mask = (
        (data_large['class'] == 'Pinopsida') |
        (data_large['family'] == 'Lauraceae') |
        (data_large['genus'].isin(['Populus', 'Arbutus'])) |
        (data_large['order'] == 'Fagales') | 
        (data_large['veg_class'] == 'Broadleaf-dominated')
    )
    data_filtered = data_large[~unwanted_mask]
    print(f"Data after filtering: {len(data_filtered)}")
    
    return data_filtered

def create_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert DataFrame to GeoDataFrame using WKT geometry.
    
    Args:
        df: Input DataFrame with geometry column
    Returns:
        GeoDataFrame with proper geometry
    """
    df['geometry'] = df['geometry'].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

def assign_grid_locations(gdf: gpd.GeoDataFrame, raster_path: str) -> gpd.GeoDataFrame:
    """
    Assign grid locations to points using a raster file.
    
    Args:
        gdf: Input GeoDataFrame
        raster_path: Path to the raster file
    Returns:
        GeoDataFrame with grid locations assigned
    """
    with rasterio.open(raster_path) as src:
        coords = [(point.x, point.y) for point in gdf.geometry]
        sampled_values = [val[0] for val in src.sample(coords)]
        gdf['grid_location'] = sampled_values
    
    return gdf.dropna(subset=['grid_location'])

def analyze_grid_distribution(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Analyze and visualize the distribution of observations across grid boxes.
    
    Args:
        gdf: GeoDataFrame with grid_location column
    Returns:
        DataFrame with grid box counts
    """
    unique_grid_boxes = gdf['grid_location'].nunique()
    print(f"Unique grid boxes: {unique_grid_boxes}")
    
    grid_box_counts = gdf['grid_location'].value_counts().reset_index()
    grid_box_counts.columns = ['grid_location', 'observation_count']
    
    # Print summary statistics
    print(f"Mean observations per grid box: {grid_box_counts['observation_count'].mean()}")
    print(f"Median observations per grid box: {grid_box_counts['observation_count'].median()}")
    print(f"Max observations in a grid box: {grid_box_counts['observation_count'].max()}")
    print(f"Min observations in a grid box: {grid_box_counts['observation_count'].min()}")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(grid_box_counts['observation_count'], bins=30, edgecolor='k')
    plt.xlabel('Number of Observations per Grid Box')
    plt.ylabel('Frequency')
    plt.title('Distribution of Observations per Grid Box')
    plt.show()
    
    print("Top grid boxes with the most observations:")
    print(grid_box_counts.head(10))
    
    return grid_box_counts

def create_species_presence_matrix(gdf: gpd.GeoDataFrame, n_top_species: int = 300) -> pd.DataFrame:
    """
    Create a species presence/absence matrix for top N species.
    
    Args:
        gdf: Input GeoDataFrame
        n_top_species: Number of top species to include
    Returns:
        Species presence/absence matrix
    """
    # Select top species
    species_occurrence_counts = gdf['species'].value_counts()
    top_species = species_occurrence_counts.head(n_top_species).index.tolist()
    
    # Filter data for top species
    data_top_species = gdf[gdf['species'].isin(top_species)]
    
    # Create presence/absence matrix
    species_presence = data_top_species.groupby(['grid_location', 'species']).size().unstack(fill_value=0)
    species_presence = (species_presence > 0).astype(int)
    
    # Ensure all grid locations are included
    full_grid_index = pd.Index(gdf['grid_location'].unique())
    species_presence = species_presence.reindex(full_grid_index, fill_value=0)
    species_presence.columns = [f"species_{col}" for col in species_presence.columns]
    
    return species_presence

def create_final_matrix(gdf: gpd.GeoDataFrame, species_presence: pd.DataFrame) -> pd.DataFrame:
    """
    Create the final data matrix combining species presence, bio variables, and labels.
    
    Args:
        gdf: Input GeoDataFrame
        species_presence: Species presence/absence matrix
    Returns:
        Final combined data matrix
    """
    full_grid_index = pd.Index(gdf['grid_location'].unique())
    
    # Calculate bio variable averages
    bio_columns = [col for col in gdf.columns if col.startswith("bio")]
    bio_averages = gdf.groupby('grid_location')[bio_columns].mean().reindex(full_grid_index)
    
    # Aggregate veg_class and VCM label
    grid_veg_class = gdf.groupby('grid_location')['veg_class'].agg(lambda x: x.mode()[0]).reindex(full_grid_index)
    grid_vcm_label = gdf.groupby('grid_location')['vcm_label'].mean().apply(lambda x: 1 if x >= 0.5 else 0).reindex(full_grid_index)
    
    # Combine all components
    data_matrix = pd.concat([species_presence, bio_averages], axis=1)
    data_matrix['veg_class'] = grid_veg_class
    data_matrix['vcm_label'] = grid_vcm_label
    
    return data_matrix

def main():
    """Main execution function."""
    # File paths
    data_csv = 'data_sources/processed-inat-data-complete_01_29_25.csv'
    raster_path = 'data_sources/Gymno800mGAM_ecoSN401520.tif'
    output_file = "inat-data-matrix-gdf.csv"
    
    # Process data
    data_filtered = load_and_filter_data(data_csv)
    data_gdf = create_geodataframe(data_filtered)
    data_gdf = assign_grid_locations(data_gdf, raster_path)
    
    # Analyze distribution
    grid_box_counts = analyze_grid_distribution(data_gdf)
    
    # Create final matrix
    species_presence = create_species_presence_matrix(data_gdf)
    data_matrix = create_final_matrix(data_gdf, species_presence)
    
    # Save results
    data_matrix.to_csv(output_file)
    print(f"Data matrix saved to {output_file}")

if __name__ == "__main__":
    main()
