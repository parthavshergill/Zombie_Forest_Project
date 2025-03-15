import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import argparse

# Define latitude and longitude increments for a 1 km x 1 km grid
# At the equator, 1 km is approximately 0.008998 degrees
LATITUDE_INCREMENT = 0.008998
LONGITUDE_INCREMENT = 0.008998

def load_and_filter_data(data_csv: str, altitude_threshold: float = None, min_species_count: int = 100) -> pd.DataFrame:
    """
    Load data from CSV and filter out unwanted taxonomic entries, by altitude if specified,
    and species with fewer than the specified minimum number of observations.
    
    Args:
        data_csv: Path to the input CSV file
        altitude_threshold: Optional maximum altitude threshold
        min_species_count: Minimum number of observations required for a species to be included
    Returns:
        Filtered DataFrame
    """
    data_large = pd.read_csv(data_csv)
    print(f"Data before filtering: {len(data_large)}")

    vcm_count = len(data_large[data_large['composite_zf_class'].str.contains('VCM')])
    non_vcm_count = len(data_large) - vcm_count
    print(f"The ratio of VCM to non-VCM observations before filtering is {vcm_count} : {non_vcm_count} ({vcm_count/non_vcm_count:.3f})")
    
    # Filter by unwanted taxonomic entries
    unwanted_mask = (
        (data_large['class'] == 'Pinopsida') |
        (data_large['family'] == 'Lauraceae') |
        (data_large['genus'].isin(['Populus', 'Arbutus'])) |
        (data_large['order'] == 'Fagales') | 
        (data_large['veg_class'] == 'Broadleaf-dominated')
    )
    data_filtered = data_large[~unwanted_mask]
    print(f"Data after taxonomic filtering: {len(data_filtered)}")
    
    # Apply altitude filtering if threshold is provided
    if altitude_threshold is not None and 'altitude' in data_filtered.columns:
        original_count = len(data_filtered)
        data_filtered = data_filtered[data_filtered['altitude'] < altitude_threshold]
        print(f"Data after altitude filtering (<{altitude_threshold}m): {len(data_filtered)}")
        print(f"Removed {original_count - len(data_filtered)} records with altitude ≥{altitude_threshold}m")
    
    # Filter by coordinate uncertainty if available
    if 'coordinateuncertaintyinmeters' in data_filtered.columns:
        original_count = len(data_filtered)
        data_filtered = data_filtered[data_filtered['coordinateuncertaintyinmeters'] < 1000]
        print(f"Removed {original_count - len(data_filtered)} records with coordinate uncertainty ≥1000m")
    
    # Filter species with fewer than min_species_count observations
    if 'species' in data_filtered.columns:
        original_count = len(data_filtered)
        species_counts = data_filtered['species'].value_counts()
        common_species = species_counts[species_counts >= min_species_count].index
        data_filtered = data_filtered[data_filtered['species'].isin(common_species)]
        
        removed_species = len(species_counts) - len(common_species)
        removed_observations = original_count - len(data_filtered)
        
        print(f"Removed {removed_species} species with fewer than {min_species_count} observations")
        print(f"This eliminated {removed_observations} records ({removed_observations/original_count*100:.1f}% of data)")
        print(f"Remaining species: {len(common_species)}")
        print(f"Remaining observations: {len(data_filtered)}")
    
    vcm_count = len(data_filtered[data_filtered['composite_zf_class'].str.contains('VCM')])
    non_vcm_count = len(data_filtered) - vcm_count
    print(f"The ratio of VCM to non-VCM observations after filtering is {vcm_count} : {non_vcm_count} ({vcm_count/non_vcm_count:.3f})")
    
    return data_filtered

def assign_grid_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate grid locations based on latitude and longitude.
    
    Args:
        df: Input DataFrame with decimallatitude and decimallongitude columns
    Returns:
        DataFrame with grid_location column added
    """
    # Define function to calculate grid box identifier
    def calculate_grid_location(lat, lon):
        lat_grid = int(lat // LATITUDE_INCREMENT)
        lon_grid = int(lon // LONGITUDE_INCREMENT)
        return f"{lat_grid}_{lon_grid}"
    
    # Apply the function to each observation
    df['grid_location'] = df.apply(
        lambda row: calculate_grid_location(row['decimallatitude'], row['decimallongitude']), 
        axis=1
    )
    
    print(f"Assigned grid locations to {len(df)} observations")
    return df

def analyze_grid_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze and visualize the distribution of observations across grid boxes.
    
    Args:
        df: DataFrame with grid_location column
    Returns:
        DataFrame with grid box counts
    """
    unique_grid_boxes = df['grid_location'].nunique()
    print(f"Unique grid boxes: {unique_grid_boxes}")
    
    grid_box_counts = df['grid_location'].value_counts().reset_index()
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

def create_species_presence_matrix(df: pd.DataFrame, n_top_species: int = 300) -> pd.DataFrame:
    """
    Create a species presence/absence matrix for top N species.
    
    Args:
        df: Input DataFrame
        n_top_species: Number of top species to include
    Returns:
        Species presence/absence matrix
    """
    # Select top species
    species_occurrence_counts = df['species'].value_counts()
    top_species = species_occurrence_counts.head(n_top_species).index.tolist()
    
    print(f"Selected top {len(top_species)} species")
    
    # Filter data for top species
    data_top_species = df[df['species'].isin(top_species)]
    
    # Create presence/absence matrix
    species_presence = data_top_species.groupby(['grid_location', 'species']).size().unstack(fill_value=0)
    species_presence = (species_presence > 0).astype(int)
    
    # Ensure all grid locations are included
    full_grid_index = pd.Index(df['grid_location'].unique())
    species_presence = species_presence.reindex(full_grid_index, fill_value=0)
    species_presence.columns = [f"species_{col}" for col in species_presence.columns]
    
    return species_presence

def create_final_matrix(df: pd.DataFrame, species_presence: pd.DataFrame) -> pd.DataFrame:
    """
    Create the final data matrix combining species presence, bio variables, and labels.
    
    Args:
        df: Input DataFrame
        species_presence: Species presence/absence matrix
    Returns:
        Final combined data matrix
    """
    full_grid_index = pd.Index(df['grid_location'].unique())
    
    # Calculate bio variable averages
    bio_columns = [col for col in df.columns if col.startswith("bio")]
    if bio_columns:
        bio_averages = df.groupby('grid_location')[bio_columns].mean().reindex(full_grid_index)
    else:
        bio_averages = pd.DataFrame(index=full_grid_index)
        print("Warning: No bio columns found in the dataset")
    
    # Aggregate veg_class and VCM label
    if 'veg_class' in df.columns:
        grid_veg_class = df.groupby('grid_location')['veg_class'].agg(lambda x: x.mode()[0]).reindex(full_grid_index)
    else:
        grid_veg_class = pd.Series(index=full_grid_index)
        print("Warning: No veg_class column found in the dataset")
    
    if 'vcm_label' in df.columns:
        grid_vcm_label = df.groupby('grid_location')['vcm_label'].mean().apply(lambda x: 1 if x >= 0.5 else 0).reindex(full_grid_index)
    else:
        # If VCM label doesn't exist, create it from composite_zf_class if available
        if 'composite_zf_class' in df.columns:
            print("Creating VCM labels from composite_zf_class...")
            df['vcm_label'] = df['composite_zf_class'].apply(lambda x: 1 if 'VCM' in str(x) else 0)
            grid_vcm_label = df.groupby('grid_location')['vcm_label'].mean().apply(lambda x: 1 if x >= 0.5 else 0).reindex(full_grid_index)
        else:
            grid_vcm_label = pd.Series(index=full_grid_index)
            print("Warning: No vcm_label or composite_zf_class column found in the dataset")
    
    # Combine all components
    data_matrix = pd.concat([species_presence, bio_averages], axis=1)
    data_matrix['veg_class'] = grid_veg_class
    data_matrix['vcm_label'] = grid_vcm_label
    
    return data_matrix

def main():
    """Main execution function with command-line arguments."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process iNaturalist data to create gridded data matrix using lat/long.')
    parser.add_argument('--input', type=str, default='data_sources/inat-table-for-parthav-alt-lat.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='inat-data-matrix-latlong.csv',
                        help='Path to output CSV file')
    parser.add_argument('--altitude', type=float, default=2500,
                        help='Maximum altitude threshold for filtering (optional)')
    parser.add_argument('--top-species', type=int, default=300,
                        help='Number of top species to include in the matrix')
    parser.add_argument('--min-species-count', type=int, default=100)
    args = parser.parse_args()
    
    # Process data
    data_filtered = load_and_filter_data(args.input, args.altitude, args.min_species_count)
    data_with_grid = assign_grid_locations(data_filtered)
    
    # Analyze distribution
    grid_box_counts = analyze_grid_distribution(data_with_grid)
    
    # Create final matrix
    species_presence = create_species_presence_matrix(data_with_grid, args.top_species)
    data_matrix = create_final_matrix(data_with_grid, species_presence)
    
    # Save results
    data_matrix.to_csv(args.output)
    print(f"Data matrix saved to {args.output}")
    
    # Save processed data with grid locations (optional)
    processed_output = args.output.replace('.csv', '-complete.csv')
    data_with_grid.to_csv(processed_output, index=False)
    print(f"Processed data with grid locations saved to {processed_output}")

if __name__ == "__main__":
    main()
