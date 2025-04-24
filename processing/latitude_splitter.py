import pandas as pd
import os
import argparse
from pathlib import Path

def split_data_by_region(data_csv: str, output_dir: str) -> tuple[str, str]:
    """
    Split the dataset into North Sierra and South Sierra based on latitude_eco_group.
    
    Args:
        data_csv: Path to the input CSV file
        output_dir: Directory to save the split datasets
        
    Returns:
        Tuple of (north_sierra_path, south_sierra_path)
    """
    print(f"\nSplitting data from {data_csv} by region...")
    
    # Load data
    data = pd.read_csv(data_csv)
    print(f"Total observations: {len(data)}")
    
    # Split based on latitude_eco_group
    north_sierra = data[data['latitude_eco_group'] == 'North Sierra']
    south_sierra = data[data['latitude_eco_group'] == 'South Sierra']
    
    print(f"North Sierra observations: {len(north_sierra)}")
    print(f"South Sierra observations: {len(south_sierra)}")
    
    # Create output paths
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(data_csv).stem
    
    north_path = os.path.join(output_dir, f"{base_name}_north_sierra.csv")
    south_path = os.path.join(output_dir, f"{base_name}_south_sierra.csv")
    
    # Save split datasets
    north_sierra.to_csv(north_path, index=False)
    south_sierra.to_csv(south_path, index=False)
    
    print(f"North Sierra data saved to: {north_path}")
    print(f"South Sierra data saved to: {south_path}")
    
    return north_path, south_path

def main():
    """Main execution function with command-line arguments."""
    parser = argparse.ArgumentParser(description='Split iNaturalist data into North and South Sierra regions.')
    parser.add_argument('--input', type=str, default="data_sources/inat-table-for-parthav-alt-lat.csv",
                      help='Path to input CSV file')
    parser.add_argument('--output-dir', type=str, default='data_sources/regional_splits',
                      help='Directory to save split datasets')
    
    args = parser.parse_args()
    
    # Split the data
    north_path, south_path = split_data_by_region(args.input, args.output_dir)
    
    print("\nData splitting complete!")
    print(f"Use these files as input for the processing pipeline:")
    print(f"North Sierra: {north_path}")
    print(f"South Sierra: {south_path}")

if __name__ == "__main__":
    main()
