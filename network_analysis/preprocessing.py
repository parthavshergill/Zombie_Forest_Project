import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple
from tqdm import tqdm
import time

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the data matrix from CSV file.
    
    Args:
        filepath: Path to the input CSV file
    Returns:
        DataFrame containing the data matrix
    """
    start_time = time.time()
    # Read CSV with first column as index
    data = pd.read_csv(filepath, index_col=0)
    
    # Convert index to a column named 'grid_location'
    data = data.reset_index()
    data = data.rename(columns={'index': 'grid_location'})
    
    elapsed = time.time() - start_time
    print(f"Loaded data with shape: {data.shape} in {elapsed:.2f} seconds")
    return data

def get_species_columns(data: pd.DataFrame) -> List[str]:
    """
    Get list of species columns from the dataset.
    
    Args:
        data: Input DataFrame
    Returns:
        List of column names representing species
    """
    return [col for col in data.columns if col.startswith('species_')]

def compute_indval_scores(data: pd.DataFrame, species_columns: List[str], vcm_label_col: str = 'vcm_label') -> Dict[str, float]:
    """
    Compute proper IndVal scores for each species.
    
    IndVal = A × B
    where:
    - A is the specificity (relative abundance in target group vs all groups)
    - B is the fidelity (relative frequency in target group)
    
    Args:
        data: Input DataFrame
        species_columns: List of species column names
        vcm_label_col: Column name for VCM label
    Returns:
        Dictionary mapping species to their IndVal scores
    """
    indval_scores = {}
    
    # Split data into VCM=1 and VCM=0 groups
    vcm_sites = data[data[vcm_label_col] == 1]
    non_vcm_sites = data[data[vcm_label_col] == 0]
    
    n_vcm = len(vcm_sites)
    n_total = len(data)
    
    print(f"Computing proper IndVal scores for {len(species_columns)} species...")
    for species in tqdm(species_columns, desc="Computing IndVal"):
        # Count presence in VCM and all sites
        vcm_presence = (vcm_sites[species] > 0).sum()
        total_presence = (data[species] > 0).sum()
        
        if total_presence == 0:
            indval_scores[species] = 0
            continue
            
        # Calculate specificity (A)
        specificity = vcm_presence / total_presence if total_presence > 0 else 0
        
        # Calculate fidelity (B)
        fidelity = vcm_presence / n_vcm if n_vcm > 0 else 0
        
        # IndVal = A × B
        indval_scores[species] = specificity * fidelity
    
    return indval_scores

def normalize_indval_scores(indval_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize IndVal scores using min-max scaling.
    
    Args:
        indval_scores: Dictionary of raw IndVal scores
    Returns:
        Dictionary of normalized IndVal scores
    """
    scaler = MinMaxScaler()
    species_list = list(indval_scores.keys())
    scores = np.array([indval_scores[s] for s in species_list]).reshape(-1, 1)
    normalized_scores = scaler.fit_transform(scores).flatten()
    
    return dict(zip(species_list, normalized_scores))

def create_presence_matrix(data: pd.DataFrame, species_columns: List[str]) -> pd.DataFrame:
    """
    Create binary presence matrix for species.
    
    Args:
        data: Input DataFrame
        species_columns: List of species column names
    Returns:
        DataFrame with binary presence values
    """
    return data[species_columns].applymap(lambda x: 1 if x > 0 else 0) 