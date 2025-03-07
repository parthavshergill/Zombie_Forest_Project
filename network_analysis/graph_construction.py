import networkx as nx
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
import time

def create_bipartite_graph(
    data: pd.DataFrame,
    species_columns: List[str],
    norm_indval: Dict[str, float]
) -> nx.Graph:
    """
    Construct bipartite graph from data.
    
    Args:
        data: Input DataFrame
        species_columns: List of species column names
        norm_indval: Dictionary of normalized IndVal scores
    Returns:
        NetworkX bipartite graph
    """
    start_time = time.time()
    B = nx.Graph()
    
    # Add grid cell nodes with vcm_label
    print("Adding grid cell nodes...")
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Adding grid cells"):
        grid_id = row['grid_location']
        vcm = row['vcm_label']
        B.add_node(grid_id, bipartite='grid', vcm_label=vcm)
    
    # Add species nodes with normalized IndVal score
    print("Adding species nodes...")
    for species, score in tqdm(norm_indval.items(), desc="Adding species"):
        B.add_node(species, bipartite='species', indval=score)
    
    # Add edges with weights
    print("Adding edges...")
    edge_count = 0
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Adding edges"):
        grid_id = row['grid_location']
        for species in species_columns:
            if row[species] > 0:
                weight = norm_indval[species]
                B.add_edge(grid_id, species, weight=weight)
                edge_count += 1
    
    elapsed = time.time() - start_time
    print(f"Created bipartite graph with {len(B.nodes)} nodes and {edge_count} edges in {elapsed:.2f} seconds")
    return B

def create_incidence_matrix(
    data: pd.DataFrame,
    species_columns: List[str],
    norm_indval: Dict[str, float]
) -> pd.DataFrame:
    """
    Create weighted incidence matrix.
    
    Args:
        data: Input DataFrame
        species_columns: List of species column names
        norm_indval: Dictionary of normalized IndVal scores
    Returns:
        DataFrame representing the incidence matrix
    """
    start_time = time.time()
    grid_ids = data['grid_location'].unique()
    incidence = pd.DataFrame(0, index=grid_ids, columns=species_columns)
    
    print(f"Creating incidence matrix for {len(grid_ids)} grid cells and {len(species_columns)} species...")
    for grid in tqdm(grid_ids, desc="Processing grid cells"):
        grid_data = data[data['grid_location'] == grid]
        for species in species_columns:
            if (grid_data[species] > 0).any():
                incidence.loc[grid, species] = norm_indval[species]
    
    elapsed = time.time() - start_time
    print(f"Created incidence matrix with shape {incidence.shape} in {elapsed:.2f} seconds")
    return incidence

def create_incidence_matrix_optimized(
    data: pd.DataFrame,
    species_columns: List[str],
    norm_indval: Dict[str, float]
) -> pd.DataFrame:
    """
    Create weighted incidence matrix using vectorized operations.
    
    Args:
        data: Input DataFrame
        species_columns: List of species column names
        norm_indval: Dictionary of normalized IndVal scores
    Returns:
        DataFrame representing the incidence matrix
    """
    start_time = time.time()
    
    # Get unique grid cells
    print("Getting unique grid cells...")
    grid_ids = data['grid_location'].unique()
    print(f"Found {len(grid_ids)} unique grid cells")
    
    # Create pivot table with grid_location as index and species columns
    print("\nCreating pivoted presence matrix...")
    pivot_start = time.time()
    
    # Process species columns in chunks for better memory management
    chunk_size = 50  # Process 50 species at a time
    presence = pd.DataFrame(index=grid_ids)
    
    for i in range(0, len(species_columns), chunk_size):
        chunk_columns = species_columns[i:i+chunk_size]
        print(f"\nProcessing species columns {i+1}-{min(i+chunk_size, len(species_columns))} of {len(species_columns)}")
        
        chunk_presence = pd.pivot_table(
            data, 
            index='grid_location',
            values=chunk_columns,
            aggfunc=lambda x: 1 if (x > 0).any() else 0
        )
        presence = pd.concat([presence, chunk_presence], axis=1)
    
    pivot_elapsed = time.time() - pivot_start
    print(f"\nPivot table creation completed in {pivot_elapsed:.2f} seconds")
    
    # Multiply each species column by its normalized IndVal score
    print("\nApplying IndVal weights to matrix...")
    weight_start = time.time()
    
    for species in tqdm(species_columns, desc="Weighting species"):
        if species in presence.columns:
            presence[species] = presence[species] * norm_indval[species]
    
    weight_elapsed = time.time() - weight_start
    print(f"Applied weights in {weight_elapsed:.2f} seconds")
    
    total_elapsed = time.time() - start_time
    print(f"\nCreated optimized incidence matrix with shape {presence.shape} in {total_elapsed:.2f} seconds")
    
    # Report memory usage
    memory_usage = presence.memory_usage(deep=True).sum() / 1024**2  # Convert to MB
    print(f"Matrix memory usage: {memory_usage:.2f} MB")
    
    return presence 