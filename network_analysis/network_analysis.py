import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from community import community_louvain
import time

def project_to_species_network(
    incidence_matrix: pd.DataFrame, 
    data: pd.DataFrame, 
    vcm_only: bool = False
) -> Tuple[pd.DataFrame, nx.Graph]:
    """
    Project bipartite graph onto species network.
    
    Args:
        incidence_matrix: Weighted incidence matrix
        data: Original data with vcm_label
        vcm_only: If True, only use grid cells with vcm_label=1
    Returns:
        Tuple of (projection matrix, NetworkX graph)
    """
    start_time = time.time()
    
    # Optionally filter by VCM
    if vcm_only:
        print("Filtering to include only VCM=1 grid cells...")
        grid_ids = data[data['vcm_label'] == 1]['grid_location'].unique()
        incidence_filtered = incidence_matrix.loc[incidence_matrix.index.isin(grid_ids)]
        print(f"Filtered from {len(incidence_matrix)} to {len(incidence_filtered)} grid cells")
    else:
        incidence_filtered = incidence_matrix
    
    print("Computing projection matrix...")
    species_projection = incidence_filtered.T.dot(incidence_filtered)
    
    print("Creating NetworkX graph from projection...")
    species_network = nx.from_pandas_adjacency(species_projection)
    
    elapsed = time.time() - start_time
    print(f"Created species network with {len(species_network.nodes)} nodes in {elapsed:.2f} seconds")
    return species_projection, species_network

def analyze_network(G: nx.Graph) -> Dict:
    """
    Perform network analysis including community detection and centrality.
    
    Args:
        G: NetworkX graph
    Returns:
        Dictionary containing analysis results
    """
    results = {}
    
    print("Detecting communities...")
    start_time = time.time()
    partition = community_louvain.best_partition(G)
    elapsed = time.time() - start_time
    print(f"Found {len(set(partition.values()))} communities in {elapsed:.2f} seconds")
    
    # Save communities as node attributes
    nx.set_node_attributes(G, partition, 'community')
    results['communities'] = partition
    
    print("Computing centrality measures...")
    start_time = time.time()
    
    print("- Computing degree centrality...")
    degree_cent = nx.degree_centrality(G)
    nx.set_node_attributes(G, degree_cent, 'degree_centrality')
    results['degree_centrality'] = degree_cent
    
    print("- Computing betweenness centrality...")
    between_cent = nx.betweenness_centrality(G)
    nx.set_node_attributes(G, between_cent, 'betweenness_centrality')
    results['betweenness_centrality'] = between_cent
    
    print("- Computing eigenvector centrality...")
    eigen_cent = nx.eigenvector_centrality(G, max_iter=1000)
    nx.set_node_attributes(G, eigen_cent, 'eigenvector_centrality')
    results['eigenvector_centrality'] = eigen_cent
    
    elapsed = time.time() - start_time
    print(f"Computed all centrality measures in {elapsed:.2f} seconds")
    
    return results

def visualize_network(G: nx.Graph, output_path: str = None, label_top_n: int = 50):
    """
    Visualize the species network with improved readability.
    
    Args:
        G: NetworkX graph
        output_path: Optional path to save the plot
        label_top_n: Number of top nodes to label by centrality
    """
    start_time = time.time()
    
    print("Creating network visualization...")
    plt.figure(figsize=(16, 16))
    
    # Create a subplot with a specific axes
    ax = plt.subplot(111)
    
    print("Computing layout...")
    pos = nx.spring_layout(G, seed=42)
    
    # Get community colors
    communities = nx.get_node_attributes(G, 'community')
    node_colors = [communities.get(node, 0) for node in G.nodes()]
    
    # Compute node sizes based on centrality
    degree_cent = nx.get_node_attributes(G, 'degree_centrality')
    node_sizes = [1000 * degree_cent.get(node, 0.1) + 50 for node in G.nodes()]
    
    print("Drawing nodes...")
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors, 
        cmap=plt.cm.tab20,
        node_size=node_sizes,
        alpha=0.8,
        ax=ax  # Add ax parameter
    )
    
    print("Drawing edges...")
    # Calculate edge weights for line thickness
    edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + 5 * (w/max_weight) for w in edge_weights]
    
    nx.draw_networkx_edges(
        G, pos, 
        alpha=0.3, 
        width=edge_widths,
        ax=ax  # Add ax parameter
    )
    
    # Label only the top N nodes by centrality
    print(f"Adding labels for top {label_top_n} nodes...")
    if degree_cent:
        top_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:label_top_n]
        top_nodes_dict = {node: node for node, _ in top_nodes}
        nx.draw_networkx_labels(
            G, pos, 
            labels=top_nodes_dict,
            font_size=8,
            font_weight='bold',
            ax=ax  # Add ax parameter
        )
    
    plt.title("Species Co-occurrence Network", fontsize=20)
    plt.axis('off')
    
    # Add a colorbar for communities
    sm = plt.cm.ScalarMappable(cmap=plt.cm.tab20, norm=plt.Normalize(
        vmin=min(communities.values()) if communities else 0, 
        vmax=max(communities.values()) if communities else 1
    ))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Community', fontsize=14)
    
    if output_path:
        print(f"Saving visualization to {output_path}...")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    elapsed = time.time() - start_time
    print(f"Created visualization in {elapsed:.2f} seconds")
    plt.close()  # Close the figure instead of showing it to avoid display issues 