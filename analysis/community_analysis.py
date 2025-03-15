import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import community as community_louvain
from matplotlib.colors import rgb2hex
from sklearn.metrics import silhouette_score
from collections import defaultdict
import argparse
from pathlib import Path

def load_data(data_matrix_path, indval_path, cooccurrence_path, frequency_path):
    """
    Load all required data sources.
    
    Args:
        data_matrix_path: Path to the main data matrix
        indval_path: Path to IndVal analysis results
        cooccurrence_path: Path to co-occurrence analysis results
        frequency_path: Path to frequency analysis results
        
    Returns:
        Tuple of DataFrames with all required data
    """
    print("\n=== Loading Data ===")
    
    # Load main data matrix
    print(f"Loading data matrix from {data_matrix_path}")
    data_matrix = pd.read_csv(data_matrix_path)
    print(f"Data matrix shape: {data_matrix.shape}")
    
    # Check for required columns
    species_columns = [col for col in data_matrix.columns if col.startswith('species_')]
    print(f"Found {len(species_columns)} species columns")
    
    if 'vcm_label' not in data_matrix.columns:
        raise ValueError("vcm_label column not found in data matrix")
    
    # Load IndVal results
    print(f"Loading IndVal results from {indval_path}")
    indval_results = pd.read_csv(indval_path)
    print(f"IndVal results shape: {indval_results.shape}")
    
    # Load co-occurrence results
    print(f"Loading co-occurrence results from {cooccurrence_path}")
    cooccurrence_results = pd.read_csv(cooccurrence_path)
    print(f"Co-occurrence results shape: {cooccurrence_results.shape}")
    
    # Load frequency analysis results
    print(f"Loading frequency analysis results from {frequency_path}")
    frequency_results = pd.read_csv(frequency_path)
    print(f"Frequency results shape: {frequency_results.shape}")
    
    return data_matrix, indval_results, cooccurrence_results, frequency_results

def build_species_cooccurrence_network(data_matrix, species_columns):
    """
    Build a species co-occurrence network from the data matrix.
    
    Args:
        data_matrix: DataFrame containing species presence and VCM labels
        species_columns: List of species column names
        
    Returns:
        NetworkX graph of species co-occurrence in VCM grid cells
    """
    print("\n=== Building Co-occurrence Network ===")
    
    # Filter to VCM grid locations
    vcm_data = data_matrix[data_matrix['vcm_label'] == 1]
    print(f"Filtered to {len(vcm_data)} VCM grid locations")
    
    # Extract species presence matrix for VCM grids
    vcm_species = vcm_data[species_columns]
    
    # Calculate co-occurrence matrix
    print("Calculating species co-occurrence matrix...")
    cooccurrence_matrix = vcm_species.T.dot(vcm_species)
    
    # Set diagonal to zero (species don't co-occur with themselves)
    np.fill_diagonal(cooccurrence_matrix.values, 0)
    
    # Create graph from co-occurrence matrix
    print("Creating network graph...")
    G = nx.from_pandas_adjacency(cooccurrence_matrix)
    
    # Remove nodes with no connections
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print(f"Removed {len(isolated_nodes)} isolated species nodes")
    
    print(f"Final network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G, cooccurrence_matrix

def detect_communities(G, resolution=1.0):
    """
    Detect communities in the species co-occurrence network.
    
    Args:
        G: NetworkX graph of species co-occurrence
        
    Returns:
        Dictionary mapping node names to community IDs
    """
    print("\n=== Detecting Communities ===")
    
    # Apply Louvain community detection
    print("Applying Louvain community detection algorithm...")
    communities = community_louvain.best_partition(G, resolution=resolution)
    
    # Count communities
    community_counts = defaultdict(int)
    for node, community_id in communities.items():
        community_counts[community_id] += 1
    
    print(f"Found {len(community_counts)} communities")
    
    # Print community sizes
    print("Community sizes:")
    for community_id, count in sorted(community_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  Community {community_id}: {count} species")
    
    return communities

def evaluate_communities(communities, G, indval_results, cooccurrence_results, frequency_results):
    """
    Evaluate communities based on indicator metrics.
    
    Args:
        communities: Dictionary mapping nodes to community IDs
        G: NetworkX graph of species co-occurrence
        indval_results: DataFrame with IndVal scores
        cooccurrence_results: DataFrame with co-occurrence ratios
        frequency_results: DataFrame with VCM preference strengths
        
    Returns:
        DataFrame with community metrics
    """
    print("\n=== Evaluating Communities ===")
    
    # Prepare lookup dictionaries
    indval_lookup = {}
    for _, row in indval_results.iterrows():
        species = row['Species']
        species_col = species
        indval_lookup[species_col] = row['IndVal']
    
    cooccurrence_lookup = {}
    for _, row in cooccurrence_results.iterrows():
        species = row['species']
        cooccurrence_lookup[species] = row['co_occurrence_ratio']
    
    frequency_lookup = {}
    for _, row in frequency_results.iterrows():
        species = row['Species']
        species_col = species
        if 'VCM_Preference_Strength' in row:
            frequency_lookup[species_col] = row['VCM_Preference_Strength']
    
    # Group nodes by community
    community_groups = defaultdict(list)
    for node, community_id in communities.items():
        community_groups[community_id].append(node)
    
    # Evaluate each community
    community_metrics = []
    
    for community_id, nodes in community_groups.items():
        # Skip communities with less than 2 species
        if len(nodes) < 2:
            continue
            
        # Calculate metrics
        indval_scores = []
        cooccurrence_ratios = []
        preference_strengths = []
        
        for node in nodes:
            # For IndVal
            if node in indval_lookup:
                indval_scores.append(indval_lookup[node])
            
            # For co-occurrence ratio
            if node in cooccurrence_lookup:
                cooccurrence_ratios.append(cooccurrence_lookup[node])
            
            # For VCM preference strength
            if node in frequency_lookup:
                preference_strengths.append(frequency_lookup[node])
        
        # Calculate average metrics
        avg_indval = np.mean(indval_scores) if indval_scores else np.nan
        avg_cooccurrence = np.mean(cooccurrence_ratios) if cooccurrence_ratios else np.nan
        avg_preference = np.mean(preference_strengths) if preference_strengths else np.nan
        
        # Calculate network metrics
        subgraph = G.subgraph(nodes)
        density = nx.density(subgraph)
        
        # Calculate modularity contribution
        internal_edges = subgraph.number_of_edges()
        total_edges = sum(dict(G.degree(nodes)).values()) / 2
        modularity_contribution = internal_edges / total_edges if total_edges > 0 else 0
        
        community_metrics.append({
            'community_id': community_id,
            'size': len(nodes),
            'mean_indval': avg_indval,
            'mean_cooccurrence_ratio': avg_cooccurrence,
            'mean_preference_strength': avg_preference,
            'density': density,
            'modularity_contribution': modularity_contribution,
            'species': ", ".join([n.replace('species_', '') for n in nodes])
        })
    
    # Create DataFrame and sort by composite score
    metrics_df = pd.DataFrame(community_metrics)
    
    # Create composite score (average of normalized ranks)
    if not metrics_df.empty:
        for col in ['mean_indval', 'mean_cooccurrence_ratio', 'mean_preference_strength', 'density']:
            if col in metrics_df.columns:
                metrics_df[f'{col}_rank'] = metrics_df[col].rank(ascending=False)
        
        rank_cols = [col for col in metrics_df.columns if col.endswith('_rank')]
        if rank_cols:
            metrics_df['composite_score'] = metrics_df[rank_cols].mean(axis=1)
            metrics_df = metrics_df.sort_values('composite_score')
    
    print(f"Evaluated {len(metrics_df)} communities with at least 2 species")
    
    return metrics_df

def visualize_community_network(G, communities, output_path):
    """
    Visualize the species co-occurrence network with communities.
    
    Args:
        G: NetworkX graph
        communities: Dictionary mapping nodes to community IDs
        output_path: Path to save the visualization
    """
    # Create a figure and axis explicitly
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Generate colormap
    unique_communities = sorted(set(communities.values()))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_communities)))
    color_map = {comm: rgb2hex(colors[i]) for i, comm in enumerate(unique_communities)}
    
    # Assign colors to nodes
    node_colors = [color_map[communities[node]] for node in G.nodes()]
    
    # Calculate node sizes based on degree
    degree_dict = dict(G.degree())
    node_sizes = [30 + 20 * degree_dict[node] for node in G.nodes()]
    
    # Calculate edge weights for width
    edge_weights = [G[u][v].get('weight', 1) * 0.1 for u, v in G.edges()]
    
    # Spring layout with seed for reproducibility
    pos = nx.spring_layout(G, k=0.3, seed=42)
    
    # Draw network - make sure to use the axis object
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.4, edge_color='gray', ax=ax)
    
    # Add labels to high-degree nodes (hubs in each community)
    hub_threshold = max(5, np.percentile(list(degree_dict.values()), 75))
    hub_nodes = {node: node.replace('species_', '') for node, degree in degree_dict.items() 
                if degree >= hub_threshold}
    
    nx.draw_networkx_labels(G, pos, labels=hub_nodes, font_size=8, font_weight='bold', ax=ax)
    
    ax.set_title("Species Co-occurrence Network with Communities", fontsize=16)
    ax.axis('off')
    
    # Add colorbar legend for communities - now using the axis
    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=0, vmax=len(unique_communities)-1))
    sm.set_array([])
    
    # Create a separate axis for the colorbar
    cbar_ax = fig.add_axes([0.92, 0.3, 0.03, 0.4])  # [x, y, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Community ID', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Network visualization saved to {output_path}")

def visualize_community_metrics(metrics_df, output_dir):
    """
    Create visualizations of community metrics.
    
    Args:
        metrics_df: DataFrame with community metrics
        output_dir: Directory to save visualizations
    """
    if metrics_df.empty:
        print("No communities with sufficient data to visualize metrics")
        return
    
    # 1. Bar plot of top communities by composite score
    plt.figure(figsize=(12, 8))
    top_n = min(10, len(metrics_df))
    top_communities = metrics_df.sort_values('composite_score').head(top_n)
    
    sns.barplot(x='community_id', y='composite_score', data=top_communities)
    plt.title(f'Top {top_n} Communities by Composite Score', fontsize=14)
    plt.xlabel('Community ID', fontsize=12)
    plt.ylabel('Composite Score (lower is better)', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add size labels
    for i, row in enumerate(top_communities.itertuples()):
        plt.text(i, row.composite_score + 0.1, f'Size: {row.size}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_communities_composite.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap of community metrics
    plt.figure(figsize=(14, 10))
    plot_cols = ['community_id', 'size', 'mean_indval', 'mean_cooccurrence_ratio', 
                'mean_preference_strength', 'density', 'modularity_contribution']
    
    # Select top communities for better visualization
    heatmap_data = top_communities[plot_cols].set_index('community_id')
    
    # Scale columns for better visualization
    scaled_data = heatmap_data.copy()
    for col in scaled_data.columns:
        if col != 'size':
            scaled_data[col] = (scaled_data[col] - scaled_data[col].min()) / \
                              (scaled_data[col].max() - scaled_data[col].min() + 1e-10)
    
    sns.heatmap(scaled_data, annot=heatmap_data.round(3), fmt='.3f', cmap='viridis')
    plt.title('Metrics of Top Communities', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'community_metrics_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plot of size vs. mean metrics
    plt.figure(figsize=(12, 8))
    plt.scatter(metrics_df['size'], metrics_df['mean_indval'], 
               alpha=0.7, label='Mean IndVal')
    plt.scatter(metrics_df['size'], metrics_df['mean_cooccurrence_ratio'], 
               alpha=0.7, label='Mean Co-occurrence')
    
    if 'mean_preference_strength' in metrics_df.columns:
        plt.scatter(metrics_df['size'], metrics_df['mean_preference_strength'], 
                   alpha=0.7, label='Mean Preference')
    
    plt.xlabel('Community Size (number of species)', fontsize=12)
    plt.ylabel('Mean Score', fontsize=12)
    plt.title('Community Size vs. Mean Indicator Scores', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'community_size_vs_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Community metric visualizations saved to {output_dir}")

def create_community_species_report(metrics_df, communities, indval_results, cooccurrence_results, frequency_results, output_path):
    """
    Create a detailed report of species in each community with their individual metrics.
    
    Args:
        metrics_df: DataFrame with community metrics
        communities: Dictionary mapping nodes to community IDs
        indval_results: DataFrame with IndVal scores
        cooccurrence_results: DataFrame with co-occurrence ratios
        frequency_results: DataFrame with VCM preference strengths
        output_path: Path to save the report
    """
    # Prepare lookup dictionaries
    indval_lookup = {}
    for _, row in indval_results.iterrows():
        species = row['Species']
        species_col = f"species_{species}"
        indval_lookup[species_col] = row['IndVal']
    
    cooccurrence_lookup = {}
    for _, row in cooccurrence_results.iterrows():
        species = row['species']
        cooccurrence_lookup[species] = row['co_occurrence_ratio']
    
    frequency_lookup = {}
    for _, row in frequency_results.iterrows():
        species = row['Species']
        species_col = f"species_{species}"
        if 'VCM_Preference_Strength' in row:
            frequency_lookup[species_col] = row['VCM_Preference_Strength']
    
    # Create report data
    report_data = []
    
    # Get top communities
    top_communities = metrics_df.sort_values('composite_score').head(10)['community_id'].tolist()
    
    for node, community_id in communities.items():
        if community_id in top_communities:
            species_name = node.replace('species_', '')
            
            entry = {
                'community_id': community_id,
                'species': species_name,
                'indval_score': indval_lookup.get(node, np.nan),
                'cooccurrence_ratio': cooccurrence_lookup.get(node, np.nan),
                'preference_strength': frequency_lookup.get(node, np.nan)
            }
            
            report_data.append(entry)
    
    # Create DataFrame and sort
    report_df = pd.DataFrame(report_data)
    report_df = report_df.sort_values(['community_id', 'indval_score'], ascending=[True, False])
    
    # Save to CSV
    report_df.to_csv(output_path, index=False)
    print(f"Community species report saved to {output_path}")
    
    return report_df

def analyze_community_cooccurrence(communities, data_matrix, species_columns):
    """
    Analyze how often species within each community co-occur together in VCM 
    grid cells versus overall.
    
    Args:
        communities: Dictionary mapping nodes to community IDs
        data_matrix: DataFrame containing species presence and VCM labels
        species_columns: List of species column names
        
    Returns:
        Dictionary with community co-occurrence statistics
    """
    print("\n=== Analyzing Community Co-occurrence Patterns ===")
    
    # Group species by community
    community_species = defaultdict(list)
    for species, community_id in communities.items():
        if species in species_columns:
            community_species[community_id].append(species)
    
    # Initialize results dictionary
    cooccurrence_stats = {}
    
    # Separate VCM and all locations
    vcm_data = data_matrix[data_matrix['vcm_label'] == 1]
    all_data = data_matrix
    
    # For each community with at least 2 species
    for community_id, species_list in community_species.items():
        if len(species_list) < 2:
            continue
            
        print(f"Analyzing co-occurrence for community {community_id} ({len(species_list)} species)")
        
        # Calculate community co-occurrence in VCM areas
        vcm_species_data = vcm_data[species_list]
        vcm_cooccurrence = 0
        
        # Count sites where at least 2 community members co-occur
        for _, row in vcm_species_data.iterrows():
            if sum(row > 0) >= 2:
                vcm_cooccurrence += 1
                
        # Calculate community co-occurrence in all areas
        all_species_data = all_data[species_list]
        all_cooccurrence = 0
        
        # Count sites where at least 2 community members co-occur
        for _, row in all_species_data.iterrows():
            if sum(row > 0) >= 2:
                all_cooccurrence += 1
        
        # Calculate average co-occurrence patterns
        vcm_cohesion = vcm_cooccurrence / len(vcm_data) if len(vcm_data) > 0 else 0
        overall_cohesion = all_cooccurrence / len(all_data) if len(all_data) > 0 else 0
        vcm_preference = vcm_cooccurrence / all_cooccurrence if all_cooccurrence > 0 else 0
        
        # Calculate pairwise co-occurrence
        pair_count = 0
        vcm_pair_count = 0
        
        # For more detailed pairwise analysis
        for i in range(len(species_list)):
            for j in range(i+1, len(species_list)):
                species1, species2 = species_list[i], species_list[j]
                
                # Count overall co-occurrences
                pair_overlap = ((all_data[species1] > 0) & (all_data[species2] > 0)).sum()
                pair_count += pair_overlap
                
                # Count VCM co-occurrences
                vcm_pair_overlap = ((vcm_data[species1] > 0) & (vcm_data[species2] > 0)).sum()
                vcm_pair_count += vcm_pair_overlap
        
        # Calculate average pairwise co-occurrence
        pair_total = len(species_list) * (len(species_list) - 1) / 2
        avg_pair_cooccurrence = pair_count / pair_total if pair_total > 0 else 0
        avg_vcm_pair_cooccurrence = vcm_pair_count / pair_total if pair_total > 0 else 0
        
        # Store results
        cooccurrence_stats[community_id] = {
            'community_size': len(species_list),
            'vcm_cooccurrence_count': vcm_cooccurrence,
            'total_cooccurrence_count': all_cooccurrence,
            'vcm_cohesion': vcm_cohesion,  # % of VCM sites with community co-occurrence
            'overall_cohesion': overall_cohesion,  # % of all sites with community co-occurrence
            'vcm_preference_ratio': vcm_preference,  # VCM co-occurrence / overall co-occurrence
            'avg_pair_cooccurrence': avg_pair_cooccurrence,  # Average co-occurrence per species pair
            'avg_vcm_pair_cooccurrence': avg_vcm_pair_cooccurrence,  # Average VCM co-occurrence per species pair
            'vcm_pair_ratio': avg_vcm_pair_cooccurrence / avg_pair_cooccurrence if avg_pair_cooccurrence > 0 else 0
        }
    
    # Print summary
    print(f"Analyzed co-occurrence patterns for {len(cooccurrence_stats)} communities")
    
    return cooccurrence_stats

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Identify multi-species indicator sets for VCM')
    parser.add_argument('--data-matrix', type=str, default='inat-data-matrix-latlong.csv',
                        help='Path to data matrix CSV file')
    parser.add_argument('--indval', type=str, default='outputs/indval_analysis/indval_full_results.csv',
                        help='Path to IndVal results CSV file')
    parser.add_argument('--cooccurrence', type=str, default='outputs/cooccurrence_analysis/cooccurrence_full_results.csv',
                        help='Path to co-occurrence results CSV file')
    parser.add_argument('--frequency', type=str, default='outputs/frequency_analysis/species_frequency_analysis.csv',
                        help='Path to frequency analysis results CSV file')
    parser.add_argument('--output-dir', type=str, default='outputs/community_analysis',
                        help='Directory to save outputs')
    parser.add_argument('--resolution', type=float, default=0.4,
                        help='Resolution parameter for community detection (higher = more communities)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data_matrix, indval_results, cooccurrence_results, frequency_results = load_data(
        args.data_matrix, args.indval, args.cooccurrence, args.frequency
    )
    
    # Get species columns
    species_columns = [col for col in data_matrix.columns if col.startswith('species_')]
    
    # Build co-occurrence network
    G, cooccurrence_matrix = build_species_cooccurrence_network(data_matrix, species_columns)
    
    # Save co-occurrence matrix
    cooccurrence_matrix.to_csv(os.path.join(args.output_dir, 'species_cooccurrence_matrix.csv'))
    
    # Detect communities
    communities = detect_communities(G, resolution=args.resolution)
    
    # Analyze community co-occurrence
    cooccurrence_stats = analyze_community_cooccurrence(communities, data_matrix, species_columns)
    
    # Evaluate communities
    metrics_df = evaluate_communities(communities, G, indval_results, cooccurrence_results, frequency_results)
    
    # Add co-occurrence stats to metrics dataframe
    for community_id, stats in cooccurrence_stats.items():
        idx = metrics_df[metrics_df['community_id'] == community_id].index
        if len(idx) > 0:
            for key, value in stats.items():
                metrics_df.loc[idx, key] = value
    
    # Save community metrics
    metrics_df.to_csv(os.path.join(args.output_dir, 'community_metrics.csv'), index=False)
    
    # Create detailed community species report
    report_df = create_community_species_report(
        metrics_df, communities, indval_results, cooccurrence_results, frequency_results,
        os.path.join(args.output_dir, 'community_species_report.csv')
    )
    
    # Visualize community network
    visualize_community_network(G, communities, os.path.join(args.output_dir, 'community_network.png'))
    
    # Visualize community metrics
    visualize_community_metrics(metrics_df, args.output_dir)
    
    print(f"\nCommunity analysis complete! Results saved to {args.output_dir}")
    
    # Print top communities with enhanced co-occurrence info
    print("\nTop 5 Multi-Species Indicator Communities with Co-occurrence Stats:")
    print("=" * 80)
    top_communities = metrics_df.sort_values('composite_score').head(5)
    for _, row in top_communities.iterrows():
        print(f"Community {row['community_id']} (Size: {row['size']} species)")
        print(f"  Mean IndVal: {row['mean_indval']:.4f}")
        print(f"  Mean Co-occurrence: {row['mean_cooccurrence_ratio']:.4f}")
        print(f"  Mean Preference: {row['mean_preference_strength']:.4f}")
        
        # Add new co-occurrence metrics
        if 'vcm_cooccurrence_count' in row:
            print(f"  VCM co-occurrence sites: {row['vcm_cooccurrence_count']}")
            print(f"  Total co-occurrence sites: {row['total_cooccurrence_count']}")
            print(f"  VCM preference ratio: {row['vcm_preference_ratio']:.4f}")
            print(f"  Average pairwise co-occurrence: {row['avg_pair_cooccurrence']:.4f}")
            print(f"  VCM pairwise ratio: {row['vcm_pair_ratio']:.4f}")
            
        print(f"  Species: {row['species']}")
        print("-" * 80)

if __name__ == "__main__":
    main()
