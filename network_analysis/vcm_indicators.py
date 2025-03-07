import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple
import os

def identify_vcm_indicators(
    species_attr_all_path: str = 'outputs/species_attributes_all.csv',
    species_attr_vcm_path: str = 'outputs/species_attributes_vcm.csv',
    species_proj_all_path: str = 'outputs/species_projection_all.csv',
    species_proj_vcm_path: str = 'outputs/species_projection_vcm.csv',
    output_dir: str = 'outputs'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Identify indicator species for vegetation climate mismatch (VCM).
    
    Args:
        species_attr_all_path: Path to species attributes for all data
        species_attr_vcm_path: Path to species attributes for VCM data
        species_proj_all_path: Path to species projection for all data
        species_proj_vcm_path: Path to species projection for VCM data
        output_dir: Directory to save outputs
        
    Returns:
        Tuple of (indicators DataFrame, top indicators by community Dict)
    """
    print("=== Identifying VCM Indicator Species ===")
    
    # Load species attributes data
    species_attr_all = pd.read_csv(species_attr_all_path, index_col=0)
    species_attr_vcm = pd.read_csv(species_attr_vcm_path, index_col=0)
    
    # Load projection matrices
    proj_all = pd.read_csv(species_proj_all_path, index_col=0)
    proj_vcm = pd.read_csv(species_proj_vcm_path, index_col=0)
    
    # Calculate the ratio of interactions in VCM vs all environments
    interaction_ratios = {}
    vcm_specificity = {}
    
    print("Analyzing interaction patterns...")
    for species in species_attr_vcm.index:
        # Skip species with no interactions in VCM areas
        if species not in proj_vcm.index or species not in proj_all.index:
            continue
            
        # Get total interaction weights in each network
        vcm_total = proj_vcm.loc[species].sum()
        all_total = proj_all.loc[species].sum()
        
        # Calculate ratio (how concentrated interactions are in VCM areas)
        if all_total > 0:
            interaction_ratios[species] = vcm_total / all_total
        else:
            interaction_ratios[species] = 0
            
        # Calculate VCM specificity: how much the species interacts differently in VCM areas
        vcm_interactions = set(proj_vcm.columns[proj_vcm.loc[species] > 0])
        all_interactions = set(proj_all.columns[proj_all.loc[species] > 0])
        
        if len(all_interactions) > 0:
            # Jaccard dissimilarity between interaction sets (1 - similarity)
            vcm_specificity[species] = 1 - len(vcm_interactions.intersection(all_interactions)) / len(vcm_interactions.union(all_interactions))
        else:
            vcm_specificity[species] = 0
    
    # Create combined indicator scores
    indicators = pd.DataFrame(index=species_attr_vcm.index)
    
    # Add network metrics
    indicators['degree_centrality_vcm'] = species_attr_vcm['degree_centrality']
    indicators['betweenness_centrality_vcm'] = species_attr_vcm['betweenness_centrality']
    indicators['eigenvector_centrality_vcm'] = species_attr_vcm['eigenvector_centrality']
    indicators['community_vcm'] = species_attr_vcm['community']
    
    # Add all-network metrics for comparison
    indicators['degree_centrality_all'] = species_attr_all['degree_centrality']
    indicators['betweenness_centrality_all'] = species_attr_all['betweenness_centrality']
    indicators['eigenvector_centrality_all'] = species_attr_all['eigenvector_centrality']
    indicators['community_all'] = species_attr_all['community']
    
    # Add VCM-specific metrics
    indicators['interaction_ratio'] = pd.Series(interaction_ratios)
    indicators['vcm_specificity'] = pd.Series(vcm_specificity)
    
    # Create composite indicator score
    # Higher score = more important in VCM network structure + more specific to VCM
    indicators['indicator_score'] = (
        indicators['degree_centrality_vcm'] * 0.3 +
        indicators['eigenvector_centrality_vcm'] * 0.3 +
        indicators['betweenness_centrality_vcm'] * 0.2 +
        indicators['interaction_ratio'] * 0.1 +
        indicators['vcm_specificity'] * 0.1
    )
    
    # Remove rows with NaN values
    indicators = indicators.dropna()
    
    # Find top indicators overall
    top_indicators = indicators.nlargest(20, 'indicator_score')
    print("\nTop 20 VCM indicator species:")
    for i, (species, score) in enumerate(top_indicators['indicator_score'].items()):
        print(f"{i+1}. {species.replace('species_', '')}: {score:.4f}")
    
    # Find top indicators by community
    top_indicators_by_community = {}
    for community in indicators['community_vcm'].unique():
        if pd.isna(community):
            continue
        
        community_species = indicators[indicators['community_vcm'] == community]
        top_community_indicators = community_species.nlargest(5, 'indicator_score')
        
        if not top_community_indicators.empty:
            top_indicators_by_community[int(community)] = top_community_indicators.index.tolist()
    
    print("\nTop indicator species by community:")
    for community, species_list in top_indicators_by_community.items():
        print(f"Community {community}:")
        for species in species_list:
            score = indicators.loc[species, 'indicator_score']
            print(f"  - {species.replace('species_', '')}: {score:.4f}")
    
    # Create visualization of indicators
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot colored by community
    scatter = sns.scatterplot(
        data=indicators,
        x='eigenvector_centrality_vcm',
        y='degree_centrality_vcm',
        hue='community_vcm',
        size='indicator_score',
        sizes=(20, 200),
        alpha=0.7,
        palette='tab20'
    )
    
    # Add labels for top indicators
    for species in top_indicators.index:
        x = indicators.loc[species, 'eigenvector_centrality_vcm']
        y = indicators.loc[species, 'degree_centrality_vcm']
        plt.annotate(
            species.replace('species_', ''),
            (x, y),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.xlabel('Eigenvector Centrality')
    plt.ylabel('Degree Centrality')
    plt.title('VCM Indicator Species by Community')
    plt.tight_layout()
    
    # Save the visualization and indicators
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'vcm_indicator_species.png'), dpi=300)
    indicators.to_csv(os.path.join(output_dir, 'vcm_indicator_scores.csv'))
    
    print(f"\nResults saved to {os.path.join(output_dir, 'vcm_indicator_scores.csv')} and {os.path.join(output_dir, 'vcm_indicator_species.png')}")
    
    # Add community-level analysis of VCM association
    print("\n=== Analyzing Community-Level VCM Association ===")
    
    # Group indicator scores by community
    community_stats = indicators.groupby('community_vcm').agg({
        'indicator_score': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'interaction_ratio': ['mean', 'median'],
        'vcm_specificity': ['mean', 'median'],
        'degree_centrality_vcm': 'mean',
        'eigenvector_centrality_vcm': 'mean'
    })
    
    # Flatten the multi-index columns
    community_stats.columns = ['_'.join(col).strip() for col in community_stats.columns.values]
    
    # Add variance to mean ratio as a measure of consistency within community
    community_stats['indicator_score_variance_to_mean'] = (community_stats['indicator_score_std'] ** 2) / community_stats['indicator_score_mean']
    
    # Calculate a community VCM association score
    community_stats['community_vcm_score'] = (
        community_stats['indicator_score_mean'] * 0.4 +
        community_stats['interaction_ratio_mean'] * 0.3 +
        community_stats['vcm_specificity_mean'] * 0.3
    )
    
    # Sort by community VCM score
    community_stats = community_stats.sort_values('community_vcm_score', ascending=False)
    
    # Print community statistics
    print("\nCommunity VCM Association Rankings:")
    for community, stats in community_stats.iterrows():
        if pd.isna(community) or stats['indicator_score_count'] < 3:
            continue  # Skip communities with too few species
        print(f"Community {int(community)}: Score {stats['community_vcm_score']:.4f} (n={int(stats['indicator_score_count'])})")
        print(f"  - Mean indicator score: {stats['indicator_score_mean']:.4f}")
        print(f"  - Mean interaction ratio: {stats['interaction_ratio_mean']:.4f}")
        print(f"  - Mean VCM specificity: {stats['vcm_specificity_mean']:.4f}")
    
    # Create visualization of community-level VCM association
    plt.figure(figsize=(12, 8))
    
    # Filter out communities with too few species
    plot_communities = community_stats[community_stats['indicator_score_count'] >= 3]
    
    # Create bar plot of community VCM scores
    communities = plot_communities.index.astype(int)
    scores = plot_communities['community_vcm_score']
    
    bars = plt.bar(communities, scores, alpha=0.7)
    
    # Color bars by community (matching the network visualization)
    cmap = plt.cm.tab20
    for i, bar in enumerate(bars):
        bar.set_color(cmap(communities[i] % 20))
    
    # Add community size as text on bars
    for i, community in enumerate(communities):
        count = int(plot_communities.loc[community, 'indicator_score_count'])
        plt.text(
            i, 
            scores.iloc[i] + 0.01, 
            f"n={count}", 
            ha='center'
        )
    
    plt.xlabel('Community ID')
    plt.ylabel('VCM Association Score')
    plt.title('Community-Level VCM Association')
    plt.xticks(communities)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save community statistics and visualization
    community_stats.to_csv(os.path.join(output_dir, 'community_vcm_stats.csv'))
    plt.savefig(os.path.join(output_dir, 'community_vcm_association.png'), dpi=300)
    
    # Create boxplot of indicator scores by community
    plt.figure(figsize=(14, 8))
    
    # Filter to communities with enough species for meaningful boxplots
    valid_communities = community_stats[community_stats['indicator_score_count'] >= 5].index
    community_data = indicators[indicators['community_vcm'].isin(valid_communities)]
    
    # Convert community to string for better display
    community_data['community_str'] = community_data['community_vcm'].astype(int).astype(str)
    
    # Sort boxplot by median indicator score
    order = community_data.groupby('community_str')['indicator_score'].median().sort_values(ascending=False).index
    
    # Create boxplot
    sns.boxplot(
        data=community_data,
        x='community_str',
        y='indicator_score',
        order=order,
        palette='tab20'
    )
    
    # Add swarmplot to show individual species
    sns.swarmplot(
        data=community_data,
        x='community_str',
        y='indicator_score',
        order=order,
        color='black',
        alpha=0.5,
        size=4
    )
    
    plt.xlabel('Community')
    plt.ylabel('Indicator Score')
    plt.title('Distribution of VCM Indicator Scores by Community')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save the boxplot
    plt.savefig(os.path.join(output_dir, 'community_indicator_distribution.png'), dpi=300)
    print(f"Saved community analysis to {os.path.join(output_dir, 'community_vcm_stats.csv')}")
    print(f"Saved community visualizations to {output_dir}")
    
    return indicators, top_indicators_by_community

def visualize_indicator_network(
    G: nx.Graph,
    indicators_df: pd.DataFrame,
    output_path: str,
    top_n: int = 30
):
    """
    Create a network visualization focused on top indicator species.
    
    Args:
        G: NetworkX graph
        indicators_df: DataFrame with indicator scores
        output_path: Path to save visualization
        top_n: Number of top indicators to include
    """
    # Get top indicator species
    top_species = indicators_df.nlargest(top_n, 'indicator_score').index
    
    # Create subgraph with just these species
    G_indicators = G.subgraph(top_species)
    
    # Create visualization
    plt.figure(figsize=(16, 16))
    
    # Get layout
    pos = nx.spring_layout(G_indicators, seed=42, k=0.3)
    
    # Get node attributes
    communities = nx.get_node_attributes(G_indicators, 'community')
    node_colors = [communities.get(node, 0) for node in G_indicators.nodes()]
    
    eigen_cent = nx.get_node_attributes(G_indicators, 'eigenvector_centrality')
    node_sizes = [3000 * eigen_cent.get(node, 0.1) + 100 for node in G_indicators.nodes()]
    
    # Get edge weights
    edge_weights = [G_indicators[u][v].get('weight', 1) for u, v in G_indicators.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + 5 * (w/max_weight) for w in edge_weights]
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G_indicators, pos,
        node_color=node_colors,
        cmap=plt.cm.tab20,
        node_size=node_sizes,
        alpha=0.8
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G_indicators, pos,
        alpha=0.4,
        width=edge_widths
    )
    
    # Draw labels
    labels = {node: node.replace('species_', '') for node in G_indicators.nodes()}
    nx.draw_networkx_labels(
        G_indicators, pos,
        labels=labels,
        font_size=10,
        font_weight='bold'
    )
    
    # Create colorbar
    unique_communities = sorted(set(communities.values()))
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.tab20,
        norm=plt.Normalize(min(unique_communities), max(unique_communities))
    )
    sm.set_array([])
    
    # Get the current axis
    ax = plt.gca()
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Community')
    
    plt.title(f'Top {top_n} VCM Indicator Species Network', fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved indicator network visualization to {output_path}") 