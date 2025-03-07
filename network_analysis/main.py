from network_analysis import preprocessing, graph_construction, network_analysis, vcm_indicators
import os
import time
import networkx as nx
import pandas as pd

def check_data_requirements(data):
    """Check if the data meets all requirements."""
    required_columns = ['grid_location', 'vcm_label']
    species_columns = [col for col in data.columns if col.startswith('species_')]
    
    if not species_columns:
        raise ValueError("No species columns found (columns starting with 'species_')")
    
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    if not all(data['vcm_label'].isin([0, 1])):
        raise ValueError("vcm_label column should only contain 0 and 1")

def main():
    total_start_time = time.time()
    
    # File paths
    input_file = "inat-data-matrix-gdf.csv"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("\n=== Loading and Preprocessing Data ===")
    try:
        data = preprocessing.load_data(input_file)
        check_data_requirements(data)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    species_columns = preprocessing.get_species_columns(data)
    print(f"Found {len(species_columns)} species columns")
    
    print("\n=== Computing IndVal Scores ===")
    indval_scores = preprocessing.compute_indval_scores(data, species_columns)
    norm_indval = preprocessing.normalize_indval_scores(indval_scores)
    
    print("\n=== Creating Presence Matrix ===")
    presence_matrix = preprocessing.create_presence_matrix(data, species_columns)
    
    # Construct graphs
    print("\n=== Building Network ===")
    B = graph_construction.create_bipartite_graph(data, species_columns, norm_indval)
    
    # Use the optimized incidence matrix function
    incidence = graph_construction.create_incidence_matrix_optimized(data, species_columns, norm_indval)
    
    # Analyze network
    print("\n=== Analyzing Network ===")
    # Create two networks - one with all cells and one with only VCM=1 cells
    species_proj, species_network = network_analysis.project_to_species_network(incidence, data, vcm_only=False)
    species_proj_vcm, species_network_vcm = network_analysis.project_to_species_network(incidence, data, vcm_only=True)
    
    # Analyze both networks
    analysis_results = network_analysis.analyze_network(species_network)
    analysis_results_vcm = network_analysis.analyze_network(species_network_vcm)
    
    # Save results
    print("\n=== Saving Results ===")
    # Save projection matrices
    species_proj.to_csv(os.path.join(output_dir, "species_projection_all.csv"))
    species_proj_vcm.to_csv(os.path.join(output_dir, "species_projection_vcm.csv"))
    
    # Save networks
    nx.write_graphml(species_network, os.path.join(output_dir, "species_network_all.graphml"))
    nx.write_graphml(species_network_vcm, os.path.join(output_dir, "species_network_vcm.graphml"))
    
    # Save node attributes
    save_node_attributes(species_network, os.path.join(output_dir, "species_attributes_all.csv"))
    save_node_attributes(species_network_vcm, os.path.join(output_dir, "species_attributes_vcm.csv"))
    
    # Visualize
    print("\n=== Creating Visualization ===")
    # Visualize both networks
    network_analysis.visualize_network(
        species_network, 
        output_path=os.path.join(output_dir, "species_network_all.png"),
        label_top_n=50
    )
    
    network_analysis.visualize_network(
        species_network_vcm, 
        output_path=os.path.join(output_dir, "species_network_vcm.png"),
        label_top_n=50
    )
    
    print("\n=== Identifying VCM Indicator Species ===")
    indicator_results, top_indicators = vcm_indicators.identify_vcm_indicators(
        species_attr_all_path=os.path.join(output_dir, "species_attributes_all.csv"),
        species_attr_vcm_path=os.path.join(output_dir, "species_attributes_vcm.csv"),
        species_proj_all_path=os.path.join(output_dir, "species_projection_all.csv"),
        species_proj_vcm_path=os.path.join(output_dir, "species_projection_vcm.csv"),
        output_dir=output_dir
    )
    
    print("\n=== Creating Indicator Species Network Visualization ===")
    vcm_indicators.visualize_indicator_network(
        species_network_vcm,
        indicator_results,
        output_path=os.path.join(output_dir, "vcm_indicator_network.png"),
        top_n=30
    )
    
    total_elapsed = time.time() - total_start_time
    print(f"\nAnalysis complete! Total runtime: {total_elapsed:.2f} seconds")
    print(f"Results saved in: {output_dir}")

def save_node_attributes(G, output_file):
    """Helper function to save node attributes to CSV"""
    node_attrs = {}
    for node in G.nodes():
        node_attrs[node] = {
            'community': G.nodes[node].get('community', 0),
            'degree_centrality': G.nodes[node].get('degree_centrality', 0),
            'betweenness_centrality': G.nodes[node].get('betweenness_centrality', 0),
            'eigenvector_centrality': G.nodes[node].get('eigenvector_centrality', 0)
        }
    
    node_df = pd.DataFrame.from_dict(node_attrs, orient='index')
    node_df.to_csv(output_file)
    print(f"Saved node attributes to {output_file}")

if __name__ == "__main__":
    main() 