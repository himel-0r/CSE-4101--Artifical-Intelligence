import json
import os
import matplotlib.pyplot as plt
from typing import Dict, List

def load_results(file_path: str = 'dcop_results/results.json') -> List[Dict]:
    """
    Load results from the JSON file.
    
    Args:
        file_path: Path to the results JSON file.
    
    Returns:
        List of result dictionaries.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file {file_path} not found. Run evaluate_dcop_algorithms_200_instances.py first.")
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Error loading {file_path}: {e}")

def plot_configuration_results(config_results: List[Dict], graph_type: str, k: int, d: int, output_dir: str):
    """
    Plot pruning percentage and average runtime for a single configuration and save to separate files.
    
    Args:
        config_results: List of results for the specific configuration.
        graph_type: 'sparse' or 'dense'.
        k: Constraint arity (2 or 4).
        d: Domain size (2 or 10).
        output_dir: Directory to save the plots.
    """
    # Sort results by number of function nodes
    config_results.sort(key=lambda x: x['num_constraints'])
    
    # Extract data
    function_counts = [r['num_constraints'] for r in config_results]
    gdp_pruning = [r['gdp_avg_pruned_percentage'] for r in config_results]
    gd2p_pruning = [r['gd2p_avg_pruned_percentage'] for r in config_results]
    fdsp_pruning = [r['fdsp_avg_pruned_percentage'] for r in config_results]
    gdp_runtime = [r['gdp_avg_runtime'] for r in config_results]
    gd2p_runtime = [r['gd2p_avg_runtime'] for r in config_results]
    fdsp_runtime = [r['fdsp_avg_runtime'] for r in config_results]
    
    # Plot pruning percentage
    plt.figure(figsize=(8, 6))
    plt.plot(function_counts, gdp_pruning, label='GDP', marker='o')
    plt.plot(function_counts, gd2p_pruning, label='GD2P', marker='s')
    plt.plot(function_counts, fdsp_pruning, label='FDSP', marker='^')
    plt.title(f'Pruning Percentage: {graph_type}, k={k}, d={d}')
    plt.xlabel('Number of Function Nodes')
    plt.ylabel('Pruning Percentage (%)')
    plt.legend()
    plt.grid(True)
    pruning_file = os.path.join(output_dir, f'pruning_{graph_type}_k{k}_d{d}.png')
    plt.savefig(pruning_file)
    plt.close()
    print(f"Saved pruning plot to {pruning_file}")
    
    # Plot average runtime
    plt.figure(figsize=(8, 6))
    plt.plot(function_counts, gdp_runtime, label='GDP', marker='o')
    plt.plot(function_counts, gd2p_runtime, label='GD2P', marker='s')
    plt.plot(function_counts, fdsp_runtime, label='FDSP', marker='^')
    plt.title(f'Average Runtime: {graph_type}, k={k}, d={d}')
    plt.xlabel('Number of Function Nodes')
    plt.ylabel('Average Runtime (seconds)')
    plt.legend()
    plt.grid(True)
    runtime_file = os.path.join(output_dir, f'runtime_{graph_type}_k{k}_d{d}.png')
    plt.savefig(runtime_file)
    plt.close()
    print(f"Saved runtime plot to {runtime_file}")

def plot_results(results: List[Dict], output_dir: str = 'dcop_results'):
    """
    Generate separate line graph files for pruning percentage and average runtime for each configuration.
    
    Args:
        results: List of result dictionaries from the JSON file.
        output_dir: Directory to save the plots.
    """
    # Define configurations
    configurations = [
        ('sparse', 2, 2), ('sparse', 2, 10), ('sparse', 4, 2), ('sparse', 4, 10),
        ('dense', 2, 2), ('dense', 2, 10), ('dense', 4, 2), ('dense', 4, 10)
    ]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot for each configuration
    for graph_type, k, d in configurations:
        # Filter results for this configuration
        config_results = [r for r in results 
                         if r['graph_type'] == graph_type and r['arity'] == k and r['domain_size'] == d]
        if not config_results:
            print(f"No results found for {graph_type}, k={k}, d={d}. Skipping.")
            continue
        
        plot_configuration_results(config_results, graph_type, k, d, output_dir)

if __name__ == "__main__":
    try:
        results = load_results()
        plot_results(results)
    except Exception as e:
        print(f"Error in plotting: {e}")