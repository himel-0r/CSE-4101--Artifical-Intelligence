import json
import matplotlib.pyplot as plt
from itertools import product

# Load the JSON data
with open('results_onekgula1.json', 'r') as file:
    data = json.load(file)

# Define the unique combinations of graph_type, domain_type, and arity_type
graph_types = ['sparse', 'dense']
domain_types = ['small', 'large']
arity_types = ['low', 'high']
combinations = list(product(graph_types, domain_types, arity_types))

# Initialize lists to store data for each combination
plots_data = {comb: {'num_constraints': [], 'gdp_runtime': [], 'gd2p_runtime': [], 'fdsp_runtime': [],
                     'gdp_pruned': [], 'gd2p_pruned': [], 'fdsp_pruned': []} for comb in combinations}

# Organize data by combination
for entry in data:
    comb = (entry['graph_type'], entry['domain_type'], entry['arity_type'])
    plots_data[comb]['num_constraints'].append(entry['num_constraints'])
    plots_data[comb]['gdp_runtime'].append(entry['gdp_avg_runtime'])
    plots_data[comb]['gd2p_runtime'].append(entry['gd2p_avg_runtime'])
    plots_data[comb]['fdsp_runtime'].append(entry['fdsp_avg_runtime'])
    plots_data[comb]['gdp_pruned'].append(entry['gdp_avg_pruned_percentage'])
    plots_data[comb]['gd2p_pruned'].append(entry['gd2p_avg_pruned_percentage'])
    plots_data[comb]['fdsp_pruned'].append(entry['fdsp_avg_pruned_percentage'])

# Create 8 plots for avg_runtime
plt.figure(figsize=(20, 12))
for i, comb in enumerate(combinations, 1):
    graph, domain, arity = comb
    plt.subplot(4, 2, i)
    plt.plot(plots_data[comb]['num_constraints'], plots_data[comb]['gdp_runtime'], label='GDP', marker='o')
    plt.plot(plots_data[comb]['num_constraints'], plots_data[comb]['gd2p_runtime'], label='GD2P', marker='s')
    plt.plot(plots_data[comb]['num_constraints'], plots_data[comb]['fdsp_runtime'], label='FDSP', marker='^')
    plt.title(f'Runtime: {graph.capitalize()} Graph, {domain.capitalize()} Domain, {arity.capitalize()} Arity')
    plt.xlabel('Number of Constraints')
    plt.ylabel('Average Runtime (seconds)')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig('runtime_plots1.png')
plt.close()

# Create 8 plots for avg_pruned_percentage
plt.figure(figsize=(20, 12))
for i, comb in enumerate(combinations, 1):
    graph, domain, arity = comb
    plt.subplot(4, 2, i)
    plt.plot(plots_data[comb]['num_constraints'], plots_data[comb]['gdp_pruned'], label='GDP', marker='o')
    plt.plot(plots_data[comb]['num_constraints'], plots_data[comb]['gd2p_pruned'], label='GD2P', marker='s')
    plt.plot(plots_data[comb]['num_constraints'], plots_data[comb]['fdsp_pruned'], label='FDSP', marker='^')
    plt.title(f'Pruned Percentage: {graph.capitalize()} Graph, {domain.capitalize()} Domain, {arity.capitalize()} Arity')
    plt.xlabel('Number of Constraints')
    plt.ylabel('Average Pruned Percentage (%)')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig('pruned_percentage_plots1.png')
plt.close()