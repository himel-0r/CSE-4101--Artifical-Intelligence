import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
with open('results_onekgula.json', 'r') as file:
    data = json.load(file)

# Define the number of constraints
num_constraints = [20, 30, 40, 50, 60, 70, 80, 90, 100]

# Initialize dictionaries to store average pruned percentages for each algorithm
avg_pruned = {
    'gdp': {nc: [] for nc in num_constraints},
    'gd2p': {nc: [] for nc in num_constraints},
    'fdsp': {nc: [] for nc in num_constraints}
}

# Collect pruned percentages for each number of constraints
for entry in data:
    nc = entry['num_constraints']
    avg_pruned['gdp'][nc].append(entry['gdp_avg_pruned_percentage'])
    avg_pruned['gd2p'][nc].append(entry['gd2p_avg_pruned_percentage'])
    avg_pruned['fdsp'][nc].append(entry['fdsp_avg_pruned_percentage'])

# Compute the average pruned percentage for each algorithm at each number of constraints
mean_pruned = {
    'gdp': [np.mean(avg_pruned['gdp'][nc]) for nc in num_constraints],
    'gd2p': [np.mean(avg_pruned['gd2p'][nc]) for nc in num_constraints],
    'fdsp': [np.mean(avg_pruned['fdsp'][nc]) for nc in num_constraints]
}

# Set up the bar plot
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.25
x = np.arange(len(num_constraints))

# Plot bars for each algorithm
ax.bar(x - bar_width, mean_pruned['gdp'], bar_width, label='GDP', color='skyblue')
ax.bar(x, mean_pruned['gd2p'], bar_width, label='GD2P', color='lightgreen')
ax.bar(x + bar_width, mean_pruned['fdsp'], bar_width, label='FDSP', color='salmon')

# Customize the plot
ax.set_xlabel('Number of Constraints')
ax.set_ylabel('Average Pruned Percentage (%)')
ax.set_title('Pruned Percentage Comparison Across Different Problem Scales')
ax.set_xticks(x)
ax.set_xticklabels(num_constraints)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('pruned_percentage_comparison.png')
plt.close()