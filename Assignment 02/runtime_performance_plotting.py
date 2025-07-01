import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
with open('results_onekgula.json', 'r') as file:
    data = json.load(file)

# Define the number of constraints
num_constraints = [20, 30, 40, 50, 60, 70, 80, 90, 100]

# Initialize dictionaries to store average runtimes for each algorithm
avg_runtimes = {
    'gdp': {nc: [] for nc in num_constraints},
    'gd2p': {nc: [] for nc in num_constraints},
    'fdsp': {nc: [] for nc in num_constraints}
}

# Collect runtimes for each number of constraints
for entry in data:
    nc = entry['num_constraints']
    avg_runtimes['gdp'][nc].append(entry['gdp_avg_runtime'])
    avg_runtimes['gd2p'][nc].append(entry['gd2p_avg_runtime'])
    avg_runtimes['fdsp'][nc].append(entry['fdsp_avg_runtime'])

# Compute the average runtime for each algorithm at each number of constraints
mean_runtimes = {
    'gdp': [np.mean(avg_runtimes['gdp'][nc]) for nc in num_constraints],
    'gd2p': [np.mean(avg_runtimes['gd2p'][nc]) for nc in num_constraints],
    'fdsp': [np.mean(avg_runtimes['fdsp'][nc]) for nc in num_constraints]
}

# Set up the bar plot
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.25
x = np.arange(len(num_constraints))

# Plot bars for each algorithm
ax.bar(x - bar_width, mean_runtimes['gdp'], bar_width, label='GDP', color='skyblue')
ax.bar(x, mean_runtimes['gd2p'], bar_width, label='GD2P', color='lightgreen')
ax.bar(x + bar_width, mean_runtimes['fdsp'], bar_width, label='FDSP', color='salmon')

# Customize the plot
ax.set_xlabel('Number of Constraints')
ax.set_ylabel('Average Runtime (seconds)')
ax.set_title('Runtime Performance Comparison Across Different Problem Scales')
ax.set_xticks(x)
ax.set_xticklabels(num_constraints)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('runtime_performance_comparison.png')
plt.close()