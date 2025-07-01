import ast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from main import graph  # Adjust this import based on your actual file location

def visualize_all_paths(testcase_file="testc"):
    with open(testcase_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().replace('\x00', '')
    array = ast.literal_eval(content)
    g = graph(array)

    algorithms = [
        "bfs", "dfs_main", "dls", "ucs",
        "ids", "best_first", "a_star", "weighted_a_star"
    ]

    results = {}
    # Creating a dictionary to store the visit frequency matrices
    visit_frequencies = {}

    for algo in algorithms:
        try:
            g = graph(array)  # Reset the graph for each algorithm
            if algo == "dls":
                g.dls(200)
            elif algo == "weighted_a_star":
                g.weighted_a_star(1.5)
            else:
                getattr(g, algo)()
        except Exception as e:
            print(f"Error in {algo}: {e}")

        # Store the found path for each algorithm
        if hasattr(g, "last_path") and g.last_path:
            results[algo] = g.last_path
        else:
            print(f"Warning: No path found/stored for {algo}")
            results[algo] = []

        # Store the visit frequency for each algorithm's search
        visit_frequencies[algo] = g.visit_frequency

    # Plotting paths for each algorithm
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, algo in enumerate(algorithms):
        ax = axes[i]
        ax.set_title(algo.upper())
        for r in range(g.row):
            for c in range(g.col):
                cell_type = array[r][c][0]
                rect_color = 'white'
                if (r, c) in results[algo]:
                    rect_color = 'blue'  # Path color
                elif (r * g.row + c) == g.start:
                    rect_color = 'green'  # Start color
                elif (r * g.row + c) == g.end:
                    rect_color = 'red'  # End color
                elif cell_type == 3:
                    rect_color = 'black'  # Obstacle color

                ax.add_patch(plt.Rectangle((c, g.row - 1 - r), 1, 1, color=rect_color))
        ax.set_xlim(0, g.col)
        ax.set_ylim(0, g.row)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig("all_paths_testcase1.png")
    plt.show()

    # Creating heatmaps based on visit frequency
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, algo in enumerate(algorithms):
        ax = axes[i]
        ax.set_title(f"{algo.upper()} Heatmap")
        
        # Retrieve the visit frequency matrix for the current algorithm
        visit_freq = visit_frequencies[algo]

        # Plot the heatmap using the frequency matrix
        ax.imshow(visit_freq, cmap="YlOrRd", origin='lower', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("search_heatmaps_with_frequency_testcase1.png")
    plt.show()

# Call the function to generate the visualizations
visualize_all_paths()
