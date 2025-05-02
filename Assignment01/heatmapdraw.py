import ast
import matplotlib.pyplot as plt
from main import graph  # Assumes main.py and heatmapdraw.py are in the same folder

def plot_heatmaps(grid, title, visit_array, ax):
    if not isinstance(visit_array, list) or not all(isinstance(row, list) for row in visit_array):
        raise ValueError(f"Visit frequency for {title} is not 2D!")

    row, col = len(visit_array), len(visit_array[0])
    max_val = max(max(row) for row in visit_array) if visit_array else 1
    norm_vals = [[val / max_val if max_val > 0 else 0 for val in row] for row in visit_array]

    for i in range(row):
        for j in range(col):
            alpha = norm_vals[i][j]
            color = (1, 0, 0, alpha)  # Red with varying opacity
            ax.add_patch(plt.Rectangle((j, row - i - 1), 1, 1, color=color))

    ax.set_xlim(0, col)
    ax.set_ylim(0, row)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

if __name__ == "__main__":
    with open("testc", "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().replace('\x00', '')
    array = ast.literal_eval(content)

    algorithms = {
        "BFS": lambda g: g.bfs(),
        "DFS": lambda g: g.dfs_main(),
        "DLS": lambda g: g.dls(100),
        "UCS": lambda g: g.ucs(),
        "IDS": lambda g: g.ids(),
        "Best-First": lambda g: g.best_first(),
        "A*": lambda g: g.a_star(),
        "Weighted A*": lambda g: g.weighted_a_star(1.5),
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (name, func) in enumerate(algorithms.items()):
        g = graph(array)
        try:
            func(g)
            visit = g.visit_frequency
            # Ensure visit is a valid 2D list
            if not isinstance(visit, list) or not all(isinstance(row, list) for row in visit):
                raise ValueError
        except Exception:
            # Use blank visit frequency if any failure occurs
            visit = [[0 for _ in range(g.col)] for _ in range(g.row)]
        plot_heatmaps(g, name, visit, axes[idx])

    plt.tight_layout()
    plt.show()
