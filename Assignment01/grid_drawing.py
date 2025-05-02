import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_grid(grid):
    n = len(grid)
    fig, ax = plt.subplots(figsize=(n, n))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])

    # Reverse y-axis so (0,0) is at top-left like a grid
    ax.invert_yaxis()

    # Directional offsets: w, x, y, z -> up, right, down, left
    dir_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    dir_positions = [(0.5, 0.05), (0.95, 0.5), (0.5, 0.95), (0.05, 0.5)]  # text position

    for i in range(n):
        for j in range(n):
            cell = grid[i][j]
            ctype, w, x, y, z = cell
            # Draw the square
            rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='white')
            ax.add_patch(rect)

            # Write the cell type at center
            ax.text(j + 0.5, i + 0.5, str(ctype), ha='center', va='center', fontsize=12, weight='bold')

            # Draw cost labels on valid edges
            costs = [w, x, y, z]
            for d in range(4):
                ni, nj = i + dir_offsets[d][0], j + dir_offsets[d][1]
                if 0 <= ni < n and 0 <= nj < n and costs[d] != -1:
                    x_text = j + dir_positions[d][0]
                    y_text = i + dir_positions[d][1]
                    ax.text(x_text, y_text, str(costs[d]), fontsize=8, color='blue')

    plt.grid(False)
    plt.savefig("grid_5.png", dpi=300, bbox_inches='tight')
    plt.show()

# Example usage:
# Paste the grid output from the generator function here
# Example small input:
grid = [[[2, -1, 6, 4, -1], [2, -1, 5, 7, 6], [2, -1, 7, 3, 5], [2, -1, -1, 7, 7], [3, -1, -1, -1, -1]],[[2, 4, 7, 10, -1], [2, 7, 7, 6, 7], [2, 3, 2, 6, 7], [2, 7, 5, 5, 2], [2, -1, 7, 7, 5]], [[0, 10, 2, 4, -1], [2, 6, 10, -1, 2], [2, 6, 4, 3, 10], [1, 5, 2, 3, 4], [2, 7, 4, -1, 2]], [[2, 4, -1, 8, -1], [3, -1, -1, -1, -1], [2, 3, 8, 1, -1], [2, 3, -1, 10, 8], [3, -1, -1, -1, -1]], [[2, 8, -1, 4, -1], [3, -1, -1, -1, -1], [2, 1, 4, 2, -1], [2, 10, 8, 10, 4], [2, -1, 6, 10, 8]]]

# grid = input()

draw_grid(grid)
