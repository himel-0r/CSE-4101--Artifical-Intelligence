import ast
import time
import tracemalloc
import pandas as pd
import matplotlib.pyplot as plt
from main import graph  # Adjust if your file is named differently

testcases = [f"testcase{i}" for i in range(1, 11)]
algorithms = [
    "bfs", "dfs_main", "dls", "ucs",
    "ids", "best_first", "a_star", "weighted_a_star"
]

# Store results
time_data = {algo: [] for algo in algorithms}
memory_data = {algo: [] for algo in algorithms}

# Run all algorithms on all test cases
for tname in testcases:
    with open(tname, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().replace('\x00', '')
    array = ast.literal_eval(content)
    g = graph(array)

    for algo in algorithms:
        tracemalloc.start()
        start_time = time.perf_counter()

        try:
            if algo == "dls":
                getattr(g, algo)(200)  # custom depth
            elif algo == "weighted_a_star":
                getattr(g, algo)(w=1.5)  # custom weight
            else:
                getattr(g, algo)()
        except Exception as e:
            print(f"Error in {algo} on {tname}: {e}")

        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed = end_time - start_time

        time_data[algo].append(elapsed)
        memory_data[algo].append(peak / 1024)  # in KB

# ---- Convert to DataFrames ----
df_time = pd.DataFrame(time_data, index=testcases)
df_memory = pd.DataFrame(memory_data, index=testcases)

# ---- Save tables ----
df_time.to_csv("execution_time.csv")
df_memory.to_csv("memory_usage.csv")

print("Execution Time (seconds):")
print(df_time.round(5))
print("\nMemory Usage (KB):")
print(df_memory.round(2))

# # ---- Plotting Time ----
# plt.figure(figsize=(12, 6))
# for algo in algorithms:
#     plt.plot(testcases, df_time[algo], label=algo)
# plt.ylabel("Time (seconds)")
# plt.title("Execution Time of Algorithms across Test Cases")
# plt.legend()
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("execution_time_plot.png")
# plt.show()

# # ---- Plotting Memory ----
# plt.figure(figsize=(12, 6))
# for algo in algorithms:
#     plt.plot(testcases, df_memory[algo], label=algo)
# plt.ylabel("Memory Usage (KB)")
# plt.title("Memory Usage of Algorithms across Test Cases")
# plt.legend()
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("memory_usage_plot.png")
# plt.show()


import numpy as np

# --- Grouped Bar Chart Function ---
def plot_grouped_bar(df, ylabel, title, filename):
    x = np.arange(len(df.index))  # positions for test cases
    width = 0.1  # width of each bar
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, algo in enumerate(df.columns):
        offset = (i - len(df.columns)/2) * width + width/2
        ax.bar(x + offset, df[algo], width, label=algo)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# --- Plot Execution Time (Bar Chart) ---
plot_grouped_bar(df_time, "Time (seconds)", "Execution Time of Algorithms (Bar Chart)", "execution_time_bar_chart.png")

# --- Plot Memory Usage (Bar Chart) ---
plot_grouped_bar(df_memory, "Memory (KB)", "Memory Usage of Algorithms (Bar Chart)", "memory_usage_bar_chart.png")
