import matplotlib.pyplot as plt
import asyncio
import platform
FPS = 60

# Data for dense graphs (average pruning % from Small and Large High Dense, Low Dense)
algorithms = ['GDP', 'GD2P', 'FDSP']
avg_pruning = [46.3, 57.7, 86.4]  # Averages for High Dense and Low Dense configurations

async def main():
    setup()  # Initialize matplotlib
    plt.bar(algorithms, avg_pruning, color=['skyblue', 'lightgreen', 'salmon'])
    plt.xlabel('Algorithms')
    plt.ylabel('Avg Pruning %')
    plt.title('Average Pruning Percentage for Dense Graphs')
    plt.ylim(0, 100)  # Set y-axis limit from 0 to 100%
    save_plot()  # Save the plot
    await asyncio.sleep(1.0 / FPS)  # Control frame rate

def setup():
    plt.figure(figsize=(8, 6))

def save_plot():
    plt.savefig('pruning_percentage_dense.png', dpi=300, bbox_inches='tight')

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())