import matplotlib.pyplot as plt
import csv
import numpy as np

def plot_benchmark_results(filename="rrt_star_benchmark_low_iter.csv"):
    step_sizes = []
    times = []
    lengths = []
    
    print(f"Reading {filename}...")
    
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            for row in reader:
                if not row: continue # Skip empty rows
                step_sizes.append(float(row[0]))
                times.append(float(row[1]))
                lengths.append(float(row[2]))
    except FileNotFoundError:
        print(f"Error: Could not find '{filename}'. Make sure you ran the benchmark script first.")
        return

    # Convert to numpy arrays for easier sorting
    step_sizes = np.array(step_sizes)
    times = np.array(times)
    lengths = np.array(lengths)

    # Sort data by Computation Time (x-axis) so the line connects properly
    sort_idx = np.argsort(times)
    times_sorted = times[sort_idx]
    lengths_sorted = lengths[sort_idx]
    steps_sorted = step_sizes[sort_idx]

    # Create Plot
    plt.figure(figsize=(12, 8))
    
    # Plot line and points
    plt.plot(times_sorted, lengths_sorted, linestyle='--', color='gray', alpha=0.5, zorder=1)
    scatter = plt.scatter(times_sorted, lengths_sorted, c=steps_sorted, cmap='viridis_r', s=100, zorder=2)
    
    # Annotate points with Step Size values
    for i, step in enumerate(steps_sorted):
        # Shift text slightly to avoid overlapping the dot
        plt.annotate(f"Step={step:.3f}", 
                     (times_sorted[i], lengths_sorted[i]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=9)

    # Labels and Title
    plt.xlabel("Average Computation Time (s)", fontsize=12)
    plt.ylabel("Average Path Length (m)", fontsize=12)
    plt.title("RRT* Benchmark: Cost vs. Speed Trade-off", fontsize=14)
    
    # Colorbar to indicate Step Size visually
    cbar = plt.colorbar(scatter)
    cbar.set_label('Step Size (m)', rotation=270, labelpad=15)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_benchmark_results()
