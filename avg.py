
import numpy as np
import matplotlib as plt

results = np.load('results_dict.npy', allow_pickle=True)
paths = np.load('paths.npy', allow_pickle=True)

for result in results:
    resolution = result['resolution']
    avg_compute_time = sum(result['compute_time']) / len(result['compute_time'])
    avg_path_length = sum(result['path_length']) / len(result['path_length'])
    print(f"Resolution: {resolution:.2f}, Average Compute Time: {avg_compute_time:.4f}, Average Path Length: {avg_path_length:.4f}")
