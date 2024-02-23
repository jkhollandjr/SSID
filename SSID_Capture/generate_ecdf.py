import os
import glob
import numpy as np
import scipy.stats as stats

directory = 'inflow_cdf/'

def compute_ecdf(data):
    sorted_data = np.sort(data)
    ecdf_values = np.arange(1, len(data) + 1) / len(data)

    def ecdf_function(new_data_point):
        return ecdf_values[np.searchsorted(sorted_data, new_data_point, side="right") - 1]

    return ecdf_function

def save_ecdf_function(original_data, filename="ecdf_function.npy"):
    ecdf_func = compute_ecdf(original_data)
    np.save(filename, np.array([original_data, ecdf_func(original_data)]))

def load_ecdf_function(filename="ecdf_function.npy"):
    data, ecdf_values = np.load(filename, allow_pickle=True)

    def ecdf_function(new_data_point):
        return ecdf_values[np.searchsorted(data, new_data_point, side="right") - 1]

    return ecdf_function


# Load files from the first directory
file_paths = [directory + f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Extract packet times
packet_times = []

for file_path in file_paths:
    with open(file_path, 'r') as f:
        for line in f:
            time, _ = map(float, line.strip().split('\t'))
            if time <= 120:
                packet_times.append(time)


data = np.array(packet_times)

save_ecdf_function(data)
