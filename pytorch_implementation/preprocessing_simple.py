import os
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split

WINDOW_SIZE = 1000

def resize_array(arr, target_size):
    # Calculate how many zeros to pad
    pad_size = max(0, target_size - len(arr))

    # Pad the array if it's smaller than the target size
    arr_padded = np.pad(arr, (0, pad_size), mode='constant')

    # Truncate the array if it's larger than the target size
    arr_resized = arr_padded[:target_size]

    return arr_resized

def convert_file_to_numpy(filename):
    windows = []
    with open(filename, 'r') as rf:
        data = rf.readlines()

        # Convert each line to (time, size) tuple
        packets = [(float(line.split('\t')[0]), float(line.split('\t')[1])) for line in data]

        # Create windows
        for start in np.arange(0, 22, 2):  # 0, 2, 4, ..., 18
            end = start + 5
            window_packets = [p for p in packets if start <= p[0] < end]
            window_packets = [(p[0]-start, p[1]) for p in window_packets]
            if len(window_packets) < WINDOW_SIZE:
                window_packets += [(0, 0)] * (WINDOW_SIZE - len(window_packets))
            else:
                window_packets = window_packets[:WINDOW_SIZE]

            # Separate times and sizes
            times, sizes = zip(*window_packets)

            # Convert to numpy arrays
            times = np.array(times)
            sizes = np.array(sizes)
            directions = np.sign(sizes)

            window = np.stack([np.abs(sizes), times, directions])
            windows.append(window)

        if(len(packets) < WINDOW_SIZE):
            packets += [(0,0)] * (WINDOW_SIZE - len(packets))
        else:
            packets = packets[:WINDOW_SIZE]

        times, sizes = zip(*packets)

        directions = np.sign(sizes)
        times = np.array(times)
        sizes = np.array(sizes)

        window = np.stack([np.abs(sizes), times, directions])
        windows.append(window)

    return np.stack(windows)

def process_directory(directory):
    file_list = sorted([os.path.join(directory, file) for file in os.listdir(directory)])
    with mp.Pool(processes=mp.cpu_count()) as pool:
        arrays = pool.map(convert_file_to_numpy, file_list)

    return np.stack(arrays)

# Process directories
inflow_directory = "/home/james/Desktop/research/SSID/CrawlE_Proc_20000/inflow/"
outflow_directory = "/home/james/Desktop/research/SSID/CrawlE_Proc_20000/outflow/"

inflow_data = process_directory(inflow_directory)
outflow_data = process_directory(outflow_directory)

# Generate indices for splits
indices = list(range(len(inflow_data)))
train_indices, val_indices = train_test_split(indices, test_size=0.25)

# Split inflow_data and outflow_data using the same indices
train_inflows = inflow_data[train_indices]
val_inflows = inflow_data[val_indices]

train_outflows = outflow_data[train_indices]
val_outflows = outflow_data[val_indices]

# Save the numpy arrays for later use
np.save('train_inflows.npy', train_inflows)
np.save('val_inflows.npy', val_inflows)
np.save('train_outflows.npy', train_outflows)
np.save('val_outflows.npy', val_outflows)

