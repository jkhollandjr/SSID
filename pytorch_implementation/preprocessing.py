import os
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split

WINDOW_SIZE = 500

def convert_file_to_numpy(filename):
    windows = []
    with open(filename, 'r') as rf:
        data = rf.readlines()

        # Convert each line to (time, size) tuple
        packets = [(float(line.split('\t')[0]), float(line.split('\t')[1])) for line in data]

        # Create 11 overlapping windows
        for start in np.arange(0, 22, 2):  # 0, 2, 4, ..., 18
            end = start + 5
            window_packets = [p for p in packets if start <= p[0] < end]
            window_packets = [(p[0]-start, p[1]) for p in window_packets]
            if len(window_packets) < 500:
                window_packets += [(0, 0)] * (500 - len(window_packets))
            else:
                window_packets = window_packets[:500]

            # Separate times and sizes
            times, sizes = zip(*window_packets)

            # Convert to numpy arrays and create 4 representations
            times = np.array(times)
            sizes = np.array(np.abs(sizes))
            directions = np.sign(sizes)

            # Handle the diff computation as described
            non_padded_diff = np.diff(times[times != 0], prepend=0)
            inter_packet_times = np.pad(non_padded_diff, (0, len(times) - len(non_padded_diff)), mode='constant')

            # Make sizes of download packets negative
            times_with_direction = times * directions
            inter_packet_times = np.abs(inter_packet_times)

            window = np.stack([sizes, inter_packet_times, times_with_direction, directions])
            windows.append(window)

    return np.stack(windows)


# Function to process a directory of files
def process_directory(directory):
    #Sort the list of input traces so that matched flows can be easily identified later
    file_list = sorted([os.path.join(directory, file) for file in os.listdir(directory)])

    # Use a pool of worker processes to convert files in parallel
    #with mp.Pool(processes=mp.cpu_count()-5) as pool:
    with mp.Pool(processes=1) as pool:
        arrays = pool.map(convert_file_to_numpy, file_list)

    return np.stack(arrays)

inflow_directory = "/home/james/Desktop/research/SSID/SSID_Capture/inflow_obfuscated/"
outflow_directory = "/home/james/Desktop/research/SSID/SSID_Capture/outflow/"

# Process directories
inflow_data = process_directory(inflow_directory)
outflow_data = process_directory(outflow_directory)

# Each of inflow_data and outflow_data are numpy arrays of shape (# traces, 10, 4, 1000)

# Generate indices for splits
indices = list(range(len(inflow_data)))
train_indices, val_indices = train_test_split(indices, test_size=0.25)

# Split inflow_data and outflow_data using the same indices
train_inflows = inflow_data[train_indices]
val_inflows = inflow_data[val_indices]

train_outflows = outflow_data[train_indices]
val_outflows = outflow_data[val_indices]

# Save the numpy arrays for later use
np.save('train_inflows_obfuscated.npy', train_inflows)
np.save('val_inflows_obfuscated.npy', val_inflows)

np.save('train_outflows_obfuscated.npy', train_outflows)
np.save('val_outflows_obfuscated.npy', val_outflows)
