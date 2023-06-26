import os
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split

def convert_file_to_numpy(filename):
    windows = []
    with open(filename, 'r') as rf:
        data = rf.readlines()

        # Convert each line to (time, size) tuple
        packets = [(float(line.split('\t')[0]), float(line.split('\t')[1])) for line in data]

        # Create 10 overlapping windows
        for start in np.arange(0, 20, 2):  # 0, 2, 4, ..., 18
            end = start + 5
            window_packets = [p for p in packets if start <= p[0] < end]

            if len(window_packets) < 1000:
                window_packets += [(0, 0)] * (1000 - len(window_packets))  # Pad with zeros
            else:
                window_packets = window_packets[:1000]  # Cut off after 1000 packets

            # Separate times and sizes
            times, sizes = zip(*window_packets)

            # Convert to numpy arrays and create 4 representations
            times = np.array(times)
            sizes = np.array(sizes)
            directions = np.sign(sizes)
            inter_packet_times = np.diff(times, prepend=0)

            # Make sizes of download packets negative
            times_with_direction = times * directions

            window = np.stack([sizes, directions, inter_packet_times, times_with_direction])
            #window = np.stack([sizes, inter_packet_times])
            windows.append(window)

    return np.stack(windows)

# Function to process a directory of files
def process_directory(directory):
    #Sort the list of input traces so that matched flows can be easily identified later
    file_list = sorted([os.path.join(directory, file) for file in os.listdir(directory)])

    # Use a pool of worker processes to convert files in parallel
    with mp.Pool(processes=mp.cpu_count()-5) as pool:
        arrays = pool.map(convert_file_to_numpy, file_list)

    return np.stack(arrays)

inflow_directory = "/home/james/Desktop/research/SSID/SSID_Capture/inflows/"
outflow_directory = "/home/james/Desktop/research/SSID/SSID_Capture/outflows/"

#inflow_directory = "/home/james/Desktop/research/SSID/CrawlE_Proc_20000/inflow"
#outflow_directory = "/home/james/Desktop/research/SSID/CrawlE_Proc_20000/outflow/"

# Process directories
inflow_data = process_directory(inflow_directory)
outflow_data = process_directory(outflow_directory)

# Each of inflow_data and outflow_data are numpy arrays of shape (# traces, 10, 4, 1000)

# Generate indices for splits
indices = list(range(len(inflow_data)))
#train_indices, test_indices = train_test_split(indices, test_size=0.1)
train_indices, val_indices = train_test_split(indices, test_size=0.25)

# Split inflow_data and outflow_data using the same indices
train_inflows = inflow_data[train_indices]
val_inflows = inflow_data[val_indices]
#test_inflows = inflow_data[test_indices]

train_outflows = outflow_data[train_indices]
val_outflows = outflow_data[val_indices]
#test_outflows = outflow_data[test_indices]

# Save the numpy arrays for later use
np.save('train_inflows.npy', train_inflows)
np.save('val_inflows.npy', val_inflows)
#np.save('test_inflows.npy', test_inflows)

np.save('train_outflows.npy', train_outflows)
np.save('val_outflows.npy', val_outflows)
#np.save('test_outflows.npy', test_outflows)
