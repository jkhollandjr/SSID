import numpy as np
import os
import multiprocessing as mp
from sklearn.model_selection import train_test_split

WINDOW_SIZE = 1000
TIME_INTERVAL = 0.1
BIN_SIZE = 0.05

def calculate_cumulative_traffic(packet_sizes, packet_times, window_duration=5.0, step=0.1):
    # Create an array representing time intervals of window_duration with step size
    time_bins = np.arange(0, window_duration, step)
    cumulative_traffic = np.zeros(len(time_bins))

    for size, time in zip(packet_sizes, packet_times):
        index = int(time // step)
        if 0 <= index < len(cumulative_traffic):
            cumulative_traffic[index] += size

    return np.cumsum(cumulative_traffic)

def count_packets_in_bins(packet_times, bin_size=0.05, window_duration=5.0):
    # Create bins of bin_size within the window_duration
    bins = np.arange(0, window_duration, bin_size)
    packet_counts = np.histogram(packet_times, bins=bins)[0]
    return packet_counts

def resize_array(arr, target_size=WINDOW_SIZE):
    return np.pad(arr, (0, max(0, target_size - len(arr))), mode='constant')[:target_size]

def process_window(packets, window_start, window_end):
    # Filter and adjust packet times for the window
    window_packets = [(t - window_start, s) for t, s in packets if window_start <= t < window_end]

    # Separate download and upload packets
    download_packets = [p for p in window_packets if p[1] < 0]
    upload_packets = [p for p in window_packets if p[1] > 0]

    # Function to process packets for each representation
    def process_representation(packets):
        if not packets:
            return np.zeros((4, WINDOW_SIZE))

        times, sizes = zip(*packets)
        times, sizes = np.array(times), np.array(sizes)

        # Inter-arrival times
        inter_arrival_times = np.diff(times, prepend=times[0])
        inter_arrival_times = resize_array(inter_arrival_times)

        # Absolute values of packet sizes
        abs_sizes = resize_array(np.abs(sizes))

        # Cumulative bytes of traffic
        cumulative_traffic = calculate_cumulative_traffic(np.abs(sizes), times)
        cumulative_traffic = resize_array(cumulative_traffic)

        # Amount of packets in bins
        packet_counts = count_packets_in_bins(times)
        packet_counts = resize_array(packet_counts)

        return np.stack([inter_arrival_times, abs_sizes, cumulative_traffic, packet_counts])

    # Process download and upload packets
    download_representations = process_representation(download_packets)
    upload_representations = process_representation(upload_packets)

    return np.vstack([download_representations, upload_representations])

def convert_file_to_numpy(filename):
    with open(filename, 'r') as file:
        packets = [(float(line.split('\t')[0]), float(line.split('\t')[1])) for line in file]
    
    max_time = 60
    # Create windows and process each
    windows = []
    for start in np.arange(0, 22, 2):
        end = start + 5
        window_representation = process_window(packets, start, end)
        windows.append(window_representation)

    # Process the entire trace as a single window
    full_trace_representation = process_window(packets, 0, max_time)
    windows.append(full_trace_representation)

    return np.array(windows)

# Function to process a directory of files
def process_directory(directory):
    file_list = sorted([os.path.join(directory, file) for file in os.listdir(directory)])
    with mp.Pool(processes=30) as pool:
        arrays = pool.map(convert_file_to_numpy, file_list)
    return np.stack(arrays)

inflow_directory = "/home/james/Desktop/research/SSID/SSID_Capture/inflow_nov30/"
outflow_directory = "/home/james/Desktop/research/SSID/SSID_Capture/outflow_nov30/"

# Process directories
inflow_data = process_directory(inflow_directory)
outflow_data = process_directory(outflow_directory)

# Split data
indices = list(range(len(inflow_data)))
train_indices, val_indices = train_test_split(indices, test_size=0.25)

train_inflows = inflow_data[train_indices]
val_inflows = inflow_data[val_indices]
train_outflows = outflow_data[train_indices]
val_outflows = outflow_data[val_indices]

# Save the numpy arrays
np.save('train_inflows.npy', train_inflows)
np.save('val_inflows.npy', val_inflows)
np.save('train_outflows.npy', train_outflows)
np.save('val_outflows.npy', val_outflows)

