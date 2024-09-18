import os
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split

WINDOW_SIZE = 1000
INTERVAL_SIZE = 0.03

def remove_right_padded_zeros(arr):
    return arr[:np.argmax(arr[::-1] != 0) or None]

def resize_array(arr, target_size):
    pad_size = max(0, target_size - len(arr))
    arr_padded = np.pad(arr, (0, pad_size), mode='constant')
    arr_resized = arr_padded[:target_size]
    return arr_resized

def pad_or_truncate(arr, size=1000):
    if len(arr) < size:
        return np.pad(arr, (0, size - len(arr)), 'constant')
    else:
        return arr[:size]

def convert_file_to_numpy(filename):
    with open(filename, 'r') as rf:
        data = rf.readlines()

        packets = [(float(line.split('\t')[0]), float(line.split('\t')[1])) for line in data]

        times, sizes = zip(*packets)
        times = np.array(times)
        sizes = np.array(sizes)
        directions = np.sign(sizes)

        times = remove_right_padded_zeros(times)
        sizes = remove_right_padded_zeros(sizes)
        directions = remove_right_padded_zeros(directions)

        upload = directions > 0
        download = ~upload
        iats = np.diff(times, prepend=0)

        num_intervals = int(np.ceil(times.max() / INTERVAL_SIZE))

        split_points = np.arange(0, num_intervals) * INTERVAL_SIZE
        split_indices = np.searchsorted(times, split_points)

        interval_dirs_up = np.zeros(num_intervals + 1)
        interval_dirs_down = np.zeros(num_intervals + 1)
        for j, tensor in enumerate(np.split(directions, split_indices)):
            size = len(tensor)
            if size > 0:
                up = (tensor >= 0).sum()
                interval_dirs_up[j] = up
                interval_dirs_down[j] = size - up

        interval_times = np.zeros(num_intervals + 1)
        for j, tensor in enumerate(np.split(times, split_indices)):
            if len(tensor) > 0:
                interval_times[j] = tensor.mean()
            elif j > 0:
                interval_times[j] = interval_times[j - 1]

        interval_times_norm = interval_times - interval_times.mean()
        interval_times_norm /= np.abs(interval_times_norm).max()

        interval_iats = np.zeros(num_intervals + 1)
        for j, tensor in enumerate(np.split(iats, split_indices)):
            if len(tensor) > 0:
                interval_iats[j] = tensor.mean()
            elif j > 0:
                interval_iats[j] = interval_iats[j - 1] + INTERVAL_SIZE

        download_iats = np.diff(times[download], prepend=0)
        upload_iats = np.diff(times[upload], prepend=0)
        flow_iats = np.zeros_like(times)
        flow_iats[upload] = upload_iats
        flow_iats[download] = download_iats
        inv_iat_logs = np.log(np.nan_to_num(1 / flow_iats + 1, nan=1e4, posinf=1e4))
        interval_inv_iat_logs = np.zeros(num_intervals + 1)
        for j, tensor in enumerate(np.split(inv_iat_logs, split_indices)):
            if len(tensor) > 0:
                interval_inv_iat_logs[j] = tensor.mean()

        size_dirs = sizes * directions
        cumul = np.cumsum(size_dirs)
        interval_cumul = np.zeros(num_intervals + 1)
        for j, tensor in enumerate(np.split(cumul, split_indices)):
            if len(tensor) > 0:
                interval_cumul[j] = tensor.mean()
            elif j > 0:
                interval_cumul[j] = interval_cumul[j - 1]

        interval_cumul_norm = interval_cumul - interval_cumul.mean()
        interval_cumul_norm /= np.abs(interval_cumul_norm).max()

        interval_dirs_sum = interval_dirs_up + interval_dirs_down
        interval_dirs_sub = interval_dirs_up - interval_dirs_down

        features = np.stack([
            pad_or_truncate(sizes),
            pad_or_truncate(times),
            pad_or_truncate(directions),
            pad_or_truncate(interval_dirs_up),
            pad_or_truncate(interval_dirs_down),
            pad_or_truncate(interval_dirs_sum),
            pad_or_truncate(interval_dirs_sub),
            pad_or_truncate(interval_iats),
            pad_or_truncate(interval_inv_iat_logs),
            pad_or_truncate(interval_cumul_norm),
            pad_or_truncate(interval_times_norm)
        ])

    return features

def process_directory(directory):
    file_list = sorted([os.path.join(directory, file) for file in os.listdir(directory)])
    with mp.Pool(processes=1) as pool:
        arrays = pool.map(convert_file_to_numpy, file_list)
    return np.stack(arrays)

# Process directories
inflow_directory = "/home/james/Desktop/research/SSID/SSID_Capture/inflow_may17_fixed/"
outflow_directory = "/home/james/Desktop/research/SSID/SSID_Capture/outflow_may17_fixed/"

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
np.save('data/train_inflows_may17_transformer.npy', train_inflows)
np.save('data/val_inflows_may17_transformer.npy', val_inflows)
np.save('data/train_outflows_may17_transformer.npy', train_outflows)
np.save('data/val_outflows_may17_transformer.npy', val_outflows)

