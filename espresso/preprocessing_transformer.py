'''
python preprocessing_transformer.py \
    --inflow_directory /path/to/inflow_directory \
    --outflow_directory /path/to/outflow_directory \
    --output_directory data \
    --window_size 1000 \
    --interval_size 0.03 \
    --test_size 0.25 \
    --num_processes 4
'''
import argparse
import os
import multiprocessing as mp

import numpy as np
from sklearn.model_selection import train_test_split


def remove_right_padded_zeros(arr):
    """Remove trailing zeros from a numpy array."""
    idx = np.where(arr != 0)[0]
    if len(idx):
        return arr[:idx[-1] + 1]
    else:
        return arr


def pad_or_truncate(arr, size):
    """Pad or truncate a numpy array to a specified size."""
    if len(arr) < size:
        return np.pad(arr, (0, size - len(arr)), 'constant')
    else:
        return arr[:size]


def convert_file_to_numpy(filename, window_size, interval_size):
    """Convert a single file to a numpy array of features."""
    with open(filename, 'r') as rf:
        data = rf.readlines()

    # Extract times and sizes from the file
    packets = [
        (float(line.split('\t')[0]), float(line.split('\t')[1])) for line in data
    ]

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

    num_intervals = int(np.ceil(times.max() / interval_size))

    split_points = np.arange(0, num_intervals) * interval_size
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
    if np.abs(interval_times_norm).max() != 0:
        interval_times_norm /= np.abs(interval_times_norm).max()

    interval_iats = np.zeros(num_intervals + 1)
    for j, tensor in enumerate(np.split(iats, split_indices)):
        if len(tensor) > 0:
            interval_iats[j] = tensor.mean()
        elif j > 0:
            interval_iats[j] = interval_iats[j - 1] + interval_size

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
    if np.abs(interval_cumul_norm).max() != 0:
        interval_cumul_norm /= np.abs(interval_cumul_norm).max()

    interval_dirs_sum = interval_dirs_up + interval_dirs_down
    interval_dirs_sub = interval_dirs_up - interval_dirs_down

    features = np.stack([
        pad_or_truncate(sizes, size=window_size),
        pad_or_truncate(times, size=window_size),
        pad_or_truncate(directions, size=window_size),
        pad_or_truncate(interval_dirs_up, size=window_size),
        pad_or_truncate(interval_dirs_down, size=window_size),
        pad_or_truncate(interval_dirs_sum, size=window_size),
        pad_or_truncate(interval_dirs_sub, size=window_size),
        pad_or_truncate(interval_iats, size=window_size),
        pad_or_truncate(interval_inv_iat_logs, size=window_size),
        pad_or_truncate(interval_cumul_norm, size=window_size),
        pad_or_truncate(interval_times_norm, size=window_size)
    ])

    return features


def process_directory(directory, window_size, interval_size, num_processes):
    """Process all files in a directory and return a stacked array of features."""
    file_list = sorted(
        [os.path.join(directory, file) for file in os.listdir(directory)]
    )
    args_list = [
        (filename, window_size, interval_size) for filename in file_list
    ]
    with mp.Pool(processes=num_processes) as pool:
        arrays = pool.starmap(convert_file_to_numpy, args_list)
    return np.stack(arrays)


def main():
    parser = argparse.ArgumentParser(
        description='Process packet capture data and save as numpy arrays.'
    )
    parser.add_argument(
        '--inflow_directory',
        type=str,
        required=True,
        help='Path to inflow directory'
    )
    parser.add_argument(
        '--outflow_directory',
        type=str,
        required=True,
        help='Path to outflow directory'
    )
    parser.add_argument(
        '--output_directory',
        type=str,
        default='data',
        help='Directory to save output numpy arrays'
    )
    parser.add_argument(
        '--max_window_size',
        type=int,
        default=1000,
        help='Window size for padding/truncating arrays'
    )
    parser.add_argument(
        '--interval_size',
        type=float,
        default=0.03,
        help='Interval size for processing'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.25,
        help='Proportion of data to use for validation set'
    )
    parser.add_argument(
        '--num_processes',
        type=int,
        default=mp.cpu_count(),
        help='Number of processes to use for multiprocessing'
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_directory, exist_ok=True)

    # Process directories
    print('Processing inflow directory...')
    inflow_data = process_directory(
        args.inflow_directory,
        args.max_window_size,
        args.interval_size,
        args.num_processes
    )
    print('Processing outflow directory...')
    outflow_data = process_directory(
        args.outflow_directory,
        args.max_window_size,
        args.interval_size,
        args.num_processes
    )

    # Generate indices for splits
    indices = list(range(len(inflow_data)))
    train_indices, val_indices = train_test_split(
        indices, test_size=args.test_size
    )

    # Split inflow_data and outflow_data using the same indices
    train_inflows = inflow_data[train_indices]
    val_inflows = inflow_data[val_indices]

    train_outflows = outflow_data[train_indices]
    val_outflows = outflow_data[val_indices]

    # Save the numpy arrays for later use
    np.save(
        os.path.join(args.output_directory, 'train_inflows.npy'), train_inflows
    )
    np.save(
        os.path.join(args.output_directory, 'val_inflows.npy'), val_inflows
    )
    np.save(
        os.path.join(args.output_directory, 'train_outflows.npy'),
        train_outflows
    )
    np.save(
        os.path.join(args.output_directory, 'val_outflows.npy'), val_outflows
    )

    print('Processing complete. Numpy arrays saved to', args.output_directory)


if __name__ == '__main__':
    main()

