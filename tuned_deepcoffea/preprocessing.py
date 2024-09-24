import os
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split
import argparse

# How to run:
# python script.py --inflow_dir /path/to/inflow --outflow_dir /path/to/outflow --save_dir /path/to/save --window_size 1000

def parse_args():
    parser = argparse.ArgumentParser(description="Process network packet data.")
    parser.add_argument('--inflow_dir', type=str, required=True, help="Path to the inflow directory.")
    parser.add_argument('--outflow_dir', type=str, required=True, help="Path to the outflow directory.")
    parser.add_argument('--save_dir', type=str, default='data/', help="Directory to save the .npy files.")
    parser.add_argument('--window_size', type=int, default=1000, help="Window size for processing.")
    return parser.parse_args()

def calculate_cumulative_traffic(packet_sizes, packet_times, window_size):
    cumulative_traffic = np.zeros(512)

    for size, time in zip(packet_sizes, packet_times):
        index = int(time // 0.1)
        if 0 <= index < len(cumulative_traffic):
            cumulative_traffic[index] += size

    cumulative_traffic = np.cumsum(cumulative_traffic)
    padded_cumulative_traffic = np.pad(cumulative_traffic, (0, window_size - len(cumulative_traffic)), 'constant')

    return padded_cumulative_traffic

def resize_array(arr, target_size):
    pad_size = max(0, target_size - len(arr))
    arr_padded = np.pad(arr, (0, pad_size), mode='constant')
    arr_resized = arr_padded[:target_size]

    return arr_resized

def convert_file_to_numpy(filename, window_size):
    windows = []
    with open(filename, 'r') as rf:
        data = rf.readlines()
        packets = [(float(line.split('\t')[0]), float(line.split('\t')[1])) for line in data]

        for start in np.arange(0, 22, 2):
            end = start + 5
            window_packets = [p for p in packets if start <= p[0] < end]
            window_packets = [(p[0] - start, p[1]) for p in window_packets]
            if len(window_packets) < window_size:
                window_packets += [(0, 0)] * (window_size - len(window_packets))
            else:
                window_packets = window_packets[:window_size]

            times, sizes = zip(*window_packets)
            times = np.array(times)
            directions = np.sign(sizes)
            sizes = np.array(np.abs(sizes)) / 100
            non_padded_diff = np.diff(times[times != 0], prepend=0)
            inter_packet_times = np.pad(non_padded_diff, (0, len(times) - len(non_padded_diff)), mode='constant')

            times_with_direction = times * directions
            inter_packet_times = np.abs(inter_packet_times)

            cusum = np.cumsum(sizes) / 1000000
            cusum = resize_array(cusum, window_size)

            window = np.stack([sizes, inter_packet_times, times_with_direction, directions, cusum])
            windows.append(window)

        if len(packets) < window_size:
            packets += [(0, 0)] * (window_size - len(packets))
        else:
            packets = packets[:window_size]
        times, sizes = zip(*packets)

        directions = np.sign(sizes)
        times = np.array(times)
        sizes = np.array(np.abs(sizes)) / 100
        non_padded_diff = np.diff(times[times != 0], prepend=0)
        inter_packet_times = np.pad(non_padded_diff, (0, len(times) - len(non_padded_diff)), mode='constant')
        inter_packet_times_inv = np.reciprocal(np.where(inter_packet_times == 0, 1, inter_packet_times), dtype=float)

        times_with_direction = times * directions
        inter_packet_times = np.abs(inter_packet_times)

        cumul = calculate_cumulative_traffic(np.abs(sizes), np.abs(times), window_size)
        window = np.stack([sizes, inter_packet_times, times_with_direction, directions, cumul])
        windows.append(window)

    return np.stack(windows)

def process_directory(directory, window_size):
    file_list = sorted([os.path.join(directory, file) for file in os.listdir(directory)])

    with mp.Pool(processes=mp.cpu_count()) as pool:
        arrays = pool.starmap(convert_file_to_numpy, [(file, window_size) for file in file_list])

    return np.stack(arrays)

def main():
    args = parse_args()

    inflow_data = process_directory(args.inflow_dir, args.window_size)
    outflow_data = process_directory(args.outflow_dir, args.window_size)

    indices = list(range(len(inflow_data)))
    train_indices, val_indices = train_test_split(indices, test_size=0.25)

    train_inflows = inflow_data[train_indices]
    val_inflows = inflow_data[val_indices]

    train_outflows = outflow_data[train_indices]
    val_outflows = outflow_data[val_indices]

    np.save(os.path.join(args.save_dir, 'train_inflows.npy'), train_inflows)
    np.save(os.path.join(args.save_dir, 'val_inflows.npy'), val_inflows)
    np.save(os.path.join(args.save_dir, 'train_outflows.npy'), train_outflows)
    np.save(os.path.join(args.save_dir, 'val_outflows.npy'), val_outflows)

if __name__ == "__main__":
    main()

