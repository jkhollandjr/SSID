import os
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

def get_keys_for_value(input_dict: Dict, value: int) -> set:
    """Get keys for a given value"""
    return {k for k, v in input_dict.items() if v == value}

def process_flow_data(csv_path: str, interval: Tuple[int, int], threshold: int) -> Dict[str, int]:
    """Process flow data in specified interval and return filenames that meet the threshold."""
    inflow_path = os.path.join(csv_path, 'inflow')
    outflow_path = os.path.join(csv_path, 'outflow')
    
    filenames = os.listdir(inflow_path)
    flow_counts = defaultdict(int)

    for filename in filenames:
        inflow_packets = count_packets_in_interval(os.path.join(inflow_path, filename), interval)
        outflow_packets = count_packets_in_interval(os.path.join(outflow_path, filename), interval)
        
        if inflow_packets > threshold and outflow_packets > threshold:
            flow_counts[filename] += 1

    return flow_counts

def count_packets_in_interval(file_path: str, interval: Tuple[int, int]) -> int:
    """Count the number of packets in the specified time interval."""
    with open(file_path) as f:
        lines = [line for line in f]
    
    packet_count = 0
    for line in lines:
        time = float(line.split('\t')[0])
        if interval[0] <= time <= interval[1]:
            packet_count += 1

    return packet_count

def analyze_overlapping_windows(csv_path: str, out_path: str, threshold: int, interval: int, num_windows: int, addnum: int):
    """Analyze packet flow data in overlapping windows and write results to output file."""
    final_counts = defaultdict(int)

    for window in range(num_windows):
        current_interval = (window * addnum, window * addnum + interval)
        flow_counts = process_flow_data(csv_path, current_interval, threshold)
        for filename, count in flow_counts.items():
            final_counts[filename] += count

    with open(out_path, 'w') as output_file:
        for name in get_keys_for_value(final_counts, num_windows):
            output_file.write(f'{name}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=False, default='CrawlE_Proc_20000/')
    parser.add_argument('--output_path', required=False, default='original_20000_alt.txt')
    parser.add_argument('--threshold', type=int, required=False, default=1)

    args = parser.parse_args()

    analyze_overlapping_windows(args.data_path, args.output_path, args.threshold, 5, 11, 2)
