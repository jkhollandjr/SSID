import pickle
import os
import argparse
import numpy as np


def convert_to_bursts(sequence):
    """
    Convert a packet sequence into a burst sequence.

    Parameters:
    sequence (list): List of packets in the format (time, size, direction)

    Returns:
    bursts (list): List of bursts in the format (start_time, size, direction)
    """
    bursts = [[0.0, 0, sequence[0][2]]]
    for packet in sequence:
        if packet[2] == bursts[-1][2]:
            bursts[-1][1] += packet[1]
        else:
            bursts.append([packet[0], packet[1], packet[2]])
    return bursts


def parse_csv(csv_path, interval, file_names):
    """
    Open a csv file and read/store trace.

    Parameters:
    csv_path (str): Path to the directory containing the csv files
    interval (tuple): Time interval to consider for each file in the format (start_time, end_time)
    file_names (list): List of csv file names to read

    Returns:
    here (list): List of incoming bursts for each file
    there (list): List of outgoing bursts for each file
    final_names (list): List of file names with non-empty bursts
    """
    here_path = os.path.join(csv_path, 'inflow')
    there_path = os.path.join(csv_path, 'outflow')
    print(here_path, there_path, interval)

    here = []
    there = []
    final_names = []

    for file_name in file_names:
        here_seq = []
        there_seq = []

        with open(os.path.join(here_path, file_name)) as f:
            pre_h_time = 0.0
            lines = f.readlines()
            if len(lines) == 0:
                continue
            big_pkt = []
            num_here_big_pkt = 0
            for line in lines:
                time_size = []
                time = float(line.split('\t')[0])
                size = int(float(line.split('\t')[1].strip()))
                if size > 0:
                    ipd = time - pre_h_time
                else:
                    ipd = -(time - pre_h_time)
                if float(time) > interval[1]:
                    break
                if float(time) < interval[0]:
                    continue
                if abs(size) > 1:  # ignore ack packets
                    if pre_h_time != 0 and ipd == 0:
                        big_pkt.append(size)
                        continue
                    if len(big_pkt) != 0:
                        last_pkt = here_seq.pop()
                        here_seq.append({"ipd": last_pkt["ipd"], "size": sum(big_pkt)+big_pkt[0]})
                        big_pkt = []
                        num_here_big_pkt += 1
                    time_size.append(ipd)
                    time_size.append(size)
                    time_size = np.array(time_size)
                    here_seq.append({"ipd": time_size[0], "size": time_size[1]})
                    pre_h_time = time

        with open(os.path.join(there_path, file_name)) as f:
            pre_h_time = 0.0
            lines = f.readlines()
            if len(lines) == 0:
                continue
            big_pkt = []
            num_there_big_pkt = 0
            for line in lines:
                time_size = []
                time = float(line.split('\t')[0])
                size = int(float(line.split('\t')[1].strip()))
                if size > 0:
                    ipd = time
                else:
                    ipd = -(time - pre_h_time)
                if float(time) > interval[1]:
                    break
                if float(time) < interval[0]:
                    continue
                if abs(size) > 1:  # ignore ack packets
                    if pre_h_time != 0 and ipd == 0:
                        big_pkt.append(size)
                        continue
                    if len(big_pkt) != 0:
                        last_pkt = there_seq.pop()
                        there_seq.append({"ipd": last_pkt["ipd"], "size": sum(big_pkt) + big_pkt[0]})
                        big_pkt = []
                        num_there_big_pkt += 1

                    time_size.append(ipd)
                    time_size.append(size)
                    time_size = np.array(time_size)
                    there_seq.append({"ipd": time_size[0], "size": time_size[1]})
                    pre_h_time = time

        if len(here_seq) > 0 and len(there_seq) > 0:
            here.append(here_seq)
            there.append(there_seq)
            final_names.append(file_name)

    here_len = [len(seq) for seq in here]
    there_len = [len(seq) for seq in there]
    num_here_big_pkt_cnt = [sum(1 for pkt in seq if pkt['size'] > 1) for seq in here]
    num_there_big_pkt_cnt = [sum(1 for pkt in seq if pkt['size'] > 1) for seq in there]
    flow_cnt = len(here)

    print(interval, 'mean', np.mean(np.array(here_len)), np.mean(np.array(there_len)), np.mean(num_here_big_pkt_cnt), np.mean(num_there_big_pkt_cnt), flow_cnt)
    print(interval, 'median', np.median(np.array(here_len)), np.median(np.array(there_len)), np.median(num_here_big_pkt_cnt), np.median(num_there_big_pkt_cnt), flow_cnt)

    return here, there, final_names


def create_overlap_window_csv(csv_path, file_list, prefix_pickle_output, interval, num_windows, addnum):
    """
    Write pickle files for each window.

    Parameters:
    csv_path (str): Path to the directory containing the csv files
    file_list (str): Path to the file containing the list of csv file names
    prefix_pickle_output (str): Prefix for the output pickle file names
    interval (int): Time interval to consider for each window
    num_windows (int): Number of windows to create
    addnum (int): Time interval between the start times of consecutive windows
    """
    file_names = [line.strip() for line in open(file_list, 'r').readlines()]
    windows_seq = []

    for win in range(num_windows):
        here, there, labels = parse_csv(csv_path, [win*addnum, win*addnum+interval], file_names)
        windows_seq.append({"tor": here, "exit": there, "label": labels})

        with open(f"{prefix_pickle_output}{interval}_win{win}_addn{addnum}_w_superpkt.pickle", 'wb') as handle:
            pickle.dump({'tor': here, 'exit': there, "label": labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=False, default='CrawlE_Proc_20000/')
    parser.add_argument('--file_list_path', required=False, default='original_20000_alt.txt')
    parser.add_argument('--prefix_pickle_output', required=False, default='original_cutoff/')
    parser.add_argument('--interval', required=False, type=int, default=5)
    parser.add_argument('--num_windows', required=False, type=int, default=11)
    parser.add_argument('--addnum', required=False, type=int, default=2)

    args = parser.parse_args()

    create_overlap_window_csv(args.data_path, args.file_list_path, args.prefix_pickle_output, args.interval, args.num_windows, args.addnum)

