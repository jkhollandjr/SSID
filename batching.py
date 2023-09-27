import os
from collections import deque
from multiprocessing import Pool, cpu_count

INPUT_DIR = '/home/james/Desktop/research/SSID/SSID_Capture/inflow/'
OUTPUT_DIR = '/home/james/Desktop/research/SSID/SSID_Capture/inflow_batched/'
MAX_DELAY = 1  # seconds
BUFFER_SIZE = 4

def flush_buffer(buffer, flush_time):
    total_latency = 0
    for i in range(len(buffer)):
        original_time, packet_size = buffer[i]
        latency = flush_time - original_time
        total_latency += latency
        buffer[i] = (flush_time, packet_size)

    dummy_packet_size = -1 if buffer and buffer[0][1] < 0 else 1
    while len(buffer) < BUFFER_SIZE:
        buffer.append((flush_time, dummy_packet_size))

    return buffer, total_latency

def process_file(filename):
    download_buffer = []
    upload_buffer = []
    output = []
    total_latency = 0

    with open(os.path.join(INPUT_DIR, filename), 'r') as f:
        packets = [line.strip().split('\t') for line in f]
        packets = [(float(ts), int(float(size))) for ts, size in packets]

    current_time = 0
    while packets or download_buffer or upload_buffer:
        next_packet_time = packets[0][0] if packets else float('inf')
        next_download_flush_time = download_buffer[0][0] + MAX_DELAY if download_buffer else float('inf')
        next_upload_flush_time = upload_buffer[0][0] + MAX_DELAY if upload_buffer else float('inf')

        next_event_time = min(next_packet_time, next_download_flush_time, next_upload_flush_time)

        while download_buffer and (len(download_buffer) == BUFFER_SIZE or next_event_time - download_buffer[0][0] >= MAX_DELAY):
            flushed, latency = flush_buffer(download_buffer, next_event_time)
            output.extend(flushed)
            total_latency += latency

        while upload_buffer and (len(upload_buffer) == BUFFER_SIZE or next_event_time - upload_buffer[0][0] >= MAX_DELAY):
            flushed, latency = flush_buffer(upload_buffer, next_event_time)
            output.extend(flushed)
            total_latency += latency

        if packets and next_event_time == packets[0][0]:
            timestamp, packet_size = packets.pop(0)
            if packet_size < 0:
                download_buffer.append((timestamp, packet_size))
            else:
                upload_buffer.append((timestamp, packet_size))

    with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
        for ts, size in output:
            f.write(f"{ts}\t{size}\n")

    avg_latency = total_latency / len(output) if output else 0

    return filename, (len(output), avg_latency)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    total_original_packets = 0
    total_obfuscated_packets = 0
    accumulated_avg_latency = 0

    filenames = os.listdir(INPUT_DIR)

    with Pool(cpu_count()) as pool:
        results = pool.map(process_file, filenames)

    for filename, (obfuscated_count, avg_latency) in results:
        total_original_packets += sum(1 for line in open(os.path.join(INPUT_DIR, filename)))
        total_obfuscated_packets += obfuscated_count
        accumulated_avg_latency += avg_latency

    dataset_avg_latency = accumulated_avg_latency / len(filenames) if filenames else 0
    bandwidth_overhead = ((total_obfuscated_packets - total_original_packets) / total_original_packets) * 100 if total_original_packets else 0

    print(f"Bandwidth Overhead: {bandwidth_overhead:.2f}%")
    print(f"Average latency for the dataset: {dataset_avg_latency:.2f} seconds")

if __name__ == "__main__":
    main()

