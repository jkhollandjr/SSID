import os
import glob
import matplotlib.pyplot as plt

dir1 = 'inflows'

# Load files from the first directory
file_paths = glob.glob(os.path.join(dir1, '*'))

# Extract trace sizes in terms of the number of packets and time
trace_sizes_packets = []
trace_sizes_time = []

for file_path in file_paths:
    with open(file_path, 'r') as f:
        lines = f.readlines()
        packet_count = len(lines)
        trace_sizes_packets.append(packet_count)

        last_packet_time = float(lines[-1].strip().split('\t')[0])
        trace_sizes_time.append(last_packet_time)

# Create histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Trace Sizes Distribution')

# Trace size distribution by number of packets
ax1.hist(trace_sizes_packets, bins='auto', edgecolor='black')
ax1.set_title('Trace Sizes by Number of Packets')
ax1.set_xlabel('Number of Packets')
ax1.set_ylabel('Frequency')

# Trace size distribution by time
ax2.hist(trace_sizes_time, bins='auto', range=(0,60),edgecolor='black')
ax2.set_title('Trace Sizes by Time')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Frequency')

# Adjust layout and display plots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('length_distribution.png')

