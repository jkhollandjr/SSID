import os
import glob
import matplotlib.pyplot as plt

dir1 = 'may_inflow'

# Load files from the first directory
file_paths = glob.glob(os.path.join(dir1, '*'))

# Extract trace sizes in terms of the number of packets
trace_sizes_packets = []

for file_path in file_paths:
    with open(file_path, 'r') as f:
        packet_count = sum(1 for line in f)
        trace_sizes_packets.append(packet_count)

# Create histogram
plt.hist(trace_sizes_packets, bins='auto', edgecolor='black')
plt.title('Trace Sizes by Number of Packets')
plt.xlabel('Number of Packets')
plt.ylabel('Frequency')
plt.savefig('length.png')
plt.show()

