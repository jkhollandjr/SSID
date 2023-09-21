import os
import glob
import matplotlib.pyplot as plt

dir1 = 'inflow'

# Load files from the first directory
file_paths = glob.glob(os.path.join(dir1, '*'))

# Extract packet times
packet_times = []

for file_path in file_paths:
    with open(file_path, 'r') as f:
        for line in f:
            time, _ = map(float, line.strip().split('\t'))
            if time <= 60:
                packet_times.append(time)

# Create a histogram of packet distribution
plt.hist(packet_times, bins=60, edgecolor='black')
plt.title('Packet Distribution in the First 30 Seconds')
plt.xlabel('Time (seconds)')
plt.ylabel('Number of Packets')
plt.show()
plt.savefig('packet_distribution.png')
