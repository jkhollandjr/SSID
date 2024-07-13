import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

inflows = np.load("data/val_inflows_may17_transformer.npy")
outflows = np.load("data/val_outflows_may17_transformer.npy")

index = 999

inflow_time = inflows[index, -1, 1, :]
inflow_dir = inflows[index, -1, 2, :]
inflow_sizes = inflows[index, -1, 0, :]

inflows = inflow_time * inflow_dir * -1

index = 999
outflow_time = outflows[index, -1, 1, :]
outflow_dir = outflows[index, -1, 2, :]
outflow_sizes = outflows[index, -1, 0, :]

outflows = outflow_time * outflow_dir * -1

def plot_packet_timings(packet_timings):
    packet_timings = np.array(packet_timings)

    # Split uploads and downloads based on sign
    uploads = packet_timings[packet_timings > 0]
    downloads = -packet_timings[packet_timings < 0]  # Make downloads positive for display

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot narrow vertical lines for each packet timing
    for time in uploads:
        ax.axvline(x=time, ymin=0.5, ymax=1, color='blue', linewidth=1, alpha=0.5)  # Uploads

    for time in downloads:
        ax.axvline(x=time, ymin=0, ymax=0.5, color='red', linewidth=1, alpha=0.5)  # Downloads

    # Setting the axis labels and title
    ax.set_xlabel('Time (s)')
    ax.set_yticks([])
    ax.set_title('Packet Timings (Uploads and Downloads)')

    # Create custom legend manually
    ax.plot([], [], color='blue', label='Uploads', linewidth=10, alpha=0.5)
    ax.plot([], [], color='red', label='Downloads', linewidth=10, alpha=0.5)
    ax.legend()

    # Show grid for better visual guidance
    ax.grid(True)

    # Display the plot
    plt.show()


def plot_packet_timings(packet_timings):
    packet_timings = np.array(packet_timings)

    # Split uploads and downloads based on sign
    uploads = packet_timings[packet_timings > 0]
    downloads = -packet_timings[packet_timings < 0]  # Make downloads positive for display

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Plot narrow vertical lines for each packet timing on the first subplot
    for time in uploads:
        axs[0].axvline(x=time, ymin=0.5, ymax=1, color='blue', linewidth=1, alpha=0.5)  # Uploads

    for time in downloads:
        axs[0].axvline(x=time, ymin=0, ymax=0.5, color='red', linewidth=1, alpha=0.5)  # Downloads

    # Setting the axis labels and title for the first plot
    axs[0].set_xlabel('Time (s)')
    axs[0].set_yticks([])
    axs[0].set_title('Packet Timings (Uploads and Downloads)')

    # Custom legend
    axs[0].plot([], [], color='blue', label='Uploads', linewidth=10, alpha=0.5)
    axs[0].plot([], [], color='red', label='Downloads', linewidth=10, alpha=0.5)
    axs[0].legend()

    # Add grid
    axs[0].grid(True)

    # Histogram for the binned packet volumes on the second subplot
    # Define bin edges as 50ms intervals
    min_time = 0
    max_time = 30
    bins = np.arange(min_time, max_time, 0.1)  # 50 ms bins

    # Plot histograms for uploads and downloads
    axs[1].hist(uploads, bins=bins, color='blue', alpha=0.5, label='Uploads', linewidth=5)
    axs[1].hist(downloads, bins=bins, color='red', alpha=0.5, label='Downloads', linewidth=5)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Packet Count')
    axs[1].legend()

    # Grid for the second plot
    axs[1].grid(True)

    # Adjust layout to prevent overlap and ensure clarity
    plt.tight_layout()
    plt.show()

def plot_download_comparison(trace1, trace2):
    trace1 = np.array(trace1)
    trace2 = np.array(trace2)

    # Filter out downloads (negative values)
    downloads_trace1 = -trace1[trace1 < 0]
    downloads_trace2 = -trace2[trace2 < 0]

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Plot narrow vertical lines for each download timing
    for time in downloads_trace1:
        axs[0].axvline(x=time, ymin=0.5, ymax=1, color='blue', linewidth=1, alpha=0.5, label='Trace 1' if time == downloads_trace1[0] else "")  # Downloads from Trace 1
    for time in downloads_trace2:
        axs[0].axvline(x=time, ymin=0, ymax=0.5, color='red', linewidth=1, alpha=0.5, label='Trace 2' if time == downloads_trace2[0] else "")  # Downloads from Trace 2

    # Setting the axis labels and title for the first plot
    axs[0].set_xlabel('Time (s)')
    axs[0].set_yticks([])
    axs[0].set_title('Download Comparison Between Two Traces')
    axs[0].legend()

    # Add grid
    axs[0].grid(True)

    # Histogram for the binned download volumes on the second subplot
    # Define bin edges as 50ms intervals
    min_time = 0
    max_time = 30
    bins = np.arange(min_time, max_time + 0.05, 0.05)  # Ensure at least one bin

    # Plot histograms for downloads from both traces
    axs[1].hist(downloads_trace1, bins=bins, color='blue', alpha=0.5, label='Trace 1', linewidth=2)
    axs[1].hist(downloads_trace2, bins=bins, color='red', alpha=0.5, label='Trace 2', linewidth=2)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Download Packet Count')
    axs[1].legend()

    # Grid for the second plot
    axs[1].grid(True)

    # Adjust layout to prevent overlap and ensure clarity
    plt.tight_layout()
    plt.show()

def plot_download_comparison(trace1, trace2, sizes_trace1, sizes_trace2, size_threshold=500):
    trace1 = np.array(trace1)
    trace2 = np.array(trace2)
    sizes_trace1 = np.array(sizes_trace1)
    sizes_trace2 = np.array(sizes_trace2)

    # Filter out downloads (negative values) and apply size filtering
    mask1 = (trace1 < 0) & (sizes_trace1 >= size_threshold)
    downloads_trace1 = -trace1[mask1]
    mask2 = (trace2 < 0) & (sizes_trace2 >= size_threshold)
    downloads_trace2 = -trace2[mask2]

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Plot narrow vertical lines for each download timing
    for time in downloads_trace1:
        axs[0].axvline(x=time, ymin=0.5, ymax=1, color='blue', linewidth=1, alpha=0.5, label='Trace 1' if time == downloads_trace1[0] else "")  # Downloads from Trace 1
    for time in downloads_trace2:
        axs[0].axvline(x=time, ymin=0, ymax=0.5, color='red', linewidth=1, alpha=0.5, label='Trace 2' if time == downloads_trace2[0] else "")  # Downloads from Trace 2

    # Setting the axis labels and title for the first plot
    axs[0].set_xlabel('Time (s)')
    axs[0].set_yticks([])
    axs[0].set_title('Download Comparison Between Two Traces (Size Threshold: {} Bytes)'.format(size_threshold))
    axs[0].legend()

    # Add grid
    axs[0].grid(True)

    # Histogram for the binned download volumes on the second subplot
    # Define bin edges as 50ms intervals
    min_time = 0
    max_time = 30
    bins = np.arange(min_time, max_time + 0.05, 0.05)  # Ensure at least one bin

    # Plot histograms for downloads from both traces
    axs[1].hist(downloads_trace1, bins=bins, color='blue', alpha=0.5, label='Trace 1', linewidth=2)
    axs[1].hist(downloads_trace2, bins=bins, color='red', alpha=0.5, label='Trace 2', linewidth=2)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Download Packet Count')
    axs[1].legend()

    # Grid for the second plot
    axs[1].grid(True)

    # Adjust layout to prevent overlap and ensure clarity
    plt.tight_layout()
    plt.show()

plot_packet_timings(inflows)
plot_packet_timings(outflows)
plot_download_comparison(inflows, outflows, inflow_sizes, outflow_sizes, 100)

def find_closest(packet_time, other_flow_times):
    """ Find the closest time in other_flow_times to packet_time. """
    idx = np.searchsorted(other_flow_times, packet_time)
    # Handle edge cases where searchsorted returns an index outside of valid range
    if idx == len(other_flow_times):
        return other_flow_times[-1]
    elif idx == 0:
        return other_flow_times[0]
    else:
        # Check the closest of the neighboring elements
        before = other_flow_times[idx - 1]
        after = other_flow_times[idx]
        if abs(packet_time - before) < abs(packet_time - after):
            return before
        else:
            return after

def calculate_correlation(flow1, flow2, threshold=0.025):
    """ Determine if two flows are correlated based on the closeness of packet timings. """
    flow1 = np.sort(flow1)  # Ensure the array is sorted
    flow2 = np.sort(flow2)  # Ensure the array is sorted
    
    close_count = 0
    
    for time in flow1:
        closest_time = find_closest(time, flow2)
        if abs(time - closest_time) <= threshold:
            close_count += 1
    
    # Optionally calculate expected close_count by chance and perform statistical test
    # Here we just return the count and leave statistical analysis for further study
    return close_count

def whether_correlated(trace1, trace2, sizes_trace1, sizes_trace2):
    sizes_trace1 = np.abs(sizes_trace1)
    sizes_trace2 = np.abs(sizes_trace2)

    size_threshold = 80
    trace1 = np.array(trace1)
    trace2 = np.array(trace2)
    sizes_trace1 = np.array(sizes_trace1)
    sizes_trace2 = np.array(sizes_trace2)

    # Filter out downloads (negative values) and apply size filtering
    mask1 = (trace1 < 0) & (sizes_trace1 >= size_threshold)
    downloads_trace1 = -trace1[mask1]
    mask2 = (trace2 < 0) & (sizes_trace2 >= size_threshold)
    downloads_trace2 = -trace2[mask2]

    trace2_packets = np.count_nonzero(downloads_trace2)
    trace1_packets = np.count_nonzero(downloads_trace1)
    if trace2_packets == 0 or trace1_packets == 0:
        return (0.0, False)
    avg_distance = (np.max(downloads_trace2) / trace2_packets) / 2
    threshold = avg_distance / 5
   
    count = calculate_correlation(downloads_trace1, downloads_trace2, threshold=threshold)
    proportion = count / trace1_packets

    if ((trace1_packets / trace2_packets) < 5) and (trace2_packets / trace1_packets) < 5 :
        qualified = True
    else:
        qualified = False

    return (proportion, qualified)
    
proportion = whether_correlated(outflows, inflows, outflow_sizes, inflow_sizes)
inflows = np.load("data/val_inflows_cumul.npy")
outflows = np.load("data/val_outflows_cumul.npy")

for i in range(100):
    for j in range(100):
        inflow_time = inflows[i, -1, 1, :]
        inflow_dir = inflows[i, -1, 2, :]
        inflow_sizes = inflows[i, -1, 0, :]

        outflow_time = outflows[j, -1, 1, :]
        outflow_dir = outflows[j, -1, 2, :]
        outflow_sizes = outflows[j, -1, 0, :]

        proportion, qualified = whether_correlated(outflow_time*outflow_dir*-1, inflow_time*inflow_dir*-1, outflow_sizes, inflow_sizes)
        if i == j:
            print(proportion)
        '''
        if i != j and proportion > .8 and qualified:
            print(i)
            print(j)
            print("")
        '''

