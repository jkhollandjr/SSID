import numpy as np
import torch
from scipy.spatial.distance import cosine, euclidean
import torch.nn.functional as F
from orig_model import DFModel, DFModelWithAttention
from sklearn.model_selection import train_test_split
from traffic_utils import insert_dummy_packets_torch, calculate_inter_packet_times, calculate_times_with_directions, calculate_cumulative_traffic_torch, insert_dummy_packets_torch_exponential

# Function to apply transformations and dummy packet insertion
def transform_and_defend_features(features):
    # Initialize lists to store transformed and defended tensors
    defended_sizes = []
    defended_times = []
    defended_directions = []
    
    # Loop through each sample in the features tensor
    for i in range(features.shape[0]):
        sizes, times, directions = features[i, 0, :], features[i, 1, :], features[i, 2, :]
        # Apply dummy packet insertion
        defended_sizes_i, defended_times_i, defended_directions_i = insert_dummy_packets_torch_exponential(sizes, times, directions, num_dummy_packets=0)
        
        defended_sizes.append(defended_sizes_i.unsqueeze(0))
        defended_times.append(defended_times_i.unsqueeze(0))
        defended_directions.append(defended_directions_i.unsqueeze(0))
    
    # Stack defended features back into tensors
    defended_sizes = torch.cat(defended_sizes, dim=0)
    defended_times = torch.cat(defended_times, dim=0)
    defended_directions = torch.cat(defended_directions, dim=0)

    # Calculate additional features based on defended traffic
    inter_packet_times = calculate_inter_packet_times(defended_times)
    times_with_directions = calculate_times_with_directions(defended_times, defended_directions)
    cumul = calculate_cumulative_traffic_torch(defended_sizes, defended_times)
    
    # Stack all features together
    transformed_features = torch.stack([defended_sizes, inter_packet_times, times_with_directions, cumul], dim=1)
    return transformed_features

# Instantiate the models
embedding_size = 64
inflow_model = DFModel()
outflow_model = DFModel()

# Load the best models
#checkpoint = torch.load('models/best_model_dcf_defened_0.00806727527074893.pth')
checkpoint = torch.load('models/best_model_live_detorrent_stride.pth')
inflow_model.load_state_dict(checkpoint['inflow_model_state_dict'])
outflow_model.load_state_dict(checkpoint['outflow_model_state_dict'])

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inflow_model.to(device)
outflow_model.to(device)

# Evaluation mode
inflow_model.eval()
outflow_model.eval()

# Load the numpy arrays
val_inflows = np.load('data/val_inflows_detorrent.npy')[:1000]
val_outflows = np.load('data/val_outflows_detorrent.npy')[:1000]

# Split the data
val_inflows, test_inflows, val_outflows, test_outflows = train_test_split(val_inflows, val_outflows, test_size=0.5, random_state=42)

np.save('data/test_inflows_detorrent.npy', test_inflows)
np.save('data/test_outflows_detorrent.npy', test_outflows)
# Initialize the outputs
val_output_array = np.zeros((len(val_inflows) * len(val_outflows), 13))
test_output_array = np.zeros((len(test_inflows) * len(test_outflows), 13))

def compute_batch_distances(inflow_traces, outflow_traces, inflow_model, outflow_model):
    all_cosine_similarities = []

    for window_idx in range(inflow_traces.shape[1]):
        inflow_window = inflow_traces[:, window_idx, :, :] 
        outflow_window = outflow_traces[:, window_idx, :, :]

        inflow_window = torch.from_numpy(inflow_window).float().to(device)
        outflow_window = torch.from_numpy(outflow_window).float().to(device)

        inflow_window = transform_and_defend_features(inflow_window)
        outflow_window = transform_and_defend_features(outflow_window)

        #inflow_window = inflow_window.reshape(inflow_window.shape[0], -1, inflow_window.shape[-1]) 
        #outflow_window = outflow_window.reshape(outflow_window.shape[0], -1, outflow_window.shape[-1])
        # 64, 4, 500

        inflow_embeddings = inflow_model(inflow_window) 
        outflow_embeddings = outflow_model(outflow_window)

        cosine_similarities = F.cosine_similarity(inflow_embeddings, outflow_embeddings).detach().cpu().numpy() #64,
        all_cosine_similarities.append(cosine_similarities)

    return np.stack(all_cosine_similarities, axis=1)

batch_size = 64

import random
'''
def process_data(inflows, outflows, output_array):
    num_inflows = len(inflows)
    num_outflows = len(outflows)
    batch_size = 256

    # Allocate an empty array for the results
    # Size: num_inflows * batch_size, 15+1 (for 15 windows and 1 match column)
    output_array = np.zeros((num_inflows * batch_size, 15))

    for idx, inflow_example in enumerate(inflows):
        # Randomly select 64 outflow examples
        selected_outflow_indices = random.sample(range(num_outflows), batch_size)
        selected_outflows = outflows[selected_outflow_indices]

        # Reshape inflow_example to have the same number of dimensions as batched data
        inflow_batch = np.repeat(inflow_example[np.newaxis, ...], batch_size, axis=0)

        distances = compute_batch_distances(inflow_batch, selected_outflows, inflow_model, outflow_model)

        for b in range(batch_size):
            output_idx = idx * batch_size + b
            output_array[output_idx, :-3] = distances[b]

            # Set match to 1 if inflow and randomly selected outflow have the same index
            output_array[output_idx, -1] = int(idx == selected_outflow_indices[b])

            output_array[output_idx, -2] = selected_outflow_indices[b]
            output_array[output_idx, -3] = idx

    return output_array
'''

def find_closest(packet_time, other_flow_times):
    """ Find the closest time in other_flow_times to packet_time. """
    if other_flow_times.size == 0:
        return 0
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

def calculate_time_differences(trace1, trace2, sizes_trace1, sizes_trace2, max_length=40):
    """ Calculate time differences for the first 87 (or fewer) packets in flow1 against the closest in flow2. """

    sizes_trace1 = np.abs(sizes_trace1)
    sizes_trace2 = np.abs(sizes_trace2)

    size_threshold = 80
    trace1 = np.array(trace1)
    trace2 = np.array(trace2)
    sizes_trace1 = np.array(sizes_trace1)
    sizes_trace2 = np.array(sizes_trace2)

    # Filter out downloads (negative values) and apply size filtering
    mask1 = (trace1 < 0) & (sizes_trace1 >= size_threshold)
    flow1 = -trace1[mask1]
    mask2 = (trace2 < 0) & (sizes_trace2 >= size_threshold)
    flow2 = -trace2[mask2]

    flow1 = np.sort(flow1)  # Ensure the array is sorted
    flow2 = np.sort(flow2)  # Ensure the array is sorted
    
    time_diffs = []
    
    for time in flow1[:max_length]:
        closest_time = find_closest(time, flow2)
        time_diffs.append(abs(time - closest_time))
    
    # Zero-pad the array if there are fewer than max_length packets in flow1
    if len(time_diffs) < max_length:
        time_diffs.extend([0] * (max_length - len(time_diffs)))
    
    return np.array(time_diffs)

def calculate_proportions(trace1, trace2, sizes_trace1, sizes_trace2, thresholds=[0.001, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]):

    sizes_trace1 = np.abs(sizes_trace1)
    sizes_trace2 = np.abs(sizes_trace2)

    size_threshold = 80
    trace1 = np.array(trace1)
    trace2 = np.array(trace2)
    sizes_trace1 = np.array(sizes_trace1)
    sizes_trace2 = np.array(sizes_trace2)

    # Filter out downloads (negative values) and apply size filtering
    mask1 = (trace1 < 0) & (sizes_trace1 >= size_threshold)
    flow1 = -trace1[mask1]
    mask2 = (trace2 < 0) & (sizes_trace2 >= size_threshold)
    flow2 = -trace2[mask2]

    flow1 = np.sort(flow1)  # Ensure the array is sorted
    flow2 = np.sort(flow2)  # Ensure the array is sorted

    """ Calculate proportions of packet time differences falling below specified thresholds. """
    if len(flow2) == 0:  # Check if flow2 is empty
        return np.zeros(len(thresholds))  # Return zero array if no data to compare
    
    flow1 = np.sort(flow1)  # Ensure the array is sorted
    flow2 = np.sort(flow2)  # Ensure the array is sorted

    time_diffs = []
    for time in flow1:
        closest_time = find_closest(time, flow2)
        if closest_time is not None:
            time_diffs.append(abs(time - closest_time))
        else:
            time_diffs.append(float(1))  # Use infinity where no comparison is possible

    if np.count_nonzero(flow1) == 0:
        return np.array([0]*8)
    # Compute proportions for each threshold
    proportions = []
    for threshold in thresholds:
        count = np.sum(np.array(time_diffs) < threshold)
        proportion = count / np.count_nonzero(flow1)
        proportions.append(proportion)

    return np.array(proportions)

def process_data(inflows, outflows, output_array):
    num_inflows = len(inflows)
    num_outflows = len(outflows)
    batch_size = 256

    # Allocate an empty array for the results
    # Size: num_inflows * batch_size, 12+1 (for 12 windows and 1 match column)
    output_array = np.zeros((num_inflows * batch_size, 29))

    for idx, inflow_example in enumerate(inflows):
        # Randomly select 64 outflow examples
        selected_outflow_indices = random.sample(range(num_outflows), batch_size)
        selected_outflows = outflows[selected_outflow_indices]

        # Reshape inflow_example to have the same number of dimensions as batched data
        inflow_batch = np.repeat(inflow_example[np.newaxis, ...], batch_size, axis=0)

        distances = compute_batch_distances(inflow_batch, selected_outflows, inflow_model, outflow_model)

        for b in range(batch_size):
            output_idx = idx * batch_size + b
            output_array[output_idx, :12] = distances[b]

            # Set match to 1 if inflow and randomly selected outflow have the same index
            output_array[output_idx, -1] = int(idx == selected_outflow_indices[b])

            outflow_index = selected_outflow_indices[b]
            inflow_index = idx

            inflow_time = inflows[inflow_index, -1, 1, :]
            inflow_dir = inflows[inflow_index, -1, 2, :]
            inflow_sizes = inflows[inflow_index, -1, 0, :]

            outflow_time = outflows[outflow_index, -1, 1, :]
            outflow_dir = outflows[outflow_index, -1, 2, :]
            outflow_sizes = outflows[outflow_index, -1, 0, :]

            # get first 87 distances between packets, if possible
            download_time_diff = calculate_proportions(outflow_time*outflow_dir, inflow_time*inflow_dir, outflow_sizes, inflow_sizes)

            upload_time_diff = calculate_proportions(outflow_time*outflow_dir*-1, inflow_time*inflow_dir*-1, outflow_sizes, inflow_sizes)

            output_array[output_idx, 12:20] = download_time_diff
            output_array[output_idx, 20:28] = upload_time_diff

    return output_array


# Process and save the results
val_output_array = process_data(val_inflows, val_outflows, val_output_array)
np.save('data/dcf_val_distances_live.npy', val_output_array)

test_output_array = process_data(test_inflows, test_outflows, test_output_array)
np.save('data/dcf_test_distances_live.npy', test_output_array)

print(val_output_array.shape)
