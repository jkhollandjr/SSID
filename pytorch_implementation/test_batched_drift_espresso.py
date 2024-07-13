import numpy as np
import torch
from scipy.spatial.distance import cosine, euclidean
import torch.nn.functional as F
from orig_model import DFModel, DFModelWithAttention
from sklearn.model_selection import train_test_split
from traffic_utils import insert_dummy_packets_torch, calculate_inter_packet_times, calculate_times_with_directions, calculate_cumulative_traffic_torch, insert_dummy_packets_torch_exponential
from espresso import EspressoNet


model_config = {
        'input_size': 1000,
        'feature_dim': 64,
        'hidden_dim': 128,
        'depth': 12,
        'input_conv_kwargs': {
            'kernel_size': 3,
            'stride': 3,
            'padding': 0,
            },
        'output_conv_kwargs': {
            'kernel_size': 60,
            #'stride': 40,
            'stride': 3,
            'padding': 0,
            },
        "mhsa_kwargs": {
            "head_dim": 16,
            "use_conv_proj": True,
            "kernel_size": 3,
            "stride": 2,
            "feedforward_style": "mlp",
            "feedforward_ratio": 4,
            "feedforward_drop": 0.0
        },
        "features": [
            "interval_dirs_up",
            "interval_dirs_down",
            "interval_dirs_sum",
            "interval_dirs_sub",
            "interval_iats",
            "interval_inv_iat_logs",
            "interval_cumul_norm",
            "interval_times_norm",
            ],
        "window_kwargs": {
            'window_count': 1,
            'window_width': 0,
            'window_overlap': 0,
            'include_all_window': True,
        },
}

# Instantiate the models
embedding_size = 64
inflow_model = EspressoNet(8, special_toks=1, **model_config)
outflow_model = EspressoNet(8, special_toks=1, **model_config)

# Load the best models
#checkpoint = torch.load('models/best_model_dcf_defened_0.00806727527074893.pth')
checkpoint = torch.load('models/best_model_live_espresso_may17_fixed.pth')
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
val_inflows = np.load('data/val_inflows_may17_transformer.npy')[:1000]
val_outflows = np.load('data/val_outflows_may17_transformer.npy')[:1000]

# Split the data
val_inflows, test_inflows, val_outflows, test_outflows = train_test_split(val_inflows, val_outflows, test_size=0.5, random_state=42)

np.save('data/test_inflows.npy', test_inflows)
np.save('data/test_outflows.npy', test_outflows)
# Initialize the outputs
val_output_array = np.zeros((len(val_inflows) * len(val_outflows), 92))
test_output_array = np.zeros((len(test_inflows) * len(test_outflows), 92))

def compute_batch_distances(inflow_traces, outflow_traces, inflow_model, outflow_model):
    
    all_cosine_similarities = []

    inflow_window = inflow_traces[:, :, :] 
    outflow_window = outflow_traces[:, :, :]

    inflow_window = torch.from_numpy(inflow_window).float()
    outflow_window = torch.from_numpy(outflow_window).float()

    inflow_embeddings, _ = outflow_model(inflow_window.to(device)[:,3:,:])
    outflow_embeddings, _ = outflow_model(outflow_window.to(device)[:,3:,:])

    #print(inflow_embeddings.shape)
    #print(_.shape)
    inflow_embeddings = inflow_embeddings.reshape(256, 92, 64)
    outflow_embeddings = outflow_embeddings.reshape(256, 92, 64)

    for i in range(92):
        inflow_window_embedding = inflow_embeddings[:,i,:]
        outflow_window_embedding = outflow_embeddings[:,i,:]

        cosine_similarities = F.cosine_similarity(inflow_window_embedding, outflow_window_embedding).detach().cpu().numpy()
        all_cosine_similarities.append(cosine_similarities)

    #cosine_similarities = F.cosine_similarity(inflow_embeddings, outflow_embeddings).detach().cpu().numpy() #64,
    #all_cosine_similarities.append(cosine_similarities)

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

def calculate_proportions(trace1, trace2, sizes_trace1, sizes_trace2, thresholds=[0.0001, 0.001, 0.01, 0.03, 0.08, 0.16, 0.32, 0.64]):

    sizes_trace1 = np.abs(sizes_trace1)
    sizes_trace2 = np.abs(sizes_trace2)

    size_threshold = 50
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
    output_array = np.zeros((num_inflows * batch_size, 109))

    for idx, inflow_example in enumerate(inflows):
        # Randomly select 64 outflow examples
        selected_outflow_indices = random.sample(range(num_outflows), batch_size)
        selected_outflows = outflows[selected_outflow_indices]

        # Reshape inflow_example to have the same number of dimensions as batched data
        inflow_batch = np.repeat(inflow_example[np.newaxis, ...], batch_size, axis=0)

        distances = compute_batch_distances(inflow_batch, selected_outflows, inflow_model, outflow_model)

        for b in range(batch_size):
            output_idx = idx * batch_size + b
            output_array[output_idx, :92] = distances[b]

            # Set match to 1 if inflow and randomly selected outflow have the same index
            output_array[output_idx, -1] = int(idx == selected_outflow_indices[b])

            outflow_index = selected_outflow_indices[b]
            inflow_index = idx

            inflow_time = inflows[inflow_index, 1, :]
            inflow_dir = inflows[inflow_index, 2, :]
            inflow_sizes = inflows[inflow_index, 0, :]

            outflow_time = outflows[outflow_index, 1, :]
            outflow_dir = outflows[outflow_index, 2, :]
            outflow_sizes = outflows[outflow_index, 0, :]

            # get first 87 distances between packets, if possible
            download_time_diff = calculate_proportions(outflow_time*outflow_dir, inflow_time*inflow_dir, outflow_sizes, inflow_sizes)

            upload_time_diff = calculate_proportions(outflow_time*outflow_dir*-1, inflow_time*inflow_dir*-1, outflow_sizes, inflow_sizes)

            '''
            if(inflow_index == outflow_index):
                print(outflow_index)
                print(inflow_index)
                print(inflow_time)
                print(outflow_time)
                print(download_time_diff)
                print(upload_time_diff)
                print("")
                exit()
            '''

            output_array[output_idx, 92:100] = download_time_diff
            output_array[output_idx, 100:108] = upload_time_diff

    return output_array


# Process and save the results
val_output_array = process_data(val_inflows, val_outflows, val_output_array)
np.save('data/dcf_val_distances_espresso_drift.npy', val_output_array)

test_output_array = process_data(test_inflows, test_outflows, test_output_array)
np.save('data/dcf_test_distances_espresso_drift.npy', test_output_array)

print(val_output_array.shape)
