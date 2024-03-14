import numpy as np
import torch
from model import Embedder
from scipy.spatial.distance import cosine, euclidean
import torch.nn.functional as F
from orig_model import DFModel, DFModelWithAttention
from sklearn.model_selection import train_test_split
from transdfnet import DFNet

def insert_dummy_packets_torch(sizes, times, directions, num_dummy_packets=0):
    if num_dummy_packets == 0:
        return sizes, times, directions

    device = sizes.device
    original_length = 1000  # Target length

    num_dummy_packets = np.random.randint(0, 2*num_dummy_packets)

    # Assume packets with direction 0 are non-existent (padding)
    real_packet_mask = directions != 0

    # Filter out non-existent packets
    real_sizes = sizes[real_packet_mask]
    real_times = times[real_packet_mask]
    real_directions = directions[real_packet_mask]

    # Generate dummy packets
    dummy_directions = torch.randint(0, 2, (num_dummy_packets,), device=device) * 2 - 1  # Converts {0, 1} to {-1, 1}
    dummy_sizes = torch.where(dummy_directions == -1, torch.full((num_dummy_packets,), 1514, device=device), torch.full((num_dummy_packets,), 609, device=device))
    dummy_times = torch.linspace(0, 5, num_dummy_packets, device=device)

    # Combine filtered real packets with dummy packets
    combined_sizes = torch.cat((real_sizes, dummy_sizes))
    combined_times = torch.cat((real_times, dummy_times))
    combined_directions = torch.cat((real_directions, dummy_directions))

    # Sort the combined packets based on times
    sorted_indices = torch.argsort(combined_times)
    sorted_sizes = combined_sizes[sorted_indices]
    sorted_times = combined_times[sorted_indices]
    sorted_directions = combined_directions[sorted_indices]

    # Check current length and pad if necessary to reach 1000
    current_length = sorted_sizes.size(0)
    
    if current_length < original_length:
        # Calculate padding length
        padding_length = original_length - current_length

        # Create padding tensors with 0.0 values
        padded_sizes = torch.zeros(padding_length, device=device)
        padded_times = torch.zeros(padding_length, device=device)
        padded_directions = torch.zeros(padding_length, device=device)

        # Append padding tensors
        final_sizes = torch.cat((sorted_sizes, padded_sizes))
        final_times = torch.cat((sorted_times, padded_times))
        final_directions = torch.cat((sorted_directions, padded_directions))
    else:
        # If the current length meets or exceeds the target, trim to 1000
        final_sizes = sorted_sizes[:original_length]
        final_times = sorted_times[:original_length]
        final_directions = sorted_directions[:original_length]

    return final_sizes, final_times, final_directions

def calculate_inter_packet_times(times):
    # Ensure 'prepend' has the same dimensionality as 'times', except for the last dimension
    batch_size, seq_length = times.shape
    prepend_tensor = torch.zeros((batch_size, 1), device=times.device)  # Match the batch dimension, add a single column for prepend
    
    non_padded_diff = torch.diff(times, dim=1, prepend=prepend_tensor)
    # Since you're computing the difference along the last dimension (time sequence), no need to pad after diff
    
    return torch.abs(non_padded_diff)

def calculate_times_with_directions(times, directions):
    return times * directions

def calculate_cumulative_traffic(sizes, times):
    # Assuming 'sizes' and 'times' are PyTorch tensors
    # This method might need adjustments based on the exact representation of 'times'
    cumulative_traffic = torch.cumsum(sizes, dim=0)
    return cumulative_traffic

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

        
        defended_sizes_i, defended_times_i, defended_directions_i = insert_dummy_packets_torch(sizes, times, directions)
        
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
    cumul = calculate_cumulative_traffic(defended_sizes, defended_times)
    
    # Stack all features together
    transformed_features = torch.stack([defended_sizes, inter_packet_times, times_with_directions, defended_directions, cumul], dim=1)
    return transformed_features

# Instantiate the models
embedding_size = 64
inflow_model = DFModel()
outflow_model = DFModel()

# Load the best models
#checkpoint = torch.load('models/best_model_dcf_defened_0.00806727527074893.pth')
checkpoint = torch.load('models/best_model_defended.pth')
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
val_inflows = np.load('val_inflows_defended.npy')[:1000]
val_outflows = np.load('val_outflows_defended.npy')[:1000]

# Split the data
val_inflows, test_inflows, val_outflows, test_outflows = train_test_split(val_inflows, val_outflows, test_size=0.5, random_state=42)

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

        inflow_embeddings = inflow_model(inflow_window[:,:4,:]) 
        outflow_embeddings = outflow_model(outflow_window[:,:4,:])

        cosine_similarities = F.cosine_similarity(inflow_embeddings, outflow_embeddings).detach().cpu().numpy() #64,
        all_cosine_similarities.append(cosine_similarities)

    return np.stack(all_cosine_similarities, axis=1)

batch_size = 64

import random
def process_data(inflows, outflows, output_array):
    num_inflows = len(inflows)
    num_outflows = len(outflows)
    batch_size = 256

    # Allocate an empty array for the results
    # Size: num_inflows * batch_size, 15+1 (for 15 windows and 1 match column)
    output_array = np.zeros((num_inflows * batch_size, 13))

    for idx, inflow_example in enumerate(inflows):
        # Randomly select 64 outflow examples
        selected_outflow_indices = random.sample(range(num_outflows), batch_size)
        selected_outflows = outflows[selected_outflow_indices]

        # Reshape inflow_example to have the same number of dimensions as batched data
        inflow_batch = np.repeat(inflow_example[np.newaxis, ...], batch_size, axis=0)

        distances = compute_batch_distances(inflow_batch, selected_outflows, inflow_model, outflow_model)

        for b in range(batch_size):
            output_idx = idx * batch_size + b
            output_array[output_idx, :-1] = distances[b]

            # Set match to 1 if inflow and randomly selected outflow have the same index
            output_array[output_idx, -1] = int(idx == selected_outflow_indices[b])

    return output_array


# Process and save the results
val_output_array = process_data(val_inflows, val_outflows, val_output_array)
np.save('dcf_val_distances_cdf.npy', val_output_array)

test_output_array = process_data(test_inflows, test_outflows, test_output_array)
np.save('dcf_test_distances_cdf.npy', test_output_array)

print(val_output_array.shape)
