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
        defended_sizes_i, defended_times_i, defended_directions_i = insert_dummy_packets_torch_exponential(sizes, times, directions, num_dummy_packets=18)
        
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
    transformed_features = torch.stack([defended_sizes, inter_packet_times, times_with_directions, defended_directions, cumul], dim=1)
    return transformed_features

# Instantiate the models
embedding_size = 64
inflow_model = DFModel()
outflow_model = DFModel()

# Load the best models
#checkpoint = torch.load('models/best_model_dcf_defened_0.00806727527074893.pth')
checkpoint = torch.load('models/best_model_live_defended_exp.pth')
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
val_inflows = np.load('data/val_inflows.npy')[:1000]
val_outflows = np.load('data/val_outflows.npy')[:1000]

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

        inflow_embeddings = inflow_model(inflow_window) 
        outflow_embeddings = outflow_model(outflow_window)

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
np.save('data/dcf_val_distances_live.npy', val_output_array)

test_output_array = process_data(test_inflows, test_outflows, test_output_array)
np.save('data/dcf_test_distances_live.npy', test_output_array)

print(val_output_array.shape)
