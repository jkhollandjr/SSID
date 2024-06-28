import numpy as np
import torch
from scipy.spatial.distance import cosine, euclidean
import torch.nn.functional as F
from orig_model import DFModel, DFModelWithAttention
from sklearn.model_selection import train_test_split
from traffic_utils import insert_dummy_packets_torch, calculate_inter_packet_times, calculate_times_with_directions, calculate_cumulative_traffic_torch, insert_dummy_packets_torch_exponential
from espresso import EspressoNet
import torch.nn as nn


def pad_or_truncate(tensor, max_len=1000):
    # Check if the length of the tensor is greater than the max_len
    if tensor.size(0) > max_len:
        # Truncate the tensor to max_len
        return tensor[:max_len]
    else:
        # Calculate the padding needed to reach max_len
        padding_size = max_len - tensor.size(0)
        # Pad the tensor on the right with zeros
        return torch.cat([tensor, torch.zeros(padding_size, dtype=tensor.dtype)], dim=0)

def rate_estimator(iats, sizes):
    """Simple/naive implementation of a running average traffic flow rate estimator
       It is entirely vectorized, so it is fast
    """
    times = torch.cumsum(iats, dim=0)
    #indices = torch.arange(1, iats.size(0) + 1)
    sizes = torch.cumsum(sizes, dim=0)
    flow_rate = torch.where(times != 0, sizes / times, torch.ones_like(times))
    return flow_rate


def remove_right_padded_zeros(tensor):
    # Check if tensor is 1-dimensional or 2-dimensional
    if tensor.dim() == 1:
        # For 1-dimensional tensor
        non_zero_indices = torch.nonzero(tensor, as_tuple=True)[0]
        if non_zero_indices.numel() == 0:
            return tensor
        last_non_zero_index = non_zero_indices[-1]
        return tensor[:last_non_zero_index + 1]
    elif tensor.dim() == 2:
        # For 2-dimensional tensor, remove right-padded zeros for each row
        result = []
        for row in tensor:
            non_zero_indices = torch.nonzero(row, as_tuple=True)[0]
            if non_zero_indices.numel() == 0:
                result.append(row)
            else:
                last_non_zero_index = non_zero_indices[-1]
                result.append(row[:last_non_zero_index + 1])
        # Find the max length of rows after removing right-padded zeros
        max_len = max(len(r) for r in result)
        # Pad rows to the same length (if necessary)
        result_padded = [torch.nn.functional.pad(r, (0, max_len - len(r)), "constant", 0) for r in result]
        return torch.stack(result_padded)
    else:
        raise ValueError("Only 1D or 2D tensors are supported")

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

class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, anchor, positive, negative):
        pos_sim = self.cosine_sim(anchor, positive)
        neg_sim = self.cosine_sim(anchor, negative)
        loss = F.relu(neg_sim - pos_sim + self.margin)
        return loss.mean()

# Function to apply transformations and dummy packet insertion
def transform_and_defend_features(features):
    # Initialize lists to store transformed and defended tensors
    interval_dirs_up_list = []
    interval_dirs_down_list = []
    interval_dirs_sum_list = []
    interval_dirs_sub_list = []
    interval_iats_list = []
    interval_inv_iat_logs_list = []
    interval_cumul_norm_list = []
    interval_times_norm_list = []

    # Loop through each sample in the features tensor
    for i in range(features.size(0)):
        sizes, times, directions = features[i, 0, :], features[i, 1, :], features[i, 2, :]

        sizes = remove_right_padded_zeros(sizes)
        times = remove_right_padded_zeros(times)
        directions = remove_right_padded_zeros(directions)
        zero_tensor = torch.tensor([0])
        upload = directions > 0
        download = ~upload
        iats = torch.diff(times, prepend=zero_tensor)


        interval_size = .03
        num_intervals = int(torch.ceil(torch.max(times) / interval_size).item())

        split_points = torch.arange(0, num_intervals) * interval_size
        split_points = torch.searchsorted(times, split_points)

        dirs_subs = torch.tensor_split(directions, split_points)
        interval_dirs_up = torch.zeros(num_intervals+1)
        interval_dirs_down = torch.zeros(num_intervals+1)
        for i, tensor in enumerate(dirs_subs):
            size = tensor.numel()
            if size > 0:
                up = (tensor >= 0).sum()
                interval_dirs_up[i] = up
                interval_dirs_down[i] = size - up

        times_subs = torch.tensor_split(times, split_points)
        interval_times = torch.zeros(num_intervals+1)
        for i,tensor in enumerate(times_subs):
            if tensor.numel() > 0:
                interval_times[i] = tensor.mean()
            elif i > 0:
                interval_times[i] = interval_times[i-1]

        interval_times_norm = interval_times.clone()
        interval_times_norm -= torch.mean(interval_times_norm)
        interval_times_norm /= torch.amax(torch.abs(interval_times_norm))

        iats_subs = torch.tensor_split(iats, split_points)
        interval_iats = torch.zeros(num_intervals+1)
        for i,tensor in enumerate(iats_subs):
            if tensor.numel() > 0:
                interval_iats[i] = tensor.mean()
            elif i > 0:
                interval_iats[i] = interval_iats[i-1] + interval_size

        download_iats = torch.diff(times[download], prepend=torch.tensor([0]))
        upload_iats = torch.diff(times[upload], prepend=torch.tensor([0]))
        flow_iats = torch.zeros_like(times)
        flow_iats[upload] = upload_iats
        flow_iats[download] = download_iats
        inv_iat_logs = torch.log(torch.nan_to_num((1 / flow_iats)+1, nan=1e4, posinf=1e4))
        inv_iat_logs_subs = torch.tensor_split(inv_iat_logs, split_points)
        interval_inv_iat_logs = torch.zeros(num_intervals+1)
        for i,tensor in enumerate(inv_iat_logs_subs):
            if tensor.numel() > 0:
                interval_inv_iat_logs[i] = tensor.mean()

        size_dirs = sizes*directions
        cumul = torch.cumsum(size_dirs, dim=0)   # raw accumulation
        cumul_subs = torch.tensor_split(cumul, split_points)
        interval_cumul = torch.zeros(num_intervals+1)
        for i,tensor in enumerate(cumul_subs):
            if tensor.numel() > 0:
                interval_cumul[i] = tensor.mean()
            elif i > 0:
                interval_cumul[i] = interval_cumul[i-1]

        interval_cumul_norm = interval_cumul.clone()
        interval_cumul_norm -= torch.mean(interval_cumul_norm)
        interval_cumul_norm /= torch.amax(torch.abs(interval_cumul_norm))

        running_rates = rate_estimator(iats, sizes)
        rates_subs = torch.tensor_split(running_rates, split_points)
        interval_rates = torch.zeros(num_intervals+1)
        for i,tensor in enumerate(rates_subs):
            if tensor.numel() > 0:
                interval_rates[i] = tensor.mean()

        # Apply dummy packet insertion
        #defended_sizes_i, defended_times_i, defended_directions_i = insert_dummy_packets_torch_exponential(sizes, times, directions, num_dummy_packets=0)

        #defended_sizes.append(defended_sizes_i.unsqueeze(0))
        #defended_times.append(defended_times_i.unsqueeze(0))
        #defended_directions.append(defended_directions_i.unsqueeze(0))
        interval_dirs_sum = interval_dirs_up + interval_dirs_down
        interval_dirs_sub = interval_dirs_up - interval_dirs_down

        interval_dirs_up_list.append(pad_or_truncate(interval_dirs_up).unsqueeze(0))
        interval_dirs_down_list.append(pad_or_truncate(interval_dirs_down).unsqueeze(0))
        interval_dirs_sum_list.append(pad_or_truncate(interval_dirs_sum).unsqueeze(0))
        interval_dirs_sub_list.append(pad_or_truncate(interval_dirs_sub).unsqueeze(0))
        interval_iats_list.append(pad_or_truncate(interval_iats).unsqueeze(0))
        interval_inv_iat_logs_list.append(pad_or_truncate(interval_inv_iat_logs).unsqueeze(0))
        interval_cumul_norm_list.append(pad_or_truncate(interval_cumul_norm).unsqueeze(0))
        interval_times_norm_list.append(pad_or_truncate(interval_times_norm).unsqueeze(0))

    interval_dirs_up_list = torch.cat(interval_dirs_up_list, dim=0)
    interval_dirs_down_list = torch.cat(interval_dirs_down_list, dim=0)
    interval_dirs_sum_list = torch.cat(interval_dirs_sum_list, dim=0)
    interval_dirs_sub_list = torch.cat(interval_dirs_sub_list, dim=0)
    interval_iats_list = torch.cat(interval_iats_list, dim=0)
    interval_inv_iat_logs_list = torch.cat(interval_inv_iat_logs_list, dim=0)
    interval_cumul_norm_list = torch.cat(interval_cumul_norm_list, dim=0)
    interval_times_norm_list = torch.cat(interval_times_norm_list, dim=0)

    # Consider splitting upload and download inter-packet times?

    # Stack all features together
    transformed_features = torch.stack([interval_dirs_up_list, interval_dirs_down_list, interval_dirs_sum_list, interval_dirs_sub_list, interval_iats_list, interval_inv_iat_logs_list, interval_cumul_norm_list, interval_times_norm_list], dim=1)
    return transformed_features

# Instantiate the models
embedding_size = 64
inflow_model = EspressoNet(8, special_toks=1, **model_config)
outflow_model = EspressoNet(8, special_toks=1, **model_config)

# Load the best models
#checkpoint = torch.load('models/best_model_dcf_defened_0.00806727527074893.pth')
checkpoint = torch.load('models/best_model_live_espresso_may17.pth')
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
val_inflows = np.load('data/val_inflows_may17.npy')[:1000]
val_outflows = np.load('data/val_outflows_may17.npy')[:1000]

# Split the data
val_inflows, test_inflows, val_outflows, test_outflows = train_test_split(val_inflows, val_outflows, test_size=0.5, random_state=42)

np.save('data/test_inflows_may17_test.npy', test_inflows)
np.save('data/test_outflows_may17_test.npy', test_outflows)

# Initialize the outputs
val_output_array = np.zeros((len(val_inflows) * len(val_outflows), 92))
test_output_array = np.zeros((len(test_inflows) * len(test_outflows), 92))

def compute_batch_distances(inflow_traces, outflow_traces, inflow_model, outflow_model):
    
    all_cosine_similarities = []

    inflow_window = inflow_traces[:, -1, :, :] 
    outflow_window = outflow_traces[:, -1, :, :]

    inflow_window = torch.from_numpy(inflow_window).float()
    outflow_window = torch.from_numpy(outflow_window).float()

    inflow_window = transform_and_defend_features(inflow_window)
    outflow_window = transform_and_defend_features(outflow_window)

    #inflow_window = inflow_window.reshape(inflow_window.shape[0], -1, inflow_window.shape[-1]) 
    #outflow_window = outflow_window.reshape(outflow_window.shape[0], -1, outflow_window.shape[-1])
    # 64, 64, 114

    #print(inflow_window.shape)

    inflow_embeddings, _ = outflow_model(inflow_window.to(device))
    outflow_embeddings, _ = outflow_model(outflow_window.to(device))

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
def process_data(inflows, outflows, output_array):
    num_inflows = len(inflows)
    num_outflows = len(outflows)
    batch_size = 256

    # Allocate an empty array for the results
    # Size: num_inflows * batch_size, 15+1 (for 15 windows and 1 match column)
    output_array = np.zeros((num_inflows * batch_size, 93))

    for idx, inflow_example in enumerate(inflows):
        print(idx)
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

            #output_array[output_idx, -2] = selected_outflow_indices[b]
            #output_array[output_idx, -3] = idx

    return output_array

# Process and save the results
val_output_array = process_data(val_inflows, val_outflows, val_output_array)
np.save('data/dcf_val_distances_espresso_live.npy', val_output_array)

test_output_array = process_data(test_inflows, test_outflows, test_output_array)
np.save('data/dcf_test_distances_espresso_live.npy', test_output_array)

print(val_output_array.shape)
