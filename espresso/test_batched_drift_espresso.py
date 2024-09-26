import numpy as np
import torch
from scipy.spatial.distance import cosine, euclidean
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from espresso import EspressoNet
import argparse
import random

# Argument parser for command-line inputs
parser = argparse.ArgumentParser(description="Process model inputs and outputs.")
parser.add_argument('--checkpoint_path', type=str, default='models/best_model.pth', help='Path to model checkpoint file.')
parser.add_argument('--val_inflows_path', type=str, default='data/val_inflows.npy', help='Path to validation inflows file (numpy).')
parser.add_argument('--val_outflows_path', type=str, default='data/val_outflows.npy', help='Path to validation outflows file (numpy).')
parser.add_argument('--output_val_path', type=str, default='data/val_output.npy', help='Path to save the validation output numpy file.')
parser.add_argument('--output_test_path', type=str, default='data/test_output.npy', help='Path to save the test output numpy file.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for processing.')
args = parser.parse_args()

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
inflow_model = EspressoNet(8, special_toks=1, **model_config)
outflow_model = EspressoNet(8, special_toks=1, **model_config)

# Load the best models
checkpoint = torch.load(args.checkpoint_path)
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
val_inflows = np.load(args.val_inflows_path)[:1000]
val_outflows = np.load(args.val_outflows_path)[:1000]

# Split the data
val_inflows, test_inflows, val_outflows, test_outflows = train_test_split(val_inflows, val_outflows, test_size=0.5, random_state=42)

# Initialize the outputs
val_output_array = np.zeros((len(val_inflows) * len(val_outflows), 92))
test_output_array = np.zeros((len(test_inflows) * len(test_outflows), 92))

def find_closest(packet_time, other_flow_times):
    if other_flow_times.size == 0:
        return 0
    idx = np.searchsorted(other_flow_times, packet_time)
    if idx == len(other_flow_times):
        return other_flow_times[-1]
    elif idx == 0:
        return other_flow_times[0]
    else:
        before = other_flow_times[idx - 1]
        after = other_flow_times[idx]
        return before if abs(packet_time - before) < abs(packet_time - after) else after

def calculate_proportions(trace1, trace2, sizes_trace1, sizes_trace2, thresholds=[0.0001, 0.001, 0.01, 0.03, 0.08, 0.16, 0.32, 0.64]):
    sizes_trace1 = np.abs(sizes_trace1)
    sizes_trace2 = np.abs(sizes_trace2)

    size_threshold = 50
    trace1 = np.array(trace1)
    trace2 = np.array(trace2)
    sizes_trace1 = np.array(sizes_trace1)
    sizes_trace2 = np.array(sizes_trace2)

    mask1 = (trace1 < 0) & (sizes_trace1 >= size_threshold)
    flow1 = -trace1[mask1]
    mask2 = (trace2 < 0) & (sizes_trace2 >= size_threshold)
    flow2 = -trace2[mask2]

    flow1 = np.sort(flow1)
    flow2 = np.sort(flow2)

    if len(flow2) == 0:
        return np.zeros(len(thresholds))
    
    time_diffs = []
    for time in flow1:
        closest_time = find_closest(time, flow2)
        if closest_time is not None:
            time_diffs.append(abs(time - closest_time))
        else:
            time_diffs.append(float(1))

    if np.count_nonzero(flow1) == 0:
        return np.array([0]*8)

    proportions = []
    for threshold in thresholds:
        count = np.sum(np.array(time_diffs) < threshold)
        proportion = count / np.count_nonzero(flow1)
        proportions.append(proportion)

    return np.array(proportions)

def compute_batch_distances(inflow_traces, outflow_traces, inflow_model, outflow_model):
    # Compute embeddings
    inflow_window = torch.from_numpy(inflow_traces[:, :, :]).float().to(device)
    outflow_window = torch.from_numpy(outflow_traces[:, :, :]).float().to(device)

    with torch.no_grad():
        inflow_embeddings, _ = inflow_model(inflow_window[:, 3:, :])
        outflow_embeddings, _ = outflow_model(outflow_window[:, 3:, :])

    B_inflow = inflow_embeddings.shape[0]
    B_outflow = outflow_embeddings.shape[0]

    inflow_embeddings = inflow_embeddings.reshape(B_inflow, 92, 64)
    outflow_embeddings = outflow_embeddings.reshape(B_outflow, 92, 64)

    all_cosine_similarities = np.zeros((B_inflow, B_outflow, 92))

    for i in range(92):
        inflow_window_embedding = inflow_embeddings[:, i, :]  # shape (B_inflow, 64)
        outflow_window_embedding = outflow_embeddings[:, i, :]  # shape (B_outflow, 64)

        # Compute pairwise cosine similarities between inflow and outflow embeddings
        inflow_exp = inflow_window_embedding.unsqueeze(1)  # shape (B_inflow, 1, 64)
        outflow_exp = outflow_window_embedding.unsqueeze(0)  # shape (1, B_outflow, 64)

        numerator = (inflow_exp * outflow_exp).sum(dim=2)  # shape (B_inflow, B_outflow)
        inflow_norm = inflow_window_embedding.norm(dim=1).unsqueeze(1)  # shape (B_inflow, 1)
        outflow_norm = outflow_window_embedding.norm(dim=1).unsqueeze(0)  # shape (1, B_outflow)
        denominator = inflow_norm * outflow_norm  # shape (B_inflow, B_outflow)
        cosine_similarities = (numerator / denominator).cpu().numpy()  # shape (B_inflow, B_outflow)

        # Store the cosine similarities
        all_cosine_similarities[:, :, i] = cosine_similarities

    return all_cosine_similarities  # shape (B_inflow, B_outflow, 92)

def process_data(inflows, outflows, batch_size):
    num_inflows = len(inflows)
    num_outflows = len(outflows)

    # The output array will store results for every inflow-outflow pair
    output_array = np.zeros((num_inflows * num_outflows, 109))

    # Determine batch sizes for inflow and outflow
    # Adjust B_inflow and B_outflow such that B_inflow * B_outflow <= batch_size
    # For example, batch_size = 256, B_inflow = 16, B_outflow = 16
    B_inflow = int(np.sqrt(batch_size))
    B_outflow = int(np.sqrt(batch_size))

    for inflow_start in range(0, num_inflows, B_inflow):
        print(inflow_start)
        inflow_end = min(inflow_start + B_inflow, num_inflows)
        inflow_batch = inflows[inflow_start:inflow_end]
        inflow_indices = list(range(inflow_start, inflow_end))

        for outflow_start in range(0, num_outflows, B_outflow):
            outflow_end = min(outflow_start + B_outflow, num_outflows)
            outflow_batch = outflows[outflow_start:outflow_end]
            outflow_indices = list(range(outflow_start, outflow_end))

            # Compute distances for the current batch of inflows against the current batch of outflows
            distances = compute_batch_distances(inflow_batch, outflow_batch, inflow_model, outflow_model)  # shape (B_inflow, B_outflow, 92)

            # Now for each pair of inflow and outflow in the batches, store the distances
            for i_inflow, inflow_idx in enumerate(inflow_indices):
                for i_outflow, outflow_idx in enumerate(outflow_indices):
                    output_idx = inflow_idx * num_outflows + outflow_idx
                    output_array[output_idx, :92] = distances[i_inflow, i_outflow, :]  # The 92 distances

                    # Set match to 1 if inflow and outflow have the same index
                    output_array[output_idx, -1] = int(inflow_idx == outflow_idx)

                    # For calculate_proportions, get the inflow and outflow traces
                    inflow_time = inflows[inflow_idx, 1, :]
                    inflow_dir = inflows[inflow_idx, 2, :]
                    inflow_sizes = inflows[inflow_idx, 0, :]

                    outflow_time = outflows[outflow_idx, 1, :]
                    outflow_dir = outflows[outflow_idx, 2, :]
                    outflow_sizes = outflows[outflow_idx, 0, :]

                    download_time_diff = calculate_proportions(outflow_time * outflow_dir, inflow_time * inflow_dir, outflow_sizes, inflow_sizes)
                    upload_time_diff = calculate_proportions(outflow_time * outflow_dir * -1, inflow_time * inflow_dir * -1, outflow_sizes, inflow_sizes)

                    output_array[output_idx, 92:100] = download_time_diff
                    output_array[output_idx, 100:108] = upload_time_diff

    return output_array


# Process and save the results
val_output_array = process_data(val_inflows, val_outflows, args.batch_size)
np.save(args.output_val_path, val_output_array)

test_output_array = process_data(test_inflows, test_outflows, args.batch_size)
np.save(args.output_test_path, test_output_array)
