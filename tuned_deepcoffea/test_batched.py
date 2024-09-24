import numpy as np
import torch
import torch.nn.functional as F
from orig_model import DFModel, DFModelWithAttention
from sklearn.model_selection import train_test_split
import argparse
import os
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on inflow and outflow data.")
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help="Path to the saved model file.")
    parser.add_argument('--data_dir', type=str, default='data/', help="Directory containing the validation and test .npy files.")
    parser.add_argument('--output_dir', type=str, default='data/', help="Directory to save the output distance files.")
    return parser.parse_args()

def compute_batch_distances(inflow_traces, outflow_traces, inflow_model, outflow_model, device):
    all_cosine_similarities = []

    for window_idx in range(inflow_traces.shape[1]):
        inflow_window = inflow_traces[:, window_idx, :5, :]
        outflow_window = outflow_traces[:, window_idx, :5, :]

        inflow_window = inflow_window.reshape(inflow_window.shape[0], -1, inflow_window.shape[-1]) 
        outflow_window = outflow_window.reshape(outflow_window.shape[0], -1, outflow_window.shape[-1])

        inflow_window = torch.from_numpy(inflow_window).float().to(device)
        outflow_window = torch.from_numpy(outflow_window).float().to(device)

        inflow_embeddings = inflow_model(inflow_window)
        outflow_embeddings = outflow_model(outflow_window)

        cosine_similarities = F.cosine_similarity(inflow_embeddings, outflow_embeddings).detach().cpu().numpy()
        all_cosine_similarities.append(cosine_similarities)

    return np.stack(all_cosine_similarities, axis=1)

def process_data(inflows, outflows, inflow_model, outflow_model, device):
    num_inflows = len(inflows)
    num_outflows = len(outflows)
    batch_size = 256

    output_array = np.zeros((num_inflows * batch_size, 13))

    for idx, inflow_example in enumerate(inflows):
        selected_outflow_indices = random.sample(range(num_outflows), batch_size)
        selected_outflows = outflows[selected_outflow_indices]

        inflow_batch = np.repeat(inflow_example[np.newaxis, ...], batch_size, axis=0)

        distances = compute_batch_distances(inflow_batch, selected_outflows, inflow_model, outflow_model, device)

        for b in range(batch_size):
            output_idx = idx * batch_size + b
            output_array[output_idx, :-1] = distances[b]
            output_array[output_idx, -1] = int(idx == selected_outflow_indices[b])

    return output_array

def main():
    args = parse_args()

    # Instantiate the models
    inflow_model = DFModel()
    outflow_model = DFModel()

    # Load the best models
    checkpoint = torch.load(args.model_path)
    inflow_model.load_state_dict(checkpoint['inflow_model_state_dict'])
    outflow_model.load_state_dict(checkpoint['outflow_model_state_dict'])

    # Move models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inflow_model.to(device)
    outflow_model.to(device)

    inflow_model.eval()
    outflow_model.eval()

    # Load the numpy arrays
    val_inflows = np.load(os.path.join(args.data_dir, 'val_inflows.npy'))[:1000]
    val_outflows = np.load(os.path.join(args.data_dir, 'val_outflows.npy'))[:1000]

    # Split the data
    val_inflows, test_inflows, val_outflows, test_outflows = train_test_split(
        val_inflows, val_outflows, test_size=0.5, random_state=42
    )

    # Initialize the outputs
    val_output_array = np.zeros((len(val_inflows) * len(val_outflows), 13))
    test_output_array = np.zeros((len(test_inflows) * len(test_outflows), 13))

    # Process and save the results
    val_output_array = process_data(val_inflows, val_outflows, inflow_model, outflow_model, device)
    np.save(os.path.join(args.output_dir, 'dcf_val_distances.npy'), val_output_array)

    test_output_array = process_data(test_inflows, test_outflows, inflow_model, outflow_model, device)
    np.save(os.path.join(args.output_dir, 'dcf_test_distances.npy'), test_output_array)

    print(val_output_array.shape)

if __name__ == "__main__":
    main()

