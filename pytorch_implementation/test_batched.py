import numpy as np
import torch
from scipy.spatial.distance import cosine, euclidean
import torch.nn.functional as F
from orig_model import DFModel, DFModelWithAttention
from sklearn.model_selection import train_test_split

# Instantiate the models
embedding_size = 64
inflow_model = DFModel()
outflow_model = DFModel()

# Load the best models
checkpoint = torch.load('models/best_model_ssid_live.pth')
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
        inflow_window = inflow_traces[:, window_idx, :5, :] #64, 15, 4, 500
        outflow_window = outflow_traces[:, window_idx, :5, :]

        inflow_window = inflow_window.reshape(inflow_window.shape[0], -1, inflow_window.shape[-1]) 
        outflow_window = outflow_window.reshape(outflow_window.shape[0], -1, outflow_window.shape[-1])
        # 64, 4, 500

        inflow_window = torch.from_numpy(inflow_window).float().to(device)
        outflow_window = torch.from_numpy(outflow_window).float().to(device)

        inflow_embeddings = inflow_model(inflow_window) #64, 64
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
