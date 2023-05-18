import numpy as np
import torch
from model import Embedder
from torch.nn.functional import cosine_similarity
from scipy.spatial.distance import cosine, euclidean
import torch.nn.functional as F

# Instantiate the models
embedding_size = 64
inflow_model = Embedder(embedding_size)
outflow_model = Embedder(embedding_size)

# Load the best models
checkpoint = torch.load('best_model_cosine.pth')
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
val_inflows = np.load('val_inflows.npy')
val_outflows = np.load('val_outflows.npy')

print(len(val_inflows))

import torch.nn.functional as F

def compute_distances(inflow_trace, outflow_trace, inflow_model, outflow_model):
    # List to store distances
    euclidean_distances = []

    # Iterate over windows in the trace
    for i in range(inflow_trace.shape[0]):
        # Get the window
        inflow_window = inflow_trace[i]
        outflow_window = outflow_trace[i]

        # Reshape windows to be 3D (1, 4, 1000)
        inflow_window = inflow_window.reshape(1, inflow_window.shape[0], inflow_window.shape[1])
        outflow_window = outflow_window.reshape(1, outflow_window.shape[0], outflow_window.shape[1])

        # Move tensors to the correct device and convert them to PyTorch tensors
        inflow_window = torch.from_numpy(inflow_window).float().to(device)
        outflow_window = torch.from_numpy(outflow_window).float().to(device)

        # Get embeddings
        inflow_embedding = inflow_model(inflow_window)
        outflow_embedding = outflow_model(outflow_window)

        # Compute euclidean distance
        euclidean_distance = F.pairwise_distance(inflow_embedding, outflow_embedding)
        euclidean_distance = euclidean_distance.item()  # Convert tensor to single number
        euclidean_distances.append(euclidean_distance)

    return euclidean_distances

from sklearn.model_selection import train_test_split

# Split the original validation set into a new validation set and a test set
val_inflows, test_inflows, val_outflows, test_outflows = train_test_split(val_inflows, val_outflows, test_size=0.5, random_state=42)

# Initialize two empty numpy arrays for the new validation set and the test set
val_output_array = np.zeros((len(val_inflows) * len(val_outflows), 11))
test_output_array = np.zeros((len(test_inflows) * len(test_outflows), 11))
print(len(val_output_array))
print(len(test_output_array))

# Generate combinations for the new validation set
for i in range(len(val_inflows)):
    print(f"Processing validation set, trace {i+1}/{len(val_inflows)}")
    for j in range(len(val_outflows)):
        distances = compute_distances(val_inflows[i], val_outflows[j], inflow_model, outflow_model)
        match = int(i == j)
        val_output_array[i * len(val_outflows) + j, :10] = distances
        val_output_array[i * len(val_outflows) + j, 10] = match

# Generate combinations for the test set
for i in range(len(test_inflows)):
    print(f"Processing test set, trace {i+1}/{len(test_inflows)}")
    for j in range(len(test_outflows)):
        distances = compute_distances(test_inflows[i], test_outflows[j], inflow_model, outflow_model)
        match = int(i == j)
        test_output_array[i * len(test_outflows) + j, :10] = distances
        test_output_array[i * len(test_outflows) + j, 10] = match

# Save the numpy arrays to files
np.save('val_distances.npy', val_output_array)
np.save('test_distances.npy', test_output_array)

