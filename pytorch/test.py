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
checkpoint = torch.load('best_model.pth')
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
test_inflows = np.load('test_inflows.npy')
test_outflows = np.load('test_outflows.npy')

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

print(compute_distances(test_inflows[0], test_outflows[0], inflow_model, outflow_model))

import numpy as np

# Prepare an empty numpy array
output_array = np.zeros((len(test_inflows) * len(test_outflows), 11))

# Iterate over all possible combinations of inflow and outflow traces
for i in range(len(test_inflows)):
    print(i)
    for j in range(len(test_outflows)):
        # Compute the distances
        distances = compute_distances(test_inflows[i], test_outflows[j], inflow_model, outflow_model)

        # Check if the inflow and outflow are a match (i.e., they have the same index)
        match = int(i == j)

        # Store the distances and the match status in the numpy array
        output_array[i * len(test_outflows) + j, :10] = distances
        output_array[i * len(test_outflows) + j, 10] = match

print(distances[0])
# Save the numpy array to a file
np.save('distances.npy', output_array)

