import numpy as np
import torch
from model import Embedder
from torch.nn.functional import cosine_similarity
from scipy.spatial.distance import cosine, euclidean

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


def compute_cosine_distance(inflow_trace, outflow_trace, inflow_model, outflow_model):
    # List to store distances
    distances = []

    # Iterate over windows in the trace
    for i in range(inflow_trace.shape[0]):
        # Get the window
        inflow_window = inflow_trace[i]
        outflow_window = outflow_trace[i]

        # Reshape windows to be 2D (1, 4*1000)
        inflow_window = inflow_window.reshape(1, 4, 1000)
        outflow_window = outflow_window.reshape(1, 4, 1000)

        # Move tensors to the correct device and convert them to PyTorch tensors
        inflow_window = torch.from_numpy(inflow_window).float().to(device)
        outflow_window = torch.from_numpy(outflow_window).float().to(device)

        # Get embeddings
        inflow_embedding = inflow_model(inflow_window)
        outflow_embedding = outflow_model(outflow_window)

        # Compute cosine distance
        inflow_embedding = inflow_embedding.detach().cpu().numpy().reshape(64)
        outflow_embedding = outflow_embedding.detach().cpu().numpy().reshape(64)
        distance = euclidean(inflow_embedding, outflow_embedding)

        distances.append(distance)

    return distances

print(compute_cosine_distance(test_inflows[0], test_outflows[0], inflow_model, outflow_model))
