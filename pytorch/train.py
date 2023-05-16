import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from model import Embedder
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class TripletDataset(Dataset):
    def __init__(self, inflow_data, outflow_data):
        self.inflow_data = inflow_data
        self.outflow_data = outflow_data

    def __len__(self):
        # Dataset length is the number of traces
        return len(self.inflow_data)

    def __getitem__(self, idx):
        # Select the window randomly
        window_idx = random.randint(0, self.inflow_data.shape[1]-1)

        anchor = self.inflow_data[idx, window_idx]
        positive = self.outflow_data[idx, window_idx]

        # Select a random negative example
        negative_idx = random.choice([j for j in range(len(self.outflow_data)) if j != idx])
        negative = self.outflow_data[negative_idx, window_idx]

        return anchor, positive, negative

# Load the numpy arrays
train_inflows = np.load('train_inflows.npy')
val_inflows = np.load('val_inflows.npy')
test_inflows = np.load('test_inflows.npy')

train_outflows = np.load('train_outflows.npy')
val_outflows = np.load('val_outflows.npy')
test_outflows = np.load('test_outflows.npy')

# Define the datasets
train_dataset = TripletDataset(train_inflows, train_outflows)
val_dataset = TripletDataset(val_inflows, val_outflows)

# Create the dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the models
embedding_size = 64
inflow_model = Embedder(embedding_size)
outflow_model = Embedder(embedding_size)

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inflow_model.to(device)
outflow_model.to(device)

# Define the loss function and the optimizer
criterion = TripletLoss()
optimizer = optim.Adam(list(inflow_model.parameters()) + list(outflow_model.parameters()), lr=0.001)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    # Training
    inflow_model.train()
    outflow_model.train()

    running_loss = 0.0
    for anchor, positive, negative in train_loader:
        # Move tensors to the correct device
        anchor = anchor.float().to(device)
        positive = positive.float().to(device)
        negative = negative.float().to(device)

        # Forward pass
        anchor_embeddings = inflow_model(anchor)
        positive_embeddings = outflow_model(positive)
        negative_embeddings = outflow_model(negative)

        # Compute the loss
        loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # Validation
    inflow_model.eval()
    outflow_model.eval()

    running_loss = 0.0
    with torch.no_grad():
        for anchor, positive, negative in val_loader:
            optimizer.zero_grad()
            # Move tensors to the correct device
            anchor = anchor.float().to(device)
            positive = positive.float().to(device)
            negative = negative.float().to(device)

            # Forward pass
            anchor_embeddings = inflow_model(anchor)
            positive_embeddings = outflow_model(positive)
            negative_embeddings = outflow_model(negative)

            # Compute the loss
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

            running_loss += loss.item()

    val_loss = running_loss / len(val_loader)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

