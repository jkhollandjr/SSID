import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from orig_model import DFModel, DFModelWithAttention
import torch.nn.functional as F
import math

torch.set_printoptions(threshold=5000)

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
    cumulative_traffic = torch.div(torch.cumsum(sizes, dim=1), 1000)
    return cumulative_traffic

# Define the learning rate schedule function
def lr_schedule(epoch):
    if epoch < 100:
        # Increase linearly
        return epoch / 100
    else:
        # Decrease gradually
        return 0.5 * (1 + math.cos(math.pi * 1 * (epoch - 1000) / (num_epochs - 1000)))

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

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


class TripletDataset(Dataset):
    def __init__(self, inflow_data, outflow_data):
        self.positive_top = True
        self.inflow_data = inflow_data
        self.outflow_data = outflow_data
        self.all_indices = list(range(len(self.inflow_data)))
        random.shuffle(self.all_indices)  # Shuffle the indices initially

        # Divide the shuffled indices into two partitions.
        cutoff = len(self.all_indices) // 2
        self.partition_1 = self.all_indices[:cutoff]
        self.partition_2 = self.all_indices[cutoff:]

    def __len__(self):
        return len(self.inflow_data)

    def __getitem__(self, idx):
        window_idx = random.randint(0, self.inflow_data.shape[1]-1)

        # Choose a positive from partition 1 and a negative from partition 2 (or vice versa).
        if self.positive_top:
            idx = random.choice(self.partition_1)
            negative_idx = random.choice([j for j in self.partition_2 if j != idx])
        else:
            idx = random.choice(self.partition_2)
            negative_idx = random.choice([j for j in self.partition_1 if j != idx])

        anchor = self.inflow_data[idx, window_idx]
        positive = self.outflow_data[idx, window_idx]
        negative = self.outflow_data[negative_idx, window_idx]

        return anchor, positive, negative

    def reset_split(self):
        self.positive_top = not self.positive_top

        # Reshuffle the indices at the start of each epoch.
        random.shuffle(self.all_indices)

        # Re-divide the shuffled indices into two partitions.
        cutoff = len(self.all_indices) // 2
        self.partition_1 = self.all_indices[:cutoff]
        self.partition_2 = self.all_indices[cutoff:]

class QuadrupleSampler(Sampler):
    """Sampler that repeats the dataset indices four times, effectively quadrupling the dataset size for each epoch."""
    
    def __init__(self, data_source):
        self.data_source = data_source
    
    def __iter__(self):
        # Repeat the dataset indices four times
        indices = list(range(len(self.data_source))) * 4
        # Shuffle indices to ensure random sampling across repeats
        np.random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        # The length is now four times the original dataset size
        return 4 * len(self.data_source)

def custom_collate_fn(batch):
    anchors, positives, negatives = zip(*batch)
    
    # Convert numpy arrays to PyTorch tensors
    anchors = [torch.tensor(anchor, dtype=torch.float32) for anchor in anchors]
    positives = [torch.tensor(positive, dtype=torch.float32) for positive in positives]
    negatives = [torch.tensor(negative, dtype=torch.float32) for negative in negatives]
    
    # Stack tensors to create batched tensors
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    # Function to apply transformations and dummy packet insertion
    def transform_and_defend_features(features):
        # Initialize lists to store transformed and defended tensors
        defended_sizes = []
        defended_times = []
        defended_directions = []
        
        # Loop through each sample in the features tensor
        for i in range(features.size(0)):
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

    # Apply transformations and defense mechanism
    transformed_anchors = transform_and_defend_features(anchors)
    transformed_positives = transform_and_defend_features(positives)
    transformed_negatives = transform_and_defend_features(negatives)
    
    return transformed_anchors, transformed_positives, transformed_negatives


# Load the numpy arrays
train_inflows = np.load('train_inflows.npy')
val_inflows = np.load('val_inflows.npy')

train_outflows = np.load('train_outflows.npy')
val_outflows = np.load('val_outflows.npy')

# Define the datasets
train_dataset = TripletDataset(train_inflows, train_outflows)
val_dataset = TripletDataset(val_inflows, val_outflows)

train_sampler = QuadrupleSampler(train_dataset)
val_sampler = QuadrupleSampler(val_dataset)

# Create the dataloaders
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=custom_collate_fn, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=custom_collate_fn, num_workers=1)

# Instantiate the models
embedding_size = 64
inflow_model = DFModel()
outflow_model = DFModel()

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inflow_model.to(device)
outflow_model.to(device)

# Define the loss function and the optimizer
criterion = CosineTripletLoss()
optimizer = optim.Adam(list(inflow_model.parameters()) + list(outflow_model.parameters()), lr=0.0001)
#optimizer = optim.SGD(list(inflow_model.parameters())+list(outflow_model.parameters()), lr=.001, weight_decay=1e-6, momentum=.9, nesterov=True)

# Create the scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

# Training loop
best_val_loss = float("inf")
num_epochs = 3000
for epoch in range(num_epochs):
    train_dataset.reset_split()
    val_dataset.reset_split()
    # Training
    inflow_model.train()
    outflow_model.train()

    running_loss = 0.0
    for anchor, positive, negative in train_loader:
        # Move tensors to the correct device
        anchor = anchor.float().to(device)
        positive = positive.float().to(device)
        negative = negative.float().to(device)

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

    scheduler.step()

    # Validation
    inflow_model.eval()
    outflow_model.eval()

    running_loss = 0.0
    with torch.no_grad():
        for anchor, positive, negative in val_loader:
            # Move tensors to the correct device
            anchor = anchor.float().to(device)
            positive = positive.float().to(device)
            negative = negative.float().to(device)

            # Forward pass
            anchor_embeddings = inflow_model(anchor[:,:,:])
            positive_embeddings = outflow_model(positive[:,:,:])
            negative_embeddings = outflow_model(negative[:,:,:])

            # Compute the loss
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

            running_loss += loss.item()

    val_loss = running_loss / len(val_loader)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

    # Save the model if it's the best one so far
    if val_loss < best_val_loss:
        print("Best model so far!")
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'inflow_model_state_dict': inflow_model.state_dict(),
            'outflow_model_state_dict': outflow_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, f'models/best_model_live_undefended.pth')

