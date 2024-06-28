import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from orig_model import DFModel, DFModelWithAttention
from traffic_utils import insert_dummy_packets_torch, calculate_inter_packet_times, calculate_times_with_directions, calculate_cumulative_traffic, calculate_cumulative_traffic_torch, calculate_inter_packet_times_torch, insert_dummy_packets_torch_exponential
import torch.nn.functional as F
import math
from espresso import EspressoNet

torch.set_printoptions(threshold=5000)

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
        #window_idx = random.randint(0, self.inflow_data.shape[1]-1)
        window_idx = -1

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

class CosineTripletLossEspresso(nn.Module):
    def __init__(self, margin=0.1):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, anchor, positive, negative):
        # anchor, positive, negative are expected to be [batch_size, embedding_size, num_segments]
        # Compute cosine similarity for each segment
        pos_sim = self.cosine_sim(anchor, positive)  # [64, 92]
        neg_sim = self.cosine_sim(anchor, negative)  # [64, 92]

        # Compute loss for each segment
        losses = F.relu(neg_sim - pos_sim + self.margin)  # [64, 92]

        # Average the losses across all segments and then across the batch
        segment_mean_loss = losses.mean(dim=1)  # Average across segments
        batch_mean_loss = segment_mean_loss.mean()  # Average across batch
        return batch_mean_loss

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

            upload = directions > 0
            download = ~upload
            iats = torch.diff(times, prepend=torch.tensor([0]))

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
        
        '''
        # Stack defended features back into tensors
        defended_sizes = torch.cat(defended_sizes, dim=0)
        defended_times = torch.cat(defended_times, dim=0)
        defended_directions = torch.cat(defended_directions, dim=0)

        # Calculate additional features based on defended traffic
        inter_packet_times = calculate_inter_packet_times(defended_times)
        times_with_directions = calculate_times_with_directions(defended_times, defended_directions)
        cumul = calculate_cumulative_traffic_torch(defended_sizes, defended_times)
        '''
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

    # Apply transformations and defense mechanism
    transformed_anchors = transform_and_defend_features(anchors)
    transformed_positives = transform_and_defend_features(positives)
    transformed_negatives = transform_and_defend_features(negatives)
    
    return transformed_anchors, transformed_positives, transformed_negatives


# Load the numpy arrays
train_inflows = np.load('data/train_inflows_may17.npy')
val_inflows = np.load('data/val_inflows_may17.npy')

train_outflows = np.load('data/train_outflows_may17.npy')
val_outflows = np.load('data/val_outflows_may17.npy')

# Define the datasets
train_dataset = TripletDataset(train_inflows, train_outflows)
val_dataset = TripletDataset(val_inflows, val_outflows)

train_sampler = QuadrupleSampler(train_dataset)
val_sampler = QuadrupleSampler(val_dataset)

# Create the dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=custom_collate_fn, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=custom_collate_fn, num_workers=8)

# Instantiate the models
embedding_size = 64
inflow_model = EspressoNet(8, special_toks=1, **model_config)
outflow_model = EspressoNet(8, special_toks=1, **model_config)

checkpoint = torch.load('models/best_model_live_espresso_single.pth')
inflow_model.load_state_dict(checkpoint['inflow_model_state_dict'])
outflow_model.load_state_dict(checkpoint['outflow_model_state_dict'])

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inflow_model.to(device)
outflow_model.to(device)

# Define the loss function and the optimizer
criterion = CosineTripletLoss()
#optimizer = optim.Adam(list(inflow_model.parameters()) + list(outflow_model.parameters()), lr=0.0001)
optimizer = optim.AdamW(list(inflow_model.parameters()) + list(outflow_model.parameters()), lr=.001, betas=(0.9, 0.999), weight_decay=0.001)
#optimizer = optim.SGD(list(inflow_model.parameters())+list(outflow_model.parameters()), lr=.001, weight_decay=1e-6, momentum=.9, nesterov=True)

# Training loop
best_val_loss = float("inf")
num_epochs = 5000
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

        #anchor_embeddings, anchor_chain = inflow_model(anchor)
        anchor_embeddings, anchor_chain = outflow_model(anchor)
        positive_embeddings, positive_chain = outflow_model(positive)
        negative_embeddings, negative_chain = outflow_model(negative)

        # Compute the loss
        loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    #scheduler.step()

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
            #anchor_embeddings, anchor_chain = inflow_model(anchor[:,:,:])
            anchor_embeddings, anchor_chain = outflow_model(anchor)
            positive_embeddings, positive_chain = outflow_model(positive[:,:,:])
            negative_embeddings, negative_chain = outflow_model(negative[:,:,:])

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
        }, f'models/best_model_live_espresso_may17.pth')

