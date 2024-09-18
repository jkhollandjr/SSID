import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import torch.nn.functional as F
import math
from espresso import EspressoNet

torch.set_printoptions(threshold=5000)

import torch
import torch.nn as nn
import torch.nn.functional as F

class OnlineHardCosineTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(OnlineHardCosineTripletLoss, self).__init__()
        self.margin = margin

    def _get_anc_pos_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have the same label.

        Args:
            labels: torch.Tensor of dtype torch.int32 with shape [batch_size]

        Returns:
            mask: torch.Tensor of dtype torch.bool with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct
        indices_equal = torch.eye(labels.size(0)).to(labels.device).bool()
        indices_not_equal = ~indices_equal

        # Check if labels[i] == labels[j]
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

        # Combine the two masks
        mask = indices_not_equal & labels_equal

        return mask

    def _get_anc_neg_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

        Args:
            labels: torch.Tensor of dtype torch.int32 with shape [batch_size]

        Returns:
            mask: torch.Tensor of dtype torch.bool with shape [batch_size, batch_size]
        """
        return labels.unsqueeze(0) != labels.unsqueeze(1)

    def forward(self, embeddings, labels, 
            use_iq_mean = False,
            use_hard_negative_loss = True):
        """
        Args:
            embeddings: torch.Tensor -- batch of feature embeddings with shape [batch_size, features] or [batch_size, windows, features]
            labels: torch.Tensor of dtype torch.int32 with shape [batch_size]
            use_ciq_mean: bool -- use the mean of the interquartile range (e.g., exclude high and low quartiles from mean)
            use_hard_negative_loss: bool -- when enabled, the positive loss component is ignored when the hardest pos is too easy (e.g. pos_sim < neg_sim)
        """

        batch_sizes = embeddings.shape[0]
        embeddings = embeddings.reshape(-1, 92, 64)
        labels = labels.reshape(batch_sizes).to(device)

        # Normalize each vector (element) to have unit norm
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)  # Compute L2 norms
        embeddings = embeddings / norms  # Divide by norms to normalize
        
        # Compute pairwise cosine similarity
        if embeddings.dim() == 2:
            all_sim = torch.mm(embeddings, embeddings.t())

        elif embeddings.dim() == 3:
            all_sim = torch.matmul(embeddings.permute(1,0,2), embeddings.permute(1,2,0))

            if use_iq_mean:
                # interquartile mean
                lower_quant = torch.quantile(all_sim, 0.25, dim=0, keepdim=True)
                upper_quant = torch.quantile(all_sim, 0.75, dim=0, keepdim=True)
                mask = (all_sim > lower_quant) & (all_sim < upper_quant)
                all_sim = all_sim * mask
                all_sim = torch.sum(all_sim, dim=0) / torch.sum(mask, dim=0)
            else:
                # standard mean
                all_sim = all_sim.mean(0)

        # find hardest positive pairs (when positive has low sim)
        # mask of all valid positives
        mask_anc_pos = self._get_anc_pos_triplet_mask(labels)
        # prevent invalid pos by increasing sim
        anc_pos_sim = all_sim + (~mask_anc_pos * 999).float()
        # select minimum sim positives
        hardest_pos_sim = anc_pos_sim.min(dim=1, keepdim=True)[0]

        # find hardest negative triplets (when negative has high sim)
        # mask of all valid negatives
        mask_anc_neg = self._get_anc_neg_triplet_mask(labels).float()
        # set invalid negatives to 0
        anc_neg_sim = all_sim * mask_anc_neg
        # select maximum sim negatives
        hardest_neg_sim = anc_neg_sim.max(dim=1, keepdim=True)[0]

        if use_hard_negative_loss:
            # selective contrastive loss
            selective_idx = hardest_neg_sim > hardest_pos_sim
            hardest_pos_sim[selective_idx] = 0.

        loss = F.relu(hardest_neg_sim - hardest_pos_sim + self.margin)

        # calculate average loss (disregarding invalid & easy triplets)
        loss = torch.sum(loss) / (torch.gt(loss, 1e-16).float().sum() + 1e-16)

        return loss

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
            "dirs",
            "times",
            "interval_cumul_norm",
            "interval_rates",
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

        anchor = self.inflow_data[idx]
        positive = self.outflow_data[idx]
        negative = self.outflow_data[negative_idx]

        return anchor, positive, negative

    def reset_split(self):
        self.positive_top = not self.positive_top

        # Reshuffle the indices at the start of each epoch.
        random.shuffle(self.all_indices)

        # Re-divide the shuffled indices into two partitions.
        cutoff = len(self.all_indices) // 2
        self.partition_1 = self.all_indices[:cutoff]
        self.partition_2 = self.all_indices[cutoff:]

class OnlineTripletDataset(Dataset):
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
        self.size = len(self.all_indices)

    def __len__(self):
        return len(self.inflow_data)

    def __getitem__(self, idx):
        # pick a random inflow, outflow pair
        window_idx = -1

        trace_idx = torch.randint(low=0, high=self.size, size=(1,), dtype=torch.int32)
        anchor = self.inflow_data[trace_idx]
        positive = self.outflow_data[trace_idx]

        return anchor, positive, trace_idx

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
    anchors = [torch.tensor(emb, dtype=torch.float32) for emb in batch]
    positives = [torch.tensor(positive, dtype=torch.float32) for positive in positives]
    negatives = [torch.tensor(negative, dtype=torch.float32) for negative in negatives]
    
    # Stack tensors to create batched tensors
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    return anchors, positives, negatives

# Load the numpy arrays
train_inflows = np.load('data/train_inflows_may17_transformer.npy')
val_inflows = np.load('data/val_inflows_may17_transformer.npy')

train_outflows = np.load('data/train_outflows_may17_transformer.npy')
val_outflows = np.load('data/val_outflows_may17_transformer.npy')

# Define the datasets
train_dataset = OnlineTripletDataset(train_inflows, train_outflows)
val_dataset = OnlineTripletDataset(val_inflows, val_outflows)

train_sampler = QuadrupleSampler(train_dataset)
val_sampler = QuadrupleSampler(val_dataset)

# Create the dataloaders
batch_size = 250
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=16)

# Instantiate the models
embedding_size = 64
inflow_model = EspressoNet(8, special_toks=1, **model_config)
outflow_model = EspressoNet(8, special_toks=1, **model_config)

'''
checkpoint = torch.load('models/best_model_live_espresso_may17_fixed.pth')
inflow_model.load_state_dict(checkpoint['inflow_model_state_dict'])
outflow_model.load_state_dict(checkpoint['outflow_model_state_dict'])
'''

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inflow_model.to(device)
outflow_model.to(device)

# Define the loss function and the optimizer
criterion = OnlineHardCosineTripletLoss()
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
    for embeddings_anc, embeddings_pos, labels in train_loader:
        '''
        # Move tensors to the correct device
        anchor = anchor.float().to(device)
        positive = positive.float().to(device)
        negative = negative.float().to(device)

        #anchor_embeddings, anchor_chain = inflow_model(anchor)
        anchor_embeddings, anchor_chain = outflow_model(anchor[:,3:,:])
        positive_embeddings, positive_chain = outflow_model(positive[:,3:,:])
        negative_embeddings, negative_chain = outflow_model(negative[:,3:,:])
        '''
        if(embeddings_anc.shape[0] != 250):
            continue

        embeddings = torch.cat((embeddings_anc, embeddings_pos), dim=0)
        embeddings = embeddings.float().to(device)
        embeddings, chain = outflow_model(embeddings[:,3:,:])

        # Compute the loss
        #loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
        labels = torch.cat((labels, labels))
        loss = criterion(embeddings, labels)

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
        for embeddings_anc, embeddings_pos, labels in val_loader:
            # Move tensors to the correct device
            if(embeddings_anc.shape[0] != 250):
                continue
            '''
            anchor = anchor.float().to(device)
            positive = positive.float().to(device)
            negative = negative.float().to(device)

            # Forward pass
            #anchor_embeddings, anchor_chain = inflow_model(anchor[:,:,:])
            anchor_embeddings, anchor_chain = outflow_model(anchor[:,3:,:])
            positive_embeddings, positive_chain = outflow_model(positive[:,3:,:])
            negative_embeddings, negative_chain = outflow_model(negative[:,3:,:])
            '''
            embeddings = torch.cat((embeddings_anc, embeddings_pos), dim=0)
            embeddings = embeddings.float().to(device)
            embeddings, chain = outflow_model(embeddings[:,3:,:])

            # Compute the loss
            #loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            labels = torch.cat((labels, labels))
            loss = criterion(embeddings, labels)

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
        }, f'models/best_model_live_espresso_may17_test.pth')

