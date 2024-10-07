'''
python train_espresso.py \
    --train_inflows data/train_inflows.npy \
    --val_inflows data/val_inflows.npy \
    --train_outflows data/train_outflows.npy \
    --val_outflows data/val_outflows.npy \
    --save_model_path models/best_model.pth \
    --batch_size 128 \
    --num_epochs 200 \
    --device cuda \
    --learning_rate 0.0001
'''
import argparse
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler

from espresso import EspressoNet

torch.set_printoptions(threshold=5000)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class OnlineHardCosineTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(OnlineHardCosineTripletLoss, self).__init__()
        self.margin = margin

    def _get_anc_pos_triplet_mask(self, labels):
        """
        Return a 2D mask where mask[a, p] is True iff a and p are distinct and have the same label.

        Args:
            labels: torch.Tensor of dtype torch.int32 with shape [batch_size]

        Returns:
            mask: torch.Tensor of dtype torch.bool with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal

        # Check if labels[i] == labels[j]
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

        # Combine the two masks
        mask = indices_not_equal & labels_equal

        return mask

    def _get_anc_neg_triplet_mask(self, labels):
        """
        Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

        Args:
            labels: torch.Tensor of dtype torch.int32 with shape [batch_size]

        Returns:
            mask: torch.Tensor of dtype torch.bool with shape [batch_size, batch_size]
        """
        return labels.unsqueeze(0) != labels.unsqueeze(1)

    def forward(
        self,
        embeddings,
        labels,
        use_iq_mean=False,
        use_hard_negative_loss=True,
    ):
        """
        Args:
            embeddings: torch.Tensor -- batch of feature embeddings with shape [batch_size, features] or [batch_size, windows, features]
            labels: torch.Tensor of dtype torch.int32 with shape [batch_size]
            use_ciq_mean: bool -- use the mean of the interquartile range (exclude high and low quartiles from mean)
            use_hard_negative_loss: bool -- when enabled, the positive loss component is ignored when the hardest pos is too easy
        """
        batch_sizes = embeddings.shape[0]
        embeddings = embeddings.reshape(-1, 92, 64)
        labels = labels.reshape(batch_sizes).to(embeddings.device)

        # Normalize each vector (element) to have unit norm
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        embeddings = embeddings / norms

        # Compute pairwise cosine similarity
        if embeddings.dim() == 2:
            all_sim = torch.mm(embeddings, embeddings.t())
        elif embeddings.dim() == 3:
            all_sim = torch.matmul(
                embeddings.permute(1, 0, 2), embeddings.permute(1, 2, 0)
            )

            if use_iq_mean:
                # Interquartile mean
                lower_quant = torch.quantile(all_sim, 0.25, dim=0, keepdim=True)
                upper_quant = torch.quantile(all_sim, 0.75, dim=0, keepdim=True)
                mask = (all_sim > lower_quant) & (all_sim < upper_quant)
                all_sim = all_sim * mask
                all_sim = torch.sum(all_sim, dim=0) / torch.sum(mask, dim=0)
            else:
                # Standard mean
                all_sim = all_sim.mean(0)

        # Find hardest positive pairs (when positive has low sim)
        mask_anc_pos = self._get_anc_pos_triplet_mask(labels)
        anc_pos_sim = all_sim + (~mask_anc_pos * 999).float()
        hardest_pos_sim = anc_pos_sim.min(dim=1, keepdim=True)[0]

        # Find hardest negative triplets (when negative has high sim)
        mask_anc_neg = self._get_anc_neg_triplet_mask(labels).float()
        anc_neg_sim = all_sim * mask_anc_neg
        hardest_neg_sim = anc_neg_sim.max(dim=1, keepdim=True)[0]

        if use_hard_negative_loss:
            # Selective contrastive loss
            selective_idx = hardest_neg_sim > hardest_pos_sim
            hardest_pos_sim[selective_idx] = 0.0

        loss = F.relu(hardest_neg_sim - hardest_pos_sim + self.margin)

        # Calculate average loss (disregarding invalid & easy triplets)
        loss = torch.sum(loss) / (torch.gt(loss, 1e-16).float().sum() + 1e-16)

        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1):
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
        random.shuffle(self.all_indices)

        # Divide the shuffled indices into two partitions
        cutoff = len(self.all_indices) // 2
        self.partition_1 = self.all_indices[:cutoff]
        self.partition_2 = self.all_indices[cutoff:]

    def __len__(self):
        return len(self.inflow_data)

    def __getitem__(self, idx):
        # Choose a positive from partition 1 and a negative from partition 2 (or vice versa)
        if self.positive_top:
            idx = random.choice(self.partition_1)
            negative_idx = random.choice([j for j in self.partition_2 if j != idx])
        else:
            idx = random.choice(self.partition_2)
            negative_idx = random.choice([j for j in self.partition_1 if j != idx])

        anchor = self.inflow_data[idx]
        positive = self.outflow_data[idx]
        negative = self.outflow_data[negative_idx]

        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
            torch.tensor(negative, dtype=torch.float32),
        )

    def reset_split(self):
        self.positive_top = not self.positive_top

        # Reshuffle the indices at the start of each epoch
        random.shuffle(self.all_indices)

        # Re-divide the shuffled indices into two partitions
        cutoff = len(self.all_indices) // 2
        self.partition_1 = self.all_indices[:cutoff]
        self.partition_2 = self.all_indices[cutoff:]

class OnlineTripletDataset(Dataset):
    def __init__(self, inflow_data, outflow_data):
        self.positive_top = True
        self.inflow_data = inflow_data
        self.outflow_data = outflow_data
        self.all_indices = list(range(len(self.inflow_data)))
        random.shuffle(self.all_indices)
        self.size = len(self.all_indices)

    def __len__(self):
        return len(self.inflow_data)

    def __getitem__(self, idx):
        # Pick a random inflow, outflow pair
        trace_idx = torch.randint(low=0, high=self.size, size=(1,), dtype=torch.int32)
        anchor = self.inflow_data[trace_idx]
        positive = self.outflow_data[trace_idx]

        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
            trace_idx,
        )

    def reset_split(self):
        self.positive_top = not self.positive_top

        # Reshuffle the indices at the start of each epoch
        random.shuffle(self.all_indices)

        # Re-divide the shuffled indices into two partitions
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
        "feedforward_drop": 0.0,
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


def main():
    parser = argparse.ArgumentParser(description='Triplet Loss Training Script')
    parser.add_argument(
        '--train_inflows', type=str, default='data/train_inflows.npy', help='Path to train inflows numpy file'
    )
    parser.add_argument(
        '--val_inflows', type=str, default='data/val_inflows.npy', help='Path to validation inflows numpy file'
    )
    parser.add_argument(
        '--train_outflows', type=str, default='data/train_outflows.npy', help='Path to train outflows numpy file'
    )
    parser.add_argument(
        '--val_outflows', type=str, default='data/val_outflows.npy', help='Path to validation outflows numpy file'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=False, help='Path to model checkpoint file'
    )
    parser.add_argument(
        '--save_model_path',
        type=str,
        default='models/best_model.pth',
        help='Path to save the best model',
    )
    parser.add_argument(
        '--batch_size', type=int, default=128, help='Batch size for training and validation'
    )
    parser.add_argument(
        '--num_epochs', type=int, default=200, help='Total number of epochs to train'
    )
    parser.add_argument(
        '--switch_loss_type', type=int, default=0, help='Switch from triplet loss to online hard triplet loss')
    parser.add_argument(
        '--learning_rate', type=float, default=.00001, help='Initial learning rate for the optimizer')
    parser.add_argument(
        '--weight_decay', type=float, default=.01, help='Weight decay (L2 penalty) for the optimizer')
    parser.add_argument(
        '--device', type=str, default='cuda', help='Device to use for training (e.g., "cuda" or "cpu")'
    )
    args = parser.parse_args()

    # Load the numpy arrays
    train_inflows = np.load(args.train_inflows)
    val_inflows = np.load(args.val_inflows)
    train_outflows = np.load(args.train_outflows)
    val_outflows = np.load(args.val_outflows)

    # Define the datasets
    train_dataset_triplet = TripletDataset(train_inflows, train_outflows)
    val_dataset_triplet = TripletDataset(val_inflows, val_outflows)

    train_dataset_online = OnlineTripletDataset(train_inflows, train_outflows)
    val_dataset_online = OnlineTripletDataset(val_inflows, val_outflows)

    train_sampler_triplet = QuadrupleSampler(train_dataset_triplet)
    val_sampler_triplet = QuadrupleSampler(val_dataset_triplet)

    train_sampler_online = QuadrupleSampler(train_dataset_online)
    val_sampler_online = QuadrupleSampler(val_dataset_online)

    # Create the dataloaders
    train_loader_triplet = DataLoader(
        train_dataset_triplet,
        batch_size=args.batch_size,
        sampler=train_sampler_triplet,
        num_workers=16,
        pin_memory=True,
    )
    val_loader_triplet = DataLoader(
        val_dataset_triplet,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler_triplet,
        num_workers=16,
        pin_memory=True,
    )

    train_loader_online = DataLoader(
        train_dataset_online,
        batch_size=args.batch_size,
        sampler=train_sampler_online,
        num_workers=16,
        pin_memory=True,
    )
    val_loader_online = DataLoader(
        val_dataset_online,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler_online,
        num_workers=16,
        pin_memory=True,
    )

    # Instantiate the models
    inflow_model = EspressoNet(8, special_toks=1, **model_config)
    outflow_model = EspressoNet(8, special_toks=1, **model_config)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        inflow_model.load_state_dict(checkpoint['inflow_model_state_dict'])
        outflow_model.load_state_dict(checkpoint['outflow_model_state_dict'])

    # Move models to device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    inflow_model.to(device)
    outflow_model.to(device)

    # Define the optimizer with weight decay
    optimizer = optim.AdamW(
        list(inflow_model.parameters()) + list(outflow_model.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,  # Adjusted weight decay for better regularization
    )

    # Define the learning rate scheduler with warm-up
    num_epochs = args.num_epochs
    warmup_epochs = 10  # Number of warm-up epochs

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        else:
            return 0.5 * (
                1
                + math.cos(
                    math.pi
                    * (current_epoch - warmup_epochs)
                    / (num_epochs - warmup_epochs)
                )
            )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        if epoch < args.switch_loss_type:
            criterion = TripletLoss(margin=0.5)
            train_loader = train_loader_triplet
            val_loader = val_loader_triplet
            train_dataset_triplet.reset_split()
        else:
            criterion = OnlineHardCosineTripletLoss(margin=0.5)
            train_loader = train_loader_online
            val_loader = val_loader_online
            train_dataset_online.reset_split()

        # Training
        inflow_model.train()
        outflow_model.train()

        running_loss = 0.0
        for batch in train_loader:
            if epoch < args.switch_loss_type:
                # TripletDataset returns anchor, positive, negative
                anchor, positive, negative = batch
                # Move tensors to device
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                # Forward pass
                anchor_embeddings, _ = outflow_model(anchor[:, 3:, :])
                positive_embeddings, _ = outflow_model(positive[:, 3:, :])
                negative_embeddings, _ = outflow_model(negative[:, 3:, :])
                # Compute loss
                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            else:
                # OnlineTripletDataset returns embeddings_anc, embeddings_pos, labels
                embeddings_anc, embeddings_pos, labels = batch
                embeddings = torch.cat((embeddings_anc, embeddings_pos), dim=0)
                embeddings = embeddings.to(device)
                embeddings, _ = outflow_model(embeddings[:, 3:, :])
                labels = torch.cat((labels, labels)).to(device)
                loss = criterion(embeddings, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Step the scheduler after each epoch
        scheduler.step()

        # Validation
        inflow_model.eval()
        outflow_model.eval()

        running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if epoch < args.switch_loss_type:
                    # TripletDataset returns anchor, positive, negative
                    anchor, positive, negative = batch
                    # Move tensors to device
                    anchor = anchor.to(device)
                    positive = positive.to(device)
                    negative = negative.to(device)
                    # Forward pass
                    anchor_embeddings, _ = outflow_model(anchor[:, 3:, :])
                    positive_embeddings, _ = outflow_model(positive[:, 3:, :])
                    negative_embeddings, _ = outflow_model(negative[:, 3:, :])
                    # Compute loss
                    loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                else:
                    # OnlineTripletDataset returns embeddings_anc, embeddings_pos, labels
                    embeddings_anc, embeddings_pos, labels = batch
                    embeddings = torch.cat((embeddings_anc, embeddings_pos), dim=0)
                    embeddings = embeddings.to(device)
                    embeddings, _ = outflow_model(embeddings[:, 3:, :])
                    labels = torch.cat((labels, labels)).to(device)
                    loss = criterion(embeddings, labels)

                running_loss += loss.item()

        val_loss = running_loss / len(val_loader)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f'Epoch {epoch + 1}/{num_epochs}, LR: {current_lr:.6f}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}'
        )

        if epoch == args.switch_loss_type:
            best_val_loss = float("inf")

        # Save the model if it's the best one so far
        if val_loss < best_val_loss:
            print("Best model so far!")
            best_val_loss = val_loss
            torch.save(
                {
                    'epoch': epoch,
                    'inflow_model_state_dict': inflow_model.state_dict(),
                    'outflow_model_state_dict': outflow_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                },
                args.save_model_path,
            )


if __name__ == '__main__':
    main()

