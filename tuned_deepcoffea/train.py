import os
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import argparse
from orig_model import DFModel, DFModelWithAttention
import torch.nn.functional as F
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model for inflow and outflow data.")
    parser.add_argument('--train_dir', type=str, default='data/', help="Directory containing the training .npy files.")
    parser.add_argument('--val_dir', type=str, default='data/', help="Directory containing the validation .npy files.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--triplet_margin', type=float, default=0.1, help="Margin for the triplet loss function.")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Initial learning rate.")
    parser.add_argument('--num_epochs', type=int, default=2000, help="Number of epochs to train.")
    parser.add_argument('--model_name', type=str, default='best_model.pth', help="Name of the saved model file.")
    return parser.parse_args()

# Define the learning rate schedule function
def lr_schedule(epoch, num_epochs, initial_lr):
    if epoch < 100:
        # Warmup phase
        return initial_lr * (epoch / 100)
    else:
        # Cosine annealing
        return initial_lr * 0.5 * (1 + math.cos(math.pi * (epoch - 100) / (num_epochs - 100)))

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
    def __init__(self, data_source):
        self.data_source = data_source
    
    def __iter__(self):
        indices = list(range(len(self.data_source))) * 4
        np.random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return 4 * len(self.data_source)

def main():
    args = parse_args()

    # Load the numpy arrays
    train_inflows = np.load(os.path.join(args.train_dir, 'train_inflows.npy'))
    val_inflows = np.load(os.path.join(args.val_dir, 'val_inflows.npy'))
    train_outflows = np.load(os.path.join(args.train_dir, 'train_outflows.npy'))
    val_outflows = np.load(os.path.join(args.val_dir, 'val_outflows.npy'))

    # Define the datasets
    train_dataset = TripletDataset(train_inflows, train_outflows)
    val_dataset = TripletDataset(val_inflows, val_outflows)

    train_sampler = QuadrupleSampler(train_dataset)
    val_sampler = QuadrupleSampler(val_dataset)

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    # Instantiate the models
    inflow_model = DFModel()
    outflow_model = DFModel()

    # Move models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inflow_model.to(device)
    outflow_model.to(device)

    # Define the loss function and the optimizer
    criterion = CosineTripletLoss(margin=args.triplet_margin)
    optimizer = optim.Adam(list(inflow_model.parameters()) + list(outflow_model.parameters()), lr=args.learning_rate)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(args.num_epochs):
        lr = lr_schedule(epoch, args.num_epochs, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        train_dataset.reset_split()
        val_dataset.reset_split()

        # Training
        inflow_model.train()
        outflow_model.train()

        running_loss = 0.0
        for anchor, positive, negative in train_loader:
            anchor = anchor.float().to(device)
            positive = positive.float().to(device)
            negative = negative.float().to(device)

            anchor_embeddings = inflow_model(anchor)
            positive_embeddings = outflow_model(positive)
            negative_embeddings = outflow_model(negative)

            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

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
                anchor = anchor.float().to(device)
                positive = positive.float().to(device)
                negative = negative.float().to(device)

                anchor_embeddings = inflow_model(anchor)
                positive_embeddings = outflow_model(positive)
                negative_embeddings = outflow_model(negative)

                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

                running_loss += loss.item()

        val_loss = running_loss / len(val_loader)

        print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

        # Save the model if it's the best one so far
        if val_loss < best_val_loss:
            print("Best model so far!")
            best_val_loss = val_loss
            save_path = os.path.join('models', args.model_name)
            torch.save({
                'epoch': epoch,
                'inflow_model_state_dict': inflow_model.state_dict(),
                'outflow_model_state_dict': outflow_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, save_path)

if __name__ == "__main__":
    main()

