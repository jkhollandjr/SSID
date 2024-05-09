import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import math
import os
from os.path import join
import pickle as pkl
from tqdm import tqdm
from torchvision import transforms, utils
import transformers
import scipy
import json
import time
import argparse
from torch.utils.data import DataLoader, Dataset

from utils.nets.transdfnet import DFNet
from utils.layers import Mlp
from utils.data import BaseDataset, TripletDataset, PairwiseDataset
from utils.processor import DataProcessor
from sklearn.metrics.pairwise import pairwise_distances



# enable if NaN or other odd behavior appears
#torch.autograd.set_detect_anomaly(True)
# disable any unnecessary logging / debugging
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# auto-optimize cudnn ops
torch.backends.cudnn.benchmark = True



def parse_args():
    parser = argparse.ArgumentParser(
                        #prog = 'WF Benchmark',
                        #description = 'Train & evaluate WF attack model.',
                        #epilog = 'Text at the bottom of help'
                        )

    # experiment configuration options
    parser.add_argument('--ckpt', 
                        default = None, 
                        type = str,
                        help = "Resume from checkpoint path.", 
                        required=True)

    return parser.parse_args()


if __name__ == "__main__":
    """
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()

    # load checkpoint (if it exists)
    checkpoint_path = args.ckpt
    checkpoint_fname = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        resumed = torch.load(checkpoint_path)
        checkpoint_fname = os.path.basename(os.path.dirname(checkpoint_path))
    else:
        print("Failed to load model checkpoint!")
        sys.exit(-1)
    # else: checkpoint path and fname will be defined later if missing

    model_name = "DF"

    model_config = resumed['config']
    features = model_config['features']
    feature_dim = model_config['feature_dim']
    print(json.dumps(model_config, indent=4))

    # traffic feature extractor
    fen = DFNet(feature_dim, len(features),
                **model_config)
    fen = fen.to(device)
    fen_state_dict = resumed['fen']
    fen.load_state_dict(fen_state_dict)
    fen.eval()

    # chain length prediction head
    head = Mlp(dim=feature_dim, out_features=2)
    head = head.to(device)
    head_state_dict = resumed['chain_head']
    head.load_state_dict(head_state_dict)
    head.eval()

    # # # # # #
    # create data loaders
    # # # # # #

    # multi-channel feature processor
    processor = DataProcessor(features)

    pklpath = '../data/ssh/processed_nov17_fixtime.pkl'
    #pklpath = '../data/ssh_socat/processed_nov30.pkl'

    # chain-based sample splitting
    te_idx = np.arange(0,1000)
    va_idx = np.arange(1000,2000)

    # stream window definitions
    window_kwargs = model_config['window_kwargs']

    # train dataloader
    va_data = BaseDataset(pklpath, processor,
                        window_kwargs = window_kwargs,
                        preproc_feats = False,
                        sample_idx = va_idx,
                        host_only = True,
                        #stream_ID_range = (1,float('inf'))
                        #stream_ID_range = (0,1)
                        )

    # test dataloader
    te_data = BaseDataset(pklpath, processor,
                        window_kwargs = window_kwargs,
                        preproc_feats = False,
                        sample_idx = te_idx,
                        host_only = True,
                        #stream_ID_range = (1,float('inf')),
                        #stream_ID_range = (0,1)
                        )


    def func(t):
        t = torch.nn.utils.rnn.pad_sequence(t, 
                                        batch_first=True, 
                                        padding_value=0.)
        return t.permute(0,2,1).float().to(device)

    def make_embed_data(data):
        all_embeds = []
        all_labels = []
        fen.eval()
        with torch.no_grad():
            for windows,chain_label,sample_ID in data:
                windows = func(windows)
                out = fen(windows)
                out = head(out)
                out = out.cpu().detach()
                #chain_label = torch.tensor(chain_label)

                all_embeds.append(out.flatten())
                all_labels.append(chain_label)

        return torch.stack(all_embeds), torch.stack(all_labels)


    va_embeds, va_targets = make_embed_data(va_data)
    te_embeds, te_targets = make_embed_data(te_data)

    print(va_embeds.size(1))

    # chain length prediction head
    head = Mlp(dim=va_embeds.size(1), out_features=2)
    head = head.to(device)

    opt_lr          = 1e-3
    opt_betas       = (0.9, 0.999)
    opt_wd          = 0.001
    optimizer = optim.AdamW(head.parameters(),
            lr=opt_lr, betas=opt_betas, weight_decay=opt_wd)


    # Create PyTorch datasets
    class MyDataset(Dataset):
        def __init__(self, inputs, targets):
            self.inputs = inputs
            self.targets = targets
        def __len__(self):
            return len(self.inputs)
        def __getitem__(self, idx):
            return self.inputs[idx], self.targets[idx]

    va_dataset = MyDataset(va_embeds, va_targets)
    te_dataset = MyDataset(te_embeds, te_targets)
    
    # Create PyTorch dataloaders
    batch_size = 128
    va_loader = DataLoader(va_dataset, batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(te_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.SmoothL1Loss(beta=1.0)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Training
        fen.train()
        running_loss = 0.0

        up_acc = 0 
        down_acc = 0
        n = 0

        for inputs, targets in tqdm(va_loader):
            # Move tensors to the correct device
            inputs, targets = inputs.to(device), targets.to(device)
    
            # Forward pass
            pred = head(inputs)
            loss = criterion(pred, targets.unsqueeze(1))

            #
            # accuracy predicted as all-or-nothing
            #
            length_pred = torch.round(pred[:,0])
            up_acc += torch.sum(length_pred == targets[:,0]).item()

            length_pred = torch.round(pred[:,1])
            down_acc += torch.sum(length_pred == targets[:,1]).item()

            acc = (up_acc + down_acc)/2
            n += len(targets)
            #
            # # # # #
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        train_loss = running_loss / len(va_loader)
        train_acc_up = up_acc / n
        train_acc_down = down_acc / n
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc_up:.4f}|{train_acc_down:.4f}')
    
    
    # Put the model in evaluation mode
    head.eval()
    
    up_acc = 0
    down_acc = 0
    n = 0
    # Pass the validation data through the model
    with torch.no_grad():
        for inputs, targets in tqdm(te_loader):
            # Move tensors to the correct device
            inputs, targets = inputs.to(device), targets.to(device)
    
            # Forward pass
            pred = head(inputs)

            #
            # accuracy predicted as all-or-nothing
            #
            length_pred = torch.round(pred[:,0])
            up_acc += torch.sum(length_pred == targets[:,0]).item()

            length_pred = torch.round(pred[:,1])
            down_acc += torch.sum(length_pred == targets[:,1]).item()

            acc = (up_acc + down_acc)/2
            n += len(targets)
            #
            # # # # #
    print(f'Test Accuracy: {up_acc / n:.4f}|{down_acc / n:.4f}')
