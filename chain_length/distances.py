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
from torch.utils.data import DataLoader

from transdfnet import DFNet
from layers import Mlp
from data import BaseDataset, TripletDataset, PairwiseDataset
from processor import DataProcessor
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
    head = Mlp(dim=feature_dim*2, out_features=2)
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
    va_data = PairwiseDataset(va_data,
                              sample_mode = 'undersample',
                              sample_ratio = 5)

    # test dataloader
    te_data = BaseDataset(pklpath, processor,
                        window_kwargs = window_kwargs,
                        preproc_feats = False,
                        sample_idx = te_idx,
                        host_only = True,
                        #stream_ID_range = (1,float('inf')),
                        #stream_ID_range = (0,1)
                        )
    te_data = PairwiseDataset(te_data, 
                              sample_mode = 'undersample',
                              sample_ratio = 1000
            )


    def func(t):
        t = torch.nn.utils.rnn.pad_sequence(t, 
                                        batch_first=True, 
                                        padding_value=0.)
        return t.permute(0,2,1).float().to(device)
    va_data.set_fen(fen, func)
    te_data.set_fen(fen, func)


    #def make_embeds(dataset):

    #    embed_vectors = []
    #    chain_ids = []

    #    with tqdm(dataset,
    #            dynamic_ncols = True) as pbar:

    #        for windows, chain_length, ID in pbar:
    #            windows = torch.nn.utils.rnn.pad_sequence(windows, 
    #                                                batch_first=True, 
    #                                                padding_value=0.)
    #            windows = windows.permute(0,2,1).float().to(device)

    #            embeds = fen(windows)

    #            embed_vectors.append(embeds.numpy(force=True))
    #            chain_ids.append(ID[0])

    #    return embed_vectors, chain_ids


    def compute_distances(flow1, flow2):
        """
        sims = []
        for i in range(len(flow1)):
            a = torch.tensor(flow1[i])
            b = torch.tensor(flow2[i])
            cosine_similarity = F.cosine_similarity(a, b, dim=0)
            cosine_similarity = cosine_similarity.item()  # Convert tensor to single number
            sims.append(cosine_similarity)
        return sims
        """
        cosine_similarity = F.cosine_similarity(flow1, flow2, dim=1)
        cosine_similarity = cosine_similarity.numpy(force=True)  # Convert tensor to single number
        return cosine_similarity


    #def build_mat(vecs, ids):
    #    # Generate combinations for the new validation set
    #    output_array = []
    #    for i in range(len(vecs)):
    #        print(f"Processing set, trace {i+1}/{len(vecs)}")
    #        for j in range(i, len(vecs)):
    #            distances = compute_distances(torch.tensor(vecs[i]), torch.tensor(vecs[j]))
    #            match = int(ids[i] == ids[j])
    #            #val_output_array[i * len(val_outflows) + j, :11] = distances
    #            #val_output_array[i * len(val_outflows) + j, 11] = match
    #            output_array.append(np.concatenate((distances, [match])))
    #    output_array = np.concatenate(output_array)
    #    return output_array
    #
    #va_vecs, va_ids = make_embeds(va_data)
    #val_output_array = build_mat(va_vecs, va_ids)
    #
    #te_vecs, te_ids = make_embeds(te_data)
    #test_output_array = build_mat(te_vecs, te_ids)

    def build_dists(dataset):
        output_array = []
        for sample1, sample2, match in tqdm(dataset):
            embeds1 = sample1[0]
            embeds2 = sample2[0]
            distances = compute_distances(embeds1, embeds2)
            output_array.append(np.concatenate((distances, [int(match)])))
        return np.stack(output_array)

    val_output_array = build_dists(va_data)
    test_output_array = build_dists(te_data)


    # Save the numpy arrays to files
    np.save('dcf_val_distances.npy', val_output_array)
    np.save('dcf_test_distances.npy', test_output_array)

    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data.dataset import random_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from sklearn import metrics
    from tqdm import tqdm
    
    # Load the data
    val_data = val_output_array
    test_data = test_output_array
    
    # Split the data into inputs and targets
    #window_count = 12
    val_inputs = val_data[:, :-1]
    val_targets = val_data[:, -1]
    test_inputs = test_data[:, :-1]
    test_targets = test_data[:, -1]
    
    # Create PyTorch datasets
    class MyDataset(Dataset):
        def __init__(self, inputs, targets):
            self.inputs = inputs
            self.targets = targets
    
        def __len__(self):
            return len(self.inputs)
    
        def __getitem__(self, idx):
            return torch.tensor(self.inputs[idx], dtype=torch.float), torch.tensor(self.targets[idx], dtype=torch.float)
    
    train_dataset = MyDataset(val_inputs, val_targets)
    val_dataset = MyDataset(test_inputs, test_targets)
    
    # Create PyTorch dataloaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    class Predictor(nn.Module):
        def __init__(self, dim=64, drop=0.5):
            super(Predictor, self).__init__()
            self.fc1 = nn.Linear(val_inputs.shape[-1], dim)
            self.fc2 = nn.Linear(dim, dim)
            self.fc3 = nn.Linear(dim, dim)
            self.fc4 = nn.Linear(dim, 1)
            self.dropout = nn.Dropout(drop)
    
        def forward(self, x):
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            x = self.dropout(F.relu(self.fc3(x)))
            x = torch.sigmoid(self.fc4(x))
            return x
    
    # Instantiate the model and move it to GPU if available
    model = Predictor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Define the loss function and the optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for inputs, targets in tqdm(train_loader):
            # Move tensors to the correct device
            inputs, targets = inputs.to(device), targets.to(device)
    
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            preds = outputs >= 0.5
            correct_predictions += (preds == targets.unsqueeze(1)).sum().item()
            total_predictions += targets.size(0)
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        train_loss = running_loss / len(train_loader)
        train_acc = correct_predictions / total_predictions
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
    
    # Put the model in evaluation mode
    model.eval()
    
    # Lists to store the model's outputs and the actual targets
    outputs_list = []
    targets_list = []
    
    # Pass the validation data through the model
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            # Move tensors to the correct device
            inputs, targets = inputs.to(device), targets.to(device)
    
            # Forward pass
            outputs = model(inputs)
    
            # Store the outputs and targets
            outputs_list.extend(outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
    
    # Compute the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(targets_list, outputs_list)
    roc_auc = metrics.auc(fpr, tpr)
    
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.6f' % roc_auc)
    plt.legend(loc = 'lower right')
    #plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.000001, .1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xscale('log')
    plt.savefig('roc.pdf')


    """
    def calc_dists(dataset):
        embed_vectors = []
        chain_ids = []

        with tqdm(dataset,
                dynamic_ncols = True) as pbar:

            for windows, chain_length, ID in pbar:
                windows = torch.nn.utils.rnn.pad_sequence(windows, 
                                                    batch_first=True, 
                                                    padding_value=0.)
                windows = windows.permute(0,2,1).float().to(device)

                embeds = fen(windows)

                embed_vectors.append(embeds.numpy(force=True))
                chain_ids.append(ID[0])

        embed_vectors = np.stack(embed_vectors)

        # calculate distances between windows of all samples
        windows = embed_vectors.shape[1]
        window_dists = []
        for i in range(windows):
            X = embed_vectors[:,i,:]
            dists = pairwise_distances(X,metric='cosine')
            window_dists.append(dists)
        window_dists = np.stack(window_dists)

        ground_truths = []
        for i,cur_chain_id in enumerate(chain_ids):
            correlated = np.array([cur_chain_id == chain_id for chain_id in chain_ids])
            ground_truths.append(correlated)
        ground_truths = np.stack(ground_truths)

        return window_dists, ground_truths


    distances, ground_truths = calc_dists(te_data)
    print(distances.shape)


    # use window cosine distances to compute correlation performance metrics
    results = []
    thresholds = np.linspace(0., 1., num=10, endpoint=False)


    total_samples = (ground_truths.size - ground_truths.shape[0]) / 2
    total_positives = np.sum(ground_truths)
    total_negatives = total_samples - total_positives
    print(total_samples)
    print(total_positives)
    print(total_negatives)

    for dist_threshold in thresholds:
        window_correlation = distances < dist_threshold
        corr_counts = np.sum(window_correlation, axis=0)

        for window_threshold in range(distances.shape[0]//2, distances.shape[0]+1):
            full_correlation = corr_counts > window_threshold

            positives = np.tril(full_correlation)[ground_truths]
            TP = np.sum(positives)
            FN = len(positives) - TP

            negatives = np.tril(full_correlation)[~ground_truths]
            FP = np.sum(negatives)
            TN = total_negatives - FP

            precision = TP / (TP + FP) if TP+FP > 0 else 0
            recall = TP / (TP + FN) if TP+FN > 0 else 0
            f1 = 2 / ((1/precision) + (1/recall)) if precision + recall > 0 else 0

            results.append({'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 
                            'pre': precision, 'rec': recall, 'f1': f1,
                            'threshold': dist_threshold, 'windows': window_threshold})
            
    for result in results:
        print(result)
    # save results to file
    #if not os.path.exists(results_dir):
    #    os.makedirs(results_dir)
    #results_fp = f'{results_dir}/{checkpoint_fname}.txt'
    #results_fp = 'distances_results.json'
    #with open(results_fp, 'w') as fi:
    #    json.dump(results, fi, indent='\t')
    #print(json.dumps(results[best_idx], indent=4))
    """

