import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from model import Embedder
from orig_model import DFModel, DFModelWithAttention
import torch.nn.functional as F
from transdfnet import DFNet

class FlowDataset(Dataset):
    def __init__(self, inflows, outflows):
        self.inflows = inflows
        self.outflows = outflows
        assert len(inflows) == len(outflows), "Inflows and outflows must have the same number of examples"

    def __len__(self):
        return len(self.inflows)

    def __getitem__(self, index):
        correlated = random.choice([True, False])
        inflow = self.inflows[index]

        if correlated:
            outflow = self.outflows[index]
            target = 1
        else:
            random_index = (index + random.randint(1, len(self.outflows) - 1)) % len(self.outflows)
            outflow = self.outflows[random_index]
            target = 0

        combined_flow = np.concatenate((inflow, outflow), axis=1)
        combined_flow = torch.from_numpy(combined_flow).float()
        return combined_flow, target

model_config = {
        'input_size': 2000,
        #'input_size': 10000,
        #'input_size': 7000,

        'filter_grow_factor': 1.3,  # filter count scaling factor between stages
        'channel_up_factor': 12,    # filter count for each input channel (first stage)

        'conv_expand_factor': 2,    # filter count expansion ratio within a stage conv. block
        'conv_dropout_p': 0.0,       # dropout used inside conv. block
        'conv_skip': True,          # add skip connections for conv. blocks (after first stage)
        'depth_wise': True,         # use depth-wise convolutions in first stage
        'use_gelu': True,
        'stem_downproj': 0.5,

        'stage_count': 5,           # number of downsampling stages
        'kernel_size': 7,           # kernel size used by stage conv. blocks
        'pool_stride_size': 4,      # downsampling pool stride
        'pool_size': 7,             # downsampling pool width
        'block_dropout_p': 0.1,     # dropout used after each stage

        'mlp_hidden_dim': 128,

        'trans_depths': 2,  # number of transformer layers used in each stage
        'mhsa_kwargs': {            # transformer layer definitions
                        'head_dim': 10,
                        'use_conv_proj': True, 'kernel_size': 7, 'stride': 4,
                        'feedforward_style': 'mlp',
                        'feedforward_ratio': 3,
                        'feedforward_drop': 0.0,
                       },

        'feature_list': [ 
                            #'dirs', 
                            #'cumul', 
                            #'times', 
                            #'iats', 
                            'time_dirs', 
                            'times_norm', 
                            'cumul_norm', 
                            'iat_dirs', 
                            #'inv_iat_log_dirs', 
                            #'running_rates', 
                            #'running_rates_diff',
                ]
    }


model = DFNet(1, 4, **model_config) 
model.load_state_dict(torch.load('best_model_lb.pth'))
# Load the numpy arrays
train_inflows = np.load('train_inflows_dcf_12.npy')
val_inflows = np.load('val_inflows_dcf_12.npy')

train_outflows = np.load('train_outflows_dcf_12.npy')
val_outflows = np.load('val_outflows_dcf_12.npy')

batch_size = 128
train_dataset = FlowDataset(train_inflows.reshape(-1, 4, 1000), train_outflows.reshape(-1, 4, 1000))
val_dataset = FlowDataset(val_inflows.reshape(-1, 4, 1000), val_outflows.reshape(-1, 4, 1000))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the best models
#checkpoint = torch.load('best_model_12.pth')
#inflow_model.load_state_dict(checkpoint['inflow_model_state_dict'])
#outflow_model.load_state_dict(checkpoint['outflow_model_state_dict'])

# Define the loss function and the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.00001)
#optimizer = optim.SGD(list(inflow_model.parameters())+list(outflow_model.parameters()), lr=.001, weight_decay=1e-6, momentum=.9, nesterov=True)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    best_val_loss = float('inf')  # Initialize the best validation loss to a high value

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.float()

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.squeeze(outputs)  # Squeeze if necessary
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            predicted_train = outputs.round()
            correct_train += (predicted_train == targets).sum().item()
            total_train += targets.size(0)

        train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # Validation Phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.float()

                outputs = model(inputs)
                outputs = torch.squeeze(outputs)
                loss = criterion(outputs, targets)

                total_val_loss += loss.item()
                predicted_val = outputs.round()
                correct_val += (predicted_val == targets).sum().item()
                total_val += targets.size(0)

        val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val

        # Check if the current validation loss is the best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model
            torch.save(model.state_dict(), 'best_model_lb_tilted.pth')
            print(f'Epoch {epoch+1}: Validation loss improved to {val_loss:.4f}, saving model.')

        # Print metrics
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

pos_weight = torch.tensor([.25]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=2000)

