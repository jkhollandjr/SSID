import numpy as np
import torch
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from transdfnet import DFNet
from orig_model import DFModel

# Function to process a batch of flows
def process_flows(inflows, outflows):
    batch_size = inflows.shape[0]
    predictions = []

    for i in range(12):  # Iterate through each window
        sub_inflows = inflows[:, i, :, :]  # Shape (batch_size, 4, 1000)
        sub_outflows = outflows[:, i, :, :]  # Shape (batch_size, 4, 1000)

        # Concatenate the sub-flows
        combined_flows = np.concatenate((sub_inflows, sub_outflows), axis=2)
        combined_flow_tensors = torch.from_numpy(combined_flows).float().to(device)

        with torch.no_grad():
            outputs = model(combined_flow_tensors)
            #batch_predictions = torch.round(outputs.squeeze()).cpu().numpy()
            batch_predictions = (outputs.squeeze() > 0.50).float()
            predictions.append(batch_predictions)

    
    final_predictions = torch.sum(torch.stack(predictions), axis=0) >= y
    return final_predictions.cpu().numpy()

# Load validation arrays
val_inflows = np.load('val_inflows_dcf_12.npy')  # Shape (x, 12, 4, 1000)
val_outflows = np.load('val_outflows_dcf_12.npy')  # Shape (x, 12, 4, 1000)

# Define a threshold for making the final prediction
y = 12 # Adjust this threshold as needed

# Model configuration and instantiation
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
#model = DFModel()
model.load_state_dict(torch.load('best_model_lb_tilted.pth'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define ratio of uncorrelated to correlated flows
uncorrelated_to_correlated_ratio = 10
num_correlated = len(val_inflows)
num_uncorrelated = num_correlated * uncorrelated_to_correlated_ratio

# Create ground truth labels
batch_size = 32
uncorrelated_batches = 10000
ground_truth_labels = [1] * num_correlated + [0] * uncorrelated_batches * batch_size
predictions = []

# Function to create batches
def create_batches(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

# Create batches
inflow_batches = create_batches(val_inflows, batch_size)
outflow_batches = create_batches(val_outflows, batch_size)

# Process correlated flow pairs
for inflow_batch, outflow_batch in zip(inflow_batches, outflow_batches):
    batch_predictions = process_flows(np.array(inflow_batch), np.array(outflow_batch))
    predictions.extend(batch_predictions)

# Function to generate a random non-correlated pair
def get_random_non_correlated_pair(length):
    idx1 = random.randint(0, length - 1)
    idx2 = random.randint(0, length - 2)
    if idx2 == idx1:
        idx2 += 1
    return idx1, idx2

# Process uncorrelated flow pairs
for i in range(uncorrelated_batches):
    if i % 100 == 0:
        print(i)
    inflow_indices, outflow_indices = zip(*[get_random_non_correlated_pair(len(val_inflows)) for _ in range(batch_size)])
    inflow_batch = val_inflows[list(inflow_indices)]
    outflow_batch = val_outflows[list(outflow_indices)]
    batch_predictions = process_flows(np.array(inflow_batch), np.array(outflow_batch))
    predictions.extend(batch_predictions)

# Calculate performance metrics
accuracy = accuracy_score(ground_truth_labels[:len(predictions)], predictions)
precision = precision_score(ground_truth_labels[:len(predictions)], predictions)
recall = recall_score(ground_truth_labels[:len(predictions)], predictions)
tn, fp, fn, tp = confusion_matrix(ground_truth_labels[:len(predictions)], predictions).ravel()
true_positive_rate = tp / (tp + fn)
false_positive_rate = fp / (fp + tn)

print(f'False positives: {fp}')
print(f'True positives: {tp}')
print(f'False negatives: {fn}')
print(f'True negatives: {tn}')

# Print metrics
print(f'Accuracy: {accuracy:.6f}')
print(f'Precision: {precision:.6f}')
print(f'Recall: {recall:.6f}')
print(f'True Positive Rate: {true_positive_rate:.6f}')
print(f'False Positive Rate: {false_positive_rate:.6f}')
