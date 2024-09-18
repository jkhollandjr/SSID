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
from sklearn.metrics import roc_curve

# Load the data
val_data = np.load('dcf_val_distances_dual.npy')
test_data = np.load('dcf_test_distances_dual.npy')

# Split the data into inputs and targets
val_inputs = val_data[:, :24]
val_targets = val_data[:, 24]
test_inputs = test_data[:, :24]
test_targets = test_data[:, 24]

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
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# Instantiate the model and move it to GPU if available
model = Predictor()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the loss function and the optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Training loop
num_epochs = 75
for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        # Move tensors to the correct device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            # Move tensors to the correct device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))

            # Compute the number of correct predictions
            preds = outputs >= 0.5
            correct_predictions += (preds == targets.unsqueeze(1)).sum().item()
            total_predictions += targets.size(0)

            running_loss += loss.item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = correct_predictions / total_predictions

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Put the model in evaluation mode
model.eval()

# Lists to store the model's outputs and the actual targets
outputs_list = []
targets_list = []

# Pass the validation data through the model
with torch.no_grad():
    for inputs, targets in val_loader:
        # Move tensors to the correct device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Store the outputs and targets
        outputs_list.extend(outputs.cpu().numpy())
        targets_list.extend(targets.cpu().numpy())

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(targets_list, outputs_list, drop_intermediate=True)

new = np.array([.8, .7, .6, .5, .4, .3, .2, .1])
thresholds = np.concatenate([new, thresholds])
print(thresholds)
counter = 0
for threshold in thresholds:
    counter += 1
    if counter == 110:
        continue
    # Convert the probabilities to binary outputs
    preds = (np.array(outputs_list) >= threshold).astype(int)

    # Compute the confusion matrix
    cm = confusion_matrix(targets_list, preds)

    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    # Calculate the rates
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (TN + FP)
    FNR = FN / (TP + FN)

    print(f"Threshold: {threshold:.7f}")
    print(f"True Positives: {TP}, True Negatives: {TN}, False Positives: {FP}, False Negatives: {FN}")
    print(f"True Positive Rate: {TPR:.7f}, False Positive Rate: {FPR:.7f}, True Negative Rate: {TNR:.7f}, False Negative Rate: {FNR:.7f}\n")

tpr = [str(x) for x in list(tpr)]
fpr = [str(x) for x in list(fpr)]

print(','.join(tpr))
print(','.join(fpr))
