import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda")
train_dataset = np.load('fv_train.npy')
train_labels = np.load('fv_labels_train.npy')

#x, 11, 64
test_dataset = np.load('fv_test.npy')
test_labels = np.load('fv_labels_test.npy')

import torch
import torch.nn as nn
import numpy as np


class EmbeddingDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out_last = lstm_out[:, -1, :]
        fc_out = self.fc(lstm_out_last)
        sigmoid_out = self.sigmoid(fc_out)
        return sigmoid_out

# Convert the data and labels to PyTorch tensors
train_data = EmbeddingDataset(train_dataset, train_labels)
test_data = EmbeddingDataset(test_dataset, test_labels)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define the hyperparameters
input_dim = 64
hidden_dim = 128
output_dim = 1
learning_rate = 0.001
num_epochs = 10

# Instantiate the model and the optimizer
model = LSTMModel(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define the loss function
criterion = nn.BCELoss()

model.load_state_dict(torch.load('lstm.pth'))
'''
# Train the model
for epoch in range(num_epochs):
    for batch_data, batch_labels in train_dataloader:
        # Forward pass
        outputs = model(batch_data)[:,0]
        loss = criterion(outputs, batch_labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Print progress
    if (epoch+1) % 1 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

torch.save(model.state_dict(), 'lstm.pth')
model.load_state_dict(torch.load('lstm.pth'))
'''

model.eval()
test_loss = 0.0
test_accuracy = 0.0
thresholds = [.01, .05, .1, .25, .5, .75, .9, .95, .99,.995, .999]
for thr in thresholds:
    TP, TN, FP, FN = 0,0,0,0
    with torch.no_grad():
        for batch_data, batch_labels in test_dataloader:
            test_outputs = model(batch_data)[:,0]
            batch_loss = criterion(test_outputs, batch_labels)
            test_loss += batch_loss.item()

            TP += ((test_outputs >= thr) & (batch_labels == 1)).sum().item()
            TN += ((test_outputs <= thr) & (batch_labels == 0)).sum().item()
            FP += ((test_outputs >= thr) & (batch_labels == 0)).sum().item()
            FN += ((test_outputs <= thr) & (batch_labels == 1)).sum().item()

            test_preds = (test_outputs >= .5).float()
            batch_accuracy = (test_preds==batch_labels).float().mean()
            test_accuracy += batch_accuracy.item()

        TPR = float(TP) / (TP + FN)
        FPR = float(FP) / (FP + TN)
        print('TPR: {}'.format(TPR))
        print('FPR: {}\n'.format(FPR))

    test_loss /= len(test_dataloader)
    test_accuracy /= len(test_dataloader)

print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(test_loss, test_accuracy))


