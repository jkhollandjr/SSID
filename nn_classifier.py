import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda")
train_dataset = np.load('similarity_train.npy')
train_labels = np.load('labels_train.npy')

test_dataset = np.load('similarity_test.npy')
test_labels = np.load('labels_test.npy')

class BinaryDataset(Dataset):
    def __init__(self, X, y, device):
        self.X = X
        self.y = y
        self.device = device

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        #data = torch.from_numpy(self.X[index]).float().to(self.device)
        #labels = torch.from_numpy(self.y[index]).float().to(self.device)
        return self.X[index], self.y[index]


train_dataset = BinaryDataset(train_dataset, train_labels, device)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = BinaryDataset(test_dataset, test_labels, device)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(11, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = BinaryClassifier()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(20):
    for i, (inputs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        inputs = inputs.float()
        labels = labels.float().unsqueeze(1)

        # Forward pass
        outputs = torch.sigmoid(model(inputs))
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print (f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

model.eval()
for i in range(20):
    TP, TN, FP, FN = 0,0,0,0
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_dataloader:
            inputs = inputs.float()
            labels = labels.float().unsqueeze(1)
            output = model(inputs)

            outputs = torch.round(torch.sigmoid(output))

            score = output
            threshold = i-5


            TP += ((score >= threshold) & (labels == 1)).sum().item()
            TN += ((score <= threshold) & (labels == 0)).sum().item()
            FP += ((score >= threshold) & (labels == 0)).sum().item()
            FN += ((score <= threshold) & (labels == 1)).sum().item()

            correct += (outputs == labels).sum().item()
            total += labels.shape[0]
        accuracy = correct / total
        print(f'Accuracy: {accuracy:.4f}')
        TPR = float(TP) / (TP + FN)
        FPR = float(FP) / (FP + TN)
        print('TPR: {}'.format(TPR))
        print('FPR: {}'.format(FPR))
        print(TP)
        print(TN)
        print(FP)
        print(FN)


