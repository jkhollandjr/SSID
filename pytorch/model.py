import torch
import torch.nn as nn

class Embedder(nn.Module):
    def __init__(self, embedding_size):
        super(Embedder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm1d(32),
            nn.ELU(alpha=1.0),
            nn.Conv1d(32, 32, 8, 1, 0),
            nn.BatchNorm1d(32),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, 8, 1, 0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 8, 1, 0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, 8, 1, 0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 8, 1, 0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1),
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(128, 256, 8, 1, 0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 8, 1, 0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.out = nn.Sequential(
            nn.Linear(512, embedding_size),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        #x = self.block4(x)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.out(x)
        return output

