import torch
from torch import nn
from torch.nn import functional as F

class DFModel(nn.Module):
    def __init__(self, input_shape=(4, 1000), emb_size=64):
        super(DFModel, self).__init__()
        
        self.block1_conv1 = nn.Conv1d(input_shape[0], 32, 8, padding='same')
        self.block1_adv_act1 = nn.ELU()
        self.block1_conv2 = nn.Conv1d(32, 32, 8, padding='same')
        self.block1_adv_act2 = nn.ELU()
        self.block1_pool = nn.MaxPool1d(8, stride=4, padding=2)
        self.block1_dropout = nn.Dropout(0.1)

        self.block2_conv1 = nn.Conv1d(32, 64, 8, padding='same')
        self.block2_act1 = nn.ReLU()
        self.block2_conv2 = nn.Conv1d(64, 64, 8, padding='same')
        self.block2_act2 = nn.ReLU()
        self.block2_pool = nn.MaxPool1d(8, stride=4, padding=2)
        self.block2_dropout = nn.Dropout(0.1)

        self.block3_conv1 = nn.Conv1d(64, 128, 8, padding='same')
        self.block3_act1 = nn.ReLU()
        self.block3_conv2 = nn.Conv1d(128, 128, 8, padding='same')
        self.block3_act2 = nn.ReLU()
        self.block3_pool = nn.MaxPool1d(8, stride=4, padding=2)
        self.block3_dropout = nn.Dropout(0.1)

        self.block4_conv1 = nn.Conv1d(128, 256, 8, padding='same')
        self.block4_act1 = nn.ReLU()
        self.block4_conv2 = nn.Conv1d(256, 256, 8, padding='same')
        self.block4_act2 = nn.ReLU()
        self.block4_pool = nn.MaxPool1d(8, stride=4, padding=2)

        self.flatten = nn.Flatten()
        self.input_shape = input_shape

        self.dense = nn.Linear(self._get_flattened_size(), emb_size)

    def _get_flattened_size(self):
        x = torch.zeros(1, *self.input_shape)
        x = self.block1_pool(self.block1_conv2(self.block1_conv1(x)))
        x = self.block2_pool(self.block2_conv2(self.block2_conv1(x)))
        x = self.block3_pool(self.block3_conv2(self.block3_conv1(x)))
        x = self.block4_pool(self.block4_conv2(self.block4_conv1(x)))
        x = self.flatten(x)
        return x.size(1)

    def forward(self, x):
        x = self.block1_pool(self.block1_adv_act2(self.block1_conv2(self.block1_adv_act1(self.block1_conv1(x)))))
        x = self.block1_dropout(x)
        x = self.block2_pool(self.block2_act2(self.block2_conv2(self.block2_act1(self.block2_conv1(x)))))
        x = self.block2_dropout(x)
        x = self.block3_pool(self.block3_act2(self.block3_conv2(self.block3_act1(self.block3_conv1(x)))))
        x = self.block3_dropout(x)
        x = self.block4_pool(self.block4_act2(self.block4_conv2(self.block4_act1(self.block4_conv1(x)))))

        x = self.flatten(x)
        x = self.dense(x)
        return x

