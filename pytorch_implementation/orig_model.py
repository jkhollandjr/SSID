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
        self.block1_pool = nn.MaxPool1d(8, stride=3, padding=2)
        self.block1_dropout = nn.Dropout(0.1)

        self.block2_conv1 = nn.Conv1d(32, 64, 8, padding='same')
        self.block2_act1 = nn.ReLU()
        self.block2_conv2 = nn.Conv1d(64, 64, 8, padding='same')
        self.block2_act2 = nn.ReLU()
        self.block2_pool = nn.MaxPool1d(8, stride=3, padding=2)
        self.block2_dropout = nn.Dropout(0.1)

        self.block3_conv1 = nn.Conv1d(64, 128, 8, padding='same')
        self.block3_act1 = nn.ReLU()
        self.block3_conv2 = nn.Conv1d(128, 128, 8, padding='same')
        self.block3_act2 = nn.ReLU()
        self.block3_pool = nn.MaxPool1d(8, stride=3, padding=2)
        self.block3_dropout = nn.Dropout(0.1)

        self.block4_conv1 = nn.Conv1d(128, 256, 8, padding='same')
        self.block4_act1 = nn.ReLU()
        self.block4_conv2 = nn.Conv1d(256, 256, 8, padding='same')
        self.block4_act2 = nn.ReLU()
        self.block4_pool = nn.MaxPool1d(8, stride=3, padding=2)

        self.flatten = nn.Flatten()
        self.input_shape = input_shape

        self.dense = nn.Linear(2816, emb_size)

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

class DFModel_Alt(nn.Module):
    def __init__(self, input_shape=(4, 500), emb_size=64):
        super(DFModel_Alt, self).__init__()
        
        self.block1_conv1 = nn.Conv1d(input_shape[0], 32, 8, padding='same')
        self.block1_adv_act1 = nn.LeakyReLU()
        self.block1_conv2 = nn.Conv1d(32, 32, 8, padding='same')
        self.block1_adv_act2 = nn.LeakyReLU()
        self.block1_pool = nn.MaxPool1d(8, stride=3, padding=2)
        self.block1_dropout = nn.LeakyReLU()

        self.block2_conv1 = nn.Conv1d(32, 64, 8, padding='same')
        self.block2_act1 = nn.LeakyReLU()
        self.block2_conv2 = nn.Conv1d(64, 64, 8, padding='same')
        self.block2_act2 = nn.LeakyReLU()
        self.block2_pool = nn.MaxPool1d(8, stride=3, padding=2)
        self.block2_dropout = nn.Dropout(0.1)

        self.block3_conv1 = nn.Conv1d(64, 128, 8, padding='same')
        self.block3_act1 = nn.LeakyReLU()
        self.block3_conv2 = nn.Conv1d(128, 128, 8, padding='same')
        self.block3_act2 = nn.LeakyReLU()
        self.block3_pool = nn.MaxPool1d(8, stride=3, padding=2)
        self.block3_dropout = nn.Dropout(0.1)

        self.block4_conv1 = nn.Conv1d(128, 256, 8, padding='same')
        self.block4_act1 = nn.LeakyReLU()
        self.block4_conv2 = nn.Conv1d(256, 256, 8, padding='same')
        self.block4_act2 = nn.LeakyReLU()
        self.block4_pool = nn.MaxPool1d(8, stride=3, padding=2)

        self.flatten = nn.Flatten()
        self.input_shape = input_shape

        self.dense = nn.Linear(2816, emb_size)

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
import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        atten = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            atten = atten.masked_fill(mask == 0, float("-1e20"))

        atten = torch.nn.functional.softmax(atten / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [atten, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class DFModelWithAttention(nn.Module):
    def __init__(self, input_shape=(4, 1000), emb_size=64):
        super(DFModelWithAttention, self).__init__()

        kernel_size = 8
        padding = (kernel_size - 1) // 2
        
        # Convolutional Blocks
        self.block1_conv1 = nn.Conv1d(input_shape[0], 32, kernel_size, padding=padding)
        self.block1_adv_act1 = nn.ELU()
        self.block1_conv2 = nn.Conv1d(32, 32, kernel_size, padding=padding)
        self.block1_adv_act2 = nn.ELU()
        self.block1_pool = nn.MaxPool1d(8, stride=3, padding=2)
        self.block1_dropout = nn.Dropout(0.1)

        self.block2_conv1 = nn.Conv1d(32, 64, kernel_size, padding=padding)
        self.block2_act1 = nn.LeakyReLU()
        self.block2_conv2 = nn.Conv1d(64, 64, kernel_size, padding=padding)
        self.block2_act2 = nn.LeakyReLU()
        self.block2_pool = nn.MaxPool1d(8, stride=3, padding=2)
        self.block2_dropout = nn.Dropout(0.1)

        self.block3_conv1 = nn.Conv1d(64, 128, kernel_size, padding=padding)
        self.block3_act1 = nn.LeakyReLU()
        self.block3_conv2 = nn.Conv1d(128, 128, kernel_size, padding=padding)
        self.block3_act2 = nn.LeakyReLU()
        self.block3_pool = nn.MaxPool1d(8, stride=2, padding=2)

        '''
        self.block4_conv1 = nn.Conv1d(128, 256, kernel_size, padding=padding)
        self.block4_act1 = nn.LeakyReLU()
        self.block4_conv2 = nn.Conv1d(256, 256, kernel_size, padding=padding)
        self.block4_act2 = nn.LeakyReLU()
        self.block4_pool = nn.MaxPool1d(8, stride=3, padding=2)
        '''

        # Attention Layer
        self.attention = SelfAttention(embed_size=25, heads=5)
        
        # Flattening and Dense Layer
        self.flatten = nn.Flatten()
        self.input_shape = input_shape
        self.dense = nn.Linear(3200, emb_size)

    def forward(self, x):
        x = self.block1_pool(self.block1_adv_act2(self.block1_conv2(self.block1_adv_act1(self.block1_conv1(x)))))
        x = self.block1_dropout(x)
        x = self.block2_pool(self.block2_act2(self.block2_conv2(self.block2_act1(self.block2_conv1(x)))))
        x = self.block2_dropout(x)
        x = self.block3_pool(self.block3_act2(self.block3_conv2(self.block3_act1(self.block3_conv1(x)))))
        #x = self.block3_dropout(x)
        
        # Apply Self-Attention
        x = self.attention(x, x, x, mask=None)
        
        x = self.flatten(x)
        x = self.dense(x)
        return x

