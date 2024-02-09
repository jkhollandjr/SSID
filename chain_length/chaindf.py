"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.ops.deform_conv import deform_conv2d
from functools import partial



class DeformConv1d(nn.Module):
    """

    Modified to support the masking of padding and 1D sequences
    """
    def __init__(
            self, in_channels, out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias: bool = True,
            offset_kernel = 1,
            drop_p = 0.,
            **kwargs,
    ):
        super(DeformConv1d, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.offset_kernel = offset_kernel

        self.dropout = nn.Dropout(drop_p)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.offset_conv = nn.Conv1d(in_channels, kernel_size,
                                     kernel_size = offset_kernel,
                                     padding = offset_kernel // 2,
                                    )
        self.masking_conv = nn.Conv1d(in_channels, kernel_size,
                                      kernel_size = offset_kernel,
                                      padding = offset_kernel // 2,
                                    )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.offset_conv.weight.data.fill_(0.)
        self.offset_conv.bias.data.fill_(0.)
        self.masking_conv.weight.data.fill_(0.)
        self.masking_conv.bias.data.fill_(0.0)

    def forward(self, x):
        """
        ATM along one dimension, the shape will not be changed
        x: [B, C, N]
        """
        B, C, N = x.size()

        offset = self.offset_conv(x)
        mask = F.sigmoid(self.masking_conv(x))

        # zero offsets for unsqueezed dimension
        offset_t = torch.zeros(B, 2 * self.kernel_size, N,
                               dtype = x.dtype,
                               layout = x.layout,
                               device = x.device)
        offset_t[:, 0::2, :] += offset

        x = self.dropout(x)

        x = deform_conv2d(
            x.unsqueeze(-1), offset_t.unsqueeze(-1),
            self.weight, self.bias,
            stride = self.stride,
            padding = (self.padding, 0),
            mask = mask.unsqueeze(-1),
        ).squeeze(-1)

        return x



class DFNet(nn.Module):
    def __init__(self, num_classes, input_channels,
                       channel_up_factor = 32,
                       filter_grow_factor = 2,
                       stage_count = 4,
                       input_size = 5000,
                       depth_wise = True,
                       kernel_size = 7,
                       pool_stride_size = 4,
                       pool_size = 7,
                       mlp_hidden_dim = 1024,
                       conv_expand_factor = 1,
                       block_dropout_p = 0.1,
                       deformable = False,
                    **kwargs):
        super(DFNet, self).__init__()

        self.input_size = input_size
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.pool_stride_size = pool_stride_size
        self.pool_size = pool_size

        self.use_deform = deformable


        self.block_dropout_p = block_dropout_p
        self.mlp_dropout_p = (0.7, 0.5)
        self.filter_grow_factor = filter_grow_factor

        self.init_filters = input_channels * channel_up_factor
        self.filter_nums = [int(self.init_filters*(self.filter_grow_factor**i)) for i in range(stage_count)]

        self.blocks = nn.ModuleList([self.__block(input_channels, self.filter_nums[0], nn.GELU(),
                                                  depth_wise = True,
                                                  expand_factor = conv_expand_factor,
                                                  drop_p = self.block_dropout_p)])
        if stage_count > 1:
            for i in range(1, stage_count):
                block = self.__block(self.filter_nums[i-1], self.filter_nums[i], nn.GELU(),
                                     depth_wise = False,
                                     deformable = self.use_deform,
                                     expand_factor = conv_expand_factor,
                                     drop_p = self.block_dropout_p)
                self.blocks.append(block)
        self.max_pool = nn.MaxPool1d(self.pool_size,
                                     stride = self.pool_stride_size,
                                     padding = self.pool_size//2)
        self.dropout = nn.Dropout(p=self.block_dropout_p)      # point-wise dropout

        # calculate flattened conv output size
        self.fmap_size = self.__fmap_size(self.input_size)
        self.fc_in_features = self.fmap_size * self.filter_nums[-1]  # flattened dim = fmap_size * fmap_count

        self.fc_size = mlp_hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, self.fc_size),
            nn.BatchNorm1d(self.fc_size),
            nn.GELU(),
            nn.Dropout(self.mlp_dropout_p[0]),
            nn.Linear(self.fc_size, self.fc_size),
            nn.BatchNorm1d(self.fc_size),
            nn.GELU(),
            nn.Dropout(self.mlp_dropout_p[1])
        )
        self.fc_out_fcount = self.fc_size

        self.multi_pred = nn.Sequential(
            nn.Linear(self.fc_out_fcount, num_classes),
            # when using CrossEntropyLoss, already computed internally
            #nn.Softmax(dim=1) # dim = 1, don't softmax batch
        )

    def __fmap_size(self, input_size):
        fmap_size = input_size
        for i in range(len(self.filter_nums)):
            fmap_size = int((fmap_size - self.pool_size + 2*(self.pool_size//2)) / self.pool_stride_size) + 1
        return fmap_size

    def __block(self, channels_in, channels, activation,
                depth_wise=False,
                expand_factor=2,
                drop_p=0.1,
                deformable=False,
        ):
        deform_conv = partial(DeformConv1d,
                    kernel_size = self.kernel_size,
                    padding = self.kernel_size // 2,
                    offset_kernel = self.kernel_size,
                )
        conv = partial(nn.Conv1d,
                    kernel_size = self.kernel_size,
                    padding = self.kernel_size // 2,
                    groups = channels_in if depth_wise else 1
                )

        return nn.Sequential(
            #conv(channels_in, channels*expand_factor),
            deform_conv(channels_in, channels*expand_factor) if deformable else conv(channels_in, channels*expand_factor),
            nn.BatchNorm1d(channels*expand_factor),
            activation,
            nn.Dropout(p = drop_p if not deformable else 0.),
            deform_conv(channels*expand_factor, channels, drop_p = drop_p) if deformable else conv(channels*expand_factor, channels),
            nn.BatchNorm1d(channels),
            activation,
        )

    def features(self, x):
        for block in self.blocks:
            x = block(x)
            x = self.max_pool(x)
            x = self.dropout(x)
        x = x.flatten(start_dim=1) # dim = 1, don't flatten batch
        return x

    def forward(self, x,
            *args, **kwargs):
        # add channel dim if necessary
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        # clip sample length to maximum supported size
        size_dif = x.shape[-1] - self.input_size
        if x.shape[-1] > self.input_size:
            x = x[..., :self.input_size]
        elif size_dif < 0:
            x = F.pad(x, (0,abs(size_dif)))

        x = self.features(x, **kwargs)
        g = self.fc(x)
        y_pred = self.multi_pred(g)
        return y_pred.squeeze()
