import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers import *
from transdfnet import TransformerBlock
from functools import partial
from timm.layers import trunc_normal_, DropPath


DEFAULT_IN_CONV_KWARGS = {
                            'kernel_size': 3, 
                            'stride': 3,
                            'padding': '0',
                          }
DEFAULT_OUT_CONV_KWARGS = {
                            'kernel_size': 30, 
                            'stride': 30,
                            'padding': '0',
                            }

class EspressoNet(nn.Module):
    """
    """
    def __init__(self, input_channels, 
                       input_size = 900, 
                       depth = 0,
                       hidden_dim = 128,
                       feature_dim = 64,
                       special_toks = 0,
                       mhsa_kwargs = {},
                       input_conv_kwargs = DEFAULT_IN_CONV_KWARGS,
                       output_conv_kwargs = DEFAULT_OUT_CONV_KWARGS,
                    **kwargs):
        super(EspressoNet, self).__init__()

        # # # #
        self.input_channels = input_channels
        self.input_conv_kwargs = input_conv_kwargs
        self.output_conv_kwargs = output_conv_kwargs

        self.input_size = input_size

        # mixing op for transformer blocks
        mhsa_mixer = partial(MHSAttention,)
        mhsa_kwargs = mhsa_kwargs
        self.mixer = partial(mhsa_mixer, **mhsa_kwargs)

        self.depth = depth
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.special_toks = special_toks

        if self.special_toks > 0:
            toks = nn.init.xavier_uniform_(torch.empty(self.hidden_dim, self.special_toks))
            self.extra_tokens = nn.Parameter(toks)
            self.special_fc = nn.Linear(self.hidden_dim, self.feature_dim)

        # construct the model using the selected params
        self.__build_model()
        # # # #

    def __build_model(self):
        """Construct the model layers
        """
        # define first conv. layer that handles feature embedding
        conv_embed = nn.Conv1d(self.input_channels, self.hidden_dim, 
                               **self.input_conv_kwargs)
        block_list = [conv_embed]

        # add transformer layers if they are enabled for the stage
        block_func = partial(TransformerBlock, 
                                    dim = self.hidden_dim, 
                                    token_mixer = self.mixer,
                                    skip_toks=self.special_toks,
                             )
        block_list += [block_func() for _ in range(self.depth)]

        self.blocks = nn.ModuleList(block_list)

        self.windowing = nn.Conv1d(self.hidden_dim, self.hidden_dim * 4,
                                   **self.output_conv_kwargs)

        self.pred = nn.Sequential(
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim * 4),
                nn.Linear(self.hidden_dim * 4, self.feature_dim),
                )

    def forward(self, x, 
            sample_sizes = None,
            *args, **kwargs):
        """forward input features through the model
        """

        # add channel dim if necessary
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        # clip sample length to maximum supported size and pad with zeros if necessary
        size_dif = x.shape[-1] - self.input_size
        if x.shape[-1] > self.input_size:
            x = x[..., :self.input_size]
        elif size_dif < 0:
            x = F.pad(x, (0,abs(size_dif)))

        # padding can be ignored during self-attention if configured with pad_masks
        # note: padding-aware self-attention does not seem to be improve performance, but reduces computation efficiency
        pad_masks = None
        if sample_sizes is not None:
            pad_masks = torch.stack([torch.cat((torch.zeros(min(s, self.input_size)), 
                                                torch.ones(max(self.input_size-s, 0)))) for s in sample_sizes])
            pad_masks = pad_masks.to(x.get_device())
            pad_masks = pad_masks.unsqueeze(1)

        # apply conv. and transformer layers
        x = self.blocks[0](x)

        if self.special_toks > 0:
            special_toks_in = self.extra_tokens.unsqueeze(0).expand(x.shape[0],-1,-1)
            x = torch.cat((special_toks_in, x), dim=2)

        for i,block in enumerate(self.blocks[1:]):
            x = block(x)

        if self.special_toks > 0:
            special_toks_out = x[:,:,:self.special_toks].permute(0,2,1)
            special_toks_out = self.special_fc(special_toks_out).permute(0,2,1)

            x = x[:,:,self.special_toks:]

        # apply windowing feature prediction
        x = self.windowing(x).permute(0,2,1)
        pred_windows = self.pred(x).permute(0,2,1)

        if self.special_toks > 0:
            return pred_windows, special_toks_out.flatten(start_dim=1)
        else:
            return pred_windows
