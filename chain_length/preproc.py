import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import math
import os
from os.path import join
import pickle as pkl
from tqdm import tqdm
import transformers
import scipy
import json
import time
import argparse
from torch.utils.data import DataLoader

from utils.nets.transdfnet import DFNet
from utils.nets.espressonet import EspressoNet
from utils.layers import Mlp
from utils.processor import DataProcessor
from utils.data import *
from utils.loss import *

if __name__ == "__main__":
    
    features = [
                    "interval_dirs_up", 
                    "interval_dirs_down", 
                    "interval_dirs_sum",
                    "interval_dirs_sub",
                    "interval_iats",
                    "interval_inv_iat_logs",
                    "interval_cumul_norm",
                    "interval_times_norm",
                    ]
    
    
    # # # # # #
    # create data loaders
    # # # # # #
    # multi-channel feature processor
    processor = DataProcessor(features)

    pklpath = '../data/ssh/processed_nov17_fixtime.pkl'
    #pklpath = '../data/ssh_socat/processed_nov30.pkl'

    # chain-based sample splitting
    #te_idx = np.arange(0,1000)
    va_idx = np.arange(0,2000)
    tr_idx = np.arange(2000,10000)
    #tr_idx = np.arange(200,400)

    # stream window definitions
    window_kwargs = {
                    'window_count': 1, 
                    'window_width': 0, 
                    'window_overlap': 0,
                    'include_all_window': True,
                }
    data_kwargs = {
            'ends_only': True,
            #'host_only': True,
            #'stream_ID_range': (1,float('inf')),
            #'stream_ID_range': (0,1),
            'stream_ID_range': (1,-1),
            }

    def make_dataset(idx, **kwargs):
        """
        """
        dataset = BaseDataset(pklpath, processor,
                            window_kwargs = window_kwargs,
                            preproc_feats = False,
                            sample_idx = idx,
                            **kwargs,
                            )
        return dataset

    def pad_and_stack(windows, target_len=1200):
        """_summary_

        Args:
            windows (_type_): _description_

        Returns:
            _type_: _description_
        """
        np_windows = []
        for i in range(len(windows)):
            window = windows[i].numpy().T
            length = window.shape[-1]
            window = np.pad(window, ((0,0), (0,max(target_len - length,0))), 'constant')
            window = window[:,:target_len]
            np_windows.append(window)
        
        return np.stack(np_windows)

    def inflow_outflow_split(dataset):
        """_summary_

        Args:
            dataset (_type_): _description_

        Returns:
            _type_: _description_
        """
        inflow_samples = []
        outflow_samples = []
        
        for chain_ID in dataset.data_chain_IDs.keys():
            inflow_sample_ID = (chain_ID, min(dataset.data_chain_IDs[chain_ID]))
            outflow_sample_ID = (chain_ID, max(dataset.data_chain_IDs[chain_ID]))
            
            inflow_windows = dataset.data_windows[inflow_sample_ID]
            inflow_samples.append(pad_and_stack(inflow_windows))
            
            outflow_windows = dataset.data_windows[outflow_sample_ID]
            outflow_samples.append(pad_and_stack(outflow_windows))
            
        return np.stack(inflow_samples), np.stack(outflow_samples)
    
    va_data = make_dataset(va_idx, **data_kwargs)
    va_inflows, va_outflows = inflow_outflow_split(va_data)
    
    tr_data = make_dataset(tr_idx, **data_kwargs)
    tr_inflows, tr_outflows = inflow_outflow_split(tr_data)
    
    np.save('data/tr_inflows.npy', tr_inflows)
    np.save('data/va_inflows.npy', va_inflows)
    np.save('data/tr_outflows.npy', tr_outflows)
    np.save('data/va_outflows.npy', va_outflows)