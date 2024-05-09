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

#from chaindf import DFNet
from utils.nets.transdfnet import DFNet
from utils.data import BaseDataset
from utils.processor import DataProcessor



# enable if NaN or other odd behavior appears
#torch.autograd.set_detect_anomaly(True)
# disable any unnecessary logging / debugging
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# auto-optimize cudnn ops
torch.backends.cudnn.benchmark = True



def collate_and_pad(batch):
    """
    convert samples to tensors and pad samples to equal length
    """
    # convert labels to tensor and get sequence lengths
    batch_windows, batch_chainlengths, batch_ids = zip(*batch)
    batch_x = [windows[0] for windows in batch_windows]
    batch_y = [chainlength for chainlength in batch_chainlengths]

    batch_y = torch.tensor(batch_y)

    # pad and fix dimension
    batch_x = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True, padding_value=0.)
    if len(batch_x.shape) < 3:  # add channel dimension if missing
        batch_x = batch_x.unsqueeze(-1)
    batch_x = batch_x.permute(0,2,1)

    return batch_x.float(), batch_y.long(),



if __name__ == "__main__":
    """
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load checkpoint (if it exists)
    checkpoint_path = None
    checkpoint_fname = None
    resumed = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        resumed = torch.load(checkpoint_path)
        checkpoint_fname = os.path.basename(os.path.dirname(checkpoint_path))
    # else: checkpoint path and fname will be defined later if missing

    model_name = "DF"
    checkpoint_dir = './ckpt'
    results_dir = './res'
    eval_only = False

    # # # # # #
    # finetune config
    # # # # # #
    mini_batch_size = 64   # samples to fit on GPU
    batch_size = 64        # when to update model
    accum = batch_size // mini_batch_size
    # # # # # #
    warmup_period   = 5
    ckpt_period     = 5
    epochs          = 30
    opt_lr          = 15e-4
    opt_betas       = (0.9, 0.999)
    opt_wd          = 0.01

    # all trainable network parameters
    params = []

    # DF model config
    if resumed:
        model_config = resumed['config']
    else:
        model_config = {
                'input_size': 1000,
                'channel_up_factor': 22,
                'filter_grow_factor': 1.3,
                'stem_downproj': 0.5,
                'stage_count': 3,
                'depth_wise': True,
                'use_gelu': True,
                'conv_skip': True,
                'kernel_size': 7,
                'pool_stride_size': 3,
                'pool_size': 7,
                'mlp_hidden_dim': 512,
                'conv_expand_factor': 3,
                'conv_dropout_p': 0.,
                'block_dropout_p': 0.1,
                "trans_depths": 3,
                "mhsa_kwargs": {
                    "head_dim": 10,
                    "use_conv_proj": True,
                    "kernel_size": 3,
                    "stride": 2,
                    "feedforward_style": "mlp",
                    "feedforward_ratio": 3,
                    "feedforward_drop": 0.0
                },
            }

    print("==> Model configuration:")
    print(json.dumps(model_config, indent=4))


    # # # # # #
    # create data loaders
    # # # # # #
    pklpath = '../data/ssh/processed_nov17_fixtime.pkl'
    #pklpath = '../data/ssh_socat/processed_nov30.pkl'

    # chain-based sample splitting
    te_idx = np.arange(0,1000)
    tr_idx = np.arange(1000,10000)

    # multi-channel feature processor
    #features = ('iat_dirs','inv_iat_log_dirs', 'time_dirs', 'inv_iat_logs', 'iats', 'times')
    features = ('inv_iat_logs', 'iats', 'times', 'sizes', 'running_rates', 'burst_edges')
    processor = DataProcessor(features)

    tr_data = BaseDataset(pklpath, processor,
                        window_kwargs = None,
                        preproc_feats = True,
                        sample_idx = tr_idx,
                        host_only = True)
                        #stream_ID_range = (1,float('inf')))
                        #stream_ID_range = (1,2))
                        #stream_ID_range = (-3,-2))
    trainloader = DataLoader(tr_data,
                            batch_size=mini_batch_size, 
                            collate_fn=collate_and_pad,
                            shuffle=True)
    te_data = BaseDataset(pklpath, processor,
                        window_kwargs = None,
                        preproc_feats = True,
                        sample_idx = te_idx,
                        host_only = True)
                        #stream_ID_range = (1,float('inf')))
                        #stream_ID_range = (1,2))
                        #stream_ID_range = (-3,-2))
    testloader = DataLoader(te_data, 
                            batch_size=mini_batch_size, 
                            collate_fn=collate_and_pad,
                            shuffle=True)
    

    # # # # # #
    # define base metaformer model
    # # # # # #
    net = DFNet(num_classes=2, 
                input_channels=len(features),
                **model_config)
    net = net.to(device)
    if resumed:
        net_state_dict = resumed['model']
        net.load_state_dict(net_state_dict)
    params += net.parameters()

    # # # # # #
    # optimizer and params, reload from resume is possible
    # # # # # #
    optimizer = optim.AdamW(params,
            lr=opt_lr, betas=opt_betas, weight_decay=opt_wd)
    if resumed and resumed.get('opt', None):
        opt_state_dict = resumed['opt']
        optimizer.load_state_dict(opt_state_dict)

    last_epoch = -1
    if resumed and resumed['epoch']:    # if resuming from a finetuning checkpoint
        last_epoch = resumed['epoch']
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = len(trainloader) * warmup_period // accum,
                                                    num_training_steps = len(trainloader) * epochs // accum,
                                                    num_cycles = 0.5,
                                                    last_epoch = ((last_epoch+1) * len(trainloader) // accum) - 1,
                                                )

    # define checkpoint fname if not provided
    if not checkpoint_fname:
        checkpoint_fname = f'{model_name}'
        checkpoint_fname += f'_{time.strftime("%Y%m%d-%H%M%S")}'

    # create checkpoint directory if necesary
    if not os.path.exists(f'{checkpoint_dir}/{checkpoint_fname}/'):
        try:
            os.makedirs(f'{checkpoint_dir}/{checkpoint_fname}/')
        except:
            pass
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except:
            pass

    # # # # # #
    # print parameter count of metaformer model (head not included)
    param_count = sum(p.numel() for p in params if p.requires_grad)
    param_count /= 1000000
    param_count = round(param_count, 2)
    print(f'=> Model is {param_count}m parameters large.')
    # # # # # #


    #criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss(beta=1.0)

    def train_iter(i, eval_only=False):
        """
        """
        train_loss = 0.
        down_acc = 0
        up_acc = 0
        n = 0
        with tqdm(trainloader,
                desc=f"Epoch {i} Train [lr={scheduler.get_last_lr()[0]:.2e}]",
                dynamic_ncols=True) as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)

                # # # # # #
                # DF prediction
                pred = net(inputs)
                #pred = torch.clamp(pred,min=1)
                loss = criterion(pred, targets)

                train_loss += loss.item()

                loss /= accum   # normalize to full batch size

                up_pred = pred[:,1]
                up_targets = targets[:,1]
                up_length_pred = torch.round(up_pred)
                up_acc += torch.sum(up_length_pred == up_targets).item()

                down_pred = pred[:,0]
                down_targets = targets[:,0]
                down_length_pred = torch.round(down_pred)
                down_acc += torch.sum(down_length_pred == down_targets).item()

                #_, y_pred = torch.max(cls_pred, 1)
                #train_acc += torch.sum(y_pred == targets).item()
                n += len(targets)

                loss.backward()

                if not eval_only:
                    # update weights, update scheduler, and reset optimizer after a full batch is completed
                    if (batch_idx+1) % accum == 0 or batch_idx+1 == len(trainloader):
                        optimizer.step()
                        scheduler.step()
                        #optimizer.zero_grad()
                        for param in params:
                            param.grad = None

                pbar.set_postfix({
                                  'down_acc': down_acc/n,
                                  'up_acc': up_acc/n,
                                  'loss': train_loss/(batch_idx+1),
                                  })
                pbar.set_description(f"Epoch {i} Train [lr={scheduler.get_last_lr()[0]:.2e}]")

        train_loss /= batch_idx + 1
        #train_acc /= n
        return train_loss#, train_acc


    def test_iter(i):
        """
        """
        test_loss = 0.
        up_acc = 0
        down_acc = 0
        test_acc2 = {}
        test_acc3 = {}
        test_acc4 = {}
        n = 0
        with tqdm(testloader, desc=f"Epoch {i} Test", dynamic_ncols=True) as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)

                # # # # # #
                # DF prediction
                pred = net(inputs)
                #pred = torch.clamp(pred,min=1)
                loss = criterion(pred, targets)

                test_loss += loss.item()

                #pred = pred[:,1]
                #targets = targets[:,1]
                #length_pred = torch.round(pred)
                #correct = length_pred == targets
                #test_acc += torch.sum(correct).item()

                up_pred = pred[:,1]
                up_targets = targets[:,1]
                up_length_pred = torch.round(up_pred)
                up_acc += torch.sum(up_length_pred == up_targets).item()

                down_pred = pred[:,0]
                down_targets = targets[:,0]
                down_length_pred = torch.round(down_pred)
                down_acc += torch.sum(down_length_pred == down_targets).item()

                n += len(targets)

                #soft_res = F.softmax(cls_pred, dim=1)
                #y_prob, y_pred = soft_res.max(1)
                #test_acc += torch.sum(y_pred == targets).item()
                #for i in range(len(targets)):
                #    lab = int(targets[i].item())
                #    err = abs((targets[i] - pred[i]).item())
                #    cor = correct[i].item()
                #    test_acc2[lab] = test_acc2.get(lab, 0) + cor
                #    test_acc3[lab] = test_acc3.get(lab, 0) + 1
                #    test_acc4[lab] = test_acc4.get(lab, []) + [err]

                pbar.set_postfix({
                                  'down_acc': down_acc/n,
                                  'up_acc': up_acc/n,
                                  'loss': test_loss/(batch_idx+1),
                                })

        #keys = sorted(list(test_acc2.keys()))
        #for key in keys:
        #    print(f'\t{key} ({test_acc3[key] / n:0.2f}): {test_acc2[key] / test_acc3[key]:0.3f} - {np.mean(test_acc4[key]):0.3f}|{np.std(test_acc4[key]):0.3f}')

        test_loss /= batch_idx + 1
        #test_acc /= n
        return test_loss#, test_acc


    # run eval only
    if eval_only:
        if resumed:
            net.eval()

            epoch = -1
            train_loss = train_iter(epoch, eval_only=True)
            print(f'[{epoch}] tr. loss ({train_loss:0.3f})')
            test_loss = test_iter(epoch)
            print(f'[{epoch}] te. loss ({test_loss:0.3f})')
        else:
            print(f'Could not load checkpoint [{checkpoint_path}]: Path does not exist')

    # do training
    else:
        history = {}
        try:
            for epoch in range(last_epoch+1, epochs):

                net.train()
                train_loss = train_iter(epoch)
                metrics = {'tr_loss': train_loss}

                if testloader is not None:
                    net.eval()
                    with torch.no_grad():
                        test_loss = test_iter(epoch)
                    metrics.update({'te_loss': test_loss})

                    if (epoch % ckpt_period) == (ckpt_period-1):
                        # save last checkpoint before restart
                        checkpoint_path_epoch = f"{checkpoint_dir}/{checkpoint_fname}/e{epoch}.pth"
                        print(f"Saving end-of-cycle checkpoint to {checkpoint_path_epoch}...")
                        torch.save({
                                        "epoch": epoch,
                                        "model": net.state_dict(),
                                        "opt": optimizer.state_dict(),
                                        "config": model_config,
                                }, checkpoint_path_epoch)

                history[epoch] = metrics

        except KeyboardInterrupt:
            pass

        finally:
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            results_fp = f'{results_dir}/{checkpoint_fname}.txt'
            with open(results_fp, 'w') as fi:
                json.dump(history, fi, indent='\t')

