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



# enable if NaN or other odd behavior appears
#torch.autograd.set_detect_anomaly(True)
# disable any unnecessary logging / debugging
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# auto-optimize cudnn ops
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
                        #prog = 'WF Benchmark',
                        #description = 'Train & evaluate WF attack model.',
                        #epilog = 'Text at the bottom of help'
                        )

    # experiment configuration options
    #parser.add_argument('--data_dir', 
    #                    default = './data', 
    #                    type = str,
    #                    help = "Set root data directory.")
    parser.add_argument('--ckpt_dir',
                        default = './checkpoint',
                        type = str,
                        help = "Set directory for model checkpoints.")
    parser.add_argument('--results_dir', 
                        default = './results',
                        type = str,
                        help = "Set directory for result logs.")
    parser.add_argument('--ckpt', 
                        default = None, 
                        type = str,
                        help = "Resume from checkpoint path.")
    parser.add_argument('--exp_name',
                        type = str,
                        default = f'{time.strftime("%Y%m%d-%H%M%S")}',
                        help = "")
    parser.add_argument('--online', 
                        default=False, action='store_true',
                        help = "Use online semi-hard triplet mining.")
    parser.add_argument('--loss_margin', default=0.1, type=float,
                        help = "Loss margin for triplet learning.")
    parser.add_argument('--w', default=0.1,
                        help = "Weight placed on the chain-loss of multi-task loss.")

    # Model architecture options
    parser.add_argument('--config',
                        default = None,
                        type = str,
                        help = "Set model config (as JSON file)")
    parser.add_argument('--input_size', 
                        default = None, 
                        type = int,
                        help = "Overwrite the config .json input length parameter.")
    parser.add_argument('--features', 
                        default=None, type=str, nargs="+",
                        help='Overwrite the features used in the config file. Multiple features can be provided.')

    return parser.parse_args()


if __name__ == "__main__":
    """
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()

    # load checkpoint (if it exists)
    checkpoint_path = args.ckpt
    checkpoint_fname = None
    resumed = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        resumed = torch.load(checkpoint_path)
        checkpoint_fname = os.path.basename(os.path.dirname(checkpoint_path))
    # else: checkpoint path and fname will be defined later if missing

    model_name = "DF"
    checkpoint_dir = args.ckpt_dir
    results_dir = args.results_dir

    # # # # # #
    # finetune config
    # # # # # #
    mini_batch_size = 128   # samples to fit on GPU
    batch_size = 128        # when to update model
    accum = batch_size // mini_batch_size
    # # # # # #
    warmup_period   = 10
    ckpt_period     = 30
    epochs          = 300
    opt_lr          = 1e-3
    opt_betas       = (0.9, 0.999)
    opt_wd          = 0.001
    save_best_epoch = True
    loss_margin = args.loss_margin
    loss_delta = float(args.w)

    # all trainable network parameters
    params = []

    # DF model config
    if resumed:
        model_config = resumed['config']
    elif args.config:
        with open(args.config, 'r') as fi:
            model_config = json.load(fi)
    else:
        model_config = {
                'input_size': 1200,
                'feature_dim': 64,
                'stage_count': 3,
                #'channel_up_factor': 24,
                "features": [
                    "iats", 
                    "sizes", 
                    "burst_edges",
                    ],
                #"window_kwargs": {
                #     'window_count': 1, 
                #     'window_width': 0, 
                #     'window_overlap': 0,
                #     "include_all_window": True,
                #     },
                "features": [
                    "interval_dirs_up", 
                    "interval_dirs_down", 
                    "interval_dirs_sum",
                    "interval_dirs_sub",
                    ],
                #"window_kwargs": {
                #     'window_count': 12, 
                #     'window_width': 6, 
                #     'window_overlap': 2,
                #     "include_all_window": True,
                #     },
                "window_kwargs": {
                     'window_count': 1, 
                     'window_width': 0, 
                     'window_overlap': 0,
                     "include_all_window": True,
                     },
            }

    if args.input_size is not None:
        model_config['input_size'] = args.input_size
    if args.features is not None:
        model_config['features'] = args.features
    features = model_config['features']

    feature_dim = model_config['feature_dim']

    print("==> Model configuration:")
    print(json.dumps(model_config, indent=4))


    # traffic feature extractor
    fen = DFNet(feature_dim, len(features),
                **model_config)
    fen = fen.to(device)
    if resumed:
        fen_state_dict = resumed['fen']
        fen.load_state_dict(fen_state_dict)
    params += fen.parameters()

    # chain length prediction head
    head = Mlp(dim=feature_dim, out_features=2)
    head = head.to(device)
    if resumed:
        head_state_dict = resumed['chain_head']
        head.load_state_dict(head_state_dict)
    params += head.parameters()

    # # # # # #
    # print parameter count of metaformer model (head not included)
    param_count = sum(p.numel() for p in params if p.requires_grad)
    param_count /= 1000000
    param_count = round(param_count, 2)
    print(f'=> Model is {param_count}m parameters large.')
    # # # # # #


    # # # # # #
    # create data loaders
    # # # # # #
    # multi-channel feature processor
    processor = DataProcessor(features)

    pklpath = '../data/ssh/processed_nov17_fixtime.pkl'
    #pklpath = '../data/ssh_socat/processed_nov30.pkl'

    # chain-based sample splitting
    te_idx = np.arange(0,1000)
    va_idx = np.arange(1000,2000)
    tr_idx = np.arange(2000,10000)

    # stream window definitions
    window_kwargs = model_config['window_kwargs']
    data_kwargs = {
            'host_only': True,
            #'stream_ID_range': (1,float('inf')),
            #'stream_ID_range': (0,1),
            }

    def make_dataloader(idx, **kwargs):
        """
        """
        dataset = BaseDataset(pklpath, processor,
                            window_kwargs = window_kwargs,
                            #preproc_feats = False,
                            preproc_feats = True,
                            sample_idx = idx,
                            **kwargs,
                            )
        if args.online:
            dataset = OnlineDataset(dataset, k=2)

        else:
            dataset = TripletDataset(dataset)
        loader = DataLoader(dataset,
                            batch_size=mini_batch_size, 
                            collate_fn=dataset.batchify,
                            shuffle=True)
        return loader, dataset


    # prepare train data
    trainloader, tr_data = make_dataloader(tr_idx, **data_kwargs)

    # prepare validation data (if enabled)
    if save_best_epoch:
        validationloader, va_data = make_dataloader(va_idx, **data_kwargs)
    else:
        validationloader = None

    # prepare test data
    testloader, te_data = make_dataloader(te_idx, **data_kwargs)
    

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

    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, 
                                                                num_warmup_steps = warmup_period * len(trainloader), 
                                                                num_training_steps = epochs * len(trainloader), 
                                                                num_cycles = epochs // ckpt_period,
                                                                #last_epoch = last_epoch * len(trainloader) if last_epoch
                                                                )

    # define checkpoint fname if not provided
    if not checkpoint_fname:
        checkpoint_fname = f'{model_name}'
        checkpoint_fname += f'_{args.exp_name}'

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


    #criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss(beta=1.0)
    if args.online:
        triplet_criterion = OnlineCosineTripletLoss(margin=loss_margin,
                                                    semihard=False)
    else:
        triplet_criterion = CosineTripletLoss(margin=loss_margin)


    def epoch_iter(dataloader, 
                    eval_only = False, 
                    desc = f"Epoch"):
        """
        Step through one epoch on the dataset
        """
        tot_loss = 0.
        trip_loss = 0.

        acc = 0
        up_acc = 0
        down_acc = 0

        n = 0
        with tqdm(dataloader,
                desc = desc,
                dynamic_ncols = True) as pbar:
            for batch_idx, data in enumerate(pbar):

                # online loss variant
                if args.online:
                    inputs, labels, targets = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    targets = targets.to(device)

                    embed = fen(inputs)

                    triplet_loss = triplet_criterion(embed, labels)
                    trip_loss += triplet_loss.item()

                    pred = head(embed)
                    chain_loss = criterion(pred, targets)


                # offline / random triplets
                else:
                    inputs_anc, inputs_pos, inputs_neg, targets = data
                    inputs_anc = inputs_anc.to(device)
                    inputs_pos = inputs_pos.to(device)
                    inputs_neg = inputs_neg.to(device)
                    targets = targets.to(device)

                    # # # # # #
                    # generate traffic feature vectors & run triplet loss
                    anc_embed = fen(inputs_anc)
                    pos_embed = fen(inputs_pos)
                    neg_embed = fen(inputs_neg)
                    triplet_loss = triplet_criterion(anc_embed, pos_embed, neg_embed)
                    trip_loss += triplet_loss.item()

                    # # # # #
                    # predict chain length with head & run loss
                    #pred = head(torch.cat((anc_embed, pos_embed), dim=-1))
                    pred = head(anc_embed)
                    #pred = head(torch.cat((anc_embed, anc_embed2), dim=-1))
                    #pred = torch.clamp(pred,min=1)
                    chain_loss = criterion(pred, targets)

                # combined multi-task loss
                loss = triplet_loss + (loss_delta * chain_loss)
                tot_loss += loss.item()

                #
                # accuracy predicted as all-or-nothing
                #
                length_pred = torch.round(pred[:,0])
                up_acc += torch.sum(length_pred == targets[:,0]).item()

                length_pred = torch.round(pred[:,1])
                down_acc += torch.sum(length_pred == targets[:,1]).item()

                acc = (up_acc + down_acc)/2
                #
                # # # # #

                n += len(targets)

                if not eval_only:
                    loss /= accum   # normalize to full batch size before computing gradients
                    loss.backward()
                    # update weights, update scheduler, and reset optimizer after a full batch is completed
                    if (batch_idx+1) % accum == 0 or batch_idx+1 == len(dataloader):
                        optimizer.step()
                        scheduler.step()
                        for param in params:
                            param.grad = None


                pbar.set_postfix({
                                  'up_acc': up_acc/n,
                                  'down_acc': down_acc/n,
                                  'triplet': trip_loss/(batch_idx+1),
                                  'tot_loss': tot_loss/(batch_idx+1),
                                  })
                pbar.set_description(desc)

        tot_loss /= batch_idx + 1
        acc /= n
        return tot_loss


    # do training
    history = {}
    try:
        for epoch in range(last_epoch+1, epochs):

            # train and update model using training data
            fen.train()
            head.train()
            train_loss = epoch_iter(trainloader, 
                                    desc = f"Epoch {epoch} Train")
            metrics = {'tr_loss': train_loss}
            if not args.online:
                tr_data.reset_split()

            # evaluate on hold-out data
            fen.eval()
            head.eval()
            if validationloader is not None:
                with torch.no_grad():
                    va_loss = epoch_iter(validationloader, 
                                            eval_only = True, 
                                            desc = f"Epoch {epoch} Val.")
                metrics.update({'va_loss': va_loss})
            with torch.no_grad():
                test_loss = epoch_iter(testloader, 
                                        eval_only = True, 
                                        desc = f"Epoch {epoch} Test")
            metrics.update({'te_loss': test_loss})

            # save model
            if (epoch % ckpt_period) == (ckpt_period-1):
                # save last checkpoint before restart
                checkpoint_path_epoch = f"{checkpoint_dir}/{checkpoint_fname}/e{epoch}.pth"
                print(f"Saving end-of-cycle checkpoint to {checkpoint_path_epoch}...")
                torch.save({
                                "epoch": epoch,
                                "fen": fen.state_dict(),
                                "chain_head": head.state_dict(),
                                "opt": optimizer.state_dict(),
                                "config": model_config,
                        }, checkpoint_path_epoch)

            if save_best_epoch:
                best_val_loss = min([999]+[metrics['va_loss'] for metrics in history.values()])
                if metrics['va_loss'] < best_val_loss:
                    checkpoint_path_epoch = f"{checkpoint_dir}/{checkpoint_fname}/best.pth"
                    print(f"Saving new best model to {checkpoint_path_epoch}...")
                    torch.save({
                                    "epoch": epoch,
                                    "fen": fen.state_dict(),
                                    "chain_head": head.state_dict(),
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

