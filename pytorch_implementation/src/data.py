import os
import pickle as pkl
import numpy as np
from os.path import join
import torch
from torch.utils import data
from torch.nn import functional as F
from torchvision import transforms, utils
from random import shuffle
from functools import partial
import math
from collections import defaultdict
from tqdm import tqdm
import shutil
import random


MIN_LENGTH_DEFAULT = 1
MAX_LENGTH_DEFAULT = 12000


class GenericWFDataset(data.Dataset):
    """
    Flexible Dataset object that can be applied to load any of the subpage-based WF datasets
    """
    def __init__(self, root, 
                mon_raw_data_name,
                unm_raw_data_name,
                mon_suffix,
                unm_suffix,
                mon_tr_count, unm_tr_count, 
                mon_te_count, unm_te_count, 
                *args, 
                unm_tr_start_idx = None,
                train = True, 
                class_selector = None,
                class_divisor = 1,
                multisample_count = 1,
                min_length = MIN_LENGTH_DEFAULT, 
                max_length = MAX_LENGTH_DEFAULT, 
                per_batch_transforms = None, 
                on_load_transforms = None,
                tmp_directory = './tmp',
                tmp_subdir = None,
                keep_tmp = False,
                **kwargs,
            ):

        te_idx = np.arange(mon_te_count)
        te_unm_idx = np.arange(unm_te_count)
        tr_idx = np.arange(mon_te_count, mon_te_count + mon_tr_count)
        unm_start_idx = unm_tr_start_idx if unm_tr_start_idx else unm_te_count + unm_tr_count
        #unm_start_idx = unm_tr_start_idx if unm_tr_start_idx else unm_te_count + 4500

        #unm_start_idx = unm_te_count
        tr_unm_idx = np.arange(unm_start_idx, unm_start_idx + unm_tr_count)
        print(tr_unm_idx)
        print(unm_start_idx)

        

        mon_suffix_t = f'{mon_te_count}-{mon_tr_count}'
        unm_suffix_t = f'{unm_te_count//1000}k-{unm_tr_count//1000}k'

        if train:
            dataset, labels, ids, classes = load_full_dataset(root,
                    mon_raw_data_name = mon_raw_data_name,
                    unm_raw_data_name = unm_raw_data_name,
                    mon_suffix = f'tr-{mon_suffix_t}{mon_suffix}',
                    unm_suffix = f'tr-{unm_suffix_t}{unm_suffix}',
                    mon_sample_idx = tr_idx, 
                    unm_sample_idx = tr_unm_idx,
                    multisample_count = multisample_count,
                    min_length = min_length, 
                    class_divisor = class_divisor,
                    class_selector = class_selector,
                    **kwargs)
        else:
            dataset, labels, ids, classes = load_full_dataset(root,
                    mon_raw_data_name = mon_raw_data_name,
                    unm_raw_data_name = unm_raw_data_name,
                    mon_suffix = f'te-{mon_suffix_t}{mon_suffix}',
                    unm_suffix = f'te-{unm_suffix_t}{unm_suffix}',
                    mon_sample_idx = te_idx, 
                    unm_sample_idx = te_unm_idx,
                    multisample_count = 1,
                    min_length = min_length, 
                    class_divisor = class_divisor,
                    class_selector = class_selector,
                    **kwargs)

        self.classes = classes
        self.ids = ids
        self.dataset = dataset
        self.labels = labels
        self.transform = per_batch_transforms
        self.max_length = max_length

        # pre-apply transformations 
        self.tmp_data = None
        if on_load_transforms:
            # setup tmp directory and filename map
            if tmp_directory is not None:
                if tmp_subdir is not None:
                    self.tmp_dir = os.path.join(tmp_directory, tmp_subdir, 'tr' if train else 'te')
                else:
                    self.tmp_dir = f'{tmp_directory}/tmp{random.randint(0, 1000)}'
                if not os.path.exists(self.tmp_dir):
                    try:
                        os.makedirs(self.tmp_dir)
                    except:
                        pass
                self.keep_tmp = keep_tmp
                self.tmp_data = dict()

            # do processing
            for ID in tqdm(self.ids, desc="Processing...", dynamic_ncols=True):
                x = self.dataset[ID]  # get sample by ID

                # check if processed sample already exists in tmp
                if self.tmp_data is not None:
                    filename = f'{self.tmp_dir}/{ID}.pt'
                    self.tmp_data[ID] = filename
                    if os.path.exists(filename):
                        continue

                # apply processing to sample
                x = on_load_transforms(x)

                # store transforms to disk 
                if self.tmp_data is not None:
                    torch.save(x, filename)
                # store in memory
                else:
                    self.dataset[ID] = x
 
    def __len__(self):
        return len(self.ids)
 
    def __getitem__(self, index):
        ID = self.ids[index]
        X = self.dataset[ID] if self.tmp_data is None else torch.load(self.tmp_data[ID])
        if self.max_length:
            X = X[:self.max_length]
        if self.transform:
            return self.transform(X), self.labels[ID]
        return X, self.labels[ID]

    def __del__(self):
        if not self.keep_tmp:
            try:
                if self.tmp_data is not None:
                    shutil.rmtree(self.tmp_dir)
            except:
                print(f">>> Failed to clear temp directory \'{self.tmp_dir}\'!!")

# # # #
#
# SingleSite (2022) Dataset definitions
#
# # # #

class AmazonSingleSite(GenericWFDataset):
    def __init__(self, root, *args, 
            mon_tr_count = 140, unm_tr_count = 78400,
            mon_te_count = 40, unm_te_count = 9800,
            subpage_as_labels = False,
            defense_mode = 'undef',
            **kwargs):

        data_dir = join(root, 'wf-singlesite')
        mon_raw_data_name = f'{defense_mode}-amazon.pkl'
        unm_raw_data_name = f'{defense_mode}-unm.pkl'

        mon_suffix = f''
        unm_suffix = f''

        class_divisor = 1 if subpage_as_labels else 490

        super().__init__(
            data_dir,
            mon_raw_data_name,
            unm_raw_data_name,
            mon_suffix, 
            unm_suffix, 
            mon_tr_count, unm_tr_count,
            mon_te_count, unm_te_count,
            *args, 
            class_divisor = class_divisor, 
            **kwargs
        )

class WebMDSingleSite(GenericWFDataset):
    def __init__(self, root, *args, 
            mon_tr_count = 60, unm_tr_count = 29700,
            mon_te_count = 10, unm_te_count = 4950,
            subpage_as_labels = True,
            defense_mode = 'undef',
            **kwargs):

        data_dir = join(root, 'wf-singlesite')
        mon_raw_data_name = f'{defense_mode}-webmd.pkl'
        unm_raw_data_name = f'{defense_mode}-unm.pkl'

        mon_suffix = f''
        unm_suffix = f''

        class_divisor = 1 if subpage_as_labels else 495

        super().__init__(
            data_dir,
            mon_raw_data_name,
            unm_raw_data_name,
            mon_suffix, 
            unm_suffix, 
            mon_tr_count, unm_tr_count,
            mon_te_count, unm_te_count,
            *args, 
            class_divisor = class_divisor, 
            **kwargs
        )


# # # #
#
# BigEnough (2022) Dataset definitions
#
# # # #

class BigEnough(GenericWFDataset):
    def __init__(self, root, *args, 
            mon_tr_count = 18, unm_tr_count = 17100,
            mon_te_count = 2, unm_te_count = 1900,
            subpage_as_labels = False,
            defense_mode = 'undef',
            **kwargs):

        data_dir = join(root, 'wf-bigenough')
        mon_raw_data_name = f'{defense_mode}-mon.pkl'
        unm_raw_data_name = f'{defense_mode}-unm.pkl'

        mon_suffix = f''
        unm_suffix = f''

        class_divisor = 1 if subpage_as_labels else 10

        super().__init__(
            data_dir,
            mon_raw_data_name,
            unm_raw_data_name,
            mon_suffix, 
            unm_suffix, 
            mon_tr_count, unm_tr_count,
            mon_te_count, unm_te_count,
            *args, 
            class_divisor = class_divisor, 
            **kwargs
        )

class Surakav(GenericWFDataset):
    def __init__(self, root, *args, 
            mon_tr_count = 90, unm_tr_count = 9000,
            mon_te_count = 10, unm_te_count = 1000,
            defense_mode = 'undef',
            **kwargs):

        data_dir = join(root, 'wf-surakav')
        mon_raw_data_name = f'{defense_mode}-mon.pkl'
        unm_raw_data_name = f'{defense_mode}-unm.pkl'

        mon_suffix = f''
        unm_suffix = f''

        class_divisor = 1

        super().__init__(
            data_dir,
            mon_raw_data_name,
            unm_raw_data_name,
            mon_suffix, 
            unm_suffix, 
            mon_tr_count, unm_tr_count,
            mon_te_count, unm_te_count,
            *args, 
            class_divisor = class_divisor, 
            **kwargs
        )


# # # #
#
# Data loading helper functions
#
# # # #

def load_full_dataset(
        data_dir = './data/wf-bigenough', 
        include_mon = True,
        include_unm = True,
        mon_sample_idx = list(range(9)), 
        unm_sample_idx = list(range(9000)),
        mon_raw_data_name = 'mon_standard.pkl',
        unm_raw_data_name = 'unm_standard.pkl',
        mon_suffix = "",
        unm_suffix = "",
        multisample_count = 1,
        min_length = MIN_LENGTH_DEFAULT,
        class_divisor = 1,
        class_selector = None,
        **kwargs
    ):
    data, labels = dict(), dict()
    IDs = []
    class_names = []

    # # # # # #
    # Monitored Website Data
    # # # # # #
    all_X = None
    all_y = None
    if include_mon:
        all_X, all_y = load_mon(data_dir, mon_raw_data_name, mon_sample_idx, 
                                min_length = min_length, 
                                mon_suffix = mon_suffix,
                                multisample_count = multisample_count,
                            )

        all_y //= class_divisor

        class_names += [f'mon-{i}' for i in range(len(np.unique(all_y)))]

    # # # # # #
    # Unmonitored Websites
    # # # # # #
    all_X_unm = None
    if include_unm:
        class_names += ['unm']
        unm_label = np.amax(all_y)+1 if (all_y is not None) else 0
                
        all_X_unm = load_unm(data_dir, unm_raw_data_name, unm_sample_idx, 
                                unm_label = unm_label,
                                min_length = min_length, 
                                unm_suffix = unm_suffix,
                                multisample_count = multisample_count,
                            )
        all_y_unm = np.ones(len(all_X_unm)) * unm_label
        if (all_X is not None):
            all_X = np.concatenate((all_X, all_X_unm))
            all_y = np.concatenate((all_y, all_y_unm))
        else:
            all_X = all_X_unm
            all_y = all_y_unm

        # add unmon label to class selector to avoid filtering unm samples
        if class_selector is not None:
            class_selector.add(unm_label)

    # convert into dictionary format
    for i in range(len(all_X)):
        if (class_selector is not None) and (all_y[i] not in class_selector):
            continue
        ID = f'{all_y[i]}-{i}'
        IDs.append(ID)
        data[ID] = all_X[i]
        labels[ID] = all_y[i]

    return data, labels, IDs, class_names

def pad_or_concatenate(lst):
    while len(lst) < 12000:
        lst += lst
    return np.array(lst[:12000] + [0] * (12000 - len(lst)))
def load_mon(data_dir, mon_raw_data_name, sample_idx,
             min_length = MIN_LENGTH_DEFAULT,
             mon_suffix = "",
             multisample_count = 1,
            ):
    """
    Load monitored samples from pickle file
    """
    MON_PATH = join(data_dir, mon_raw_data_name)

    data = {}
    labels = {}
    IDs = []

    print(f"Loading mon data from {MON_PATH}...")
    with open(MON_PATH, 'rb') as fi:
        raw_data = pkl.load(fi)
    
    all_X = []
    all_y = []
    for i,key in enumerate(raw_data.keys()):

        print(f'{i}', end='\r', flush=True)
        sample_idx = sample_idx[sample_idx < len(raw_data[key])]
        #samples = np.array(raw_data[key], dtype=object)[sample_idx].tolist()
        samples = np.array(raw_data[key], dtype=object)[sample_idx].tolist()
        #samples = raw_data[key]

        '''
        print(sample_idx)
        print(len(samples)) 50
        print(len(samples[0])) 3
        '''
        for multisample in samples:
            i = 0
            while i < len(multisample) and i < multisample_count:
                sample = np.around(multisample[i], decimals=2)
                
                i += 1
                #sample = np.array([np.abs(sample), np.ones(len(sample))*512, np.sign(sample)]).T
                if len(sample) < min_length: continue
                all_X.append(sample)
                all_y.append(key)

    del raw_data

    all_X = np.array(all_X, dtype=object)
    all_y = np.array(all_y)
    return all_X, all_y


def load_unm(data_dir, raw_data_name, sample_idx,
             min_length = MIN_LENGTH_DEFAULT, 
             max_samples = 0, 
             unm_label = 1,
             unm_suffix = "",
             multisample_count = 1,
            ):
    """
    Load unmonitored samples from pickle file
    """
    UNM_PATH = join(data_dir, raw_data_name)

    # save time and load prepared data if possible
    print(f"Loading unm data from {UNM_PATH}...")
    with open(UNM_PATH, 'rb') as fi:
        raw_data = pkl.load(fi)

    sample_idx = sample_idx[sample_idx < len(raw_data)]
    samples = np.array(raw_data, dtype=object)[sample_idx].tolist()
    '''
    print(multisample_count)
    print(len(samples))
    print(len(samples[0]))
    print(len(samples[0][0]))
    exit()
    '''

    all_X_umn = []
    for i,multisample in enumerate(samples):
        if max_samples > 0 and len(all_X_umn) == max_samples:
            break

        j = 0
        while j < len(multisample) and j < multisample_count:
            #multisample = [multisample]
            sample = np.around(multisample[j], decimals=2)
            #sample = np.array([np.abs(sample), np.ones(len(sample))*512, np.sign(sample)]).T
            j += 1
            if len(sample) < min_length: continue
            all_X_umn.append(sample)

            if max_samples > 0 and len(all_X_umn) == max_samples:
                break

    del raw_data

    all_X_umn = np.array(all_X_umn, dtype=object)
    return all_X_umn


# # # #
#
# Transforms / Augments
#
# # # #

class ToProcessed(object):
    """
    Apply processing function to sample metadata
    """
    def __init__(self, process_func, **kwargs):
        self.func = process_func
        self.kwargs = kwargs

    def __call__(self, sample):
        proc = self.func(sample, **self.kwargs)
        return proc


class ToTensor(object):
    """
    Transpose numpy sample and convert to pytorch tensor
    """
    def __call__(self, sample, transpose=True):
        return torch.tensor(sample).float()


def collate_and_pad(batch, return_sample_sizes=True):
    """
    convert samples to tensors and pad samples to equal length
    """
    # convert labels to tensor and get sequence lengths
    batch_x, batch_y = zip(*batch)
    batch_y = torch.tensor(batch_y)
    if return_sample_sizes:
        sample_sizes = [t.shape[0] for t in batch_x]

    # pad and fix dimension
    batch_x = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True, padding_value=0.)
    if len(batch_x.shape) < 3:  # add channel dimension if missing
        batch_x = batch_x.unsqueeze(-1)
    batch_x = batch_x.permute((0,2,1))  # B x C x S

    if return_sample_sizes:
        return batch_x.float(), batch_y.long(), sample_sizes
    else:
        return batch_x.float(), batch_y.long(),



DATASET_CHOICES = ['be', 'be-front', 'be-interspace', 'be-regulator', 'be-ts2', 'be-ts5', 
                   'amazon', 'amazon-300k', 'amazon-front', 'amazon-front-300k', 'amazon-interspace', 'amazon-interspace-300k',
                   'webmd', 'webmd-300k', 'webmd-front', 'webmd-front-300k', 'webmd-interspace', 'webmd-interspace-300k',
                   'gong', 'gong-surakav4', 'gong-surakav6', 'gong-front', 'gong-tamaraw',
                   'gong-50k', 'gong-surakav4-50k', 'gong-surakav6-50k', 'gong-front-50k', 'gong-tamaraw-50k',
                   ]


def load_data(dataset, 
        batch_size = 32, 
        tr_transforms = (), te_transforms = (),     # apply on-load
        tr_augments = (), te_augments = (),         # apply per-batch
        root = './data', 
        workers = 4,
        collate_return_sample_sizes = True,
        **kwargs,
    ):
    """
    Prepare training and testing PyTorch dataloaders
    """

    tr_transforms = {'per_batch_transforms': transforms.Compose(tr_augments), 
                     'on_load_transforms': transforms.Compose(tr_transforms)}
    te_transforms = {'per_batch_transforms': transforms.Compose(te_augments), 
                     'on_load_transforms': transforms.Compose(te_transforms)}

    if dataset == 'be':
        data_obj = partial(BigEnough, 
                            root, 
                            **kwargs,
                )

    elif dataset == 'be-front':
        data_obj = partial(BigEnough, 
                            root, 
                            defense_mode = 'front',
                            **kwargs,
                )

    elif dataset == 'be-interspace':
        data_obj = partial(BigEnough, 
                            root, 
                            defense_mode = 'interspace',
                            **kwargs,
                )

    elif dataset == 'be-regulator':
        data_obj = partial(BigEnough, 
                            root, 
                            defense_mode = 'regulator',
                            **kwargs,
                )

    elif dataset == 'be-ts2':
        data_obj = partial(BigEnough, 
                            root, 
                            defense_mode = 'ts2',
                            **kwargs,
                )

    elif dataset == 'be-ts5':
        data_obj = partial(BigEnough, 
                            root, 
                            defense_mode = 'ts5',
                            **kwargs,
                )

    elif dataset == 'amazon':
        data_obj = partial(AmazonSingleSite, 
                            root, 
                            defense_mode = 'undef',
                            **kwargs,
                )

    elif dataset == 'amazon-300k':
        data_obj = partial(AmazonSingleSite, 
                            root, 
                            defense_mode = 'undef',
                            unm_te_count = 294000,
                            **kwargs,
                )

    elif dataset == 'amazon-front':
        data_obj = partial(AmazonSingleSite, 
                            root, 
                            defense_mode = 'front',
                            **kwargs,
                )

    elif dataset == 'amazon-front-300k':
        data_obj = partial(AmazonSingleSite, 
                            root, 
                            defense_mode = 'front',
                            unm_te_count = 294000,
                            **kwargs,
                )

    elif dataset == 'amazon-interspace':
        data_obj = partial(AmazonSingleSite, 
                            root, 
                            defense_mode = 'interspace',
                            **kwargs,
                )

    elif dataset == 'amazon-interspace-300k':
        data_obj = partial(AmazonSingleSite, 
                            root, 
                            defense_mode = 'interspace',
                            unm_te_count = 294000,
                            **kwargs,
                )

    elif dataset == 'webmd':
        data_obj = partial(WebMDSingleSite, 
                            root, 
                            defense_mode = 'undef',
                            **kwargs,
                )

    elif dataset == 'webmd-front':
        data_obj = partial(WebMDSingleSite, 
                            root, 
                            defense_mode = 'front',
                            **kwargs,
                )

    elif dataset == 'webmd-front-300k':
        data_obj = partial(WebMDSingleSite, 
                            root, 
                            defense_mode = 'front',
                            unm_te_count = 294000,
                            **kwargs,
                )

    elif dataset == 'webmd-interspace':
        data_obj = partial(WebMDSingleSite, 
                            root, 
                            defense_mode = 'interspace',
                            **kwargs,
                )

    elif dataset == 'webmd-interspace-300k':
        data_obj = partial(WebMDSingleSite, 
                            root, 
                            defense_mode = 'interspace',
                            unm_te_count = 294000,
                            **kwargs,
                )
    elif dataset == 'webmd-300k':
        data_obj = partial(WebMDSingleSite, 
                            root, 
                            defense_mode = 'undef',
                            unm_te_count = 294000,
                            **kwargs,
                )


    elif dataset == "gong":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'undef',
                            **kwargs,
                )

    elif dataset == "gong-surakav4":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'surakav-0.4',
                            **kwargs,
                )

    elif dataset == "gong-surakav6":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'surakav-0.6',
                            **kwargs,
                )

    elif dataset == "gong-front":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'front',
                            **kwargs,
                )

    elif dataset == "gong-tamaraw":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'tamaraw',
                            **kwargs,
                )

    elif dataset == "gong-50k":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'undef',
                            unm_te_count = 50000,
                            **kwargs,
                )

    elif dataset == "gong-surakav4-50k":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'surakav-0.4',
                            unm_te_count = 50000,
                            **kwargs,
                )

    elif dataset == "gong-surakav6-50k":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'surakav-0.6',
                            unm_te_count = 50000,
                            **kwargs,
                )

    elif dataset == "gong-front-50k":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'front',
                            unm_te_count = 50000,
                            **kwargs,
                )

    elif dataset == "gong-tamaraw-50k":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'tamaraw',
                            unm_te_count = 50000,
                            **kwargs,
                )


    trainset = data_obj(train=True, **tr_transforms) 
    testset = data_obj(train=False, **te_transforms) 
    classes = len(testset.classes)

    # prepare dataloaders without sampler
    trainloader = torch.utils.data.DataLoader(
        trainset, num_workers = workers, 
        collate_fn = collate_and_pad,
        batch_size = batch_size,
        shuffle = True,
        pin_memory = True,
    )
    testloader = None
    if testset is not None:
        testloader = torch.utils.data.DataLoader(
            testset, num_workers = workers, 
            collate_fn = collate_and_pad,
            batch_size = batch_size,
            pin_memory = True,
        )


    print(f'Train: {len(trainset)} samples across {len(trainloader)} batches')
    print(f'Test: {len(testset)} samples across {len(testloader)} batches')

    return trainloader, testloader, classes
