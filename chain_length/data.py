import numpy as np
import torch
from torch.utils import data
from processor import DataProcessor
from tqdm import tqdm
import pickle
import itertools
import random


class BaseDataset(data.Dataset):
    """
    Dataset object to process and hold windowed samples in a convenient package.

    Attributes
    ----------
    data_ID_tuples : list
        List of stream-level identifiers as unique tuples, (chain_no, stream_no)
    data_windows : dict
        Dictionary containing lists of stream windows for streams within each chain
    data_chainlengths : dict
        Dictionary containing lists of chain-length 'labels' for use in chain-length prediction
    data_chain_IDs : dict
        Dictonary that maps chain numbers to all correlated stream-level IDs
    """
    def __init__(self, filepath, 
                       stream_processor,
                       sample_idx = None,
                       window_kwargs = dict(),
                       stream_ID_range = (0, float('inf')),
                       host_only = False,
                       preproc_feats = False,
                       ):
        """
        Load the metadata for samples collected in our SSID data. 

        Parameters
        ----------
        filepath : str
            The path to the data dictionary saved as a pickle file.
        stream_processor : DataProcessor
            The processor object that convert raw samples into their feature representations.
        window_kwargs : dict
            Dictionary containing the keyword arguments for the window processing function
        stream_ID_range : 2-tuple
            Tuple of int or floats that can be used to control the stream hops loaded.
            First value is lower range. Second value is upper range (inclusive).
        preproc_feats : bool
            If True, the data processor will be applied on samples before windowing.
        """

        self.data_ID_tuples = []
        self.data_windows = dict()
        self.data_chainlengths = dict()
        self.data_chain_IDs = dict()

        times_processor = DataProcessor(('times',))

        if host_only:
            stream_ID_range = (1, -2)  # include only streams from stepping-stones
        self.host_only = host_only

        # load and enumeratechains in the dataset
        chains = load_dataset(filepath, sample_idx)
        for chain_ID, chain in tqdm(enumerate(chains)):

            # total hops in chain
            hops = (len(chain) // 2) + 1

            stream_ID_list = []

            # enumerate each stream in the chain
            # Note: streams are expected to be ordered
            for stream_ID, stream in enumerate(chain):
                if (stream_ID_range[0] > 0 and stream_ID < stream_ID_range[0]) or \
                        (stream_ID_range[0] < 0 and (stream_ID - len(chain)) < stream_ID_range[0]): 
                    continue
                if (stream_ID_range[1] > 0 and stream_ID > stream_ID_range[1]) or \
                        (stream_ID_range[1] < 0 and (stream_ID - len(chain)) > stream_ID_range[1]): 
                    continue

                # sample ID definitions
                sample_ID = (chain_ID, stream_ID)
                self.data_ID_tuples.append(sample_ID)
                stream_ID_list.append(stream_ID)

                # time-only representation
                times = times_processor(stream)

                if preproc_feats:
                    # multi-channel feature representation of stream
                    stream = stream_processor(stream)

                # chunk stream into windows
                if window_kwargs is not None:
                    windows = create_windows(times, stream, **window_kwargs)

                    if not preproc_feats:
                        # create multi-channel feature representation of windows independently
                        for i in range(len(windows)):
                            if len(windows[i]) <= 0:
                                windows[i] = torch.empty((0,stream_processor.input_channels))
                            else:
                                windows[i] = stream_processor(windows[i])

                    self.data_windows[sample_ID] = windows

                else:
                    self.data_windows[sample_ID] = [stream]

                # chain-length label
                downstream_hops = (stream_ID+1) // 2
                upstream_hops = hops - downstream_hops - 1
                self.data_chainlengths[sample_ID] = (downstream_hops, upstream_hops)

            if len(stream_ID_list) > 1:
                self.data_chain_IDs[chain_ID] = stream_ID_list

    def __len__(self):
        """
        Count of all streams within the dataset.
        """
        return len(self.data_ID_tuples)

    def __getitem__(self, index):
        """
        Generate a Triplet sample.

        Parameters
        ----------
        index : int
            The index of the sample to use as the anchor. 
            Note: Index values to sample mappings change after every reset_split()

        Returns
        -------
        windows : list
            List of windows for stream as torch tensors
        chain_label : 2-tuple
            A tuple containing ints that represent the downstream and upstream hop counts.
            Downstream represents the number of hosts between current host and victim.
            Upstream represents the number of hosts between current host and attacker.
        sample_ID : 2-tuple
            A tuple containing ints that acts as the sample ID.
            First value is the chain number. 
            Second value is the stream number within the chain.
        """
        sample_ID = self.data_ID_tuples[index]
        windows = self.data_windows[sample_ID]
        chain_label = self.data_chainlengths[sample_ID]
        return windows, chain_label, sample_ID


class PairwiseDataset(BaseDataset):
    """
    """
    def __init__(self, dataset, 
            sample_mode = 'oversample', 
            sample_ratio = None,
            sample_seed = 0):
        """
        """
        vars(self).update(vars(dataset))  # load dataset attributes

        self.return_features = False

        self.all_chain_IDs = np.array(list(self.data_chain_IDs.keys()))

        # create all possible combinations of chains within the dataset
        self.all_combos = list(itertools.combinations_with_replacement(self.all_chain_IDs, 2))

        # build complete list of sample ID pairs
        self.correlated_pairs = []
        self.uncorrelated_pairs = []
        for chain1_ID, chain2_ID in self.all_combos:

            # build all possible pairs of streams between the two chains

            if chain1_ID == chain2_ID:
                correlated = True
            else:
                correlated = False

            if self.host_only and correlated:
                # consider only host-wise pairs if host_only is enabled
                stream_IDs = self.data_chain_IDs[chain1_ID]
                for i in range(len(stream_IDs)//2):
                    hostwise_pair = ((chain1_ID, stream_IDs[i]), (chain1_ID, stream_IDs[i+1]), correlated)
                    self.correlated_pairs.append(hostwise_pair)

            else:
                chain1_ID_tuples = [(chain1_ID, stream_ID) for stream_ID in self.data_chain_IDs[chain1_ID]]
                chain2_ID_tuples = [(chain2_ID, stream_ID) for stream_ID in self.data_chain_IDs[chain2_ID]]

                # create all sample pairs
                for ID1 in chain1_ID_tuples:
                    for ID2 in chain2_ID_tuples:
                        if correlated:
                            self.correlated_pairs.append((ID1, ID2, correlated))
                        else:
                            self.uncorrelated_pairs.append((ID1, ID2, correlated))
                        #self.all_pairs.append((ID1, ID2, correlated))

        if sample_ratio is not None:
            random.seed(sample_seed)
            if sample_mode == 'oversample':
                k = int(len(self.uncorrelated_pairs) * sample_ratio)
                self.all_pairs = random.choices(self.correlated_pairs, k=k) + self.uncorrelated_pairs
            elif sample_mode == 'undersample':
                k = int(len(self.correlated_pairs) * sample_ratio)
                self.all_pairs = self.correlated_pairs + random.choices(self.uncorrelated_pairs, k=k)

            self.all_sample_IDs = set()
            for sample1,sample2,_ in self.all_pairs:
                self.all_sample_IDs.add(sample1)
                self.all_sample_IDs.add(sample2)
        else:
            self.all_pairs = self.correlated_pairs + self.uncorrelated_pairs
            self.all_sample_IDs = self.data_ID_tuples


    def set_fen(self, fen, proc):
        self.return_features = True
        self.sample_features = dict()
        fen.eval()
        for sample_ID in tqdm(self.all_sample_IDs):
            windows = self.data_windows[sample_ID]
            embeds = fen(proc(windows)).detach().cpu()
            self.sample_features[sample_ID] = embeds

    def unset_fen(self):
        self.return_features = False

    def __len__(self):
        """
        """
        return len(self.all_pairs)

    def __getitem__(self, index):
        """
        """
        sample1_ID, sample2_ID, correlated = self.all_pairs[index]

        if not self.return_features:
            sample1_windows = self.data_windows[sample1_ID]
            sample2_windows = self.data_windows[sample2_ID]
        else:
            sample1_windows = self.sample_features[sample1_ID]
            sample2_windows = self.sample_features[sample2_ID]

        sample1_chainlength = self.data_chainlengths[sample1_ID]
        sample2_chainlength = self.data_chainlengths[sample2_ID]

        sample1 = (sample1_windows, sample1_chainlength, sample1_ID)
        sample2 = (sample2_windows, sample2_chainlength, sample2_ID)

        return sample1, sample2, correlated


class TripletDataset(BaseDataset):
    """
    Dataset object for generating triplets for triplet learning.
    """
    def __init__(self, dataset):
        """
        Initialize triplet dataset from an existing SSI dataset object.
        """
        vars(self).update(vars(dataset))  # load dataset attributes

        # create and shuffle indices for samples
        self.all_indices = np.array(list(self.data_chain_IDs.keys()))
        np.random.shuffle(self.all_indices)

        # divide indices into two partitions for triplet generation
        cutoff = len(self.all_indices) // 2
        self.partition_1 = self.all_indices[:cutoff]
        self.partition_2 = self.all_indices[cutoff:]

    def __len__(self):
        """
        An epoch of the TripletDataset iterates over all samples within partition_1
        """
        return len(self.partition_1)

    def __getitem__(self, index):
        """
        Generate a Triplet sample.

        Anchor samples are selected from parition_1
        Positive samples are selected at random from the chain of the anchor sample
        Negative samples are selected from partition_2

        Parameters
        ----------
        index : int
            The index of the sample to use as the anchor. 
            Note: Index values to sample mappings change after every reset_split()

        Returns
        -------
        anc : tensor
            Window to represent the anchor sample
        pos : tensor
            Correlated window to represent positive sample
        neg : tensor
            Uncorrelated window to represent the negative sample
        """
        # chain ID for anchor & positive
        anc_chain_ID = self.partition_1[index]

        if not self.host_only:
            # randomly sample (w/o replacement) two streams from the chain for anchor & positive (e.g. network-based SSID)
            anc_stream_IDs = self.data_chain_IDs[anc_chain_ID]
            anc_stream_IDs = np.random.choice(anc_stream_IDs, size=2, replace=False)
        else:
            # randomly select a pair of streams, both of which were collected on the same host (e.g. host-based SSID)
            anc_stream_IDs = self.data_chain_IDs[anc_chain_ID]
            host_num = np.random.randint(0,len(anc_stream_IDs)//2)
            anc_stream_IDs = [anc_stream_IDs[host_num], anc_stream_IDs[host_num+1]]
            np.random.shuffle(anc_stream_IDs)

        anc_ID = (anc_chain_ID, anc_stream_IDs[0])
        pos_ID = (anc_chain_ID, anc_stream_IDs[1])

        # randomly select chain from partition 2 to be the negative
        neg_chain_ID = np.random.choice(self.partition_2)
        # randomly sample a stream from the negative stream
        neg_stream_ID = np.random.choice(self.data_chain_IDs[neg_chain_ID])
        neg_ID = (neg_chain_ID, neg_stream_ID)

        # get windows for anc, pos, neg
        anc = self.data_windows[anc_ID]
        pos = self.data_windows[pos_ID]
        neg = self.data_windows[neg_ID]

        # randomly select a window for the triplet
        candidate_idx = [i for i,window in enumerate(anc) if len(window) > 0]
        #window_idx = np.random.randint(0, len(anc)-1)
        window_idx = np.random.choice(candidate_idx)

        anc_tup = (anc[window_idx], self.data_chainlengths[anc_ID], anc_ID)
        pos_tup = (pos[window_idx], self.data_chainlengths[pos_ID], pos_ID)
        neg_tup = (neg[window_idx], self.data_chainlengths[neg_ID], neg_ID)
        #print(len(anc_tup[0]), len(pos_tup[0]), len(neg_tup[0]))
        return anc_tup, pos_tup, neg_tup
                    

    def reset_split(self):
        """
        Reshuffle sample indices and select a new split for triplet generation.
        """
        # Reshuffle the indices at the start of each epoch.
        np.random.shuffle(self.all_indices)

        # Re-divide the shuffled indices into two partitions.
        cutoff = len(self.all_indices) // 2
        self.partition_1 = self.all_indices[:cutoff]
        self.partition_2 = self.all_indices[cutoff:]



def create_windows(times, features,
                    window_width = 5,
                    window_count = 11,
                    window_overlap = 2,
                    include_all_window = True,
                ):
    """
    Slice a sample's full stream into time-based windows.

    Parameters
    ----------
    times : ndarray
    features : ndarray
    window_width : int
    window_count : int
    window_overlap : int

    Returns
    -------
    list
        A list of stream windows as torch tensors.
    """
    window_features = []

    window_step = min(window_width - window_overlap, 1)

    # Create overlapping windows
    for start in np.arange(0, stop = window_count * window_step, 
                              step = window_step):

        end = start + window_width

        window_idx = torch.where(torch.logical_and(times >= start, times < end))[0]
        window_features.append(features[window_idx])

    # add full stream as window
    if include_all_window:
        window_features.append(features)

    return window_features


def load_dataset(filepath, idx_selector=None):
    """
    Load the metadata for samples collected in our SSID data. 

    Parameters
    ----------
    filepath : str
        The path to the data dictionary saved as a pickle file.
    idx_selector : ndarray
        A 1D numpy array that identifies which samples from the file to load.
        Samples are sorted before selection is applied to insure consistency of loading.

    Returns
    -------
    list
        A nested list of processed streams
        The outer list contains lists of correlated processed streams, while the inner lists contain individual instances
        (with all instances within the list being streams produced by hosts within the same multi-hop tunnel)
    """

    with open(filepath, "rb") as fi:
        all_data = pickle.load(fi)


    IP_info = all_data['IPs']   # extra src. & dst. IP info available for each stream
    data = all_data['data']     # stream metadata organized by sample and hosts (per sample)


    # list of all sample idx
    sample_IDs = sorted(list(data.keys()))   # sorted so that it is reliably ordered
    if idx_selector is not None:
        sample_IDs = np.array(sample_IDs, dtype=object)[idx_selector].tolist()  # slice out requested idx


    # fill with lists of correlated samples
    all_streams = []

    # each 'sample' contains a variable number of hosts (between 3 and 6 I believe)
    for s_idx in sample_IDs:
        sample = data[s_idx]
        host_IDs = list(sample.keys())

        # first and last hosts represent the attacker's machine and target endpoint of the chain respectively
        # these hosts should contain only one SSH stream in their sample
        attacker_ID = 1
        target_ID   = len(host_IDs)

        # the stepping stone hosts are everything in-between
        # these hosts should each contain two streams
        steppingstone_IDs = list(filter(lambda x: x not in [attacker_ID, target_ID], host_IDs))

        # loop through each host, process stream metadata into vectors, and add to list
        correlated_streams = []
        for h_idx in host_IDs:
            correlated_streams.extend([torch.tensor(x).T for x in sample[h_idx]])

        # add group of correlated streams for the sample into the data list
        all_streams.append(correlated_streams)

    return all_streams


if __name__ == "__main__":

    pklpath = '../processed.pkl'

    # chain-based sample splitting
    te_idx = np.arange(0,1300)
    tr_idx = np.arange(1300,13000)

    # stream window definitions
    window_kwargs = {
                     'window_width': 5, 
                     'window_count': 11, 
                     'window_overlap': 2
                     }

    # multi-channel feature processor
    processor = DataProcessor(('sizes', 'iats', 'time_dirs', 'dirs'))

    # load SSI dataset object
    for idx in (te_idx, tr_idx):
        print(f'Chains: {len(idx)}')

        # build base dataset object
        data = BaseDataset(pklpath, processor,
                              window_kwargs = window_kwargs,
                              sample_idx = idx,
                              stream_ID_range = (0,1))
        print(f'Streams: {len(data)}')
        for windows, chain_label, sample_ID in data:
            pass

        # construct a triplets dataset object, derived from the base dataset object
        # Note: changes to the base dataset propogate to the triplet object once initialized (I think?)
        triplets = TripletDataset(data)
        print(f'Triplets: {len(triplets)}')
        for anc, pos, neg in triplets:
            pass
        triplets.reset_split()
