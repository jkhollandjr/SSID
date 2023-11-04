import numpy as np
import torch
from torch.utils import data
from processor import DataProcessor
from tqdm import tqdm
import pickle


class BaseDataset(data.Dataset):
    """
    """
    def __init__(self, filepath, 
                       stream_processor,
                       sample_idx = None,
                       window_kwargs = dict(),
                       stream_ID_min = 0,
                       stream_ID_max = 5,
                       ):

        self.window_kwargs = window_kwargs
        self.data_features = dict()
        self.data_windows = dict()
        self.data_chainlengths = dict()
        self.data_ID_tuples = []
        self.data_chain_IDs = dict()

        times_processor = DataProcessor(('times',))

        chains = load_dataset(filepath, sample_idx)

        for chain_ID, chain in tqdm(enumerate(chains)):

            # total hops in chain
            hops = len(chain) // 2
            self.data_chain_IDs[chain_ID] = []

            # enumerate each stream in the chain
            # Note: streams are expected to be ordered
            for stream_ID, stream in enumerate(chain):
                if stream_ID < stream_ID_min: continue
                if stream_ID > stream_ID_max: continue

                sample_ID = (chain_ID, stream_ID)
                self.data_ID_tuples.append(sample_ID)
                self.data_chain_IDs[chain_ID].append(stream_ID)

                # multi-channel feature representation of stream
                features = stream_processor(stream)
                self.data_features[sample_ID] = features

                # stream windows
                times = times_processor(stream)
                windows = create_windows(times, features, **self.window_kwargs)
                self.data_windows[sample_ID] = windows

                # chain-length label
                upstream_hops = (stream_ID+1) // 2
                downstream_hops = hops - upstream_hops
                self.data_chainlengths[sample_ID] = (downstream_hops, upstream_hops)



    def __len__(self):
        return len(self.data_ID_tuples)

    def __getitem__(self, index):
        sample_ID = self.data_ID_tuples[index]
        windows = self.data_windows[sample_ID]
        chain_label = self.data_chainlengths[sample_ID]
        return windows, chain_label, sample_ID


class TripletDataset(BaseDataset):
    """
    """
    def __init__(self, dataset):
        vars(self).update(vars(dataset))

        self.all_indices = np.array(list(self.data_chain_IDs.keys()))
        np.random.shuffle(self.all_indices)

        cutoff = len(self.all_indices) // 2
        self.partition_1 = self.all_indices[:cutoff]
        self.partition_2 = self.all_indices[cutoff:]

    def __len__(self):
        """
        """
        return len(self.partition_1)

    def __get_item__(self, index):
        """
        """
        # chain ID for anchor & positive
        anc_chain_ID = self.partition_1[index]

        # randomly sample (w/o replacement) two streams from the chain for anchor & positive
        anc_stream_IDs = self.data_chain_IDs[anchor_chain_ID]
        anc_stream_IDs = np.random.choice(anchor_stream_IDs, size=2, replace=False)

        anc_ID = (anc_chain_ID, anc_stream_IDs[0])
        pos_ID = (anc_chain_ID, anc_stream_IDs[1])

        # randomly select chain from partition 2 to be the negative
        neg_chain_ID = np.random.choice(self.partition_2)
        # randomly sample a stream from the negative stream
        neg_stream_ID = np.random.choice(self.data_chain_IDs[negative_chain_ID])
        neg_ID = (neg_chain_ID, neg_stream_ID)

        # get windows for anc, pos, neg
        anc = self.data_windows[anc_ID]
        pos = self.data_windows[pos_ID]
        neg = self.data_windows[neg_ID]

        # randomly select a window for the triplet
        window_idx = np.random.randint(0, len(anc)-1)

        return anc[window_idx], pos[window_idx], neg[window_idx]

    def reset_split(self):
        """
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
                    window_overlap = 2):
    """
    """
    window_features = []

    # Create overlapping windows
    for start in np.arange(0, stop = window_count * window_overlap, 
                              step = window_overlap):

        end = start + window_width

        window_idx = torch.where(torch.logical_and(times >= start, times < end))
        window_features.append(features[window_idx])

    return window_features


def load_dataset(filepath, idx_selector=None):
    """
    Load the metadata for all samples collected in our SSID data, and process them using the process() function.

    Returns: a nested list of processed streams
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
    te_idx = np.arange(0,1300)
    tr_idx = np.arange(1300,13000)
    window_kwargs = {
                     'window_width': 5, 
                     'window_count': 11, 
                     'window_overlap': 2
                     }

    processor = DataProcessor(('sizes', 'iats', 'time_dirs', 'dirs'))

    # load SSI dataset object
    te_data = BaseDataset(pklpath, processor,
                          window_kwargs = window_kwargs,
                          sample_idx = te_idx,
                          stream_ID_min = 0,
                          stream_ID_max = 1)
    print(len(te_data))
    for windows, chain_label, sample_ID in te_data:
        pass

    # construct a triplets dataset object, derived from the base dataset object
    # Note: changes to the base dataset propogate to the triplet object once initialized (I think?)
    te_triplets = TripletDataset(te_data)
    print(len(te_triplets))
    for anc, pos, neg in te_triplets:
        pass
    te_triplets.reset_split()
