import pickle
import numpy as np


def process(x):
    """
    Simple example function to use when processing 
    """
    timestamps   = x[0]
    packet_sizes = x[1]
    directions   = x[2]

    iats = np.diff(timestamps)
    iats = np.concatenate(([0], iats))

    #In case DeepCoFFEA struggles with the small total time
    MULTIPLIER = 40
    output = [(MULTIPLIER*t, d*s) for t,d,s in zip(timestamps, directions, packet_sizes)]

    return output


def load_data(fp = './processed_May.pkl'):
    """
    Load the metadata for all samples collected in our SSID data, and process them using the process() function.

    Returns: a nested list of processed streams
        The outer list contains lists of correlated processed streams, while the inner lists contain individual instances 
        (with all instances within the list being streams produced by hosts within the same multi-hop tunnel)
    """

    with open("./processed.pkl", "rb") as fi:
        all_data = pickle.load(fi)
    
    
    IP_info = all_data['IPs']   # extra src. & dst. IP info available for each stream
    data = all_data['data']     # stream metadata organized by sample and hosts (per sample)
    
    
    # list of all sample idx
    sample_IDs = list(data.keys())
    
    
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
            correlated_streams.extend([process(x) for x in sample[h_idx]])
    
        # add group of correlated streams for the sample into the data list
        all_streams.append(correlated_streams)
    
    return all_streams


if __name__ == "__main__":

    print(f'Total groups of correlated streams: {len(all_streams)}')
    print(f'Total count of streams: {sum([len(group) for group in all_streams])}')
