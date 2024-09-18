from scapy.all import *
import os
import pickle
import numpy as np


def get_ssh_streams(filepath):

    # Read in PCAP file
    pcap = rdpcap(filepath)
    
    # Initialize dictionary to store SSH streams
    ssh_streams = {}
    
    # Loop through packets in PCAP file
    for pkt in pcap:
        # Check if packet is SSH traffic
        if pkt.haslayer('TCP') and pkt.haslayer('Raw') and (pkt['TCP'].dport == 22 or pkt['TCP'].sport == 22):
            # Get source and destination IP addresses
            src = pkt['IP'].src
            dst = pkt['IP'].dst
            
            # Check if this is a new SSH stream
            if (src, dst) not in ssh_streams and (dst, src) not in ssh_streams:
                ssh_streams[(src, dst)] = []
            
            # Add packet to appropriate SSH stream list
            if (src, dst) in ssh_streams:
                ssh_streams[(src, dst)].append(pkt)
            else:
                ssh_streams[(dst, src)].append(pkt)
                
    # Print number of SSH streams found
    print("Number of SSH streams:", len(ssh_streams))
    
    # Print number of packets in each SSH stream
    for stream in ssh_streams:
        print("Number of packets in SSH stream:", len(ssh_streams[stream]))

    return ssh_streams


data = {}       # metadata for SSH streams
IP_data = {}    # extra IP information for each stream

rt_dir = './tcpdump'
time_cutoff = 11.      # fix erroneous time gaps caused by sleeps
pckt_cutoff = 10

for dirpath, dirnames, fnames in os.walk(rt_dir):
    sample_ID = os.path.basename(dirpath)
    if sample_ID.isnumeric():

        data[sample_ID] = {}
        IP_data[sample_ID] = {}
        for fname in fnames:

            # get the host number for the pcap sample
            host_ID = fname.replace('dev','').replace('.pcap', '')
            if host_ID.isnumeric():
                host_ID = int(host_ID)
            else:
                continue

            # load and split the pcap into distinct ssh streams (defined by IP src/dst tuples)
            pth = os.path.join(dirpath, fname)
            try:
                streams = get_ssh_streams(pth)
                if len(streams) < 1:
                    continue
            except:
                continue

            data_t = []
            IP_t = {}

            # process the stream scapy packets into metadata per stream
            for src_ip,dst_ip in streams:
                stream = streams[(src_ip, dst_ip)]

                metadata = []
                init_time = float(stream[0].time)
                for pkt in stream:
                    cur_time = float(pkt.time) - init_time
                    if cur_time > time_cutoff:
                        pkt_dir = 1. if pkt['IP'].src == src_ip else -1.
                        pkt_size = len(pkt)
                        metadata += [(cur_time, pkt_size, pkt_dir)]
                if len(metadata) < pckt_cutoff:
                    continue

                metadata = np.array(metadata).T
                metadata[0,:] -= metadata[0,0]      # adjust timestamp sequence to begin at zero
                data_t.append(metadata)

                # store IP information in case it's needed
                IP_t['src'] = src_ip
                IP_t['dst'] = dst_ip

            data[sample_ID][host_ID] = data_t
            IP_data[sample_ID][host_ID] = IP_t



# filter out chain samples with odd stream counts per host
# (first & last hosts should have one SSH stream, stepping stones should have two)
for sample_ID in list(data.keys()):

    if len(data[sample_ID]) <= 2:
        del data[sample_ID]
        del IP_data[sample_ID]
        continue

    host_IDs = set(data[sample_ID].keys())

    # if first and last hosts do not have exactly one stream, something is odd with the sample
    if (len(data[sample_ID][min(host_IDs)]) != 1) or (len(data[sample_ID][max(host_IDs)]) != 1):
        del data[sample_ID]
        del IP_data[sample_ID]
        continue

    host_IDs.remove(min(host_IDs))
    host_IDs.remove(max(host_IDs))

    # check if stepping stones all correctly have two streams
    for host_ID in host_IDs:
        if len(data[sample_ID][host_ID]) != 2:
            del data[sample_ID]
            del IP_data[sample_ID]
            break


with open('./processed.pkl', 'wb') as fi:
    pickle.dump({'data': data, 'IPs': IP_data}, fi)
