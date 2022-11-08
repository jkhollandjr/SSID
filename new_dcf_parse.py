import numpy as np
import pickle
import os
def convert_to_bursts(sequence):
    """
    converts a packet sequence into a burst sequence
    """
    bursts = [[0.0, 0, sequence[0][2]]]
    for packet in sequence:
        if packet[2] == bursts[-1][2]:
            bursts[-1][1] += packet[1]
        else:
            bursts.append([packet[0], packet[1], packet[2]])
    return bursts

def load_vf_burst(directory, delimiter='\t', file_split="-", max_length=None):
    """
    Load data from ascii files
    """
    X, X_ibt, y = [], [], []
    y_count = dict()
    for root, dirs, files in os.walk(directory):
        print(len(files))
        for fname in files:
            #if len(X) > 2000: break
            #print('len(X)', len(X))
            try:
                # trace_class is derived from file name (eg. 'C-n' where C is class and n is instance)
                trace_class = int(fname.split(file_split)[0])
                #if trace_class >= 10: continue
                #count number of instances of given class
                y_count[trace_class] = y_count.get(trace_class, 0) + 1

                # build direction sequence
                sequence = load_trace(os.path.join(root, fname), seperator=delimiter)


                if sequence is not None and len(sequence) > 50:

                    # adapt packet sequence into burst sequence
                    origin_sequence = convert_to_bursts(sequence)

                    sequence = [p[1]*p[2] for p in origin_sequence]
                    time_sequence = [p[0] for p in origin_sequence]
                    inter_time_sequence = np.diff(np.array(time_sequence))

                    # add sequence and label
                    sequence = np.array(sequence)
                    inter_time_sequence = np.array(inter_time_sequence)
                    if max_length is not None:
                        sequence.resize((max_length, 1))
                        inter_time_sequence.resize((max_length, 1))
                    X.append(sequence)
                    X_ibt.append(inter_time_sequence)
                    y.append(trace_class)


            except Exception as e:
                print(e)


    # wrap as numpy array
    X, X_ibt, Y = np.array(X), np.array(X_ibt), np.array(y)
    # rescaling X to [-1,1]
    #X = ((X+1249.)/1742.)*2-1
    # shuffle
    s = np.arange(Y.shape[0])
    np.random.seed(0)
    np.random.shuffle(s)
    #m = len(s) * (5/6) # for train
    X, X_ibt, Y = X[s], X_ibt[s], Y[s]
    np.savez('vf_burst_new_'+str(max_length)+'.npz', size=X, ibt=X_ibt, label=Y)
    return X, X_ibt, Y

def parse_csv(csv_path, interval,file_names):#option: 'sonly', 'tonly', 'both'
    HERE_PATH = csv_path+'inflow'
    THERE_PATH = csv_path+'outflow'
    print(HERE_PATH,THERE_PATH,interval)
    #here
    here=[]
    there=[]
    here_len=[]
    there_len=[]
    h_cnt = 0
    t_cnt = 0
    flow_cnt = 0
    final_names = []
    #for txt_file in os.listdir(HERE_PATH):
    #    file_names.append(txt_file)


    for i in range(len(file_names)):
        here_seq = []
        there_seq = []
        num_here_big_pkt_cnt = []
        num_there_big_pkt_cnt = []
        with open(HERE_PATH+'/'+file_names[i]) as f:
            #print(HERE_PATH+'/'+txt_file)
            pre_h_time = 0.0
            lines=f.readlines()
            if len(lines) == 0:
                continue
            #print('befoer filter',len(lines))
            big_pkt = []
            num_here_big_pkt = 0
            for line in lines:
                time_size = []
                time=float(line.split('\t')[0])
                size =int(float(line.split ('\t')[1].strip()))
                if size > 0:
                    ipd = time - pre_h_time
                else:
                    ipd = -(time - pre_h_time)
                if float(time) > interval[1]:
                    break
                if float(time) < interval[0]:
                    continue
                if abs(size) > 0:# ignore ack packets
                    if (pre_h_time != 0) and (ipd == 0):
                        big_pkt.append(size)
                        continue
                    if len(big_pkt)!=0:
                        last_pkt=here_seq.pop()
                        here_seq.append({"ipd": last_pkt["ipd"], "size": sum(big_pkt)+big_pkt[0]})
                        big_pkt = []
                        num_here_big_pkt += 1
                    time_size.append (ipd)
                    time_size.append(size)
                    time_size = np.array (time_size)
                    here_seq.append ({"ipd": time_size[0], "size": time_size[1]})
                    pre_h_time = time
        #print('after filter',len (here_seq))

        with open (THERE_PATH + '/' + file_names[i]) as f:
            pre_h_time = 0.0
            lines = f.readlines ()
            if len (lines) == 0:
                continue
            big_pkt = []
            num_there_big_pkt = 0
            for line in lines:
                time_size = []
                time = float (line.split ('\t')[0])
                size = int (float(line.split ('\t')[1].strip()))
                if size > 0:
                    ipd = time - pre_h_time
                else:
                    ipd = -(time - pre_h_time)
                if float(time) > interval[1]:
                    break
                if float(time) < interval[0]:
                    continue
                if abs (size) > 66:  # ignore ack packets
                    if (pre_h_time != 0) and (ipd == 0):
                        big_pkt.append (size)
                        continue
                    if len (big_pkt) != 0:
                        last_pkt = there_seq.pop ()
                        there_seq.append ({"ipd": last_pkt["ipd"], "size": sum (big_pkt)+big_pkt[0]})
                        big_pkt = []
                        num_there_big_pkt += 1

                    time_size.append (ipd)
                    time_size.append (size)
                    time_size = np.array (time_size)
                    there_seq.append ({"ipd": time_size[0], "size": time_size[1]})
                    pre_h_time = time

        #print (len (there_seq))
        if (len(here_seq)!=0) and (len(there_seq)!=0):
            here_len.append (len (here_seq))
            num_here_big_pkt_cnt.append (num_here_big_pkt)
            there_len.append (len (there_seq))
            num_there_big_pkt_cnt.append (num_there_big_pkt)
            here.append (here_seq)
            there.append (there_seq)
            final_names.append(file_names[i])
            flow_cnt += 1
        #h_cnt += len (here_seq)
        #t_cnt += len (there_seq)

    print(interval,'mean',np.mean(np.array(here_len)), np.mean(np.array(there_len)),np.mean(num_here_big_pkt_cnt),np.mean(num_there_big_pkt_cnt),flow_cnt)
    print (interval,'median', np.median (np.array (here_len)), np.median (np.array (there_len)),np.median(num_here_big_pkt_cnt),np.median(num_there_big_pkt_cnt),flow_cnt)
    return np.array(here), np.array(there), np.array(final_names)

def create_overlap_window_csv(csv_path, file_list, prefix_pickle_output, interval, num_windows, addnum):
    windows_seq = []
    file_names = []
    for txt_file in open(file_list,'r').readlines():
        file_names.append(txt_file.strip())
    for win in range(num_windows):
        here, there, labels = parse_csv(csv_path, [win*addnum,win*addnum+interval],file_names)
        windows_seq.append({"tor": here, "exit": there, "label": labels})
        #np.savez_compressed('/project/hoppernj/research/seoh/new_dcf_data/new_overlap_interval' + str(interval) + '_win' + str(win) + '_addn' + str(addnum) + '.npz',
        #         tor=here, exit=there)

        with open (prefix_pickle_output + str(interval) + '_win' + str(win) + '_addn' + str(addnum) + '_w_superpkt.pickle', 'wb') as handle:
            pickle.dump ({'tor':here, 'exit':there, "label": labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #with open ('filename.pickle', 'rb') as handle:
        #    b = pickle.load (handle)
        #d2=np.load("d1.npy")
        #f = gzip.GzipFile ('/project/hoppernj/research/seoh/new_dcf_data/new_overlap_interval' + str(interval) + '_win' + str(win) + '_addn' + str(addnum) + '.npy.gz', "w")
        #np.save(f,d1)
    #windows_seq = np.array(windows_seq)

    #return windows_seq

data_path = '/home/james/Desktop/research/outside_repos/DCF/CrawlE_Proc_20000/'
file_list_path = '/home/james/Desktop/research/outside_repos/DCF/src/DCF/original.txt'
prefix_pickle_output = '/home/james/Desktop/research/outside_repos/DCF/original/'
create_overlap_window_csv(data_path, file_list_path, prefix_pickle_output, 5, 11, 2)
create_overlap_window_csv(data_path, file_list_path, prefix_pickle_output, 4, 11, 2)
create_overlap_window_csv(data_path, file_list_path, prefix_pickle_output, 3, 11, 2)
#create_overlap_window_csv(npz_path, 5, 11, 1)
#create_overlap_window_csv(npz_path, 3, 11, 2)
