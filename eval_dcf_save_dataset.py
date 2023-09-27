import csv
import time
import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dot
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from sklearn.metrics.pairwise import cosine_similarity

from model import create_model

total_time = 0
total_emb = 0
total_vot = 0
total_cos = 0

parser = argparse.ArgumentParser()
parser.add_argument('-test', default='capture/5_test11addn2_w_superpkt.npz')
parser.add_argument('-flow', default=1000)
parser.add_argument('-tor_len', default=500)
parser.add_argument('-exit_len', default=800)
parser.add_argument('-model1', default='models/model1_best.h5')
parser.add_argument('-model2', default='models/model2_best.h5')
parser.add_argument('-output', default="capture_eval.csv")
args = parser.parse_args()


def get_session(gpu_fraction=0.85):
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
    return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def ini_cosine_output(single_output_l, input_number):
    for pairs in range(0, (input_number * input_number)):
        single_output_l.append(0)  # ([0])

def Cosine_Similarity_eval(tor_embs, exit_embs, similarity_threshold, single_output_l, evaluating_window, last_window,
                           correlated_shreshold, cosine_similarity_all_list, muti_output_list):
    global total_time
    global total_vot

    number_of_lines = tor_embs.shape[0]
    start_emd = time.time()
    for tor_emb_index in range(0, number_of_lines):
        t = similarity_threshold[tor_emb_index]
        constant_num = int(tor_emb_index * number_of_lines)
        for exit_emb_index in range(0, number_of_lines):
            if cosine_similarity_all_list[tor_emb_index][exit_emb_index] >= t:
                single_output_l[constant_num + exit_emb_index] = single_output_l[constant_num + exit_emb_index] + 1

    if evaluating_window == last_window:
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        TP_overlap_first = 0
        TP_overlap_Second = 0
        FP_overlap_first = 0
        FP_overlap_Second = 0

        for tor_eval_index in range(0, tor_embs.shape[0]):
            for exit_eval_index in range(0, tor_embs.shape[0]):
                cos_condithon_a = (tor_eval_index == exit_eval_index)
                number_of_ones = (single_output_l[(tor_eval_index * (tor_embs.shape[0])) + exit_eval_index])
                cos_condition_b = (number_of_ones >= correlated_shreshold)
                cos_condition_c = (number_of_ones < correlated_shreshold)

                if (cos_condithon_a and cos_condition_b):
                    TP = TP + 1
                if (cos_condithon_a and cos_condition_c):
                    FN = FN + 1
                if (not (cos_condithon_a)) and cos_condition_b:
                    FP = FP + 1
                if (not (cos_condithon_a)) and cos_condition_c:
                    TN = TN + 1

        if (TP + FN) != 0:
            TPR = float(TP) / (TP + FN)
        else:
            TPR = -1

        if (TN + FP) != 0:
            TNR = float(TN) / (TN + FP)
        else:
            TNR = -1

        if (FP + TN) != 0:
            FPR = float(FP) / (FP + TN)
        else:
            FPR = -1

        if (FN + TP) != 0:
            FNR = float(FN) / (FN + TP)
        else:
            FNR = -1

        if (TP + TN + FP + FN) != 0:
            ACC = float(TP + TN) / (TP + TN + FP + FN)
        else:
            ACC = -1

        if (TP + FP) != 0:
            PPV = float(TP) / (TP + FP)
        else:
            PPV = -1

        if (PPV + TPR) != 0:
            F_ONE = float(PPV * TPR) / (PPV + TPR)
        else:
            F_ONE = -1

        muti_output_list.append(TPR)
        muti_output_list.append(FPR)
        muti_output_list.append(calculate_bdr(TPR, FPR))

    end_time = time.time()
    total_time = total_time + (end_time - start_emd)
    total_vot = total_vot + (end_time - start_emd)

def calculate_bdr(tpr, fpr):
    TPR = tpr
    FPR = fpr
    c = 1 / int(args.flow)
    u = (int(args.flow)-1) / int(args.flow)
    if ((TPR * c) + (FPR * u)) != 0:
        BDR = (TPR * c) / ((TPR * c) + (FPR * u))
    else:
        BDR = -1
    return BDR

# Every tor flow will have a unique threshold
def threshold_finder(input_similarity_list, curr_win, gen_ranks, thres_seed, use_global):
    output_shreshold_list = []
    for simi_list_index in range(0, len(input_similarity_list)):
        correlated_similarity = input_similarity_list[simi_list_index][simi_list_index]
        temp = list(input_similarity_list[simi_list_index])
        temp.sort(reverse=True)

        cut_point = int((len(input_similarity_list[simi_list_index]) - 1) * ((thres_seed) / 100))
        if use_global == 1:
            output_shreshold_list.append(thres_seed)
        elif use_global != 1:
            output_shreshold_list.append(temp[cut_point])
    return output_shreshold_list


def eval_model(full_or_half, five_or_four, model1_path, model2_path, test_path, thr, use_global,
               use_softmax, muti_output_list, soft_muti_output_list):
    global total_time
    global total_emb
    global total_vot
    global total_cos

    test_data = np.load(test_path, allow_pickle=True)
    print(test_data['tor'][0].shape)
    print(test_data['exit'][0].shape)
    pad_t = int(args.tor_len)*2
    pad_e = int(args.exit_len)*2

    tor_model = create_model(input_shape=(pad_t, 1), emb_size=64, model_name='tor')
    exit_model = create_model(input_shape=(pad_e, 1), emb_size=64, model_name='exit')

    # load triplet models for tor and exit traffic
    from tensorflow import keras
    
    tor_model = tf.keras.models.load_model(model1_path)
    exit_model = tf.keras.models.load_model(model2_path)
    tor_model.compile()
    exit_model.compile()

    # print('Get logits for 5 windows')

    # This list will be the output list of cosine similarity approach.
    # should have 2093*2093 sub-sublists which are corrsponding to the number of pairs like t0e0, t0e1...
    # And each sub-sub-sublist should have 5 elements which are corrsponding to the status of correlation
    single_output = []

    # This list should have 2093 sub-sublists for each tor flow.
    # Each sublist should have 2093 elements which corrsponding to 2093 similarities of each tor flow
    # No need to use 5 list for each window. We can just overwrite the old table
    # used by threshold_finder()
    cosine_similarity_table = []
    threshold_result = []

    # below are the code that are used for controlling the behavior of the program

    activated_windows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    last_activated_window = 10
    correlated_shreshold_value = five_or_four
    thres_seed = thr

    #save data for processing 
    tor_data = []
    exit_data = []
    for win in range(11):
        test_data_tor = test_data['tor'][win][0:500]
        test_data_exit = test_data['exit'][win][0:500]

        tor_embs = tor_model.predict(test_data_tor)
        exit_embs = exit_model.predict(test_data_exit)

        tor_data.append(tor_embs)
        exit_data.append(exit_embs)
        

    dataset = []
    labels = []
    for i in range(500):
        if(i%10==0):
            print(i)
        for j in range(500):
            #if(i != j):
            #    if(np.random.rand() > .001):
            #        continue
            feature_vector = []
            all_differences = []
            for k in range(11):
                tor = tor_data[k][i].reshape(1, -1)
                exit = exit_data[k][j].reshape(1, -1)

                similarity = cosine_similarity(tor, exit)
                feature_vector.append(similarity.reshape(1)[0])

                if(i==j):
                    label = 1.0
                else:
                    label = 0.0
            
                difference = tor - exit
                all_differences.append(similarity)
            dataset.append(np.array(all_differences))
            labels.append(label)

    dataset = np.asarray(dataset)
    labels = np.asarray(labels).astype(np.float32)
    np.save('capture_train.npy', dataset)
    np.save('capture_labels_train.npy', labels)

    #save data for processing 
    tor_data = []
    exit_data = []
    for win in range(11):
        test_data_tor = test_data['tor'][win][500:1000]
        test_data_exit = test_data['exit'][win][500:1000]

        tor_embs = tor_model.predict(test_data_tor)
        exit_embs = exit_model.predict(test_data_exit)

        tor_data.append(tor_embs)
        exit_data.append(exit_embs)
        
    dataset = []
    labels = []
    for i in range(500):
        if(i%10==0):
            print(i)
        for j in range(500):
            #if(i != j):
            #    if(np.random.rand() > .001):
            #        continue
            feature_vector = []
            all_differences = []
            for k in range(11):
                tor = tor_data[k][i].reshape(1, -1)
                exit = exit_data[k][j].reshape(1, -1)

                similarity = cosine_similarity(tor, exit)
                feature_vector.append(similarity.reshape(1)[0])

                if(i==j):
                    label = 1.0
                else:
                    label = 0.0
            
                difference = tor - exit
                all_differences.append(similarity)
            dataset.append(np.array(all_differences))
            labels.append(label)

    dataset = np.asarray(dataset)
    labels = np.asarray(labels).astype(np.float32)
    np.save('capture_test.npy', dataset)
    np.save('capture_labels_test.npy', labels)
    exit()

if __name__ == "__main__":

    test_path = args.test
    model1_path = args.model1
    model2_path = args.model2

    rank_thr_list = [60, 50, 47, 43, 40, 37, 33, 28, 24, 20, 16.667, 14, 12.5, 11, 10, 9, 8.333, 7, 6.25, 5, 4.545, 3.846, 2.941, 1.667, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

    num_of_thr = len(rank_thr_list)
    flow_length = int(args.flow)

    five_or_four = 9

    rank_multi_output = []
    five_rank_multi_output = []
    for i in range(0, num_of_thr):
        rank_multi_output.append([(rank_thr_list[i])])
        five_rank_multi_output.append([(rank_thr_list[i])])

    epoch_index = 0
    use_global = 0
    use_softmax = 0
    use_new_data = 0

    for thr in rank_thr_list:
        eval_model(flow_length, five_or_four, model1_path, model2_path, test_path, thr, use_global, use_softmax, rank_multi_output[epoch_index], [])
        epoch_index = epoch_index + 1

    with open(args.output, "w", newline="") as rank_f:
        writer = csv.writer(rank_f)
        writer.writerows(rank_multi_output)

    print("total_time: " + str(total_time))
    print("total_cos: " + str(total_cos))
    print("total_vot: " + str(total_vot))
    print("total_emb: " + str(total_emb))

