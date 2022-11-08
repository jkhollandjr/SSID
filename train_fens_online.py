from __future__ import print_function

try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle


#from keras.callbacks import LambdaCallback
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.callbacks import LambdaCallback
from new_model import create_model, create_model_2d
#import keras.backend.tensorflow_backend as ktf
import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dot
from tensorflow.keras import optimizers
import sys
import numpy as np
import argparse
import random
import pickle5
from tensorflow.keras import backend as K
#### Stop the model training when 0.002 to get the best result in the paper!!!!

#os.environ["CUDA_VISIBLE_DEVICES"] = "0";
parser = argparse.ArgumentParser()

def get_params():
    parser.add_argument ('--tor_len', required=False, default=500)
    parser.add_argument ('--exit_len', required=False, default=800)
    parser.add_argument ('--win_interval', required=False, default=5)
    parser.add_argument ('--num_window', required=False, default=11)
    parser.add_argument ('--alpha', required=False, default=0.1)  # 96 for DF, 101 for pfp, 201 for awf
    parser.add_argument ('--input', required=False, default='/home/james/Desktop/research/outside_repos/DCF/original/'
)
    parser.add_argument ('--test', required=False, default='/home/james/Desktop/research/outside_repos/DCF/original/')  # 100 for DF, 30 for pfp, 200 for awf
    parser.add_argument ('--model', required=False, default="/home/james/Desktop/research/outside_repos/DCF/src/DCF/models/original_online_")
    args = parser.parse_args ()
    return args

def get_session(gpu_fraction=0.85):
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                #allow_growth=True)
    #return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
    return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def load_whole_seq_new(option,tor_seq,exit_seq,circuit_labels,test_c,train_c,model_gb):
    train_index = circuit_labels[circuit_labels%2==1]
    test_index = circuit_labels[circuit_labels%2==0][:2093]
    train_window1=[]
    train_window2=[]
    test_window1=[]
    test_window2=[]
    window_tor=[]
    window_exit=[]

    if option == 'sonly':
        print("extract size-only features...")
        for i in range(len(tor_seq)):
            window_tor.append([pair["size"] for pair in tor_seq[i]])
            window_exit.append([pair["size"] for pair in exit_seq[i]])
        #print('window_tor',len(window_tor))

        #print('window_exit', len(window_exit))
        window_tor = np.array(window_tor)
        window_exit = np.array(window_exit)
        train_window1.append(window_tor[train_index])
        train_window2.append(window_exit[train_index])
        test_window1.append(window_tor[test_index])
        test_window2.append(window_exit[test_index])
    elif option == 'tonly':
        print("extract ipd-only features...")
        for trace in tor_seq:
            window_tor.append([pair["ipd"] for pair in trace])
        print('window_tor', len(window_tor))
        for trace in exit_seq:
            window_exit.append([pair["ipd"] for pair in trace])
        print('window_exit', len(window_exit))
        window_tor = np.array(window_tor)
        window_exit = np.array(window_exit)
        train_window1.append(window_tor[train_index])
        train_window2.append(window_exit[train_index])
        test_window1.append(window_tor[test_index])
        test_window2.append(window_exit[test_index])
    elif option == "both":
        window_tor_size = []
        window_exit_size = []
        window_tor_ipd = []
        window_exit_ipd = []
        print("extract both ipd and size features...")
        for i in range(len(tor_seq)):

            window_tor_size.append([float(pair["size"])/1000.0 for pair in tor_seq[i]])
            window_exit_size.append([float(pair["size"]) / 1000.0 for pair in exit_seq[i]])
            temp_tor_ipd = []
            temp_exit_ipd = []
            for pair in tor_seq[i]:
                if float(pair["size"]) > 0:
                    temp_tor_ipd.append(float(pair["ipd"])* 1000.0)
                else:
                    temp_tor_ipd.append(float(pair["ipd"])* 1000.0)
            for pair in exit_seq[i]:
                if float(pair["size"]) > 0:
                    temp_exit_ipd.append(float(pair["ipd"])* 1000.0)
                else:
                    temp_exit_ipd.append(float(pair["ipd"])* 1000.0)
            window_tor_ipd.append(temp_tor_ipd)
            window_exit_ipd.append(temp_exit_ipd)

        print('window_tor_size', np.array(window_tor_size).shape)
        print('window_exit_size', np.array(window_exit_size).shape)
        print('window_tor_ipd', np.array(window_tor_ipd).shape)
        print('window_exit_ipd', np.array(window_exit_ipd).shape)
        window_tor_ipd = np.array(window_tor_ipd)
        window_exit_ipd = np.array(window_exit_ipd)

        # Change the first idp to 0 across all windows.
        new_window_tor_ipd = []
        new_window_exit_ipd = []
        for trace in window_tor_ipd:
            new_trace = [0]+list(trace[1:])
            new_window_tor_ipd.append([ipd for ipd in new_trace])
        for trace in window_exit_ipd:
            new_trace = [0]+list(trace[1:])
            new_window_exit_ipd.append([ipd for ipd in new_trace])

        window_tor_ipd = new_window_tor_ipd
        window_exit_ipd = new_window_exit_ipd
        print('window_tor_ipd',window_tor_ipd[10][:10])
        print('window_exit_ipd',window_exit_ipd[10][:10])

        if model_gb == 'cnn1d':
            for i in range(len(window_tor_ipd)):
                window_tor.append(np.concatenate((window_tor_ipd[i], window_tor_size[i]), axis=None))
                window_exit.append(np.concatenate((window_exit_ipd[i], window_exit_size[i]), axis=None))
        else:
            for i in range(len(window_tor_ipd)):
                window_tor.append([window_tor_ipd[i], window_tor_size[i]])
                window_exit.append([window_exit_ipd[i], window_exit_size[i]])

        window_tor = np.array(window_tor)
        window_exit = np.array(window_exit)
        print('window_tor', window_tor.shape)
        print('window_exit', window_exit.shape)

        for w, c in zip (window_tor, circuit_labels):
            if c in train_c:
                train_window1.append(w)
            elif c in test_c:
                test_window1.append(w)

        for w, c in zip (window_exit, circuit_labels):
            if c in train_c:
                train_window2.append(w)
            elif c in test_c:
                test_window2.append(w)

        print ('train_window1', np.array(train_window1).shape)
        print ('train_window2', np.array(train_window2).shape)

    else:
        print("Wrong option!")

    return np.array(train_window1), np.array(train_window2), np.array(test_window1), np.array(test_window2), np.array(test_window1), np.array(test_window2)


if __name__ == '__main__':
    args = get_params()
    #ktf.set_session(get_session())
    #set_session(get_session())

    model_gb = 'cnn1d'
    isOpen = 'both_corr'

    ## Params for time-based window
    interval = args.win_interval#5
    t_flow_size = int(args.tor_len)#500#400#238  # 238#150#184  # 238#264
    e_flow_size = int(args.exit_len)#800#330#140
    num_windows = int(args.num_window)#11#21#5
    window_index_list = np.arange(num_windows)
    print(isOpen)
    if (model_gb == 'mlp') or (model_gb == 'cnn1d'):
        pad_t = t_flow_size * 2
        pad_e = e_flow_size * 2
    else:
        pad_t = t_flow_size
        pad_e = e_flow_size

    alpha_value = float(args.alpha)#0.1

    train_windows1 = []
    valid_windows1 = []
    test_windows1 = []
    train_windows2 = []
    valid_windows2 = []
    test_windows2 = []
    train_labels = []
    test_labels = []
    valid_labels = []


    for window_index in window_index_list:
        option = 'both'
        addn = 2
        pickle_path = args.input+str(interval)+'_win'+ str(window_index) +'_addn'+ str(
            addn) +'_w_superpkt.pickle'


        with open (pickle_path, 'rb') as handle:
            traces = pickle5.load(handle, encoding='latin1')
            tor_seq = traces["tor"]
            exit_seq = traces["exit"]
            labels = traces["label"]
            circuit_labels = np.array ([int (labels[i].split ('_')[0]) for i in range (len (labels))])

            #print (tor_seq[0])

            circuit = {}
            for i in range(len(labels)):
                if labels[i].split ('_')[0] not in circuit.keys ():
                    circuit[labels[i].split ('_')[0]] = 1
                else:
                    circuit[labels[i].split ('_')[0]] += 1

            global test_c
            global train_c
            if window_index == 0:
                test_c = []
                train_c = []
                sum_ins = 2093
                keys = list (circuit.keys ())
                random.shuffle (keys)
                for key in keys:
                    if sum_ins > 0:
                        sum_ins -= circuit[key]
                        test_c.append (key)
                    else:
                        train_c.append (key)
                test_c = np.array (test_c).astype ('int')
                train_c = np.array (train_c).astype ('int')
            # print (train_c)
            #print ('test_c', test_c)
            #print ('train_c', train_c)
        ###########
        train_set_x1, train_set_x2, test_set_x1, test_set_x2, valid_set_x1, valid_set_x2 = load_whole_seq_new(option,
                                                                                                          tor_seq,exit_seq,circuit_labels,test_c,train_c,
                                                                                                          model_gb)

        temp_test1 = []
        temp_test2 = []

        #print(train_set_x1.shape)
        #print(valid_set_x1.shape)
        #print(test_set_x1.shape)
        if model_gb != 'cnn2d':
            temp_test1 = []
            temp_test2 = []
        else:
            temp_test1 = np.zeros((test_set_x1.shape[0], 2, t_flow_size, 1))
            temp_test2 = np.zeros((test_set_x2.shape[0], 2, e_flow_size, 1))

        # For cnn models, we don't need padding here

        #print('train_set_x1', train_set_x1.shape)
        for x in train_set_x1:

            if model_gb == 'mlp':
                train_windows1.append(np.pad(x[:pad_t], (0, pad_t - len(x[:pad_t])), 'constant'))
            elif model_gb == 'cnn1d':
                train_windows1.append(
                    np.reshape(np.pad(x[:pad_t], (0, pad_t - len(x[:pad_t])), 'constant'), [-1, 1]))


        for x in valid_set_x1:
            if model_gb == 'mlp':
                valid_windows1.append(np.pad(x[:pad_t], (0, pad_t - len(x[:pad_t])), 'constant'))
            elif model_gb == 'cnn1d':
                valid_windows1.append(
                    np.reshape(np.pad(x[:pad_t], (0, pad_t - len(x[:pad_t])), 'constant'), [-1, 1]))


        idx = 0
        for x in test_set_x1:
            if model_gb == 'mlp':
                temp_test1.append(np.pad(x[:pad_t], (0, pad_t - len(x[:pad_t])), 'constant'))
            elif model_gb == 'cnn1d':
                temp_test1.append(np.reshape(np.pad(x[:pad_t], (0, pad_t - len(x[:pad_t])), 'constant'), [-1, 1]))
            else:
                temp_test1[idx, 0, :, 0] = np.pad(x[0][:pad_t], (0, pad_t - len(x[0][:pad_t])), 'constant')
                temp_test1[idx, 1, :, 0] = np.pad(x[1][:pad_t], (0, pad_t - len(x[1][:pad_t])), 'constant')

                idx += 1

        for x in train_set_x2:

            if model_gb == 'mlp':
                train_windows2.append(np.pad(x[:pad_e], (0, pad_e - len(x[:pad_e])), 'constant'))
            elif model_gb == 'cnn1d':
                train_windows2.append(
                    np.reshape(np.pad(x[:pad_e], (0, pad_e - len(x[:pad_e])), 'constant'), [-1, 1]))


        for x in valid_set_x2:
            if model_gb == 'mlp':
                valid_windows2.append(np.pad(x[:pad_e], (0, pad_e - len(x[:pad_e])), 'constant'))
            elif model_gb == 'cnn1d':
                valid_windows2.append(
                    np.reshape(np.pad(x[:pad_e], (0, pad_e - len(x[:pad_e])), 'constant'), [-1, 1]))


        idx = 0
        for x in test_set_x2:
            if model_gb == 'mlp':
                temp_test2.append(np.pad(x[:pad_e], (0, pad_e - len(x[:pad_e])), 'constant'))
            elif model_gb == 'cnn1d':
                temp_test2.append(np.reshape(np.pad(x[:pad_e], (0, pad_e - len(x[:pad_e])), 'constant'), [-1, 1]))
            else:
                temp_test2[idx, 0, :, 0] = np.pad(x[0][:pad_e], (0, pad_e - len(x[0][:pad_e])), 'constant')
                temp_test2[idx, 1, :, 0] = np.pad(x[1][:pad_e], (0, pad_e - len(x[1][:pad_e])), 'constant')

                idx += 1

        #print('temp_test1: ', np.array(temp_test1).shape)
        #print('temp_test2: ', np.array(temp_test2).shape)
        test_windows1.append(np.array(temp_test1))
        test_windows2.append(np.array(temp_test2))


    np.savez_compressed(args.test+str(interval)+'_test' + str(num_windows) + 'addn'+str(addn)+'_w_superpkt.npz',
             tor=np.array(test_windows1),
             exit=np.array(test_windows2))
    #np.savez('/data/seoh/DeepCCA_model/crawle_overlap_new2021_interal'+str(interval)+'_train' + str(num_windows) + 'addn'+str(addn)+'_w_superpkt.npz',
    #         tor=np.array(train_windows1), exit=np.array(train_windows2))


    train_windows1 = np.array(train_windows1)
    valid_windows1 = np.array(valid_windows1)

    train_windows2 = np.array(train_windows2)
    valid_windows2 = np.array(valid_windows2)

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    valid_labels = np.array(valid_labels)


    #print('train_windows1: ', np.array(train_windows1).shape)
    #print('train_windows2: ', np.array(train_windows2).shape)
    #print('test_windows1: ', np.array(test_windows1).shape)
    #print('test_windows2: ', np.array(test_windows2).shape)


    if model_gb == 'cnn1d':
        input_shape1 = (pad_t, 1)  # X_train[:, :, np.newaxis]
        input_shape2 = (pad_e, 1)  # X_valid[:, :, np.newaxis]
        print('cnn1d')
    else:
        input_shape1 = (2, t_flow_size, 1)  # X_train[:, :, np.newaxis]
        input_shape2 = (2, e_flow_size, 1)  # X_valid[:, :, np.newaxis]
        print('else')

    # Building, training, and producing the new features by DCCA
    # TO-DO: use different emb_size here!
    shared_model1 = create_model(input_shape=input_shape1, emb_size=64, model_name='tor')  ##
    shared_model2 = create_model(input_shape=input_shape2, emb_size=64, model_name='exit')  ##

    anchor = Input(input_shape1, name='anchor')
    positive = Input(input_shape2, name='positive')
    negative = Input(input_shape2, name='negative')

    a = shared_model1(anchor)
    p = shared_model2(positive)
    n = shared_model2(negative)


    #print('a shape', a.shape)
    #print('p shape', p.shape)
    #print('n shape', n.shape)

    #try different distance metrics
    pos_sim = Dot(axes=-1, normalize=True)([a, p])
    neg_sim = Dot(axes=-1, normalize=True)([a, n])
    #print('pos_sim shape', pos_sim.shape)
    #print('neg_sim shape', neg_sim.shape)




    # customized loss
    # alpha_value = 0.05
    def cosine_triplet_loss(X):
        _alpha = alpha_value
        positive_sim, negative_sim = X

        losses = K.maximum(0.0, negative_sim - positive_sim + _alpha)
        # if similarity is based on the distance functions, use below
        # losses = K.maximum(0.0, positive_sim - negative_sim + _alpha)
        return K.mean(losses)


    def temp_loss(X):
        a, p, n = X

        pos_sim = Dot(axes=-1, normalize=True)([a,p])
        neg_sim = Dot(axes=-1, normalize=True)([a,n])

        pos_sim = tf.reshape(pos_sim, [-1])
        neg_sim = tf.reshape(neg_sim, [-1])

        #TODO: check that the negatives aren't from the same class as the anchors/positives (look earlier in code)

        triplet_loss = K.maximum(0.0, neg_sim - pos_sim + .1)

        lower_tensor = tf.greater(triplet_loss, 1e-16)
        upper_tensor = tf.less(triplet_loss, .1)
        in_range = tf.logical_and(lower_tensor, upper_tensor)

        mask = tf.cast(lower_tensor, tf.float32)
        triplet_loss = tf.multiply(mask, triplet_loss)

        num_valid_triplets = tf.reduce_sum(mask)
        return tf.reduce_sum(triplet_loss) / (num_valid_triplets + 1e-16)

        #Consider: B^2 matrix to compute more losses

    loss = Lambda(cosine_triplet_loss, output_shape=(1,))([pos_sim, neg_sim])
    loss = Lambda(temp_loss, output_shape=(1,))([a,p,n])

    model_triplet = Model(
        inputs=[anchor, positive, negative],
        outputs=loss)
    #print(model_triplet.summary())

    opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


    def identity_loss(y_true, y_pred):
        #tf.print(y_true, sys.stdout)
        return K.mean(y_pred - 0 * y_true)


    model_triplet.compile(loss=identity_loss, optimizer=opt)

    batch_size = 512  # batch_size_value


    def intersect(a, b):
        return list(set(a) & set(b))


    def build_similarities2(conv, all_imgs):
        embs = conv.predict(all_imgs)
        embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)  # take the norm
        all_sims = np.dot(embs, embs.T)
        return all_sims


    def build_similarities(conv1, conv2, tor_t, exit_t):

        tor_embs = conv1.predict(tor_t)
        exit_embs = conv2.predict(exit_t)
        all_embs = np.concatenate((tor_embs, exit_embs), axis=0)
        all_embs = all_embs / np.linalg.norm(all_embs, axis=-1, keepdims=True)
        mid = int(len(all_embs) / 2)
        all_sims = np.dot(all_embs[:mid], all_embs[mid:].T)
        return all_sims


    import random


    def build_negatives(anc_idxs, pos_idxs, similarities, neg_imgs_idx, num_retries=50):
        # If no similarities were computed, return a random negative
        if similarities is None:
            # print(neg_imgs_idx)
            # print(anc_idxs)
            anc_idxs = list(anc_idxs)
            valid_neg_pool = neg_imgs_idx  # .difference(anc_idxs)
            #print('valid_neg_pool', valid_neg_pool.shape)
            return np.random.choice(valid_neg_pool, len(anc_idxs), replace=False)
        final_neg = []
        # for each positive pair
        for (anc_idx, pos_idx) in zip(anc_idxs, pos_idxs):
            anchor_class = anc_idx
            # print('anchor_class',anchor_class)
            valid_neg_pool = neg_imgs_idx  # .difference(np.array([anchor_class]))
            # positive similarity
            sim = similarities[anc_idx, pos_idx]
            # find all negatives which are semi(hard)

            '''
            #final_neg.append(np.random.choice(neg_imgs_idx, 1)[0])
            found = False
            for i in range(50):
                neg_idx = np.random.choice(neg_imgs_idx, 1)[0]
                if(similarities[anc_idx][neg_idx] + alpha_value > sim):
                    final_neg.append(neg_idx)
                    found = True
                    break
            
            if(not found):
                final_neg.append(np.random.choice(neg_imgs_idx, 1)[0])
            '''            
            possible_ids = np.where((similarities[anc_idx] + alpha_value) > sim)[0]
            possible_ids = intersect(valid_neg_pool, possible_ids)
            appended = False
            for iteration in range(num_retries):
                if len(possible_ids) == 0:
                    break
                idx_neg = np.random.choice(possible_ids, 1)[0]
                if idx_neg != anchor_class:
                    final_neg.append(idx_neg)
                    appended = True
                    break
            if not appended:
                final_neg.append(np.random.choice(valid_neg_pool, 1)[0])
        return final_neg


    class SemiHardTripletGenerator():
        def __init__(self, Xa_train, Xp_train, batch_size, neg_traces_train_idx, Xa_train_all, Xp_train_all, conv1,
                     conv2):
            #first half tor, second half exit, batch size, positive indices, all anchor, all positive, model1, model1
            self.batch_size = batch_size  # 128

            self.Xa = Xa_train
            self.Xp = Xp_train
            self.Xa_all = Xa_train_all
            self.Xp_all = Xp_train_all
            self.Xp = Xp_train
            self.cur_train_index = 0
            self.num_samples = Xa_train.shape[0]
            self.neg_traces_idx = neg_traces_train_idx

            #if conv1:
            if False:
                self.similarities = build_similarities(conv1, conv2, self.Xa_all,
                                                       self.Xp_all)  # build_similarities(conv, self.traces) # compute all similarities including cross pairs
            else:
                self.similarities = None


        def next_train(self):
            while 1:
                self.cur_train_index += self.batch_size
                if self.cur_train_index >= self.num_samples:  # 50k
                    self.cur_train_index = 0  # initialize the index for the next epoch
                
                if self.cur_train_index + self.batch_size >= self.num_samples:
                    break

                # fill one batch
                traces_a = np.array(range(self.cur_train_index,
                                          self.cur_train_index + self.batch_size))  # self.Xa[self.cur_train_index:self.cur_train_index + self.batch_size]
                traces_p = np.array(range(self.cur_train_index,
                                          self.cur_train_index + self.batch_size))  # self.Xp[self.cur_train_index:self.cur_train_index + self.batch_size]

                traces_n = build_negatives(traces_a, traces_p, self.similarities, self.neg_traces_idx)
                yield ([self.Xa[traces_a],
                        self.Xp[traces_p],
                        self.Xp_all[traces_n]],
                       np.zeros(shape=(traces_a.shape[0]))
                       )


    # At first epoch we don't generate hard triplets
    all_traces_train_idx = np.array(range(len(train_windows1)))
    gen_hard = SemiHardTripletGenerator(train_windows1, train_windows2, batch_size, all_traces_train_idx,
                                        train_windows1, train_windows2, None, None)
    nb_epochs = 10000
    description = 'coffeh2'

    best_loss = sys.float_info.max


    def saveModel(epoch, logs):
        global best_loss

        loss = logs['loss']

        if loss < best_loss:
            print("loss is improved from {} to {}. save the model".format(str(best_loss),
                                                                          str(loss)))

            best_loss = loss
            shared_model1.save_weights( 
                args.model + str(num_windows) + "_alt"+str(interval)+ '_addn'+str(addn)+"_model1_w_superpkt.h5")
            shared_model2.save_weights( 
                args.model + str(num_windows) + "_alt"+str(interval)+'_addn'+str(addn)+"_model2_w_superpkt.h5")
            tf.keras.models.save_model(shared_model1, args.model + str(loss) + "_best_model1.h5")
            tf.keras.models.save_model(shared_model2, args.model + str(loss) + "_best_model2.h5")
        else:
            print("loss is not improved from {}.".format(str(best_loss)))


    # csv_logger = CSVLogger('log/Training_Log_%s.csv' % description, append=True, separator=';')
    for epoch in range(nb_epochs):
        '''
        if(epoch % 25 == 0):
            shared_model1.save_weights(args.model + "_10000_epoch_" + str(epoch) + "_model1.h5")
            shared_model2.save_weights(args.model + "_10000_epoch_" + str(epoch) + "_model2.h5")
        '''
        if(epoch == 5000):
            exit()

        print("built new hard generator for epoch " + str(epoch))
        # if epoch != 0:
        if epoch % 2 == 0:
            if epoch == 0:
                model_triplet.fit_generator(generator=gen_hard.next_train(),
                                            steps_per_epoch=train_windows1.shape[0] // batch_size - 1,
                                            epochs=1, verbose=2)
            else:
                model_triplet.fit_generator(generator=gen_hard_even.next_train(),
                                            steps_per_epoch=(train_windows1.shape[0] // 2) // batch_size - 1,
                                            epochs=1, verbose=2, callbacks=[LambdaCallback(on_epoch_end=saveModel)])
        else:
            model_triplet.fit_generator(generator=gen_hard_odd.next_train(),
                                        steps_per_epoch=(train_windows1.shape[0] // 2) // batch_size - 1,
                                        epochs=1, verbose=2, callbacks=[LambdaCallback(on_epoch_end=saveModel)])

        mid = int(len(train_windows1) / 2)
        random_ind = np.array(range(len(train_windows1)))
        np.random.shuffle(random_ind)
        X1 = np.array(random_ind[:mid])
        X2 = np.array(random_ind[mid:])


        print("Re-creating generators")
        gen_hard_odd = SemiHardTripletGenerator(train_windows1[X1], train_windows2[X1], batch_size, X2, train_windows1,
                                                train_windows2,
                                                shared_model1, shared_model2)
        gen_hard_even = SemiHardTripletGenerator(train_windows1[X2], train_windows2[X2], batch_size,
                                                 X1, train_windows1, train_windows2,
                                                 shared_model1, shared_model2)
