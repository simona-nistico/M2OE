import logging
import pickle
from itertools import chain
from time import time
import os
from sklearn import preprocessing

import numpy as np
import tensorflow as tf
import pandas as pd
from pyod.models.sod import SOD
from sklearn.neighbors import NearestNeighbors
from math import log2

from explainers import TabularMM_groups as AETabularMMmd_ae
from utils.utils import _sod_2, calcola_knn_score


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[0:1], 'GPU')


def run_test(path, **kwargs):
    epochs_exp = kwargs.pop('exp_epochs')
    batch_exp = kwargs.pop('exp_batch')
    loss_weights = kwargs.pop('loss_weights')
    no = kwargs.pop('n_other')
    lr = kwargs.pop('learning_rate')
    threshold = kwargs.pop('threshold')
    n_runs = kwargs.pop('n_runs')
    gpu_to_use = kwargs.pop('gpu_to_use')
    data_path = 'datasets/synthetic/processed/'

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[gpu_to_use: gpu_to_use+1], 'GPU')

    logging.basicConfig(filename=os.path.join(path, 'run_log.log'), level=logging.INFO, format='%(message)s')

    files = os.listdir(data_path)
    files.sort()
    #print(files)
    for f in files:  # [45:]
        if not (f.startswith('info_') or ('100' in f)):
            for r in range(n_runs):
                print(f'FILE: {f}')
    
                data = pd.read_csv(data_path + f)
                x_train = data.iloc[:, :data.shape[1] - 1].to_numpy()
                y_train = data['class'].to_numpy()
    
                in_shape = x_train.shape[1]
    
                near_neigh = NearestNeighbors(n_neighbors=no)
                near_neigh.fit(x_train[y_train == 0])
    
                prec_f = []
                sens_f = []
                times = []
    
                for c in np.unique(data['class'].to_numpy())[np.unique(data['class'].to_numpy())>0]:
                    print(f'EXPLAINING CLASS: {c}, num points: {x_train[y_train==c].shape[0]}')
    
                    x_train_sub = x_train[y_train == 0]
                    x_train_sub = np.append(x_train_sub, x_train[y_train==c], axis=0)
                    y_train_sub = y_train[y_train == 0]
                    y_train_sub = np.append(y_train_sub, np.ones(x_train[y_train==c].shape[0]), axis=0)
    
                    # -------------------- Try with more samples ---------------------------
                    _, y_normal = near_neigh.kneighbors(x_train[y_train==c][:1])
                    x_normal = x_train[y_train == 0][y_normal[0]]
                    #print('SHAPE: ', x_normal.shape)
    
                    mm_data_o = np.full_like(x_normal, fill_value=x_train[y_train==c][0])
                    mm_data_i = x_normal
                
                    for op in range(1, x_train[y_train==c].shape[0]):
                        _, y_normal = near_neigh.kneighbors(x_train[y_train==c][op:op+1])
                        x_normal = x_train[y_train == 0][y_normal[0]]
                        #print('SHAPE: ', x_normal.shape)
    
                        mm_data_o = np.append(mm_data_o, np.full_like(x_normal, x_train[y_train==c][op]), axis=0)
                        mm_data_i = np.append(mm_data_i, x_normal, axis=0)
                    mm_data = [mm_data_o, mm_data_i]
                    #print('SHAPE: ', mm_data_o.shape)
    
                    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    
                    explainer = AETabularMMmd_ae.TabularMM(in_shape, x_train[y_train == 0], loss_weights)
                    explainer.compile(optimizer=opt, run_eagerly=True)
    
                    start_time = time()
                    explainer.fit(mm_data, mm_data_i, batch_size=batch_exp, epochs=epochs_exp, verbose=0)
                    tot_time = time() - start_time
    
                    times.append(tot_time)
    
                    # Our method
                    new_sample, choose = explainer.explain([mm_data[0], mm_data[1]], threshold)
                    choose = np.argwhere(choose >= threshold).reshape(-1)
                    #print(choose)
                    #print('CLASS: ', c,'CHOSEN: ', choose)
                    
                    # print('NEW POINT: ', new_sample)
                    # print('ORIGINAL POINT: ', mm_data[0][0])
                    
                    # EVALUATION
                    info_df = pd.read_csv(data_path + f.replace('synth_', 'info_synth_'))
                    subspaces = pickle.load(open(data_path + f.replace('synth_', 'info_synth_').replace('.csv', '.joblib'), 'rb'))
                    features = subspaces[int(log2(c))]
                    d_found = set(choose)
                    d_real = set(features)
                    sens = len(d_real.intersection(d_found)) / len(d_real)
                    if len(d_found) > 0:
                        prec = len(d_real.intersection(d_found)) / len(d_found)
                    else:
                        prec = 0
                    
                    prec_f.append(prec)
                    sens_f.append(sens)
                
                print(f'PREC {np.array(prec_f).mean()}, REC {np.array(sens_f).mean()}')
                logging.info(f'{f} {prec_f} {sens_f} {times}')
                    
        else:
            print(f'File {f} skipped!')
