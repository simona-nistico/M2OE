import logging
import pickle
from itertools import chain
from time import time
import os
from sklearn import preprocessing
from sklearn.datasets import make_blobs

import numpy as np
import numpy.random
import tensorflow as tf
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from explainers import TabularMM_groups as AETabularMMmd_ae

gpus = tf.config.list_physical_devices('GPU')
if gpus:
      for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)


def run_test(path, **kwargs):
    epochs_exp = kwargs.pop('exp_epochs')
    batch_exp = kwargs.pop('exp_batch')
    loss_weights = kwargs.pop('loss_weights')
    no = kwargs.pop('n_other')
    lr = kwargs.pop('learning_rate')
    threshold = kwargs.pop('threshold')
    n_runs = kwargs.pop('n_runs')
    gpu_to_use = kwargs.pop('gpu_to_use')
    dataset = kwargs.pop('dataset')
    data_path = 'datasets/synthetic/processed/'
    
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[gpu_to_use: gpu_to_use+1], 'GPU')


    print(f'Create dataset...')
    centers = [(0, 0), (20, 20)]
    cluster_std = [0.5, 0.5]

    X, Y = make_blobs(n_samples=400, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
    X = np.append(X, np.array([[0., 20.]]), axis=0)
    Y = np.append(Y, [2], axis=0)
 
    y_train = np.zeros(X.shape[0])
    y_train[-1] = 1.
    x_train = np.random.rand(X.shape[0], 17)
    x_train[:,3] = X[:,0]
    x_train[:,10] = X[:, 1]
    print('dataset created!')

    in_shape = x_train.shape[1]

    near_neigh = NearestNeighbors(n_neighbors=no)
    near_neigh.fit(x_train[y_train == 0])
    
    for i in range(1):
        f_choose = []
        new_points = []
        for ete in np.argwhere(y_train != 0):
            print(f'EXPLAINING POINT: {ete}')
            x_train_sub = x_train[y_train == 0]
            x_train_sub = np.append(x_train_sub, x_train[ete], axis=0)
            y_train_sub = y_train[y_train == 0]
            y_train_sub = np.append(y_train_sub, [1.], axis=0)
            # -------------------- Try with more samples ---------------------------
            #_, y_normal = near_neigh.kneighbors(x_train[ete])
            #x_normal = x_train[y_train == 0][y_normal[0]]
            x_normal = x_train[Y==0][:5]
            print('SHAPE: ', x_normal.shape)
            x_normal = np.append(x_normal, x_train[Y==1][:no-5], axis=0)
            #
            mm_data_o = np.full_like(x_normal, fill_value=x_train[ete])
            mm_data_i = x_normal
            mm_data = [mm_data_o, mm_data_i]
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
            explainer = AETabularMMmd_ae.TabularMM(in_shape, x_train[y_train == 0], loss_weights)
            explainer.compile(optimizer=opt, run_eagerly=True)
            start_time = time()
            explainer.fit(mm_data, mm_data_i, batch_size=batch_exp, epochs=epochs_exp, verbose=0)
            tot_time = time() - start_time
            print('Elapsed time: ', tot_time)
            logging.info(f'Elapsed time explanation: {tot_time}')
            # Our method
            new_sample, choose = explainer.explain([mm_data[0], mm_data[1]], threshold, combine=False)
            #explainer.save_weights(os.path.join(path, f'exp_{dataset}_{ete[0]}_{i}.hdf5'))
            f_choose.append(choose)
            new_points.append(new_sample)
            print(choose[0])
            print(explainer.WEIGHTS(mm_data[0]))
            #print(explainer.CHOOSE([np.ones((1,17)), np.ones((1,17))]))
            print(explainer.MASK([mm_data[0], mm_data[1]])[:,3])
            print(explainer.MASK([mm_data[0], mm_data[1]])[:,10])

        pickle.dump(f_choose, open(os.path.join(path, f'choose_{i}.joblib'), 'wb'))
        pickle.dump(new_points, open(os.path.join(path, f'new_points_{i}.joblib'), 'wb'))