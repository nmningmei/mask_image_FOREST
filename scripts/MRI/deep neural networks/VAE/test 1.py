#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:52:17 2019

@author: nmei
"""
import tensorflow as tf
tf.enable_eager_execution()

import os
import pickle
import gc
import pandas as pd
import numpy  as np

from glob                    import glob
from sklearn.utils           import shuffle
from shutil                  import copyfile
copyfile('../../../utils.py','utils.py')
from utils                   import (groupy_average,
                                     check_train_balance,
                                     build_model_dictionary)
from sklearn.metrics         import roc_auc_score
from sklearn.preprocessing   import MinMaxScaler
from nilearn.decoding        import SpaceNetClassifier
from nilearn.input_data      import NiftiMasker

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

sub                 = 'sub-01'
stacked_data_dir    = '../../../../data/BOLD_no_average/{}/'.format(sub)
split_dir           = '../../../../results/MRI/customized_partition'
output_dir          = '../../../../results/MRI/DNN/VAE'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
BOLD_data           = glob(os.path.join(stacked_data_dir,'*BOLD.npy'))
event_data          = glob(os.path.join(stacked_data_dir,'*.csv'))
label_map           = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}
average             = True
n_splits            = 10

conscious_state = 'unconscious'
conscious_state
with open(os.path.join(split_dir,f'{conscious_state}.pkl'),'rb') as f:
    # disable garbage collection to speed up pickle
    gc.disable()
    train_test_split = pickle.load(f)
    f.close()

idx = 0
np.random.seed(12345)
BOLD_name,df_name = BOLD_data[idx],event_data[idx]
BOLD            = np.load(BOLD_name)
df_event        = pd.read_csv(df_name)
roi_name        = df_name.split('/')[-1].split('_events')[0]
idx_unconscious = df_event['visibility'] == conscious_state
data            = BOLD[idx_unconscious]
df_data         = df_event[idx_unconscious].reset_index(drop=True)
df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
targets         = np.array([label_map[item] for item in df_data['targets'].values])[:,-1]
idxs_train,idxs_test = train_test_split['train'],train_test_split['test']

idx_train,idx_test = idxs_train[0],idxs_test[0]
X_train,y_train = data[idx_train],targets[idx_train]

X_test,y_test   = data[idx_test],targets[idx_test]
df_test         = df_data.iloc[idx_test].reset_index(drop=True)
X_test_ave,temp = groupy_average(X_test,df_test,groupby=['id'])
y_test          = np.array([label_map[item] for item in temp['targets']])[:,-1]

import tensorflow.contrib.distributions as tfd
class VAE(object):
    def __init__(self, kpi, z_dim=None, n_dim=None, hidden_layer_sz=None):
        """
        Args:
          z_dim : dimension of latent space.
          n_dim : dimension of input data.
        """
        if not z_dim or not n_dim:
            raise ValueError("You should set z_dim"
                             "(latent space) dimension and your input n_dim."
                             " \n            ")

        tf.reset_default_graph()
        
        def make_prior(code_size):
            loc = tf.zeros(code_size)
            scale = tf.ones(code_size)
            return tfd.MultivariateNormalDiag(loc, scale)

        self.z_dim = z_dim
        self.n_dim = n_dim
        self.kpi = kpi
        self.dense_size = hidden_layer_sz
        
        self.input = tf.placeholder(dtype=tf.float32,shape=[None, n_dim], name='KPI_data')
        self.batch_size = tf.placeholder(tf.int64, name="init_batch_size")

        # tf.data api
        dataset = tf.data.Dataset.from_tensor_slices(self.input).repeat() \
            .batch(self.batch_size)
        self.ite = dataset.make_initializable_iterator()
        self.x = self.ite.get_next()
        
        # Define the model.
        self.prior = make_prior(code_size=self.z_dim)
        x = tf.contrib.layers.flatten(self.x)
        x = tf.layers.dense(x, 512, tf.nn.relu)
        x = tf.layers.dense(x, 128, tf.nn.relu)
        loc = tf.layers.dense(x, self.z_dim)
        scale = tf.layers.dense(x, self.z_dim , tf.nn.softplus)
        self.posterior = tfd.MultivariateNormalDiag(loc, scale)
        self.code = self.posterior.sample()

        # Define the loss.
        x = self.code
        x = tf.layers.dense(x, 128, tf.nn.relu)
        x = tf.layers.dense(x, 512, tf.nn.relu)
        loc = tf.layers.dense(x, self.n_dim)
        scale = tf.layers.dense(x, self.n_dim , tf.nn.softplus)
        self.decoder = tfd.MultivariateNormalDiag(loc, scale)
        self.likelihood = self.decoder.log_prob(self.x)
        self.divergence = tf.contrib.distributions.kl_divergence(self.posterior, self.prior)
        self.elbo = tf.reduce_mean(self.likelihood - self.divergence)
        self._cost = -self.elbo
        
        self.saver = tf.train.Saver()
        self.sess = tf.Session()    

    def fit(self, Xs, learning_rate=0.001, num_epochs=10, batch_sz=200, verbose=True):
        
        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(self._cost)

        batches_per_epoch = int(np.ceil(len(Xs[0]) / batch_sz))
        print("\n")
        print("Training anomaly detector/dimensionalty reduction VAE for KPI",self.kpi)
        print("\n")
        print("There are",batches_per_epoch, "batches per epoch")
        
        self.sess.run(tf.global_variables_initializer())
        
        for epoch in range(num_epochs):
            train_error = 0

            
            self.sess.run(
                self.ite.initializer,
                feed_dict={
                    self.input: Xs,
                    self.batch_size: batch_sz})

            for step in range(batches_per_epoch):
                _, loss = self.sess.run([self.optimize, self._cost])
                train_error += loss
                if step == (batches_per_epoch - 1):
                        mean_loss = train_error / batches_per_epoch   
            if verbose:
                print(
                    "Epoch {:^6} Loss {:0.5f}"  .format(
                        epoch + 1, mean_loss))
                
            if train_error == np.nan:
                return False
        print("\n")
        return True
























































