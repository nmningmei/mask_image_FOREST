#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:16:55 2019

@author: nmei
"""

import os
import gc
import pickle

from shutil import copyfile
copyfile('../../../utils.py','utils.py')
import utils
# Dependency imports
from glob import glob
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def build_input_pipeline(x, y, batch_size):
    """Build a Dataset iterator for supervised classification.
      Args:
    x: Numpy `array` of features, indexed by the first dimension.
    y: Numpy `array` of labels, with the same first dimension as `x`.
    batch_size: Number of elements in each training batch.
      Returns:
    batch_features: `Tensor` feed  features, of shape
      `[batch_size] + x.shape[1:]`.
    batch_labels: `Tensor` feed of labels, of shape
      `[batch_size] + y.shape[1:]`.
    """
    training_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    training_batches = training_dataset.repeat().batch(batch_size)
    training_iterator = tf.compat.v1.data.make_one_shot_iterator(training_batches)
    batch_features, batch_labels = training_iterator.get_next()
    return batch_features, batch_labels

sub                 = 'sub-01'
stacked_data_dir    = '../../../../data/BOLD_no_average/{}/'.format(sub)
split_dir           = '../../../../results/MRI/customized_partition'
mask_dir            = '../../../../data/MRI/{}/anat/ROI_BOLD'.format(sub)
output_dir          = '../../../../results/MRI/decoding'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
masks               = glob(os.path.join(mask_dir,'*.nii.gz'))
label_map           = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}
average             = False
n_splits            = 10
BOLD_data           = glob(os.path.join(stacked_data_dir,'*BOLD.npy'))
event_data          = glob(os.path.join(stacked_data_dir,'*.csv'))

conscious_state ='unconscious'
mask_file = masks[0]
with open(os.path.join(split_dir,'{conscious_state}.pkl'.format(conscious_state=conscious_state)),'rb') as f:
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
for fold,(idx_train,idx_test) in enumerate(zip(idxs_train[:n_splits],idxs_test)):
    # check balance 
    idx_train = utils.check_train_balance(df_data,idx_train,list(label_map.keys()))
    if average:
        X_,df_ = utils.groupy_average(data[idx_train],df_data.iloc[idx_train].reset_index(drop=True),groupby=['id'])
        X,y = X_,np.array([label_map[item] for item in df_['targets'].values])[:,-1]
    else:
        X,y             = data[idx_train],targets[idx_train]
    X,y             = shuffle(X,y)
    X_test,y_test   = data[idx_test],targets[idx_test]
    df_test         = df_data.iloc[idx_test].reset_index(drop=True)
    X_test_ave,temp = utils.groupy_average(X_test,df_test,groupby=['id'])
    y_test          = np.array([label_map[item] for item in temp['targets']])[:,-1]

    features, labels = build_input_pipeline(X, y, 32)
    
    # Define a logistic regression model as a Bernoulli distribution
    # parameterized by logits from a single linear layer. We use the Flipout
    # Monte Carlo estimator for the layer: this enables lower variance
    # stochastic gradients than naive reparameterization.
    with tf.compat.v1.name_scope("logistic_regression", values=[features]):
        layer = tfp.layers.DenseFlipout(
            units=1,
            activation=None,
            kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
            bias_posterior_fn=tfp.layers.default_mean_field_normal_fn())
        logits = layer(features)
        labels_distribution = tfd.Bernoulli(logits=logits)
    
    # Compute the -ELBO as the loss, averaged over the batch size.
    neg_log_likelihood = -tf.reduce_mean(input_tensor=labels_distribution.log_prob(labels))
    kl = sum(layer.losses) / X.shape[0]
    elbo_loss = neg_log_likelihood + kl
    
    # Build metrics for evaluation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    predictions = tf.cast(logits > 0, dtype=tf.int32)
    accuracy, accuracy_update_op = tf.metrics.accuracy(labels=labels, predictions=predictions)
    
    with tf.compat.v1.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(
                learning_rate=1e-3)
        train_op = optimizer.minimize(elbo_loss)

    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                       tf.compat.v1.local_variables_initializer())
    
    with tf.Session() as sess:
        sess.run(init_op)
        
        # Fit the model to data.
        for step in range(3000):
            _ = sess.run([train_op, accuracy_update_op])
            if step % 100 == 0:
                loss_value, accuracy_value = sess.run([elbo_loss, accuracy])
                print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
                        step, loss_value, accuracy_value))

        # Visualize some draws from the weights posterior.
        w_draw = layer.kernel_posterior.sample()
        b_draw = layer.bias_posterior.sample()
        candidate_w_bs = []
        for _ in range(50):
            w, b = sess.run((w_draw, b_draw))
            candidate_w_bs.append((w, b))
    




















































