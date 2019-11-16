#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:28:23 2019

@author: nmei

deep learn util funtions

"""


import mne
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers,Model, optimizers,losses,regularizers,callbacks#,Sequential
from keras import metrics as k_metrics
import tensorflow.keras.backend as K

def make_CallBackList(model_name,monitor='val_loss',mode='min',verbose=0,min_delta=1e-4,patience=50,frequency = 1):
    from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
    """
    Make call back function lists for the keras models
    
    Inputs
    -------------------------
    model_name: directory of where we want to save the model and its name
    monitor: the criterion we used for saving or stopping the model
    mode: min --> lower the better, max --> higher the better
    verboser: printout the monitoring messages
    min_delta: minimum change for early stopping
    patience: temporal windows of the minimum change monitoring
    frequency: temporal window steps of the minimum change monitoring
    
    Return
    --------------------------
    CheckPoint: saving the best model
    EarlyStopping: early stoppi....
    """
    checkPoint = ModelCheckpoint(model_name,# saving path
                                 monitor          = monitor,# saving criterion
                                 save_best_only   = True,# save only the best model
                                 mode             = mode,# saving criterion
#                                 save_freq        = 'epoch',# frequency of check the update 
                                 verbose          = verbose,# print out (>1) or not (0)
#                                 load_weights_on_restart = True,
                                 )
    earlyStop = EarlyStopping(   monitor          = monitor,
                                 min_delta        = min_delta,
                                 patience         = patience,
                                 verbose          = verbose, 
                                 mode             = mode,
#                                 restore_best_weights = True,
                                 )
    return [checkPoint,earlyStop]
def preprocess_features(X,vectorizer = None,scaler = None):
    if vectorizer is None:
        vectorizer          = mne.decoding.Vectorizer()
        X_vec               = vectorizer.fit_transform(X)
    else:
        X_vec               = vectorizer.transform(X)
    if scaler is None:
        scaler              = StandardScaler()
        X_vec_standard      = scaler.fit_transform(X_vec)
    else:
        X_vec_standard      = scaler.transform(X_vec)
    X_vec_tran          = vectorizer.inverse_transform(X_vec_standard)
    
    return X_vec_tran,vectorizer,scaler
def prepare_data_batch(X,y,batch_size = 32):
    """
    prepare the data for training, validating, and testing
    make sure the data in a certain range and fit to the batch size
    
    Inputs 
    -------------------------------
    X: input features, (n_sample x n_channels x n_timesteps)
    y: input labels, (n_samples x n_categories)
    batch_size: int, batch size
    Return
    -------------------------------
    processed X,y
    """
    X       = (X - X.min(0)) / (X.max(0) - X.min(0))
    remain_ = X.shape[0] % batch_size
    if remain_ != 0:
        np.random.seed(12345)
        idx_    = np.random.choice(X.shape[0],size = X.shape[0] - remain_)
        X,y     = X[idx_],y[idx_]
        
    return X,y
def call_back_dict(classifier,care = 'loss'):
    if care == 'loss':
        return dict(
                monitor     = 'val_{}'.format(classifier.metrics_names[0]),
                mode        = 'min',
                patience    = 3,
                min_delta   = 1e-3)
    elif care == 'metric':
        return dict(
                monitor     = 'val_{}'.format(classifier.metrics_names[-1]),
                mode        = 'max',
                patience    = 4,
                min_delta   = 1e-4)
    else:
        print('why?')

def build_model(timesteps,
                data_dim,
                n_units     = 1,
                batch_size  = 32,
                n_layers    = 1,
                drop        = True,
                l1          = 1e-4,
                l2          = 1e-4,):
    K.clear_session()
    inputs          = layers.Input(
                                   shape        = (timesteps,data_dim,),
                                   batch_size   = batch_size,#(batch_size,timesteps,data_dim),
                                   name         = 'inputs')
    inputs_         = inputs
    RNN,state_h,state_c = layers.LSTM(units              = n_units,
                                      activation         = 'sigmoid',
                                      return_state       = True,
                                      return_sequences   = True,
                                      kernel_regularizer = regularizers.l2(l2),
                                      recurrent_regularizer = regularizers.l2(l2),
                                      activity_regularizer = regularizers.l1(l1),
                                      name               = 'rnn{}'.format(1))(inputs_)
    RNN         = layers.BatchNormalization(
                             name               = 'norm{}'.format(1))(RNN)
    if n_units == 1:
        RNN         = layers.Lambda(lambda x: K.squeeze(x, 2))(RNN)
    else:
        RNN         = layers.GlobalAveragePooling1D(data_format = 'channels_first',
                                                    name = 'pool_units{}'.format(1))(RNN)
    for n_temp in range(n_layers - 1):
        l1 /= 10
        l2 /= 10
        RNN,state_h,state_c = layers.LSTM(units              = n_units,
                                          activation         = 'SELU',
                                          return_state       = True,
                                          return_sequences   = True,
                                          dropout            = 0.1,
                                          recurrent_dropout  = 0.25,
                                          kernel_regularizer = regularizers.l2(l2),
                                          recurrent_regularizer = regularizers.l2(l2),
                                          activity_regularizer = regularizers.l1(l1),
                                          name               = 'rnn{}'.format(n_temp + 2))(RNN,initial_state = [state_h,state_c])
        RNN         = layers.BatchNormalization(
                                 name               = 'norm{}'.format(n_temp + 2))(RNN)
        if n_units == 1:
            RNN         = layers.Lambda(lambda x: K.squeeze(x, 2))(RNN)
        else:
            RNN         = layers.GlobalAveragePooling1D(data_format = 'channels_first',
                                                        name = 'pool_units{}'.format(n_temp + 2))(RNN)
        if drop:
            RNN         = layers.Dropout(0.25,
                                         name           = 'drop{}'.format(n_temp + 2))(RNN)
    outputs             = layers.Dense(2,
                                       activation           = 'softmax',
                                       activity_regularizer = regularizers.l1(1e-6),
                                       name                 = 'outputs',)(RNN)
    classifier = Model(inputs,outputs,name = 'clf')
    return classifier