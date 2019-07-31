#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 15:52:28 2019

@author: nmei
"""
import os
import mne
import re
import numpy as np
import pandas as pd
from datetime import datetime
from glob import glob
from shutil import copyfile
copyfile('../utils.py','utils.py')
from utils import get_frames
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
import tensorflow as tf
from keras import layers,Model, optimizers,losses,regularizers,callbacks#,Sequential
from keras import metrics as k_metrics
import keras.backend as K

def make_CallBackList(model_name,monitor='val_loss',mode='min',verbose=0,min_delta=1e-4,patience=50,frequency = 1):
    from keras.callbacks import ModelCheckpoint,EarlyStopping
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
                                 period           = frequency,# frequency of check the update 
                                 verbose          = verbose# print out (>1) or not (0)
                                 )
    earlyStop = EarlyStopping(   monitor          = monitor,
                                 min_delta        = min_delta,
                                 patience         = patience,
                                 verbose          = verbose, 
                                 mode             = mode,
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

#def call_back_dict(classifier,temp,care = 'loss'):
#    if care == 'loss':
#        c = (temp[0] + 1) * 1e-3
#        print(f'criterion = {c:.4f}')
#        return dict(
#                monitor = 'val_{}'.format(classifier.metrics_names[0]),
#                mode = 'min',
#                patience = 1,
#                min_delta = c)
#    elif care == 'metric':
#        c = (temp[1] + 1) * 1e-4
#        print(f'criterion = {c:.4f}')
#        return dict(
#                monitor = 'val_{}'.format(classifier.metrics_names[-1]),
#                mode = 'max',
#                patience = 1,
#                min_delta = c)
#    else:
#        print('why?')

def build_model(timesteps,data_dim,n_units = 1,batch_size = 10,n_layers = 3,drop = True):
    K.clear_session()
    inputs      = layers.Input(
                       shape = (timesteps,data_dim,),
                       batch_shape = (batch_size,timesteps,data_dim),
                       name = 'inputs')
    inputs_ = inputs
    RNN,state_h = layers.GRU(units = n_units,
                             return_state = True,
                             return_sequences = True,
                             name = 'rnn{}'.format(1))(inputs_)
    for n_temp in range(n_layers - 1):
        RNN,state_h = layers.GRU(units = n_units,
                                 return_state = True,
                                 return_sequences = True,
                                 dropout = 0.1,
                                 recurrent_dropout = 0.25,
                                 name = 'rnn{}'.format(n_temp + 2))(RNN,initial_state = [state_h])
        if drop:
            RNN = layers.Dropout(0.25,name = 'drop{}'.format(n_temp + 1))(RNN)
    RNN = layers.GlobalAveragePooling1D(data_format = 'channels_first',name = 'average')(RNN) #TODO: what is the rational of this?
    outputs = layers.Dense(2,activation = 'softmax',
                           activity_regularizer = regularizers.l1(1),
                           name = 'outputs',)(RNN)
    classifier = Model(inputs,outputs,name = 'clf')
    return classifier

all_subjects = [
                'matie_5_23_2019',
                'pedro_5_14_2019',
                'aingere_5_16_2019',
                'inaki_5_9_2019',
                'clara_5_22_2019',
                'ana_5_21_2019',
                'leyre_5_13_2019',
                'lierni_5_20_2019',
                'xabier_5_15_2019',
                'maria_6_5_2019',
                'jesica_6_7_2019',
                ]
all_subjects = np.sort(all_subjects)

for subject in all_subjects:
    working_dir         = f'../../data/clean EEG/{subject}'
    working_data        = glob(os.path.join(working_dir,'*-epo.fif'))
    # there was a bug in the csv file, so the early behavioral is treated differently
    date                = '/'.join(re.findall(r'\d+',subject))
    date                = datetime.strptime(date,'%m/%d/%Y')
    breakPoint          = datetime(2019,3,10)
    if date > breakPoint:
        new             = True
    else:
        new             = False
    frames,_            = get_frames(directory = f'../../data/behavioral/{subject}',new = new)
    n_splits            = 25
    n_epochs            = int(2e2)
    print_model         = True
    
    
    for epoch_file in working_data:
        epochs  = mne.read_epochs(epoch_file)
        # resample at 100 Hz to fasten the decoding process
        print('resampling')
        epochs.resample(100)
        
        conscious   = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if (' conscious' in name)])
        see_maybe   = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('glimpse' in name)])
        unconscious = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('unconscious' in name)])
        del epochs
        results = []
        for ii,(epochs,conscious_state) in enumerate(zip([unconscious.copy(),
                                                          see_maybe.copy(),
                                                          conscious.copy()],
                                                         ['unconscious',
                                                          'glimpse',
                                                          'conscious'])):
            epochs
            epochs = epochs.pick_types(eeg=True)
            
            X_,y_               = epochs.get_data(),epochs.events[:,-1]
            y_                  = y_ //100 - 2
            
            
            X,targets           = X_.copy(),y_.copy()
            targets             = np.vstack([targets,1-targets]).T
            
            X                   = mne.decoding.Scaler(epochs.info).fit_transform(X)
            
            X                   = np.swapaxes(X,1,2)
            ss                  = []
            cv                  = StratifiedShuffleSplit(n_splits=n_splits,test_size = 0.15,random_state=12345)
            for fold,(idx_,idx_test) in enumerate(cv.split(X,targets)):
                
                X_train,X_valid,y_train,y_valid = train_test_split(
                        X[idx_],targets[idx_],
                        test_size               = 0.15,
                        random_state            = 12345,
                        shuffle                 = True,)
                
#                X_train,vectorizer,scaler = preprocess_features(X_train)
#                X_valid,_,_ = preprocess_features(X_valid,vectorizer,scaler)
                X_train = X_train / np.abs(X_train.max())
                X_valid = X_valid / np.abs(X_valid.max())
                
                
                model_name  = 'RNN.hdf5'
                batch_size  = 10
                timesteps   = X.shape[1]
                data_dim    = X.shape[2]
                n_units     = 10
                n_layers    = 1
                dropout     = True
                
                remain_train = X_train.shape[0] % batch_size
                remain_valid = X_valid.shape[0] % batch_size
                
                if remain_train != 0:
                    np.random.seed(12345)
                    idx_train = np.random.choice(X_train.shape[0],size = X_train.shape[0] - remain_train)
                    X_train,y_train = X_train[idx_train],y_train[idx_train]
                
                if remain_valid != 0:
                    np.random.seed(12345)
                    idx_valid = np.random.choice(X_valid.shape[0],size = X_valid.shape[0] - remain_valid)
                    X_valid,y_valid = X_valid[idx_valid],y_valid[idx_valid]
                
                from collections import Counter
                class_weight = dict(Counter(y_train[:,0]))
                class_weight = {key:(y_train.shape[0] - value)/value for key,value in class_weight.items()}
                print(class_weight)
                sample_weight = [class_weight[item] for item in y_train[:,0]]
                
                classifier = build_model(timesteps=timesteps,
                                         data_dim=data_dim,
                                         n_units=n_units,
                                         batch_size=batch_size,
                                         n_layers = n_layers,
                                         drop=dropout)
                print(f'the model has {classifier.count_params()} parameters')
                if print_model:
                    classifier.summary()
                    print_model = False
                classifier.compile(optimizer                = optimizers.Adam(lr = 1e-4),
                                   loss                     = losses.categorical_crossentropy,
                                   metrics                  = [k_metrics.categorical_accuracy])
                
                temp = [1,5]
                def call_back_dict(care = 'loss'):
                    if care == 'loss':
                        return dict(
                                monitor = 'val_{}'.format(classifier.metrics_names[0]),
                                mode = 'min',
                                patience = 3,
                                min_delta = 1e-2)
                    elif care == 'metric':
                        return dict(
                                monitor = 'val_{}'.format(classifier.metrics_names[-1]),
                                mode = 'max',
                                patience = 4,
                                min_delta = 1e-4)
                    else:
                        print('why?')
                callBackList = make_CallBackList(
                       model_name,
                       verbose          = 0,# print out the process
                       frequency        = 1,
                       **call_back_dict('metric')
                       )
                
                np.random.seed(12345)
                X_train,y_train = shuffle(X_train,y_train)
                
                classifier.fit(X_train,y_train,
                               batch_size               = batch_size,
                               epochs                   = n_epochs,
                               validation_data          = (X_valid,y_valid),
                               callbacks                = callBackList,
                               shuffle                  = True,
                               sample_weight            = np.array(sample_weight), # this is the key !
                               )
                
                
                classifier.load_weights(model_name)
                
                
#                X_test,_,_ = preprocess_features(X[idx_test],vectorizer,scaler)
                X_test = X[idx_test] / np.abs(X[idx_test].max())
                remain_test = X_test.shape[0] % batch_size
                if remain_test != 0:
                    X_test,y_test = X_test[:-remain_test],targets[idx_test][:-remain_test]
                else:
                    y_test = targets[idx_test]
                
                
                preds = classifier.predict(X_test,batch_size=batch_size,verbose = 0)
                
                ss.append(metrics.roc_auc_score(y_test,
                                                preds))
                print(f'{conscious_state},fold {fold+1},{ss[-1]:.4f}')
                K.clear_session()
            results.append([conscious_state,ss])
    print([np.mean(item[1]) for item in results])
    
    report_dir = '../../results/EEG/DNN_reports'
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    with open(os.path.join(report_dir,f'RNN_{subject}.txt'),'w') as f:
        [f.write(f'{item[0]:10} {np.mean(item[1]):.4f} +/- {np.std(item[1]):.4f}\n') for item in results]
    #    f.write(classifier.summary())
        f.close()




















