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
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from keras import layers,Model, optimizers,losses,regularizers#,Sequential
from keras import metrics as k_metrics
import keras.backend as K
import tensorflow as tf
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

def binary_crossentropy_with_ranking(y_true, y_pred):
    """ Trying to combine ranking loss with numeric precision"""
    # first get the log loss like normal
    logloss = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
    
    # next, build a rank loss
    
    # clip the probabilities to keep stability
    y_pred_clipped = K.clip(y_pred, K.epsilon(), 1-K.epsilon())

    # translate into the raw scores before the logit
    y_pred_score = K.log(y_pred_clipped / (1 - y_pred_clipped))

    # determine what the maximum score for a zero outcome is
    y_pred_score_zerooutcome_max = K.max(y_pred_score * K.cast(y_true < 1,dtype = 'float32'))

    # determine how much each score is above or below it
    rankloss = y_pred_score - y_pred_score_zerooutcome_max

    # only keep losses for positive outcomes
    rankloss = rankloss * y_true

    # only keep losses where the score is below the max
    rankloss = K.square(K.clip(rankloss, -100, 0))

    # average the loss for just the positive outcomes
    rankloss = K.sum(rankloss, axis=-1) / (K.sum(K.cast(y_true > 0,dtype = 'float32')) + 1)

    # return (rankloss + 1) * logloss - an alternative to try
    return rankloss + logloss

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
        scaler              = MinMaxScaler()
        X_vec_standard      = scaler.fit_transform(X_vec)
    else:
        X_vec_standard      = scaler.transform(X_vec)
    X_vec_tran          = vectorizer.inverse_transform(X_vec_standard)
    
    return X_vec_tran,vectorizer,scaler
def cnn_block(x,filters = 128 , kernel_size = 10,idx_ = 0,lamda = 1e-2,beta = 1e-2):
    cnn       = layers.Conv1D(
                                 filters                = filters,
                                 kernel_size            = int(kernel_size),
                                 padding                = 'valid',
                                 data_format            = 'channels_last',
                                 activation             = 'selu',
                                 kernel_regularizer     = regularizers.l2(lamda),
                                 activity_regularizer   = regularizers.l1(beta),
                                 name                   = f'cnn_{idx_}')(x)
    cnn         = layers.AveragePooling1D(data_format = 'channels_last',name = f'pool_{idx_ + 1}')(cnn)
    cnn         = layers.BatchNormalization(name = f'norm{idx_ + 1}')(cnn)
    cnn         = layers.Dropout(0.5,name = f'drop_{idx_ + 1}')(cnn)
    return cnn
def build_model(timesteps,data_dim,batch_size = 10,n_cnn_layers = 4,
                n_rnn_layers = 2,initial_beta = 1e-10,n_filters = 64,
                n_units = 1,kernel_size = 5):
    
    inputs      = layers.Input(
                       shape                    = (timesteps,data_dim),
                       batch_shape              = (batch_size,timesteps,data_dim),
                       name                     = 'inputs'
                       )
    inputs_     = inputs 
    beta        = initial_beta
    for n_cnn in range(n_cnn_layers):
        beta *= 10
        inputs_ = cnn_block(inputs_,
                            kernel_size = kernel_size,
                            filters = int(n_filters),
                            idx_ = n_cnn,
                            beta = beta,)
#        n_filters /= 2
    RNN,state_h = layers.GRU(units = n_units,
                             return_state = True,
                             return_sequences = True,
                             name = 'rnn{}'.format(1))(inputs_)
    for n_rnn in range(n_rnn_layers - 1):
            RNN,state_h = layers.GRU(units = n_units,
                                     return_state = True,
                                     return_sequences = True,
                                     dropout = 0.,
                                     recurrent_dropout = 0.2,
                                     name = 'rnn{}'.format(n_rnn + 2))(RNN,
                                                     initial_state = [state_h])
    
    RNN = layers.GlobalAveragePooling1D(data_format = 'channels_first',name = 'average')(RNN) #TODO: what is the rational of this?
#    RNN = layers.Dropout(0.5,name = 'drop')(RNN)
    outputs = layers.Dense(1,activation = 'sigmoid',
               activity_regularizer = regularizers.l1(1),
               name = 'outputs',)(RNN)
    classifier = Model(inputs,outputs,name = 'clf')
    return classifier
# subsample imbalanced classes
def subsampling(X_train,y_train,X_valid,y_valid,class_weight):
    idx_ = {value:key for key,value in class_weight.items()}
    idx_ = idx_[np.max(list(class_weight.values()))]
    print('major class is {}'.format(idx_))
    ii = y_train == idx_
    X_major = X_train[ii]
    y_major = y_train[ii]
    ii = y_train != idx_
    X_minor = X_train[ii]
    y_minor = y_train[ii]
    
    np.random.seed(12345)
    idx_subsample = np.random.choice(X_major.shape[0],size = X_minor.shape[0],replace = False)
    
    X_train = np.concatenate([X_major[idx_subsample],X_minor])
    y_train = np.concatenate([y_major[idx_subsample],y_minor])
    
    X_valid = np.concatenate([X_valid,X_major[[item for item in np.arange(X_major.shape[0]) if (item not in idx_subsample)]]])
    y_valid = np.concatenate([y_valid,y_major[[item for item in np.arange(X_major.shape[0]) if (item not in idx_subsample)]]])
    
    return X_train,y_train,X_valid,y_valid
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
    n_epochs            = int(400)
    print_model         = True
    
    
    
    for epoch_file in working_data:
        epochs  = mne.read_epochs(epoch_file)
        # resample at 100 Hz to fasten the decoding process
#        print('resampling')
#        epochs.resample(100)
        report_dir = '../../results/EEG/DNN_reports'
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        with open(os.path.join(report_dir,f'CRNN_{subject}.txt'),'w') as f:
            f.close()
        conscious   = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if (' conscious' in name)])
        see_maybe   = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('glimpse' in name)])
        unconscious = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('unconscious' in name)])
        del epochs
        results = []
        for ii,(epochs,conscious_state) in enumerate(zip([
                                                          unconscious.copy(),
                                                          see_maybe.copy(),
                                                          conscious.copy()],
                                                         [
                                                          'unconscious',
                                                          'glimpse',
                                                          'conscious'])):
            epochs
            
            K.clear_session()
            X_,y_               = epochs.get_data(),epochs.events[:,-1]
            y_                  = y_ //100 - 2
            
            
            X,targets           = X_.copy(),y_.copy()
#            targets             = np.vstack([targets,1-targets]).T
            
            
            
            X                   = np.swapaxes(X,1,2)
            ss                  = []
            cv                  = StratifiedShuffleSplit(n_splits=n_splits,test_size = 0.15,random_state=12345)
            
            for fold,(idx_TRAIN,idx_test) in enumerate(cv.split(X,targets)):
                
                X_TRAIN,target_TRAIN = X[idx_TRAIN],targets[idx_TRAIN]
                
                
                from collections import Counter
                f = lambda x:  x * 5 if x > 1 else x 
                try:
                    class_weight = dict(Counter(target_TRAIN[:,-1]))
#                    class_weight = {key:(target_TRAIN.shape[0] - value)/value for key,value in class_weight.items()}
#                    class_weight = {key:f(value) for key,value in class_weight.items()}
                    print(class_weight)
                except:
                    class_weight = dict(Counter(target_TRAIN))
#                    class_weight = {key:(target_TRAIN.shape[0] - value)/value for key,value in class_weight.items()}
#                    class_weight = {key:f(value) for key,value in class_weight.items()}
                    print(class_weight)
                
                
                cv_valid = StratifiedShuffleSplit(n_splits = 2,test_size = 0.2,random_state = 12345)
                for idx_train_,idx_valid_ in cv_valid.split(X_TRAIN,target_TRAIN):
                    idx_train_
                
                X_train,X_valid,y_train,y_valid = (X_TRAIN[idx_train_],
                                                   X_TRAIN[idx_valid_],
                                                   target_TRAIN[idx_train_],
                                                   target_TRAIN[idx_valid_])
                
                X_train,y_train,X_valid,y_valid = subsampling(X_train,y_train,X_valid,y_valid,class_weight)
#                X_train,vectorizer,scaler = preprocess_features(X_train)
#                X_valid,_,_ = preprocess_features(X_valid,vectorizer,scaler)
                X_train = X_train / np.abs(X_train.max())
                X_valid = X_valid / np.abs(X_valid.max())
                
                model_name  = 'CRNN.hdf5'
                batch_size  = 10
                timesteps   = X.shape[1]
                data_dim    = X.shape[2]
                n_units     = 1
                
                
                classifier = build_model(timesteps,data_dim,
                                         n_cnn_layers = 4,
                                         n_rnn_layers = 1,
                                         initial_beta = 1e-8,
                                         n_filters = 64,
                                         kernel_size = 1,
                                         batch_size = batch_size,
                                         n_units=n_units)
                
                print(f'the model has {classifier.count_params()} parameters')
                if print_model:
                    classifier.summary()
                    print_model = False
                
                
                classifier.compile(optimizer                = optimizers.adam(lr = 1e-1),
                                   loss                     = losses.binary_crossentropy,
                                   metrics                  = [k_metrics.mse])
                
#                from collections import Counter
#                f = lambda x:  x * 5 if x > 1 else x 
#                try:
#                    class_weight = dict(Counter(y_train[:,-1]))
#                    class_weight = {key:(y_train.shape[0] - value)/value for key,value in class_weight.items()}
#                    class_weight = {key:f(value) for key,value in class_weight.items()}
#                    print(class_weight)
#                except:
#                    class_weight = dict(Counter(y_train))
#                    class_weight = {key:(y_train.shape[0] - value)/value for key,value in class_weight.items()}
#                    class_weight = {key:f(value) for key,value in class_weight.items()}
#                    print(class_weight)
                    
                
                temp = [1,5]
                def call_back_dict(care = 'loss'):
                    if care == 'loss':
                        return dict(
                                monitor = 'val_{}'.format(classifier.metrics_names[0]),
                                mode = 'min',
                                patience = 5,
                                min_delta = 1e-1)
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
                       **call_back_dict('loss')
                       )
                
                
                
                remain_train = X_train.shape[0] % batch_size
                remain_valid = X_valid.shape[0] % batch_size
                
                if remain_train != 0:
                    np.random.seed(12345)
                    idx_train = np.random.choice(X_train.shape[0],size = X_train.shape[0] - remain_train,replace = False)
                    X_train,y_train = X_train[idx_train],y_train[idx_train]
                
                if remain_valid != 0:
                    np.random.seed(12345)
                    idx_valid = np.random.choice(X_valid.shape[0],size = X_valid.shape[0] - remain_valid,replace = False)
                    X_valid,y_valid = X_valid[idx_valid],y_valid[idx_valid]
                
                
                np.random.seed(12345)
                X_train,y_train = shuffle(X_train,y_train)
                X_valid,y_valid = shuffle(X_valid,y_valid)
                
                
#                sample_weight = [class_weight[item] for item in y_train[:,-1]]
                classifier.fit(X_train,y_train,
                               batch_size               = batch_size,
                               epochs                   = n_epochs,
                               validation_data          = (X_valid,y_valid),
                               callbacks                = callBackList,
                               shuffle                  = True,
#                               sample_weight            = np.array(sample_weight), # this is the key !
                               )
                
                classifier.load_weights(model_name)
                
#                X_test,_,_ = preprocess_features(X[idx_test],vectorizer,scaler)
                X_test = X[idx_test] / np.abs(X[idx_test].max())
                remain_test = X_test.shape[0] % batch_size
                if remain_test != 0:
                    X_test,y_test = X_test[:-remain_test],targets[idx_test][:-remain_test]
                else:
                    y_test = targets[idx_test]
                
                preds = classifier.predict(X_test,batch_size = batch_size)
                score_fold = metrics.roc_auc_score(y_test,preds,average = 'micro')
                print(f'{conscious_state},fold {fold+1},{score_fold:.4f}')
                K.clear_session()
#            results.append([conscious_state,ss])
#    print([np.mean(item[1]) for item in results])
    
                
                with open(os.path.join(report_dir,f'CRNN_{subject}.txt'),'a') as f:
                    f.write(f'\n{conscious_state} fold {fold + 1}, {score_fold:.4f}\n')
                #    f.write(classifier.summary())
                    f.close()




















