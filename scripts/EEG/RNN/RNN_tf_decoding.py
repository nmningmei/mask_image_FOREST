#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 15:52:28 2019

@author: nmei
"""
import os
import mne
import numpy as np
import pandas as pd
from glob import glob
from shutil import copyfile
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
import tensorflow as tf
from tensorflow.keras import layers,Model, optimizers,losses,regularizers,callbacks#,Sequential
from tensorflow.keras import metrics as k_metrics
import tensorflow.keras.backend as K
from collections import Counter
copyfile('../utils_deep.py','utils_deep.py')
import utils_deep

all_subjects = ['aingere_5_16_2019',
                'alba_6_10_2019',
                'alvaro_5_16_2019',
                'clara_5_22_2019',
                'ana_5_21_2019',
                'inaki_5_9_2019',
                'jesica_6_7_2019',
                'leyre_5_13_2019',
                'lierni_5_20_2019',
                'maria_6_5_2019',
                'matie_5_23_2019',
                'out_7_19_2019',
                'mattin_7_12_2019',
                'pedro_5_14_2019',
                'xabier_5_15_2019',
                ]
all_subjects = np.sort(all_subjects)
saving_dir = '../../../results/EEG/RNN'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)


subject = all_subjects[4]
working_dir         = f'../../../data/clean EEG highpass detrend/{subject}'
working_data        = glob(os.path.join(working_dir,'*-epo.fif'))
n_splits            = 100
n_epochs            = int(2e2)
print_model         = True

df = dict(conscious_state = [],
          score = [],
          fold = [],
          )

for epoch_file in working_data:
    epochs  = mne.read_epochs(epoch_file)
    # resample at 100 Hz to fasten the decoding process
    print('resampling')
    epochs.resample(100)
    
    conscious   = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if (' conscious' in name)])
    see_maybe   = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('glimpse' in name)])
    unconscious = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('unconscious' in name)])
    del epochs
    
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
        # because RNN take input's last dimension as the feature while the second the last dimension as time step
        X                   = np.swapaxes(X,1,2)
        ss                  = []
        cv                  = StratifiedShuffleSplit(n_splits=n_splits,test_size = 0.15,random_state=12345)
        for fold,(idx_,idx_test) in enumerate(cv.split(X,targets)):
            
            X_train,X_valid,y_train,y_valid = train_test_split(
                                X[idx_],targets[idx_],
                                test_size           = 0.15,
                                random_state        = 12345,
                                shuffle             = True,)
            
            model_name  = f'RNN_{subject}.hdf5'
            print(f'model name: {model_name}')
            batch_size  = 10
            timesteps   = X.shape[1]
            data_dim    = X.shape[2]
            n_units     = 1
            n_layers    = 1
            dropout     = True
            l2          = 1e-4
            # make the model
            classifier = utils_deep.build_model(timesteps   = timesteps,
                                                data_dim    = data_dim,
                                                n_units     = n_units,
                                                batch_size  = batch_size,
                                                n_layers    = n_layers,
                                                drop        = dropout,
                                                l2          = l2,)
            print(f'the model has {classifier.count_params()} parameters')
            if print_model:
                classifier.summary()
                print_model = False
                
            # compile the model with optimizer, loss function
            classifier.compile(optimizer                = optimizers.Adam(lr = 1e-3),
                               loss                     = losses.categorical_crossentropy,
                               metrics                  = [k_metrics.categorical_accuracy])
            # early stopping
            callBackList = utils_deep.make_CallBackList(
                               model_name,
                               verbose                  = 0,# print out the process
                               frequency                = 1,
                               **utils_deep.call_back_dict(classifier,'metric'))
            
            # prepare the data
            X_train,y_train = utils_deep.prepare_data_batch(X_train,y_train,batch_size = batch_size)
            X_valid,y_valid = utils_deep.prepare_data_batch(X_valid,y_valid,batch_size = batch_size)
            # put weights on minor class
            class_weight    = dict(Counter(y_train[:,0]))
            class_weight    = {key:(y_train.shape[0] - value)/value for key,value in class_weight.items()}
            sample_weight   = [class_weight[item] for item in y_train[:,0]]
            print(class_weight)
            
            # train and validate
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
            
            # test the trained model
            classifier.load_weights(model_name)
            
            X_test,y_test = utils_deep.prepare_data_batch(X[idx_test],targets[idx_test],batch_size = batch_size)
            
            
            preds = classifier.predict(X_test,batch_size=batch_size,verbose = 0)
            
            ss.append(metrics.roc_auc_score(y_test,preds,average = 'micro'))
            print(f'{conscious_state},fold {fold+1},{ss[-1]:.4f} - {np.mean(ss):.4f}')
            K.clear_session()
            df['conscious_state'].append(conscious_state)
            df['score'].append(metrics.roc_auc_score(y_test,preds,average = 'micro'))
            df['fold'].append(fold + 1)
            df_to_save = pd.DataFrame(df)
            df_to_save.to_csv(os.path.join(saving_dir,f'{subject}.csv'),index = False)




















