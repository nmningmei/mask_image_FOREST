#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:48:45 2019

@author: nmei
"""

import os
import re
import mne
from datetime import datetime
from glob import glob
from shutil import copyfile
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
copyfile('../utils.py','utils.py')
from utils import get_frames

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
                ]
all_subjects = np.sort(all_subjects)

torch.manual_seed(12345)
for subject in all_subjects:
    working_dir         = f'../../data/clean EEG/{subject}'
    working_data        = glob(os.path.join(working_dir,'*-epo.fif'))
    # there was a bug in the csv file, so the early behavioral is treated differently
    date                = '/'.join(re.findall('\d+',subject))
    date                = datetime.strptime(date,'%m/%d/%Y')
    breakPoint          = datetime(2019,3,10)
    if date > breakPoint:
        new             = True
    else:
        new             = False
    frames,_            = get_frames(directory = f'../../data/behavioral/{subject}',new = new)
    n_splits            = 25
    n_epochs            = int(2e2)
    
    
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
            
            X                   = mne.decoding.Scaler(epochs.info).fit_transform(X)
            
            X                   = np.swapaxes(X,1,2)
            ss                  = []
            cv                  = StratifiedShuffleSplit(n_splits=n_splits,test_size = 0.2,random_state=12345)
            for fold,(idx_,idx_test) in enumerate(cv.split(X,targets)):
                
                X_train,X_valid,y_train,y_valid = train_test_split(
                        X[idx_],targets[idx_],
                        test_size               = 0.2,
                        random_state            = 12345,
                        shuffle                 = True,)
                
                X_train,vectorizer,scaler = preprocess_features(X_train)
                X_valid,_,_ = preprocess_features(X_valid,vectorizer,scaler)


















































