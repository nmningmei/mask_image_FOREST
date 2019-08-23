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
import torch.autograd as autograd


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

torch.manual_seed(12345)
for subject in all_subjects:
    working_dir         = f'../../data/clean EEG highpass detrend/{subject}'
    working_data        = glob(os.path.join(working_dir,'*-epo.fif'))
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
            
            X_,y_               = epochs.crop(0,1.).get_data(),epochs.events[:,-1]
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
                
                time_step = X_train.shape[1]
                n_features = X_train.shape[-1]
                
                class RNN_clf(nn.Module):
                    def __init__(self,
                                 batch_size,
                                 n_layers = 1,
                                 input_size = 60,
                                 hidden_size = 32,
                                 dropout = 0.5,
                                 bidriectional = False
                                 ):
                        super(RNN_clf, self).__init__()
                        self.input_size = input_size
                        self.hidden_size = hidden_size
                        self.num_layers = n_layers
                        self.dropout = dropout
                        self.bidriectional = bidriectional
                        self.batch_size = batch_size
                        self.rnn = nn.LSTM(input_size = self.input_size,
                                           hidden_size = self.hidden_size,
                                           num_layers = self.num_layers,
                                           bias = True,
                                           batch_first = True,
                                           bidirectional = self.bidriectional,)
                        if self.bidriectional:
                            self.linear = nn.Linear(self.hidden_size*2,2)
                            self.pool = nn.AdaptiveAvgPool2d(output_size = (1,self.hidden_size*2))
                        else:
                            self.linear = nn.Linear(self.hidden_size,2)
                            self.pool = nn.AdaptiveAvgPool2d(output_size = (1,self.hidden_size))
                        self.softmax = nn.LogSoftmax(dim = 1)
#                        self.activation = nn.LeakyReLU(inplace = True)
                        self.batchnormalization = nn.BatchNorm1d(self.hidden_size)
                        self.hidden = self.init_hidden()
                        self.exp = torch.exp
                        self.squeeze = torch.squeeze
                        
                    def init_hidden(self):
                        if self.bidriectional:
                            return (autograd.Variable(torch.randn(self.num_layers*2, self.batch_size, self.hidden_size)),
                                    autograd.Variable(torch.randn(self.num_layers*2, self.batch_size, self.hidden_size)))
                        else:
                            # This is what we'll initialise our hidden state as
                            return (autograd.Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size)),
                                    autograd.Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size)))
                    def forward(self, x):
                        lstm_out,self.hidden = self.rnn(x,self.hidden)
                        lstm_out = self.batchnormalization(lstm_out)
                        lstm_out = self.squeeze(self.pool(lstm_out))
                        y_pred = self.linear(lstm_out)
                        y_pred = self.softmax(y_pred)
#                        y_pred = self.exp(self.softmax(y_pred))
#                        y_pred = self.squeeze(y_pred)[-1]
                        return y_pred
                
                batch_size = 1
                model = RNN_clf(batch_size = batch_size,
                                n_layers = 1,
                                input_size = n_features,
                                hidden_size = time_step,
                                bidriectional=True)
                model
                optimizer = optim.Adam(model.parameters(),lr = 1e-3,)
                loss_fn = nn.CrossEntropyLoss()
                train_loss = 0.
                
                for ii,(x,y) in enumerate(zip(X_train,y_train)):
                    if batch_size > 1:
                        x = x.astype('float32')
                    else:
                        x = x.reshape(-1,time_step,n_features).astype('float32')
                        y = y.reshape(-1,2)
#                    y = np.vstack([y for _ in range(time_step)]).reshape(-1,time_step,2).astype('float32')
                    
                    x = autograd.Variable(torch.from_numpy(x))
                    y = autograd.Variable(torch.from_numpy(y).type(torch.LongTensor))
                    
                    optimizer.zero_grad()
                    output = model(x)
                    loss = loss_fn(output.view(-1)[:,np.newaxis],y.view(-1)[:,np.newaxis])
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    
                    train_loss += loss.data
                    
                    if ii % 50 == 0:
                        with torch.no_grad():
                            valid_output = torch.squeeze(torch.stack([model(autograd.Variable(torch.from_numpy(temp.reshape(-1,time_step,n_features).astype('float32')))) for temp in X_valid]))
                            lo = loss_fn(valid_output,
                                         autograd.Variable(torch.from_numpy(y_valid).type(torch.LongTensor)))
                            print(ii,lo.data)
                
                X_test = X[idx_test] / np.abs(X[idx_test].max())
                y_test = targets[idx_test]
                
                with torch.no_grad():
                    test_output = torch.squeeze(torch.stack([model(autograd.Variable(torch.from_numpy(temp.reshape(-1,time_step,n_features).astype('float32')))) for temp in X_test]))
                sadf















































