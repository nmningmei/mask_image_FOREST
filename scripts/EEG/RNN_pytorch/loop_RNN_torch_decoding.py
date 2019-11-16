#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:48:45 2019

@author: nmei
"""

import os
from shutil import copyfile
from sklearn import metrics
from sklearn.utils import shuffle
from collections import Counter

#import mkl
#mkl.set_num_threads(12)

import torch
from torch import nn,no_grad
from torch.nn import functional
from torch import optim
from torch.autograd import Variable

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
    X       = (X - X.min()) / (X.max() - X.min())
#    remain_ = X.shape[0] % batch_size
#    if remain_ != 0:
#        np.random.seed(12345)
#        idx_    = np.random.choice(X.shape[0],size = X.shape[0] - remain_)
#        X,y     = X[idx_],y[idx_]
        
    return X,y

class simple_RNN(nn.Module):
    def __init__(self, 
                 batch_size = 16,
                 n_features = 60,
                 n_timesteps = 150,
                 hidden_dim = 1,# unit in tensorflow
                 num_layers = 1,
                 dropout_rate = 0.25,
                 bidirectional = False
                 ):
        super(simple_RNN, self).__init__()
        
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_timesteps = n_timesteps
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        if self.num_layers > 1:
            self.dropout_rate = dropout_rate
        else:
            self.dropout_rate = 0.
        self.bidirectional = bidirectional
        
        self.LSTM_layer = nn.LSTM(input_size = self.n_features,
                                  hidden_size = self.hidden_dim,
                                  num_layers = self.num_layers,
                                  bias = True,
                                  batch_first = True,
                                  dropout = self.dropout_rate,
                                  bidirectional = self.bidirectional,
                                  )
        if self.bidirectional:
            self.norm = nn.BatchNorm1d(num_features = self.n_timesteps * 2)
        else:
            self.norm = nn.BatchNorm1d(num_features = self.n_timesteps)
        self.linear = nn.Linear(self.n_timesteps,2,bias = True)
        self.output_activation = nn.Softmax(dim = 1)
    
    def forward(self,x):
        rnn_out1,(hidden_state1,cell_state1) = self.LSTM_layer(x)
        rnn_out1 = self.norm(rnn_out1)
        rnn_out1 = torch.squeeze(rnn_out1)
        
        linear_out = self.linear(rnn_out1)
        out = self.output_activation(linear_out)
        
        return out

def train(net,dataloader):
    
    return None
def test():
    return None

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
    from sklearn.metrics import roc_auc_score
    import mne
    import numpy as np
    import pandas as pd
    from glob import glob
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'currently use {device}')
    all_subjects = [
#                    'aingere_5_16_2019',
#                    'alba_6_10_2019',
#                    'alvaro_5_16_2019',
#                    'clara_5_22_2019',
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
    saving_dir = '../../../results/EEG/RNN_pytorch'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    
    
    for subject in all_subjects:
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
                    batch_size  = 16
                    timesteps   = X.shape[1]
                    data_dim    = X.shape[2]
                    n_units     = 1
                    n_layers    = 1
                    dropout     = True
                    l1          = 1e-4
                    
                    
                    # prepare the data
                    X_train,y_train = prepare_data_batch(X_train,y_train,batch_size = batch_size)
                    X_valid,y_valid = prepare_data_batch(X_valid,y_valid,batch_size = batch_size)
                    X_test, y_test  = prepare_data_batch(X[idx_test], targets[idx_test], batch_size = batch_size)
                    
                    class_weight    = dict(Counter(y_train[:,0]))
                    class_weight    = {key:(y_train.shape[0] - value)/value for key,value in class_weight.items()}
                    sample_weight   = np.array([class_weight[item] for item in y_train[:,0]])
                    
                    from torch.utils.data import DataLoader,TensorDataset
                    dataset_train = TensorDataset(torch.from_numpy(X_train),torch.from_numpy(y_train))
                    dataset_valid = TensorDataset(torch.from_numpy(X_valid),torch.from_numpy(y_valid))
                    dataset_test  = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
                    
                    dataloader_train = DataLoader(dataset_train,batch_size = batch_size,shuffle = True,
                                                  num_workers = 2,drop_last = True,)
                    dataloader_valid = DataLoader(dataset_valid,batch_size = batch_size,shuffle = False,
                                                  num_workers = 2,drop_last = True,)
                    dataloader_test  = DataLoader(dataset_test,batch_size = batch_size, shuffle = False,
                                                  num_workers = 2,drop_last = True,)
                    
                    torch.manual_seed(12345)
                    classifier = simple_RNN(batch_size = batch_size,
                                            n_features = data_dim,
                                            n_timesteps = timesteps,
                                            hidden_dim = 1,# unit in tensorflow
                                            num_layers = 1,
                                            dropout_rate = 0.25,).to(device).float()
                    loss_func = nn.BCELoss(weight = torch.from_numpy(np.array(list(class_weight.values()))).float().to(device))
                    optimizer = optim.Adam(classifier.parameters(),lr = 1e-4)
                    
                    current_valid_loss = torch.from_numpy(np.array(np.inf))
                    count = 0
                    for idx_epoch in range(n_epochs):
                        train_loss = 0
                        for jj,(features,labels) in enumerate(dataloader_train):
                            features = features.to(device).float()
                            labels = labels.to(device).float()
                            
                            optimizer.zero_grad()
                            
                            preds = classifier(Variable(features))
                            
                            loss_batch = loss_func(preds,labels)
                            loss_batch += l1 * torch.norm(preds,1) + 1e-12
                            loss_batch.backward()
                            optimizer.step()
                            train_loss += loss_batch.data
    #                        print('training ...')
    #                        print(f'epoch {idx_epoch}-{ii + 1:3.0f}/{100*(ii+1)/ len(dataloader_train):2.3f}%,loss = {train_loss/(jj+1):.6f}')
                        valid_preds = []
                        with no_grad():
                            valid_loss = 0.
                            for jj,(features,labels) in enumerate(dataloader_valid):
                                features = features.to(device).float()
                                labels = labels.to(device).float()
                                
                                preds = classifier(Variable(features))
                                valid_preds.append(preds.detach().cpu().numpy())
                                loss_batch = loss_func(preds,labels)
                                valid_loss += loss_batch.data
                                denominator = jj
                            valid_loss = valid_loss / (denominator + 1)
                        valid_preds = np.concatenate(valid_preds)
                        score = roc_auc_score(y_valid[:valid_preds.shape[0]],valid_preds,average = 'micro')
                        print(f'epoch {idx_epoch:2d}, validation loss = {valid_loss:.6f}, score = {score:.4f}')
                        if valid_loss.cpu().clone().detach().type(torch.float64) < current_valid_loss:
                            current_valid_loss = valid_loss.cpu().clone().detach().type(torch.float64)
                            count = 0
                        else:
                            count += 1
                        if count > 5:
                            print(f'early stop, current valid loss = {current_valid_loss:.6f}, score = {score:.4f}\n')
                            break
                    predictions = []
                    with no_grad():
                        for ii,(features,labels) in enumerate(dataloader_test):
                            features = features.to(device).float()
                            labels = labels.to(device).float()
                            
                            preds = classifier(Variable(features))
                            predictions.append(preds.detach().cpu().numpy())
                    predictions = np.concatenate(predictions)
                    score = roc_auc_score(y_test[:predictions.shape[0]],predictions,average = 'micro')
                    ss.append(score)
                    print(f'{conscious_state}, fold {fold},score = {score:.4f} - {np.mean(ss):.4f}\n')
                    df['conscious_state'].append(conscious_state)
                    df['score'].append(score)
                    df['fold'].append(fold + 1)
                    df_to_save = pd.DataFrame(df)
                    df_to_save.to_csv(os.path.join(saving_dir,f'{subject}.csv'),index = False)
                del classifier
                for _ in range(15):
                    import gc
                    gc.collect()











































