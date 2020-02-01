#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:20:02 2020

@author: nmei
"""

import os
import gc
import utils_deep
import torch
gc.collect()
import numpy as np
import pandas as pd
from glob                      import glob
from shutil                    import copyfile
copyfile('../../../utils.py','utils.py')
from utils                     import (LOO_partition)
from sklearn.model_selection   import GroupShuffleSplit
from sklearn                   import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing     import MinMaxScaler
from sklearn.utils             import shuffle as sk_shuffle
from torch.utils.data          import TensorDataset,DataLoader
from torch                     import optim
from scipy.spatial             import distance
from tqdm                      import tqdm

if __name__ == '__main__':
    sub                 = 'sub-01'
    target_folder       = 'encoding_VAE_LOO'
    data_folder         = 'BOLD_average' # BOLD_average_prepeak
    stacked_data_dir    = '../../../../data/{}/{}/'.format(data_folder,sub)
    MRI_dir             = '../../../../data/MRI/{}/'.format(sub)
    background_dir      = '../../../../data/computer_vision_background'
    feature_dir         = '../../../../data/computer_vision_features'
    output_dir          = '../../../../results/MRI/nilearn/{}/{}'.format(sub,target_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    BOLD_data           = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))
    event_data          = np.sort(glob(os.path.join(stacked_data_dir,'*.csv')))
    computer_models     = os.listdir(feature_dir)
    
    label_map           = {'Nonliving_Things':[0,1],
                           'Living_Things':   [1,0]}
    n_splits            = 50
    n_jobs              = -1
    patience            = 3
    n_epochs            = int(3e3)
    batch_size          = 16
    verbose             = 1
    n_jobs              = 8
    
    idx = 0
    np.random.seed(12345)
    BOLD_name,df_name   = BOLD_data[idx],event_data[idx]
    BOLD                = np.load(BOLD_name)
    df_event            = pd.read_csv(df_name)
    roi_name            = df_name.split('/')[-1].split('_events')[0]
    collection          = []
    for conscious_source in ['unconscious','glimpse','conscious']:
        for conscious_target in ['unconscious','glimpse','conscious']:
            idx_source              = df_event['visibility'] == conscious_source
            data_source             = BOLD[idx_source]
            VT                      = VarianceThreshold()
            scaler                  = MinMaxScaler((0,1))
            BOLD_norm_source        = VT.fit_transform(data_source)
            BOLD_sc_source          = scaler.fit_transform(BOLD_norm_source)
            df_data_source          = df_event[idx_source].reset_index(drop=True)
            df_data_source['id']    = df_data_source['session'] * 1000 + df_data_source['run'] * 100 + df_data_source['trials']
            targets_source          = np.array([label_map[item] for item in df_data_source['targets'].values])[:,-1]
            groups_source           = df_data_source['labels'].values
            
            idx_target              = df_event['visibility'] == conscious_target
            data_target             = BOLD[idx_target]
#            VT                      = VarianceThreshold()
#            scaler                  = MinMaxScaler((-1,1))
            BOLD_norm_target        = VT.transform(data_target)
            BOLD_sc_target          = scaler.transform(BOLD_norm_target)
            df_data_target          = df_event[idx_target].reset_index(drop=True)
            df_data_target['id']    = df_data_target['session'] * 1000 + df_data_target['run'] * 100 + df_data_target['trials']
            targets_target          = np.array([label_map[item] for item in df_data_target['targets'].values])[:,-1]
            groups_target           = df_data_target['labels'].values
            
            if n_splits > 300:
                idxs_train_source,idxs_test_source  = LOO_partition(df_data_source)
                
            else:
                cv = GroupShuffleSplit(n_splits = n_splits,
                                       test_size = .1,
                                       random_state = 12345,
                                       )
                idxs_train_source,idxs_test_source = [],[]
                for idx_train_,idx_test_ in cv.split(BOLD_sc_source,targets_source,groups = groups_source):
                    idxs_train_source.append(idx_train_)
                    idxs_test_source.append(idx_test_)
            # construct the test set of the target data based on the source training data
            idxs_target = []
            idxs_train_source_ = []
            idxs_test_source_ = []
            for idx_train_source,idx_test_source in zip(idxs_train_source,idxs_test_source):
                words_source = np.unique(df_data_source['labels'].values[idx_train_source])
                words_target = [word for word in np.unique(df_data_target['labels'].values) if (word not in words_source)]
                if len(words_target) > 1:
                    idx_words_target, = np.where(df_data_target['labels'].apply(lambda x: x in words_target) == True)
                    idxs_target.append(idx_words_target)
                    idxs_train_source_.append(idx_train_source)
                    idxs_test_source_.append(idx_test_source)
            # class check: sometime, the left-out 2 items belong to the same category
            idxs_target_temp = []
            idxs_train_source_temp = []
            idxs_test_source_temp = []
            for idx_,idx_source_train,idx_source_test in zip(
                                    idxs_target,
                                    idxs_train_source_,
                                    idxs_test_source_):
                temp = df_data_target.iloc[idx_]
                if len(np.unique(targets_target[idx_])) < 2:
                    print(pd.unique(temp['targets']),pd.unique(temp['labels']),targets_target[idx_],)#np.array([label_map[item] for item in temp['targets'].values])[:,-1])
                else:
                    idxs_target_temp.append(idx_)
                    idxs_train_source_temp.append(idx_source_train)
                    idxs_test_source_temp.append(idx_source_test)
            
            idxs_target_test = idxs_target_temp
            idxs_train_source = idxs_train_source_temp
            idxs_test_source = idxs_test_source_temp
            
            n_splits = len(idxs_target_test)
            print(f'{conscious_source} --> {conscious_target},perform {n_splits} cross validation')
            
            
            image_source = df_data_source['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
            image_target = df_data_target['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
            
            
            columns = ['weight_sum', 
                       'alphas', 
                       'mean_variance', 
                       'fold', 
                       'conscious_source',
                       'conscious_target', 
                       'roi_name', 
                       'positive voxels', 
                       'n_parameters',
                       'corr',
                       ]
            results = {name:[] for name in columns}
            
            for encoding_model in ['DenseNet169']:
                saving_name = f"encoding model {conscious_source} {conscious_target} {roi_name} {encoding_model}.csv"
                processed   = glob(os.path.join(output_dir,"*.csv"))
                print(saving_name)
                features_source         = np.array([np.load(os.path.join(feature_dir,
                                                                        encoding_model,
                                                                        item)) for item in image_source])
                features_target         = np.array([np.load(os.path.join(feature_dir,
                                                                        encoding_model,
                                                                        item)) for item in image_target])
                
                background_source       = np.array([np.load(os.path.join(background_dir,
                                                                         encoding_model,
                                                                         item)) for item in image_source])
                background_target       = np.array([np.load(os.path.join(background_dir,
                                                                         encoding_model,
                                                                         item)) for item in image_source])
                
                for fold,(idx_train_source,idx_test_target) in tqdm(enumerate(
                                    zip(idxs_train_source,idxs_target_test))):
                    X_train,X_test = features_source[idx_train_source],features_target[idx_test_target]
                    y_train,y_test = BOLD_sc_source[idx_train_source],BOLD_sc_target[idx_test_target]
                    
                    np.random.seed(12345)
                    X_train,y_train = sk_shuffle(X_train,y_train)
                    idx_cut = int(X_train.shape[0] * .1)
                    X_valid,y_valid = X_train[idx_cut:],y_train[idx_cut:]
                    X_train,y_train = X_train[:idx_cut],y_train[:idx_cut]
                    
                    X_in = torch.Tensor(X_valid)
                    y_in = y_valid.copy()
                    
                    torch.manual_seed(12345)
                    device = "cpu"
                    model = utils_deep.VAE(
                            input_dim = X_train.shape[1],
                            output_dim = y_train.shape[1],
                            encode_dims = [],
                            decode_dims = [],
                            dropout_rate = 0.,
                            vae_dim = y_train.shape[1],
                                            ).to(device)
                    loss_function = utils_deep.VEA_loss_function
                    
                    Bayesian_optimization_params = {
                            'n_iters':20,
                            'bounds':np.array([[1,int(1e3)]]),
                            'patience':6,
#                            'gp_params':{'random_state':12345},
                            'x0':np.array([25,50,100]).reshape(3,-1),
                            }
                    print('fitting ...')
                    model,test_losses = utils_deep.black_box_process(X_train,y_train,
                                                                     X_valid,y_valid,
                                                                     model,loss_function,device,
                                                                     patience = patience,
                                                                     batch_size = batch_size,
                                                                     n_epochs = n_epochs,
                                                                     learning_rate = 1e-4,
                                                                     print_train = False,)
                    print('done fitting')
                    
                    pred,score,corr = utils_deep.search_for_n_repeats(X_in,y_in,
                                                                      X_test,y_test,
                                                                      model,
                                                                      Bayesian_optimization_params = Bayesian_optimization_params,
                                                                      print_train = False,
                                                                      n_jobs = n_jobs,
                                                                      verbose = verbose,
                                                                      )
                    
                    print(f"{roi_name} on image [fold {fold}]: {conscious_source} --> {conscious_target},{encoding_model},VE = {score[score >=0].mean():.4f},corr = {np.mean(corr):.4f}, on {np.sum(score >= 0)} voxels\n")
                    collection.append(f"{roi_name} on image [fold {fold}]: {conscious_source} --> {conscious_target},{encoding_model},VE = {score[score >=0].mean():.4f},corr = {np.mean(corr):.4f}, on {np.sum(score >= 0)} voxels\n")
                    
                    X_train,X_test = background_source[idx_train_source],background_target[idx_test_target]
                    y_train,y_test = BOLD_sc_source[idx_train_source],BOLD_sc_target[idx_test_target]
                    np.random.seed(12345)
                    X_train,y_train = sk_shuffle(X_train,y_train)
                    idx_cut = int(X_train.shape[0] * .1)
                    X_valid,y_valid = X_train[idx_cut:],y_train[idx_cut:]
                    X_train,y_train = X_train[:idx_cut],y_train[:idx_cut]
                    
                    X_in = torch.Tensor(X_valid)
                    y_in = y_valid.copy()
                    
                    torch.manual_seed(12345)
                    device = "cpu"
                    model = utils_deep.VAE(
                            input_dim = X_train.shape[1],
                            output_dim = y_train.shape[1],
                            encode_dims = [],
                            decode_dims = [],
                            dropout_rate = 0.,
                            vae_dim = y_train.shape[1],
                                            ).to(device)
                    loss_function = utils_deep.VEA_loss_function
                    
                    print('fitting ...')
                    model,test_losses = utils_deep.black_box_process(X_train,y_train,
                                                                     X_valid,y_valid,
                                                                     model,loss_function,device,
                                                                     patience = patience,
                                                                     batch_size = batch_size,
                                                                     n_epochs = n_epochs,
                                                                     learning_rate = 1e-4,
                                                                     print_train = False,)
                    print('done fitting')
                    
                    pred,score,corr = utils_deep.search_for_n_repeats(X_in,y_in,
                                                                      X_test,y_test,
                                                                      model,
                                                                      Bayesian_optimization_params = Bayesian_optimization_params,
                                                                      print_train = False,
                                                                      n_jobs = n_jobs,
                                                                      verbose = verbose,
                                                                      )
                    
                    print(f"{roi_name} on background [fold {fold}]: {conscious_source} --> {conscious_target},{encoding_model},VE = {score[score >=0].mean():.4f},corr = {np.mean(corr):.4f}, on {np.sum(score >= 0)} voxels\n")
                    collection.append(f"{roi_name} on background [fold {fold}]: {conscious_source} --> {conscious_target},{encoding_model},VE = {score[score >=0].mean():.4f},corr = {np.mean(corr):.4f}, on {np.sum(score >= 0)} voxels\n")
                









































