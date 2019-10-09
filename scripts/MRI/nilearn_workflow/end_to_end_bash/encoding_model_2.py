#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:54:57 2019

@author: nmei
"""
import os
import gc
import numpy as np
import pandas as pd
print(os.getcwd())
from glob                      import glob
from tqdm                      import tqdm
from sklearn.utils             import shuffle
from shutil                    import copyfile
copyfile('../../../utils.py','utils.py')
from utils                     import (LOO_partition
                                       )
from sklearn.model_selection   import GroupShuffleSplit
from sklearn                   import metrics as sk_metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing     import MinMaxScaler

import tensorflow as tf
from tensorflow.keras                           import applications,layers,models,optimizers,losses,regularizers,metrics,initializers

from scipy.spatial             import distance

np.random.seed(12345)

sub                 = 'sub-01'
image_fold          = 'greyscaled'
stacked_data_dir    = '../../../../data/BOLD_average/{}/'.format(sub)
feature_dir         = '../../../../data/computer_vision_features'
output_dir          = '../../../../results/MRI/nilearn/{}/encoding_DNN'.format(sub)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
weight_dir          = '../../../../results/MRI/DNN/{}/encoding_DNN'.format(sub)
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)
BOLD_data           = glob(os.path.join(stacked_data_dir,'*BOLD.npy'))
event_data          = glob(os.path.join(stacked_data_dir,'*.csv'))
computer_models     = os.listdir(feature_dir)

label_map           = {'Nonliving_Things':[0,1],
                       'Living_Things':   [1,0]}
n_splits            = 100
batch_size          = 1
image_resize        = 224
n_epochs            = int(3e3)
validation_split    = 0.15

all_images          = glob(os.path.join(f'../../../../data/{image_fold}',
                                        '*','*','*.jpg'))

model_names     = ['DenseNet169',           # 1024
#                   'InceptionResNetV2',     # 1536
#                   'InceptionV3',           # 2048
                   'MobileNetV2',           # 1280
#                   'NASNetMobile',          # 1024
#                   'ResNet50',              # 1536
                   'VGG19',                 # 2048
#                   'Xception',              # 1280
                   ]

pretrained_models = [applications.DenseNet169,
#                     applications.InceptionResNetV2,
#                     applications.InceptionV3,
                     applications.MobileNetV2,
#                     applications.NASNetMobile,
#                     applications.ResNet50,
                     applications.VGG19,
#                     applications.Xception,
                     ]

def build_model(features:np.ndarray,BOLD:np.ndarray,l1:float = 0.001,l2:float = 0.1,lr:float = 1e-3,drop_rate:float = 0.5) -> models:
    try:
        tf.random.set_seed(12345)
    except:
        pass
    inputs = layers.Input(
                       shape                                            = (features.shape[1],),
                       batch_size                                       = batch_size,
                       name                                             = 'inputs')
    if drop_rate > 0:
        hidden = layers.Dropout(drop_rate,
                                seed                                    = 12345,
                                name                                    = 'drop2',
                                )(inputs)
    else:
        hidden = inputs
    hidden = layers.Dense(300,
                          activation                                    = tf.keras.activations.selu,
                          kernel_initializer                            = 'lecun_normal', # seggested in documentation
                          kernel_regularizer                            = regularizers.l2(l2),
                          activity_regularizer                          = regularizers.l1(l1),
                          name                                          = 'feature',
                          )(hidden)
    if drop_rate > 0:
        hidden = layers.AlphaDropout(drop_rate,
                                     seed                               = 12345,
                                     name                               = 'drop3'
                                     )(hidden) # suggested in documentation
    outputs = layers.Dense(BOLD.shape[1],
                           activation                                   = 'tanh',
                           kernel_initializer                           = initializers.he_normal(seed = 12345),
                           kernel_regularizer                           = regularizers.l2(l2),
                           activity_regularizer                         = regularizers.l1(l1),
                           name                                         = 'predict',
                           )(hidden)
    reg             = models.Model(inputs,outputs)
    
    reg.compile(optimizers.Adam(lr = lr,),
                losses.mae,
                metrics = ['mse'])
    return reg

def make_CallBackList(model_name,monitor='val_loss',mode='min',verbose=0,min_delta=1e-4,patience=50,frequency = 1):
    from tensorflow.keras.callbacks             import ModelCheckpoint,EarlyStopping
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
    checkPoint = ModelCheckpoint(model_name,                    # saving path
                                 monitor          = monitor,    # saving criterion
                                 save_best_only   = True,       # save only the best model
                                 mode             = mode,       # saving criterion
                                 save_freq        = 'epoch',    # frequency of check the update 
                                 verbose          = verbose,    # print out (>1) or not (0)
                                 )
    earlyStop = EarlyStopping(   monitor          = monitor,
                                 min_delta        = min_delta,
                                 patience         = patience,
                                 verbose          = verbose, 
                                 mode             = mode,
                                 restore_best_weights = True,
                                 )
    return [checkPoint,earlyStop]

idx                 = 1
BOLD_name,df_name   = BOLD_data[idx],event_data[idx]
BOLD                = np.load(BOLD_name)
df_event            = pd.read_csv(df_name)
roi_name            = df_name.split('/')[-1].split('_events')[0]

results             = dict(conscious_state  = [],
                           mean_variance    = [],
                           fold             = [],
                           model            = [],
                           positive_voxels  = [],
                           roi_name         = [],
                           weight_sum       = [],
                           l2               = [],
                           corr             = [],
                           )
try:
    tf.random.set_seed(12345)
except:
    pass
np.random.seed(12345)
for conscious_state in ['unconscious','glimpse','conscious']:
    idx_unconscious = df_event['visibility'] == conscious_state
    data            = BOLD[idx_unconscious]
    VT              = VarianceThreshold()
    scaler          = MinMaxScaler((-1,1))
    BOLD_sc         = VT.fit_transform(data)
    BOLD_norm       = scaler.fit_transform(BOLD_sc)
    df_data         = df_event[idx_unconscious].reset_index(drop=True)
    df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
    targets         = np.array([label_map[item] for item in df_data['targets'].values])[:,-1]
    groups          = df_data['labels'].values
    
    if n_splits <= 300:
        cv          = GroupShuffleSplit(n_splits = n_splits,
                                        test_size = 0.2,
                                        random_state = 12345)
        idxs_train,idxs_test = [],[]
        for idx_train,idx_test in cv.split(BOLD_norm,targets,groups = groups):
            idxs_train.append(idx_train)
            idxs_test.append(idx_test)
    else:
        idxs_train,idxs_test = LOO_partition(df_data)
        n_splits = len(idxs_train)
    
    image_names     = df_data['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
    
    for encoding_model_name in model_names:
        saving_name = f"encoding model {conscious_state} {roi_name} {encoding_model_name}.csv"
        processed   = glob(os.path.join(output_dir,"*.csv"))
        
        if not os.path.join(output_dir,saving_name) in processed:
            
            features_norm        = np.array([np.load(os.path.join(feature_dir,
                                                             encoding_model_name,
                                                             item)) for item in image_names])
            
            for fold,(train,test) in enumerate(zip(idxs_train,idxs_test)):
                tf.keras.backend.clear_session()
                gc.collect()
                X,y         = features_norm[train],BOLD_norm[train]
                for train,validation in GroupShuffleSplit(n_splits      = 5, 
                                                          test_size     = validation_split,
                                                          random_state  = 12345
                                                          ).split(X,y,
                                                              groups    = groups[train]):
                    train,validation
                    
                X_train,y_train = X[train],y[train]
                X_valid,y_valid = X[validation],y[validation]
                
                X_test,y_test   = features_norm[test],BOLD_norm[test]
                
                temp_results    = dict(score = [],
                                       voxel = [],
                                       l1 = [],
                                       l2 = [],
                                       corr = [],
                                       )
                l1s = [1e-5]#np.logspace(-8,2,11)
                l2s = np.logspace(-8,2,11)
                regularizations = [[l1,l2] for l1 in l1s for l2 in l2s]
                for l1,l2 in regularizations:
                    model_weights = os.path.join(weight_dir,saving_name.replace('csv',f'_{l1}_{l2}.h5'))
                    print('building model ...')
                    reg = build_model(features_norm,BOLD_norm,
                                      lr            = 1e-4,
                                      l1            = l1,
                                      l2            = l2,
                                      drop_rate     = 0.,
                                      )
                    
                    callbacks = make_CallBackList(model_weights,
                                                  monitor       = 'val_loss',
                                                  mode          = 'min',
                                                  verbose       = 0,
                                                  min_delta     = 1e-3,
                                                  patience      = 5,
                                                  )
                    print(f'training with l1 = {l1} l2 = {l2}')
                    reg.fit(X_train,y_train,
                            batch_size              = batch_size,
                            validation_data         = (X_valid,y_valid),
                            epochs                  = n_epochs,
                            callbacks               = callbacks,
                            verbose                 = 0,
                            )
                    
                    reg.load_weights(model_weights)
                    preds_valid = reg.predict(X_valid)
                    score_valid = sk_metrics.r2_score(y_valid,preds_valid,multioutput = 'raw_values')
                    corr    = np.mean([distance.cdist(a.reshape(1,-1) - a.mean(),b.reshape(1,-1) - b.mean(),'correlation',
                                                      ).flatten()[0] for a,b in zip(y_valid,preds_valid)])
                    
                    if all(score_valid <= 0):
                        score_valid = np.zeros(score_valid.shape)
                    else:
                        score_valid[score_valid <= 0] = np.nan
                    print(f'MV = {np.nanmean(score_valid):.4f},corr = {corr:.3f},PV = {len(score_valid[~np.isnan(score_valid)])}')
                    
                    temp_results['score'].append(np.nanmean(score_valid))
                    temp_results['voxel'].append(len(score_valid[~np.isnan(score_valid)]))
                    temp_results['l1'].append(l1)
                    temp_results['l2'].append(l2)
                    temp_results['corr'].append(corr)
                
                temp_results            = pd.DataFrame(temp_results)
                print(temp_results)
                p = 0.3
                q = 0.1
                # higher the better, but log of fractions is negative
                # lower the better, but log of fractions is negative and the 
                # megnitude is inverse proportional to the fraction
                # higher the better, and it is a positive integer
                temp_results['rank']    = temp_results['score'].apply(np.log10)* p +\
                            -temp_results['corr'].apply(np.log10) * q +\
                            temp_results['voxel'].apply(np.log10) * (1 - p)
                temp_ = temp_results.replace([np.inf,-np.inf],np.nan).dropna()
                temp_            = temp_.sort_values(['rank'])
                print(temp_)
                
                l1 = temp_['l1'].values[temp_['rank'].values.argmax()]
                l2  = temp_['l2'].values[temp_['rank'].values.argmax()]
                reg = build_model(features_norm,BOLD_norm,
                                  lr            = 1e-4,
                                  l1            = l1,
                                  l2            = l2,
                                  drop_rate     = 0.,
                                  )
                model_weights = os.path.join(weight_dir,saving_name.replace('csv',f'_{l1}_{l2}.h5'))
                reg.load_weights(model_weights)
                
                preds = reg.predict(X_test)
                score = sk_metrics.r2_score(y_test,preds,multioutput = 'raw_values')
                corr    = np.mean([distance.cdist(a.reshape(1,-1) - a.mean(),b.reshape(1,-1) - b.mean(),'cosine',
                                                  ).flatten()[0] for a,b in zip(y_test,preds)])
                
                if all(score < 0):
                    mean_variance = np.zeros(score.shape)
                else:
                    mean_variance = score.copy()
                    mean_variance[mean_variance <= 0] = np.nan
                print(f'{conscious_state},l1 = {l1},l2 = {l2},MV = {np.nanmean(mean_variance):.5f},corr = {corr:.4f},|weight| = {np.sum(np.abs(reg.layers[-1].get_weights()[0])):.2f}\n')
                
                results['conscious_state'].append(conscious_state)
                results['mean_variance'].append(np.nanmean(mean_variance))
                results['fold'].append(fold + 1)
                results['model'].append(encoding_model_name)
                results['positive_voxels'].append(len(mean_variance[~np.isnan(mean_variance)]))
                results['roi_name'].append(roi_name)
                results['weight_sum'].append(np.sum(np.abs(reg.layers[-1].get_weights()[0])))
                results['l2'].append(l2)
                results['corr'].append(corr)
                results_to_save = pd.DataFrame(results)
                results_to_save.to_csv(os.path.join(output_dir,saving_name), index = False)
                del reg
        else:
            print(saving_name)












