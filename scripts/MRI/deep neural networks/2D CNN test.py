#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:13:27 2019

@author: nmei
"""

import os
import gc
import pandas as pd
import numpy  as np

from glob                    import glob
from tqdm                    import tqdm
from sklearn.utils           import shuffle
from nibabel                 import load as load_fmri
from nilearn.image           import index_img
from shutil                  import copyfile
copyfile('../../utils.py','utils.py')
copyfile('../../lr_finder.py','lr_finder.py')
from utils                   import (groupby_average,
                                     check_train_balance,
                                     check_train_test_splits,
                                     customized_partition,
                                     build_model_dictionary)
from sklearn.metrics         import roc_auc_score
from nilearn.input_data      import NiftiMasker
from sklearn.model_selection import GroupShuffleSplit#StratifiedKFold

from lr_finder import LRFinder

import warnings
warnings.filterwarnings('ignore') 

# the most important helper function: early stopping and model saving
def make_CallBackList(model_name,monitor='val_loss',mode='min',verbose=0,min_delta=1e-4,patience=50,frequency = 1):
    from tensorflow.keras.callbacks             import ModelCheckpoint,EarlyStopping
    """
    Make call back function lists for the keras models
    
    Inputs
    -------------------------
    model_name: directory of where we want to save the model and its name
    monitor:    the criterion we used for saving or stopping the model
    mode:       min --> lower the better, max --> higher the better
    verboser:   printout the monitoring messages
    min_delta:  minimum change for early stopping
    patience:   temporal windows of the minimum change monitoring
    frequency:  temporal window steps of the minimum change monitoring
    
    Return
    --------------------------
    CheckPoint:     saving the best model
    EarlyStopping:  early stoppi
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

sub                 = 'sub-01'
first_session       = 2
stacked_data_dir    = '../../../data/BOLD_whole_brain_averaged/{}/'.format(sub)
mask_dir            = '../../../data/MRI/{}/anat/ROI_BOLD'.format(sub)
output_dir          = '../../../results/MRI/nilearn/spacenet/{}'.format(sub)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
masks               = glob(os.path.join(mask_dir,'*.nii.gz'))
whole_brain_mask    = f'../../../data/MRI/{sub}/func/session-0{first_session}/{sub}_unfeat_run-01/outputs/func/mask.nii.gz'
label_map           = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}
average             = True
n_splits            = 100
n_jobs              = 1
l1 = 1e-4
l2 = 1e-4

s = dict(
        conscious_state     = [],
        score               = [],
        fold                = [],)
for conscious_state in ['unconscious','glimpse','conscious']:
    df_data         = pd.read_csv(os.path.join(stacked_data_dir,
                                               f'whole_brain_{conscious_state}.csv'))
    df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
    idxs_test       = customized_partition(df_data,n_splits = n_splits)
    
    
    BOLD_file       = os.path.join(stacked_data_dir,
                                   f'whole_brain_{conscious_state}.nii.gz')
    BOLD_image      = load_fmri(BOLD_file)
    targets         = np.array([label_map[item] for item in df_data['targets']])
    groups          = df_data['labels'].values
    
    masker          = NiftiMasker(mask_img = whole_brain_mask,standardize = True)
    masker.fit(BOLD_file)
    
    whole_brain     = BOLD_image.get_data()
    whole_brain_std = (whole_brain - whole_brain.min()) / (whole_brain.max() - whole_brain.min())
    whole_brain_norm= whole_brain_std * (1 - -1) + -1
    whole_brain_norm= np.swapaxes(whole_brain_norm,3,2,)
    whole_brain_norm= np.swapaxes(whole_brain_norm,2,1,)
    whole_brain_norm= np.swapaxes(whole_brain_norm,1,0,)
    
    cv = GroupShuffleSplit(n_splits = 5,test_size = 0.15, random_state = 12345)
    
    for train_, test in cv.split(whole_brain_norm,targets[:,-1],groups = groups):
        X_,y_,groups_ = whole_brain_norm[train_],targets[train_],groups[train_]
        X_test,y_test = whole_brain_norm[test],targets[test]
        
        X_,y_,groups_ = shuffle(X_,y_,groups_)
        
        for train,valid in cv.split(X_,y_[:,-1],groups_):
            X_train,y_train = X_[train],y_[train]
            X_valid,y_valid = X_[valid],y_[valid]
        
        import tensorflow as tf
        from tensorflow.keras import applications,layers,models,optimizers,losses,regularizers
        
        tf.keras.backend.clear_session()
        gc.collect()
        tf.random.set_seed(12345)
        base_model = applications.densenet.DenseNet121(input_shape=(88,88,66), 
                                                       include_top=False, 
                                                       weights=None, 
                                                       input_tensor=None, 
                                                       pooling='max', 
                                                       )
        drop1 = layers.Dropout(0.5,name = 'drop1')(base_model.output)
        hidden1 = layers.Dense(512,
                              activation = 'selu',
                              kernel_initializer = 'lecun_normal',
                              kernel_regularizer = regularizers.l2(l2),
                              name = 'hidden1',
                              )(drop1)
        drop2 = layers.Dropout(0.5, name = 'drop2')(hidden1)
        hidden2 = layers.Dense(300,
                              activation = 'selu',
                              kernel_initializer = 'lecun_normal',
                              kernel_regularizer = regularizers.l2(l2),
                              name = 'hidden2',
                              )(drop2)
        drop3 = layers.Dropout(0.5, name = 'drop3')(hidden2)
        outputs = layers.Dense(2,
                               activity_regularizer = regularizers.l1(l1),
                               activation = 'softmax',
                               name = 'prediction')(drop3)
        
        clf = models.Model(base_model.input,outputs,name = 'clf')
        
        clf.compile(optimizers.Adam(lr = 1e-4,),
                    losses.binary_crossentropy,
                    metrics = ['categorical_accuracy'])
        saving_model_name   = 'temp.h5'
        callbacks           = make_CallBackList(saving_model_name,
                                                monitor                 = 'val_{}'.format(clf.metrics_names[-2]),
                                                mode                    = 'min',
                                                verbose                 = 0,
                                                min_delta               = 1e-2,
                                                patience                = 5,
                                                frequency               = 1)
        try:
            lr_finder = LRFinder(clf)
            lr_finder.find(X_train, y_train, start_lr=0.0001, end_lr=1, batch_size=16, epochs=1)
#            lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
            lr = lr_finder.get_best_lr(16,n_skip_beginning=10,n_skip_end=5,)
            tf.keras.backend.set_value(clf.optimizer.lr, lr)
        except:
            pass
        print(f'\ntraining with learning rate {lr:.5f}...\n')
        clf.fit(X_train,
                y_train,
                epochs = 3000,
                batch_size = 16,
                validation_data = (X_valid,y_valid,),
                callbacks = callbacks,
                verbose = 1,
                )
        
        preds = clf.predict(X_test)
        
        print(roc_auc_score(y_test,preds))






































