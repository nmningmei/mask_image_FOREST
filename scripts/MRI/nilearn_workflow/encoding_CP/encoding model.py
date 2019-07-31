#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:54:57 2019

@author: nmei
"""
import os
import numpy as np
import pandas as pd
os.chdir('..')
from glob                    import glob
from tqdm                    import tqdm
from sklearn.utils           import shuffle
from shutil                  import copyfile
copyfile('../../utils.py','utils.py')
from utils                   import (customized_partition,
                                     check_train_test_splits,
                                     check_train_balance,
                                     build_model_dictionary,
                                     Find_Optimal_Cutoff)
from sklearn.model_selection import cross_validate
from sklearn                 import metrics
from sklearn.linear_model    import RidgeCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing   import StandardScaler
from functools               import partial

sub                 = 'sub-01'
stacked_data_dir    = '../../../data/BOLD_average/{}/'.format(sub)
feature_dir         = '../../../data/computer vision features'
output_dir          = '../../../results/MRI/nilearn/{}/encoding_CP'.format(sub)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
BOLD_data           = glob(os.path.join(stacked_data_dir,'*BOLD.npy'))
event_data          = glob(os.path.join(stacked_data_dir,'*.csv'))
computer_models     = os.listdir(feature_dir)

label_map           = {'Nonliving_Things':[0,1],
                       'Living_Things':   [1,0]}
n_splits            = 300

idx = 0
np.random.seed(12345)
BOLD_name,df_name   = BOLD_data[idx],event_data[idx]
BOLD                = np.load(BOLD_name)
df_event            = pd.read_csv(df_name)
roi_name            = df_name.split('/')[-1].split('_events')[0]
for conscious_state in ['unconscious','glimpse','conscious']:
    idx_unconscious = df_event['visibility'] == conscious_state
    data            = BOLD[idx_unconscious]
    VT              = VarianceThreshold()
    scaler          = StandardScaler()
    BOLD_norm       = VT.fit_transform(data)
    BOLD_sc         = scaler.fit_transform(BOLD_norm)
    df_data         = df_event[idx_unconscious].reset_index(drop=True)
    df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
    targets         = np.array([label_map[item] for item in df_data['targets'].values])[:,-1]
    
    idxs_test       = customized_partition(df_data,['id','labels'],n_splits = n_splits)
    while check_train_test_splits(idxs_test): # just in case
        idxs_test   = customized_partition(df_data,['id','labels'],n_splits = n_splits)
    idxs_train      = [shuffle(np.array([idx for idx in df_data.index.tolist() if (idx not in idx_test)])) for idx_test in idxs_test]
    idxs_train      = [check_train_balance(df_data,idx_train,list(label_map.keys())) for idx_train in tqdm(idxs_train)]
    
    image_names     = df_data['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
    for encoding_model in computer_models:
        features    = np.array([np.load(os.path.join(feature_dir,
                                         encoding_model,
                                         item)) for item in image_names])
        score_func  = partial(metrics.r2_score,multioutput = 'variance_weighted')
        scorer      = metrics.make_scorer(score_func)
        reg         = RidgeCV(alphas = np.logspace(0,5,6),
                            normalize = True,
                            cv = 10,
#                            scoring = scorer,
                            )
        cv          = zip(idxs_train,idxs_test)
        res         = cross_validate(reg,
                                     features,
                                     BOLD_sc,
                                     cv = cv,
                                     return_estimator = True,
                                     n_jobs = 4)
        regs        = res['estimator']
        preds       = [est.predict(features[idx_test]) for est,idx_test in zip(regs,idxs_test)]
        results     = np.array([metrics.r2_score(BOLD_norm[idx_test],pred,multioutput = 'raw_values') for idx_test,pred in zip(idxs_test,preds)])
        
        scores      = results.copy()
        scores[scores < 0] = np.nan
        scores_mean = np.nanmean(scores,axis = 1)
        khlkj




















