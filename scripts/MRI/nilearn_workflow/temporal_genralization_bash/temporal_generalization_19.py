#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:13:19 2019

@author: nmei
"""

import os
import gc
import warnings
warnings.filterwarnings('ignore') 
import pandas as pd
import numpy  as np

from glob                    import glob
from tqdm                    import tqdm
from sklearn.utils           import shuffle
from shutil                  import copyfile
copyfile('../../../utils.py','utils.py')
from utils                   import (customized_partition,
                                     check_train_test_splits,
                                     check_train_balance,
                                     build_model_dictionary,
                                     Find_Optimal_Cutoff,
                                     LOO_partition)
from sklearn.model_selection import cross_validate
from sklearn                 import metrics
from sklearn.exceptions      import ConvergenceWarning
from sklearn.utils.testing   import ignore_warnings
from collections             import OrderedDict


sub                 = 'sub-01'
source_data_dir     = '../../../../data/BOLD_average/{}/'.format(sub)
target_data_dir     = '../../../../data/BOLD_average_postresp/{}/'.format(sub)
output_dir          = '../../../../results/MRI/nilearn/{}/temporal_generalization_CP'.format(sub)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
BOLD_data_source    = np.sort(glob(os.path.join(source_data_dir,'*BOLD.npy')))
event_data_source   = np.sort(glob(os.path.join(source_data_dir,'*.csv')))
BOLD_data_target    = np.sort(glob(os.path.join(target_data_dir,'*BOLD.npy')))
event_data_target   = np.sort(glob(os.path.join(target_data_dir,'*.csv')))

model_names         = [
        'None + Dummy',
        'None + Linear-SVM',
#        'None + Ensemble-SVMs',
#        'None + KNN',
#        'None + Tree',
        'PCA + Dummy',
        'PCA + Linear-SVM',
#        'PCA + Ensemble-SVMs',
#        'PCA + KNN',
#        'PCA + Tree',
#        'Mutual + Dummy',
#        'Mutual + Linear-SVM',
#        'Mutual + Ensemble-SVMs',
#        'Mutual + KNN',
#        'Mutual + Tree',
#        'RandomForest + Dummy',
#        'RandomForest + Linear-SVM',
#        'RandomForest + Ensemble-SVMs',
#        'RandomForest + KNN',
#        'RandomForest + Tree',
        ]
#build_model_dictionary().keys()
label_map           = {'Nonliving_Things':[0,1],
                       'Living_Things':   [1,0]}
average             = True
n_jobs              = 32
verbose             = 0

idx = 18
np.random.seed(12345)
BOLD_source_name,df_source_name     = BOLD_data_source[idx],event_data_source[idx]
BOLD_target_name,df_target_name     = BOLD_data_target[idx],event_data_target[idx]
BOLD_source                         = np.load(BOLD_source_name)
BOLD_target                         = np.load(BOLD_target_name)
df_event_source                     = pd.read_csv(df_source_name)
df_event_target                     = pd.read_csv(df_target_name)
roi_name                            = df_source_name.split('/')[-1].split('_events')[0]

for conscious_state in ['unconscious','glimpse','conscious']:
    idx_unconscious = df_event_source['visibility'] == conscious_state
    data_source = BOLD_source[idx_unconscious]
    data_target = BOLD_target[idx_unconscious]
    df_data_source = df_event_source[idx_unconscious].reset_index(drop=True)
    df_data_target = df_event_target[idx_unconscious].reset_index(drop=True)
    df_data_source['id'] = df_data_source['session'] * 1000 +\
                           df_data_source['run'] * 100 +\
                           df_data_source['trials']
    df_data_target['id'] = df_data_target['session'] * 1000 +\
                           df_data_target['run'] * 100 +\
                           df_data_target['trials']
    targets_source = np.array([label_map[item] for item in df_data_source['targets'].values])#[:,-1]
    targets_target = np.array([label_map[item] for item in df_data_target['targets'].values])#[:,-1]
    
    groups = df_data_source['labels'].values
    
    from sklearn.model_selection import StratifiedShuffleSplit
    idxs_train,idxs_test = [],[]
    for idx_train,idx_test in StratifiedShuffleSplit(n_splits = 500,
                                                     test_size = 0.2,
                                                     random_state = 12345).split(
                                                        data_source,
                                                        targets_source,):
        idxs_train.append(idx_train)
        idxs_test.append(idx_test)
#    idxs_train,idxs_test = LOO_partition(df_data_source)
    n_splits = len(idxs_train)
        
    for model_name in model_names:
        gc.collect()
        saving_name = f'decoding {roi_name} {conscious_state} {model_name}.csv'
        if not os.path.exists(os.path.join(output_dir,saving_name)):
            np.random.seed(12345)
            
            pipeline    = build_model_dictionary(n_jobs            = 4,
                                                 remove_invariant  = True)[model_name]
        
            with ignore_warnings(category = ConvergenceWarning):
                res = cross_validate(pipeline,
                                     data_source,
                                     targets_source[:,-1],
                                     scoring            = 'roc_auc',
                                     cv                 = zip(idxs_train,idxs_test),
                                     return_estimator   = True,
                                     n_jobs             = n_jobs,
                                     verbose            = verbose,
                                     )
            clfs                = [clf for clf in res['estimator']]
            preds               = [clf.predict_proba(        data_target[idx_test]) for clf,idx_test in zip(clfs,idxs_test)]
            roc_auc             = [metrics.roc_auc_score(    targets_target[idx_test],y_pred,average = 'micro') for idx_test,y_pred in zip(idxs_test,preds)]
            threshold_          = [Find_Optimal_Cutoff(      targets_target[ii,-1],y_pred[:,-1]) for ii,y_pred in zip(idxs_test,preds)]
            mattews_correcoef   = [metrics.matthews_corrcoef(targets_target[ii,-1],y_pred[:,-1] > thres_) for ii,y_pred,thres_ in zip(idxs_test,preds,threshold_)]
            f1_score            = [metrics.f1_score(         targets_target[ii,-1],y_pred[:,-1] > thres_) for ii,y_pred,thres_ in zip(idxs_test,preds,threshold_)]
            log_loss            = [metrics.log_loss(         targets_target[ii,-1],y_pred[:,-1]) for ii,y_pred in zip(idxs_test,preds)]
            
            temp                = np.array([metrics.confusion_matrix(targets_target[ii,-1],y_pred[:,-1] > thres_).ravel() for ii,y_pred,thres_ in zip(idxs_test,preds,threshold_)])
            tn, fp, fn, tp      = temp[:,0],temp[:,1],temp[:,2],temp[:,3]
            
            results                         = OrderedDict()
            results['fold']                 = np.arange(n_splits) + 1
            results['sub']                  = [sub] * n_splits
            results['roi']                  = [roi_name] * n_splits
            results['roc_auc']              = roc_auc
            results['mattews_correcoef']    = mattews_correcoef
            results['f1_score']             = f1_score
            results['log_loss']             = log_loss
            results['model']                = [model_name] * n_splits
            results['condition_target']     = [conscious_state] * n_splits
            results['condition_source']     = [conscious_state] * n_splits
            results['flip']                 = [False] * n_splits
            results['language']             = ['Image'] * n_splits
            results['transfer']             = [False] * n_splits
            results['tn']                   = tn
            results['tp']                   = tp
            results['fn']                   = fn
            results['fp']                   = fp
            
            print(f'{conscious_state}, {roi_name}, {model_name}, roc_auc = {np.mean(roc_auc):.4f}+/-{np.std(roc_auc):.4f}')
            results_to_save = pd.DataFrame(results)
            results_to_save.to_csv(os.path.join(output_dir,saving_name),index = False)
            print(f'saving {os.path.join(output_dir,saving_name)}')
        else:
            print(saving_name)




































