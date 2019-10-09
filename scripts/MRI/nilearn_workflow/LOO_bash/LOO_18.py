#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:49:33 2019

@author: nmei

This script is decoding with LOO methods

"""

import os
import warnings
warnings.filterwarnings('ignore') 
print(os.getcwd())
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


sub                 = 'sub-06'
stacked_data_dir    = '../../../../data/BOLD_average/{}/'.format(sub)
output_dir          = '../../../../results/MRI/nilearn/{}/LOO'.format(sub)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
BOLD_data           = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))
event_data          = np.sort(glob(os.path.join(stacked_data_dir,'*.csv')))

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
        'Mutual + Dummy',
        'Mutual + Linear-SVM',
#        'Mutual + Ensemble-SVMs',
#        'Mutual + KNN',
#        'Mutual + Tree',
        'RandomForest + Dummy',
        'RandomForest + Linear-SVM',
#        'RandomForest + Ensemble-SVMs',
#        'RandomForest + KNN',
#        'RandomForest + Tree',
        ]
#build_model_dictionary().keys()
label_map           = {'Nonliving_Things':[0,1],
                       'Living_Things':   [1,0]}
average             = True
n_jobs              = 20
idx = 17
np.random.seed(12345)
BOLD_name,df_name   = BOLD_data[idx],event_data[idx]
BOLD                = np.load(BOLD_name)
df_event            = pd.read_csv(df_name)
roi_name            = df_name.split('/')[-1].split('_events')[0]
for conscious_state in ['unconscious','glimpse','conscious']:
    idx_unconscious = df_event['visibility'] == conscious_state
    data            = BOLD[idx_unconscious]
    df_data         = df_event[idx_unconscious].reset_index(drop=True)
    df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
    targets         = np.array([label_map[item] for item in df_data['targets'].values])[:,-1]
    
    idxs_train,idxs_test = LOO_partition(data,df_data)
    n_splits = len(idxs_train)
    
    for model_name in model_names:
        file_name   = f'4 models decoding ({sub} {roi_name} {conscious_state} {model_name}).csv'
        if not os.path.exists(os.path.join(output_dir,file_name)):
            np.random.seed(12345)
            
            pipeline    = build_model_dictionary(n_jobs            = 4,
                                                 remove_invariant  = True)[model_name]
            
            features    = data.copy()
            targets     = targets.copy()
            
            with ignore_warnings(category = ConvergenceWarning):
                res = cross_validate(pipeline,
                                     features,
                                     targets,
                                     scoring            = 'roc_auc',
                                     cv                 = zip(idxs_train,idxs_test),
                                     return_estimator   = True,
                                     n_jobs             = n_jobs,
                                     verbose            = 0,
                                     )
            
            preds               = [estimator.predict_proba(  features[ii])[:,-1] for ii,estimator in zip(idxs_test,res['estimator'])]
            roc_auc             = [metrics.roc_auc_score(    targets[ii],y_pred,average = 'micro') for ii,y_pred in zip(idxs_test,preds)]
            threshold_          = [Find_Optimal_Cutoff(      targets[ii],y_pred) for ii,y_pred in zip(idxs_test,preds)]
            mattews_correcoef   = [metrics.matthews_corrcoef(targets[ii],y_pred > thres_) for ii,y_pred,thres_ in zip(idxs_test,preds,threshold_)]
            f1_score            = [metrics.f1_score(         targets[ii],y_pred > thres_) for ii,y_pred,thres_ in zip(idxs_test,preds,threshold_)]
            log_loss            = [metrics.log_loss(         targets[ii],y_pred) for ii,y_pred in zip(idxs_test,preds)]
            
            temp                = np.array([metrics.confusion_matrix(targets[ii],y_pred > thres_).ravel() for ii,y_pred,thres_ in zip(idxs_test,preds,threshold_)])
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
            results_to_save.to_csv(os.path.join(output_dir,file_name),index = False)
            print(f'saving {os.path.join(output_dir,file_name)}')
        else:
            print(file_name)

