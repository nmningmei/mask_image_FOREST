#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:49:33 2019

@author: nmei

This script is decoding with Leave p groups out methods -- groups are defined by sessions and runs

"""

import os
import gc
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
from utils                   import (build_model_dictionary,
                                     Find_Optimal_Cutoff)
from sklearn.model_selection import (cross_validate,
                                     LeavePGroupsOut,
                                     permutation_test_score)
from sklearn                 import metrics
from sklearn.exceptions      import ConvergenceWarning
from sklearn.utils.testing   import ignore_warnings
from sklearn.base            import clone
from joblib                  import Parallel, delayed
from collections             import OrderedDict


sub                 = 'sub-01'
stacked_data_dir    = '../../../../data/BOLD_average/{}/'.format(sub)
output_dir          = '../../../../results/MRI/nilearn/{}/LSRO'.format(sub)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
BOLD_data           = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))
event_data          = np.sort(glob(os.path.join(stacked_data_dir,'*.csv')))

model_names         = [
        'None + Dummy',
        'None + Linear-SVM',
        ]
#build_model_dictionary().keys()
label_map           = {'Nonliving_Things':[0,1],
                       'Living_Things':   [1,0]}
average             = True
n_jobs              = -1
n_permutations      = int(1e4)

idx = 15
np.random.seed(12345)
BOLD_name,df_name   = BOLD_data[idx],event_data[idx]
BOLD                = np.load(BOLD_name)
df_event            = pd.read_csv(df_name)
roi_name            = df_name.split('/')[-1].split('_events')[0]
for conscious_state in ['unconscious','glimpse','conscious']:
#conscious_state     = 'unconscious' # condition_change <- making it for batch process
#if True: # meaningless
    idx_unconscious = df_event['visibility'] == conscious_state
    data            = BOLD[idx_unconscious]
    df_data         = df_event[idx_unconscious].reset_index(drop=True)
    df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
    targets         = np.array([label_map[item] for item in df_data['targets'].values])[:,-1]
    groups          = df_data['session'] * 10 + df_data['run']
    
    cv = LeavePGroupsOut(n_groups = 3) # <- large number
    
    n_splits = cv.get_n_splits(data,targets,groups)
    
    for model_name in model_names:
        file_name   = f'decoding {sub} {roi_name} {conscious_state} {model_name}.csv'
        print(file_name)
        if not os.path.exists(os.path.join(output_dir,file_name)):
            np.random.seed(12345)
            
            pipeline    = build_model_dictionary(n_jobs            = 4,
                                                 remove_invariant  = True)[model_name]
            
            features    = data.copy()
            targets     = targets.copy()
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                gc.collect()
                # get the cross validation scores
                res = cross_validate(pipeline,
                                     features,
                                     targets,
                                     groups             = groups,
                                     scoring            = 'roc_auc',
                                     cv                 = cv,
                                     return_estimator   = True,
                                     n_jobs             = n_jobs,
                                     verbose            = 1,
                                     )
                
                gc.collect()
                
            idxs_test = []
            for _,idx_test in cv.split(features,targets,groups = groups):
                idxs_test.append(idx_test)
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

