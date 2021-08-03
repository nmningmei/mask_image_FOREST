#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 05:10:53 2021

@author: nmei
This script is decoding with LOO methods to fit in one of the conscious state, 
and then test in the other consciousness states

"""
def warn(*args,**kwargs):
    pass

import os
import gc
import warnings
warnings.warm = warn
# warnings.filterwarnings('ignore') 
import pandas as pd
import numpy  as np
import multiprocessing

print(f'availabel cpus = {multiprocessing.cpu_count()}')
from glob                    import glob
#from sklearn.utils           import shuffle
from shutil                  import copyfile
copyfile('../../../utils.py','utils.py')
from utils                   import (
                                     build_model_dictionary,
                                     Find_Optimal_Cutoff
                                     )
from sklearn.model_selection import cross_validate,StratifiedShuffleSplit
from sklearn                 import metrics
from sklearn.exceptions      import ConvergenceWarning,UndefinedMetricWarning
from collections             import OrderedDict

# interchangable part:
sub                     = 'sub-05'
conscious_state_source  = 'conscious'
conscious_state_target  = 'conscious'

stacked_data_dir        = '../../../../data/BOLD_average_BOLD_average_lr/{}/'.format(sub)
output_dir              = '../../../../results/MRI/nilearn/decoding_stratified/{}'.format(sub)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
BOLD_data               = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))
event_data              = np.sort(glob(os.path.join(stacked_data_dir,'*.csv')))

model_names             = [
        'None + Linear-SVM',
#        'None + Dummy',
#        'None + Ensemble-SVMs',
#        'None + KNN',
#        'None + Tree',
#        'PCA + Dummy',
#        'PCA + Linear-SVM',
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

scorer = metrics.make_scorer(metrics.log_loss,needs_proba = True,greater_is_better = False)
#build_model_dictionary().keys()
label_map               = {'Nonliving_Things':[0,1],
                           'Living_Things':   [1,0]}
average                 = True
n_jobs                  = -1
n_splits                = int(1e3)


np.random.seed(12345)
for BOLD_name,df_name in zip(BOLD_data,event_data):
    BOLD                    = np.load(BOLD_name)
    df_event                = pd.read_csv(df_name)
    roi_name                = df_name.split('/')[-1].split('_events')[0]
    print(roi_name)
    
    
    idx_unconscious_source  = df_event['visibility'] == conscious_state_source
    data_source             = BOLD[idx_unconscious_source]
    df_data_source          = df_event[idx_unconscious_source].reset_index(drop=True)
    df_data_source['id']    = df_data_source['session'] * 1000 + df_data_source['run'] * 100 + df_data_source['trials']
    targets_source          = np.array([label_map[item] for item in df_data_source['targets'].values])
    
    idx_unconscious_target  = df_event['visibility'] == conscious_state_target
    data_target             = BOLD[idx_unconscious_target]
    df_data_target          = df_event[idx_unconscious_target].reset_index(drop=True)
    df_data_target['id']    = df_data_target['session'] * 1000 + df_data_target['run'] * 100 + df_data_target['trials']
    targets_target          = np.array([label_map[item] for item in df_data_target['targets'].values])
    
    cv = StratifiedShuffleSplit(n_splits = n_splits,test_size = 0.1,random_state = 12345)
    print(f'partitioning target: {conscious_state_target}')
    idxs_train_target,idxs_test_target  = [],[]
    for train,test in cv.split(data_target,targets_target[:,-1]):
        idxs_train_target.append(train)
        idxs_test_target.append(test)
    print(f'partitioning source: {conscious_state_source}')
    # for each fold of the train-test, we throw away the subcategories that exist in the target
    idxs_train_source,idxs_test_source = [],[]
    for train,test in cv.split(data_source,targets_source[:,-1]):
        idxs_train_source.append(train)
        idxs_test_source.append(test)
    
    
    for model_name in model_names:
        file_name           = f'{sub}_{roi_name}_{conscious_state_source}_{conscious_state_target}_{model_name}.csv'.replace(' + ','_')
        print(f'{roi_name} {model_name} {conscious_state_source} --> {conscious_state_target}')
        if not os.path.exists(os.path.join(output_dir,file_name)):
            np.random.seed(12345)
            features        = data_source.copy()
            targets         = targets_source.copy()[:,-1]
            
            pipeline        = build_model_dictionary(n_jobs            = 4,
                                                     remove_invariant  = True,
                                                     l1                = True,
                                                     )[model_name]
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", 
                                        category = ConvergenceWarning,
                                        module = "sklearn")
                gc.collect()
                res = cross_validate(pipeline,
                             features,
                             targets,
                             scoring            = 'roc_auc',
                             cv                 = zip(idxs_train_source,idxs_test_source),
                             return_estimator   = True,
                             n_jobs             = n_jobs,
                             verbose            = 1,
                             )
                gc.collect()
                
                warnings.filterwarnings("ignore", 
                                        category = UndefinedMetricWarning)
                regs                = res['estimator']
                y_true              = [targets_target[idx_test] for idx_test in idxs_test_target]
                y_pred              = [estimator.predict_proba(data_target[idx_test]) for idx_test,estimator in zip(idxs_test_target,regs)]
                
                
                roc_auc             = [metrics.roc_auc_score(y_true_,y_pred_,average = 'micro') for y_true_,y_pred_ in zip(y_true,y_pred)]
                threshold_          = [Find_Optimal_Cutoff(y_true_[:,-1],y_pred_[:,-1]) for y_true_,y_pred_ in zip(y_true,y_pred)]
                mattews_correcoef   = [metrics.matthews_corrcoef(y_true_[:,-1],y_pred_[:,-1]>thres_) for y_true_,y_pred_,thres_ in zip(y_true,y_pred,threshold_)]
                f1_score            = [metrics.f1_score(y_true_[:,-1],y_pred_[:,-1]>thres_) for y_true_,y_pred_,thres_ in zip(y_true,y_pred,threshold_)]
                log_loss            = [metrics.log_loss(y_true_,y_pred_) for y_true_,y_pred_ in zip(y_true,y_pred)]
                
                
                temp                = np.array([metrics.confusion_matrix(y_true_[:,-1],y_pred_[:,-1]>thres_).ravel() for y_true_,y_pred_,thres_ in zip(y_true,y_pred,threshold_)])
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
            results['condition_target']     = [conscious_state_target] * n_splits
            results['condition_source']     = [conscious_state_source] * n_splits
            results['tn']                   = tn
            results['tp']                   = tp
            results['fn']                   = fn
            results['fp']                   = fp
            results['y_true']               = [','.join(y_true_[:,-1].astype(int).astype(str)) for y_true_ in y_true]
            gc.collect()
            print(f'{conscious_state_source}-->{conscious_state_target}, {roi_name}, {model_name}, roc_auc = {np.mean(roc_auc):.4f}+/-{np.std(roc_auc):.4f}')
            results_to_save                 = pd.DataFrame(results)
            results_to_save.to_csv(os.path.join(output_dir,file_name),index = False)
            print(f'saving {os.path.join(output_dir,file_name)}')
