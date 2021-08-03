#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:49:33 2019

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
                                     check_LOO_cv,
                                     build_model_dictionary,
                                     Find_Optimal_Cutoff,
                                     LOO_partition
                                     )
from sklearn.model_selection import cross_validate
from sklearn                 import metrics
from sklearn.exceptions      import ConvergenceWarning,UndefinedMetricWarning
from joblib                  import Parallel,delayed

# interchangable part:
sub                     = 'sub-05'
conscious_state_source  = 'conscious'
conscious_state_target  = 'conscious'

stacked_data_dir        = '../../../../data/BOLD_average_BOLD_average_lr/{}/'.format(sub)
output_dir              = '../../../../results/MRI/nilearn/decoding_instance/{}'.format(sub)
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
    
    print(f'partitioning target: {conscious_state_target}')
    idxs_train_target,idxs_test_target  = LOO_partition(df_data_target)
    n_splits                            = len(idxs_test_target)
    print(f'{n_splits} folds of testing')
    
    print(f'partitioning source: {conscious_state_source}')
    # for each fold of the train-test, we throw away the subcategories that exist in the target
    cv_warning,idxs_train_source,idxs_test_source = check_LOO_cv(
            idxs_test_target,df_data_target,df_data_source)
    idxs_train_target,idxs_test_target=[],[]
    idxs_train_source,idxs_test_source=[],[]
    from sklearn.model_selection import StratifiedShuffleSplit
    cv = StratifiedShuffleSplit(n_splits=300,test_size = 0.1,random_state=12345)
    for train,test in cv.split(data_source,targets_source[:,-1]):
        idxs_train_target.append(train)
        idxs_train_source.append(train)
        idxs_test_target.append(test)
        idxs_test_source.append(test)
    
    if not cv_warning:
        for model_name in model_names:
            file_name           = f'({sub}_{roi_name}_{conscious_state_source}_{conscious_state_target}_{model_name}).csv'.replace(' + ','_')
            print(f'{model_name} {conscious_state_source} --> {conscious_state_target}')
            if not os.path.exists(os.path.join(output_dir,file_name)):
                np.random.seed(12345)
                features        = data_source.copy()
                targets         = targets_source[:,-1].copy()
                
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
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", 
                                            category = UndefinedMetricWarning)
                    regs                = res['estimator']
                    y_true              = np.concatenate([targets_target[idx_test] for idx_test in idxs_test_target])
                    y_pred              = np.concatenate([estimator.predict_proba(data_target[idx_test]) for idx_test,estimator in zip(idxs_test_target,regs)])
                    score               = metrics.roc_auc_score(y_true,y_pred)
                # get p value
                def _gen(y_true):
                    y_pred_ = np.random.uniform(size = y_pred.shape[0])
                    y_pred_ = np.vstack([y_pred_,1-y_pred_]).T
                    return metrics.roc_auc_score(y_true,y_pred_)
                null_dist = Parallel(n_jobs = n_jobs,verbose = 1,)(delayed(_gen)(**{
                        'y_true':y_true}) for _ in range(int(1e4)))
                gc.collect()
                pval = (np.sum(null_dist >= score) + 1) / (int(1e4) + 1)
                df = pd.DataFrame(dict(sub=[sub],
                                       roi_name=[roi_name],
                                       conscious_state_source=[conscious_state_source],
                                       conscious_state_target=[conscious_state_target],
                                       score=[score],
                                       pval=[pval],))
                df.to_csv(os.path.join(output_dir,file_name),index=False)
            else:
                print(file_name)
    else:
        print('cross validation partition is wrong')
