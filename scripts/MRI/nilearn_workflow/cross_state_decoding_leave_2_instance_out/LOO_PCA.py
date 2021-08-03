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
from tqdm                    import tqdm
#from sklearn.utils           import shuffle
from shutil                  import copyfile
copyfile('../../../utils.py','utils.py')
from utils                   import (
#                                     customized_partition,
#                                     check_train_test_splits,
#                                     check_train_balance,
                                     build_model_dictionary,
                                     Find_Optimal_Cutoff,
                                     LOO_partition
                                     )
from sklearn.model_selection import cross_validate,GridSearchCV,StratifiedShuffleSplit
from sklearn                 import metric
from sklearn.exceptions      import ConvergenceWarning,UndefinedMetricWarning
#from sklearn.utils.testing   import ignore_warnings
from collections             import OrderedDict
from joblib                  import Parallel,delayed


sub                 = 'sub-01'
stacked_data_dir    = '../../../../data/BOLD_average_BOLD_average_lr/{}/'.format(sub)
output_dir          = '../../../../results/MRI/nilearn/{}/LOO_lr_cross_state'.format(sub)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
BOLD_data           = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))
event_data          = np.sort(glob(os.path.join(stacked_data_dir,'*.csv')))

model_names         = [
        
        # 'None + Linear-SVM','None + Dummy',
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
param_grid = {
              # 'calibratedclassifiercv__base_estimator__penalty':['l1','l2'],
              'calibratedclassifiercv__base_estimator__C':np.logspace(0,5,6)}
#build_model_dictionary().keys()
label_map           = {'Nonliving_Things':[0,1],
                       'Living_Things':   [1,0]}
average             = True
n_jobs              = -1

idx = 0
np.random.seed(12345)
BOLD_name,df_name   = BOLD_data[idx],event_data[idx]
BOLD                = np.load(BOLD_name)
df_event            = pd.read_csv(df_name)
roi_name            = df_name.split('/')[-1].split('_events')[0]
print(roi_name)
conscious_state_source = 'unconscious'
conscious_state_target = 'unconscious'
if True:#for conscious_state_source in ['unconscious','glimpse','conscious']:
    if True:#for conscious_state_target in ['unconscious','glimpse','conscious']:
        if True:#conscious_state_source != conscious_state_target:
            idx_unconscious_source = df_event['visibility'] == conscious_state_source
            data_source            = BOLD[idx_unconscious_source]
            df_data_source         = df_event[idx_unconscious_source].reset_index(drop=True)
            df_data_source['id']   = df_data_source['session'] * 1000 + df_data_source['run'] * 100 + df_data_source['trials']
            targets_source         = np.array([label_map[item] for item in df_data_source['targets'].values])[:,-1]
            
            idx_unconscious_target = df_event['visibility'] == conscious_state_target
            data_target            = BOLD[idx_unconscious_target]
            df_data_target         = df_event[idx_unconscious_target].reset_index(drop=True)
            df_data_target['id']   = df_data_target['session'] * 1000 + df_data_target['run'] * 100 + df_data_target['trials']
            targets_target         = np.array([label_map[item] for item in df_data_target['targets'].values])[:,-1]
            
            print(f'partitioning source: {conscious_state_source}')
            idxs_source,idxs_test  = LOO_partition(df_data_source)
            n_splits = len(idxs_source)
            
            idxs_target = []
            idxs_source_ = []
            idxs_test_ = []
            for idx_source,idx_test in zip(idxs_source,idxs_test):
                words_source = np.unique(df_data_source['labels'].values[idx_source])
                words_target = [word for word in np.unique(df_data_target['labels'].values) if (word not in words_source)]
                if len(words_target) > 1:
#                    print(words_target)
                    idx_words_target, = np.where(df_data_target['labels'].apply(lambda x: x in words_target) == True)
                    idxs_target.append(idx_words_target)
                    idxs_source_.append(idx_source)
                    idxs_test_.append(idx_test)
            # class check
            idxs_target_temp = []
            idxs_source_temp = []
            idxs_source_test_temp = []
            for idx_,idx_source_train,idx_source_test in zip(
                                    idxs_target,
                                    idxs_source_,
                                    idxs_test_):
                temp = df_data_target.iloc[idx_]
                if len(np.unique(targets_target[idx_])) < 2:
                    print(pd.unique(temp['targets']),pd.unique(temp['labels']),targets_target[idx_],)#np.array([label_map[item] for item in temp['targets'].values])[:,-1])
                else:
                    idxs_target_temp.append(idx_)
                    idxs_source_temp.append(idx_source_train)
                    idxs_source_test_temp.append(idx_source_test)
            
            idxs_target = idxs_target_temp
            idxs_source_ = idxs_source_temp
            idxs_test_ = idxs_source_test_temp
            
            n_splits = len(idxs_target)
            print(f'perform {n_splits} cross validation')
            for model_name in model_names:
                file_name   = f'decoding ({sub} {roi_name} {conscious_state_source}->{conscious_state_target} {model_name}).csv'
                print(f'{model_name} {conscious_state_source} --> {conscious_state_target}')
                if not os.path.exists(os.path.join(output_dir,file_name)):
                    np.random.seed(12345)
                    features    = data_source.copy()
                    targets     = targets_source.copy()
                    
                    pipeline    = build_model_dictionary(n_jobs            = 4,
                                                         remove_invariant  = True)[model_name]
                    
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", 
                                                category = ConvergenceWarning,
                                                module = "sklearn")
                        gc.collect()
                        res = cross_validate(pipeline,
                                     features,
                                     targets,
                                     scoring            = 'roc_auc',
                                     cv                 = zip(idxs_source_,idxs_test_),
                                     return_estimator   = True,
                                     n_jobs             = n_jobs,
                                     verbose            = 1,
                                     )
                        gc.collect()
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", 
                                                category = UndefinedMetricWarning)
                        regs = res['estimator']
                        preds = [estimator.predict_proba(data_target[idx_test])[:,-1] for idx_test,estimator in zip(idxs_target,regs)]
                        roc_auc = [metrics.roc_auc_score(targets_target[idx_test],y_pred,average = 'micro') for idx_test,y_pred in zip(idxs_target,preds)]
                        threshold_ = [Find_Optimal_Cutoff(targets_target[idx_test],y_pred) for idx_test,y_pred in zip(idxs_target,preds)]
                        mattews_correcoef = [metrics.matthews_corrcoef(targets_target[idx_test],y_pred> thres_) for idx_test,y_pred,thres_ in zip(idxs_target,preds,threshold_)]
                        f1_score = [metrics.f1_score(targets_target[idx_test],y_pred > thres_) for idx_test,y_pred,thres_ in zip(idxs_target,preds,threshold_)]
                        log_loss = [metrics.log_loss(targets_target[idx_test],y_pred) for idx_test,y_pred in zip(idxs_target,preds)]
                        
                        
                        temp                = np.array([metrics.confusion_matrix(targets_target[idx_test],y_pred > thres_).ravel() for idx_test,y_pred,thres_ in zip(idxs_target,preds,threshold_)])
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
                    results['flip']                 = [False] * n_splits
                    results['language']             = ['Image'] * n_splits
                    results['transfer']             = [True] * n_splits
                    results['tn']                   = tn
                    results['tp']                   = tp
                    results['fn']                   = fn
                    results['fp']                   = fp
                    gc.collect()
                    print(f'{conscious_state_source}-->{conscious_state_target}, {roi_name}, {model_name}, roc_auc = {np.mean(roc_auc):.4f}+/-{np.std(roc_auc):.4f}')
                    
                    results_to_save = pd.DataFrame(results)
                    results_to_save.to_csv(os.path.join(output_dir,file_name),index = False)
                    print(f'saving {os.path.join(output_dir,file_name)}')
                else:
                    print(file_name)

