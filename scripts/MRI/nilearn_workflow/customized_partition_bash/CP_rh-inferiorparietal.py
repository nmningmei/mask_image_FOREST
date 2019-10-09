#!/usr/bin/env python3 -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:59:34 2019

@author: nmei

decoding pipeline with multiple models and multiple rois, using customized partition
cross validation method

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
from utils                   import (
                                     build_model_dictionary,
                                     Find_Optimal_Cutoff,
                                     get_label_category_mapping)
from sklearn.model_selection import cross_validate,LeavePGroupsOut
from sklearn                 import metrics
from sklearn.exceptions      import ConvergenceWarning
from sklearn.utils.testing   import ignore_warnings
from collections             import OrderedDict


sub                 = 'sub-04'
stacked_data_dir    = '../../../../data/BOLD_average/{}/'.format(sub)
output_dir          = '../../../../results/MRI/nilearn/{}/L2O'.format(sub)
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
#build_model_dictionary().keys()
label_map           = {'Nonliving_Things':[0,1],
                       'Living_Things':   [1,0]}
target_map = {key:ii for ii,key in enumerate(get_label_category_mapping().keys())}
average             = True
n_jobs = 16

idx = 13
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
    groups          = df_data['labels'].values
    
    cv = LeavePGroupsOut(n_groups = 2)
    
    idxs_train,idxs_test = [],[]
    for idx_train,idx_test in cv.split(data,targets,groups):
        idxs_train.append(idx_train)
        idxs_test.append(idx_test)
    
    for model_name in model_names:
        file_name   = f'4 models decoding ({sub} {roi_name} {conscious_state} {model_name}).csv'
        if True:#not os.path.exists(os.path.join(output_dir,file_name)):
            np.random.seed(12345)
            
            pipeline    = build_model_dictionary(n_jobs            = 4,
                                                 remove_invariant  = True)[model_name]
            
            features    = data.copy()
            targets     = targets.copy()
            with ignore_warnings(category = ConvergenceWarning):
                res = cross_validate(pipeline,
                                     features,
                                     targets,
                                     groups             = groups,
                                     scoring            = 'accuracy',
                                     cv                 = cv,
                                     return_estimator   = True,
                                     n_jobs             = n_jobs,
                                     verbose            = 0,
                                     )
            
            preds               = [estimator.predict(features[ii]) for ii,estimator in zip(idxs_test,res['estimator'])]
            scores              = [metrics.accuracy_score(targets[idx_test],y_pred) for idx_test,y_pred in zip(idxs_test,preds)]
            
            n_splits            = len(idxs_test)
            results                         = OrderedDict()
            results['fold']                 = np.arange(n_splits) + 1
            results['sub']                  = [sub] * n_splits
            results['roi']                  = [roi_name] * n_splits
            results['model']                = [model_name] * n_splits
            results['condition_target']     = [conscious_state] * n_splits
            results['condition_source']     = [conscious_state] * n_splits
            results['flip']                 = [False] * n_splits
            results['language']             = ['Image'] * n_splits
            results['transfer']             = [False] * n_splits
            results['accuracy']             = scores
            results['test1']                = [pd.unique(df_data.loc[idx_test,:]['labels'])[0] for idx_test in idxs_test]
            results['test2']                = [pd.unique(df_data.loc[idx_test,:]['labels'])[1] for idx_test in idxs_test]
            
            print(f'{conscious_state}, {roi_name}, {model_name}, roc_auc = {np.mean(scores):.4f}+/-{np.std(scores):.4f}')
            
            results_to_save = pd.DataFrame(results)
            results_to_save.to_csv(os.path.join(output_dir,file_name),index = False)
            print(f'saving {os.path.join(output_dir,file_name)}')











