#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:49:33 2019

@author: nmei
"""

import os
print(os.getcwd())
import pandas as pd
import numpy  as np

from glob                    import glob
from sklearn.utils           import shuffle
from shutil                  import copyfile
copyfile('../../../utils.py','utils.py')
from utils                   import (groupy_average,
                                     check_train_balance,
                                     build_model_dictionary,
                                     LOO_partition)
from sklearn.metrics         import roc_auc_score
from sklearn.exceptions      import ConvergenceWarning
from sklearn.utils.testing   import ignore_warnings
from collections             import Counter


sub                 = 'sub-01'
stacked_data_dir    = '../../../../data/BOLD_no_average/{}/'.format(sub)
output_dir          = '../../../../results/MRI/LOO'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
BOLD_data           = glob(os.path.join(stacked_data_dir,'*BOLD.npy'))
event_data          = glob(os.path.join(stacked_data_dir,'*.csv'))
model_name          = [
#        'None + Dummy',
        'None + Linear-SVM',
#        'None + Ensemble-SVMs',
#        'None + KNN',
#        'None + Tree',
#        'PCA + Dummy',
        'PCA + Linear-SVM',
#        'PCA + Ensemble-SVMs',
#        'PCA + KNN',
#        'PCA + Tree',
#        'Mutual + Dummy',
        'Mutual + Linear-SVM',
#        'Mutual + Ensemble-SVMs',
#        'Mutual + KNN',
#        'Mutual + Tree',
#        'RandomForest + Dummy',
        'RandomForest + Linear-SVM',
#        'RandomForest + Ensemble-SVMs',
#        'RandomForest + KNN',
#        'RandomForest + Tree',
        ]
#build_model_dictionary().keys()
label_map           = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}
average             = True



idx = 1
np.random.seed(12345)
BOLD_name,df_name = BOLD_data[idx],event_data[idx]
BOLD            = np.load(BOLD_name)
df_event        = pd.read_csv(df_name)
roi_name        = df_name.split('/')[-1].split('_events')[0]
for conscious_state in ['unconscious','glimpse','conscious']:
    file_name = f'4 models decoding ({sub} {roi_name} {conscious_state}).csv'
    if not os.path.exists(os.path.join(output_dir,file_name)):
        results = dict(
        sub             = [],
        roi             = [],
        conscious_state = [],
        fold            = [],
        roc_auc         = [],
        model_name      = [],
        cv_method       = [],
        behavioral      = [],
        label1          = [],
        label2          = [],
        )
        idx_unconscious = df_event['visibility'] == conscious_state
        data            = BOLD[idx_unconscious]
        df_data         = df_event[idx_unconscious].reset_index(drop=True)
        df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
        targets         = np.array([label_map[item] for item in df_data['targets'].values])[:,-1]
        
        idxs_train,idxs_test = LOO_partition(data,df_data)
        if os.path.exists(os.path.join(output_dir,file_name.replace('.csv','_temp.csv'))):
            results_to_save = pd.read_csv(os.path.join(output_dir,file_name.replace('.csv','_temp.csv')))
            results = results_to_save.to_dict()
            fold = results['fold'][-1] - 1
            idxs_train,idxs_test = idxs_train[fold:],idxs_test[fold:]
        temp_score      = []
        for fold,(idx_train,idx_test) in enumerate(zip(idxs_train,idxs_test)):
            for name in model_name:
                # check balance 
                idx_train = check_train_balance(df_data,idx_train,list(label_map.keys()))
                print(Counter(df_data.iloc[idx_train]['targets']))
                if average:
                    X_,df_ = groupy_average(data[idx_train],df_data.iloc[idx_train].reset_index(drop=True),groupby=['id'])
                    X,y = X_,np.array([label_map[item] for item in df_['targets'].values])[:,-1]
                else:
                    X,y             = data[idx_train],targets[idx_train]
                X,y             = shuffle(X,y)
                X_test,y_test   = data[idx_test],targets[idx_test]
                df_test         = df_data.iloc[idx_test].reset_index(drop=True)
                X_test_ave,temp = groupy_average(X_test,df_test,groupby=['id'])
                y_test          = np.array([label_map[item] for item in temp['targets']])[:,-1]
                
                pipeline    = build_model_dictionary(n_jobs = 1)[name]
                with ignore_warnings(category=ConvergenceWarning):
                    pipeline.fit(X,y)
                preds       = pipeline.predict_proba(X_test_ave)[:,-1]
                score       = roc_auc_score(y_test,preds)
                unique_pair = pd.unique(temp['labels'])
                results['sub'               ].append(sub)
                results['roi'               ].append(roi_name)
                results['conscious_state'   ].append(conscious_state)
                results['fold'              ].append(fold + 1)
                results['model_name'        ].append(name)
                results['roc_auc'           ].append(score)
                results['cv_method'         ].append('LOO')
                results['behavioral'        ].append(df_data['correct'].mean())
                results['label1'            ].append(unique_pair[0])
                results['label2'            ].append(unique_pair[1])
                temp_score.append(score)
                
                print(f'\n{roi_name:15}, {name:28},{conscious_state}\nfold {fold + 1} = {score:.4f}-{np.mean(temp_score):.4f}')
                results_to_save = pd.DataFrame(results)
                
                results_to_save.to_csv(os.path.join(output_dir,file_name.replace('.csv','_temp.csv')),index=False)
                results_to_save.to_csv(os.path.join(output_dir,file_name),index=False)
os.remove(os.path.join(output_dir,file_name.replace('.csv','_temp.csv')))