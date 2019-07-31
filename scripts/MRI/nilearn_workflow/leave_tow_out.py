#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:59:34 2019

@author: nmei

decoding pipeline with multiple models and multiple rois, using leave 2 instances out
cross validation method

"""
import os
import pandas as pd
import numpy  as np

from glob                    import glob
from sklearn.utils           import shuffle
from shutil                  import copyfile
copyfile('../../utils.py','utils.py')
from utils                   import (groupy_average,
                                     build_model_dictionary,
                                     get_blocks)
from sklearn.metrics         import roc_auc_score
from sklearn.exceptions      import ConvergenceWarning
from sklearn.utils.testing   import ignore_warnings


sub                 = 'sub-01'
stacked_data_dir    = '../../../data/BOLD_no_average/{}/'.format(sub)
BOLD_data           = glob(os.path.join(stacked_data_dir,'*BOLD*.npy'))
event_data          = glob(os.path.join(stacked_data_dir,'*.csv'))
model_name          = [
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
label_map           = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}
average = True

results = dict(
        sub             = [],
        roi             = [],
        conscious_state = [],
        fold            = [],
        score           = [],
        model_name      = [],
        cv_method       = [],
        behavioral      = [],
        )

np.random.seed(12345)
for BOLD_name,df_name in zip(BOLD_data,event_data):
    BOLD_name,df_name
    BOLD            = np.load(BOLD_name)
    df_event        = pd.read_csv(df_name)
    roi_name        = df_name.split('/')[-1].split('_events')[0]
    for conscious_state in ['unconscious','glimpse','conscious']:
        idx_unconscious = df_event['visibility'] == conscious_state
        data            = BOLD[idx_unconscious]
        df_data         = df_event[idx_unconscious].reset_index()
        df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
        targets         = np.array([label_map[item] for item in df_data['targets'].values])[:,-1]
        
        for name in model_name:
            temp_score      = []
            for fold,(train,test) in enumerate():
                if average:
                    X_,df_ = groupy_average(data[train],df_data.iloc[train].reset_index(),groupby=['id'])
                    X,y = X_,np.array([label_map[item] for item in df_['targets'].values])[:,-1]
                else:
                    X,y             = data[train],targets[train]
                X,y             = shuffle(X,y)
                X_test,y_test   = data[test],targets[test]
                df_test         = df_data.iloc[test].reset_index()
                X_test_ave,temp = groupy_average(X_test,df_test,groupby=['id'])
                y_test          = np.array([label_map[item] for item in temp['targets']])[:,-1]
                
                pipeline    = build_model_dictionary(n_jobs = 6)[name]
                with ignore_warnings(category=ConvergenceWarning):
                    pipeline.fit(X,y)
                preds       = pipeline.predict_proba(X_test_ave)[:,-1]
                score       = roc_auc_score(y_test,preds)
                
                results['sub'               ].append(sub)
                results['roi'               ].append(roi_name)
                results['conscious_state'   ].append(conscious_state)
                results['fold'              ].append(fold + 1)
                results['model_name'        ].append(name)
                results['score'             ].append(score)
                results['cv_method'         ].append('LOO')
                results['behavioral'        ].append(df_data['correct'].mean())
                temp_score.append(score)
                
                print(f'\n{roi_name:15}, {name:28},{conscious_state}\nfold {fold + 1} = {score:.4f}-{np.mean(temp_score):.4f}')


#pick_test_classes = [[label1,label2] for label1 in make_class['Living_Things'] for label2 in make_class['Nonliving_Things']]

def get_blocks(df__,label_map,):
    ids = df_data['id'].values
    chunks = df_data['session'].values
    words = df_data['labels'].values
    labels = np.array([label_map[item] for item in df_data['targets'].values])[:,-1]
    sample_indecies = np.arange(len(labels))
    blocks = [np.array([ids[ids == target],
                        chunks[ids == target],
                        words[ids == target],
                        labels[ids == target],
                        sample_indecies[ids == target]
                       ]) for target in np.unique(ids)
                ]
    block_labels = np.array([np.unique(ll[-2]) for ll in blocks]).ravel()
    return blocks,block_labels
df = df_data.copy()

idxs_train,idxs_test = get_train_test_splits(df,label_map,10)
check_train_test_splits(idxs_test)
idx_train = idxs_train[3]
c=check_train_balance(df,idx_train,list(label_map.keys()))









