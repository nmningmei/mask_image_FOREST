#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:06:18 2019

@author: nmei


unconscious ,probe presented for 1.822+/-0.956, p(correct) = 0.504 for 766 trials
glimpse     ,probe presented for 2.980+/-1.103, p(correct) = 0.868 for 456 trials
conscious   ,probe presented for 4.062+/-1.251, p(correct) = 0.994 for 502 trials
missing data,probe presented for 4.250+/-0.829, p(correct) = 1.000 for   4 trials


But this dataset contains the short run

"""

import utils
import os
from glob                    import glob
from tqdm                    import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate#,permutation_test_score,cross_val_score
from sklearn                 import metrics
from sklearn.utils           import shuffle
from sklearn.exceptions      import ConvergenceWarning
from sklearn.utils.testing   import ignore_warnings
from collections             import OrderedDict



working_dir = '../../../data/BOLD_stacked/'
working_data = np.sort(glob(os.path.join(working_dir,'*.npy')))
working_df = np.sort(glob(os.path.join(working_dir,'*.csv')))
output_dir = '../../../results/pymvpa'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
label_map = {'Living_T':[1,0],'Nonlivin':[0,1]}
vis_map = {'1':'unconscious','2':'glimpse','3':'conscious'}
model_names          = [
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

idx_ = 23
n_jobs = 8
n_splits = 100

ds = np.load(working_data[idx_])
df = pd.read_csv(working_df[idx_])
roi_name = working_data[idx_].split('/')[-1].split('.')[0]

for vis in ['1','2','3']:
    data = ds[df['visibility'] == int(vis)]
    conscious_state = vis_map[vis]
    df_data = df[df['visibility'] == int(vis)].reset_index()
    
    idxs_test = utils.customized_partition(df_data,'labels',n_splits)
    while utils.check_train_test_splits(idxs_test): # just in case
        idxs_test   = utils.customized_partition(df_data,n_splits = n_splits)
    idxs_train      = [shuffle(np.array([idx for idx in df_data.index.tolist() if (idx not in idx_test)])) for idx_test in idxs_test]
    idxs_train      = [utils.check_train_balance(df_data,idx_train,list(label_map.keys())) for idx_train in tqdm(idxs_train)]
    
    
    features = data.copy()
    targets = np.array([label_map[item] for item in df_data['targets']])[:,-1]
    
    for model_name in model_names:
        
        np.random.seed(12345)
        pipeline    = utils.build_model_dictionary(n_jobs = 4,)[model_name]
        features_,targets_ = features.copy(),targets.copy()
        
        with ignore_warnings(category = ConvergenceWarning):
            cv  = zip(idxs_train,idxs_test)
            res = cross_validate(pipeline,
                                 features_,
                                 targets_,
                                 scoring            = 'roc_auc',
                                 cv                 = zip(idxs_train,idxs_test),
                                 return_estimator   = True,
                                 n_jobs             = n_jobs,
                                 verbose            = 2,
                                 )
#            score, permutation_scores, pvalue = permutation_test_score(
#                                 utils.build_model_dictionary(n_jobs = 4,)[model_name],
#                                 features_,
#                                 targets_,
#                                 scoring            = 'roc_auc',
#                                 cv                 = zip(idxs_train,idxs_test),
#                                 n_permutations     = 100,
#                                 n_jobs             = n_jobs,
#                                 random_state       = 12345,
#                                 verbose            = 2,
#                                 )
        preds               = [estimator.predict_proba(  features_[ii])[:,-1] for ii,estimator in zip(idxs_test,res['estimator'])]
        roc_auc             = [metrics.roc_auc_score(    targets_[ii],y_pred) for ii,y_pred in zip(idxs_test,preds)]
        threshold_          = [utils.Find_Optimal_Cutoff(      targets_[ii],y_pred) for ii,y_pred in zip(idxs_test,preds)]
        mattews_correcoef   = [metrics.matthews_corrcoef(targets_[ii],y_pred > thres_) for ii,y_pred,thres_ in zip(idxs_test,preds,threshold_)]
        f1_score            = [metrics.f1_score(         targets_[ii],y_pred > thres_) for ii,y_pred,thres_ in zip(idxs_test,preds,threshold_)]
        log_loss            = [metrics.log_loss(         targets_[ii],y_pred) for ii,y_pred in zip(idxs_test,preds)]
        
        temp                = np.array([metrics.confusion_matrix(targets_[ii],y_pred > thres_).ravel() for ii,y_pred,thres_ in zip(idxs_test,preds,threshold_)])
        tn, fp, fn, tp      = temp[:,0],temp[:,1],temp[:,2],temp[:,3]
        
        results                         = OrderedDict()
        results                         = OrderedDict()
        results['fold']                 = np.arange(n_splits) + 1
        results['sub']                  = ['01'] * n_splits
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
        
        print('{conscious_state}, {roi_name}, {model_name}, roc_auc = {mean_roc_auc:.4f}+/-{std_roc_auc:.4f}'.format(
                conscious_state = conscious_state,roi_name = roi_name, model_name = model_name,
                mean_roc_auc = np.mean(roc_auc),std_roc_auc = np.std(roc_auc)))
#        print('permu-score = {:.4f},p = {:.4f}'.format(score,pvalue))
        results_to_save = pd.DataFrame(results)
        
        results_to_save.to_csv(os.path.join(output_dir,'decoding ({} {} {}).csv'.format(
                roi_name,vis_map[vis],model_name)),index=False)











