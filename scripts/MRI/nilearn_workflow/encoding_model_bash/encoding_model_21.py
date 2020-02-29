#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:54:57 2019

@author: nmei

encoding model that is interchangable btw 2 source folders: BOLD-average, BOLD-average-prestim

"""
import os
import gc
gc.collect()
import numpy as np
import pandas as pd
from glob                      import glob
#from tqdm                      import tqdm
from shutil                    import copyfile
copyfile('../../../utils.py','utils.py')
from utils                     import (LOO_partition,
                                       cross_validation,
                                       fill_results)
from sklearn.model_selection   import StratifiedShuffleSplit#,cross_validate
from sklearn                   import metrics
#from sklearn                   import linear_model
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing     import MinMaxScaler
#from sklearn.model_selection   import GridSearchCV

from collections               import OrderedDict

from scipy.spatial             import distance
#from joblib                    import Parallel,delayed
from scipy                     import stats

def score_func(y, y_pred,):
    temp        = metrics.r2_score(y,y_pred,multioutput = 'raw_values')
    if np.sum(temp > 0):
        return temp[temp > 0].mean()
    else:
        return 0
custom_scorer      = metrics.make_scorer(score_func,greater_is_better = True)
if __name__ == '__main__':
    sub                 = 'sub-01'
    target_folder       = 'encoding_CP'
    data_folder         = 'BOLD_average' # BOLD_average_prepeak
    stacked_data_dir    = '../../../../data/{}/{}/'.format(data_folder,sub)
    background_dir      = '../../../../data/computer_vision_background'
    feature_dir         = '../../../../data/computer_vision_features'
    output_dir          = '../../../../results/MRI/nilearn/{}/{}'.format(sub,target_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    BOLD_data           = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))
    event_data          = np.sort(glob(os.path.join(stacked_data_dir,'*.csv')))
    computer_models     = os.listdir(feature_dir)
    
    label_map           = {'Nonliving_Things':[0,1],
                           'Living_Things':   [1,0]}
    n_splits            = 300
    n_jobs              = -1
    
    idx                 = 20
    BOLD_name,df_name   = BOLD_data[idx],event_data[idx]
    BOLD                = np.load(BOLD_name)
    df_event            = pd.read_csv(df_name)
    roi_name            = df_name.split('/')[-1].split('_events')[0]
    
    np.random.seed(12345)
    for conscious_source in ['unconscious','glimpse','conscious']:
        for conscious_target in ['unconscious','glimpse','conscious']:
            idx_source              = df_event['visibility'] == conscious_source
            data_source             = BOLD[idx_source]
            VT                      = VarianceThreshold()
            scaler                  = MinMaxScaler((-1,1))
            BOLD_norm_source        = VT.fit_transform(data_source)
            BOLD_sc_source          = scaler.fit_transform(BOLD_norm_source)
            df_data_source          = df_event[idx_source].reset_index(drop=True)
            df_data_source['id']    = df_data_source['session'] * 1000 + df_data_source['run'] * 100 + df_data_source['trials']
            targets_source          = np.array([label_map[item] for item in df_data_source['targets'].values])[:,-1]
            groups_source           = df_data_source['labels'].values
            
            idx_target              = df_event['visibility'] == conscious_target
            data_target             = BOLD[idx_target]
#            VT                      = VarianceThreshold()
#            scaler                  = MinMaxScaler((-1,1))
            BOLD_norm_target        = VT.transform(data_target)
            BOLD_sc_target          = scaler.transform(BOLD_norm_target)
            df_data_target          = df_event[idx_target].reset_index(drop=True)
            df_data_target['id']    = df_data_target['session'] * 1000 + df_data_target['run'] * 100 + df_data_target['trials']
            targets_target          = np.array([label_map[item] for item in df_data_target['targets'].values])[:,-1]
            groups_target           = df_data_target['labels'].values
            
            if n_splits > 300:
                idxs_train_source,idxs_test_source  = LOO_partition(df_data_source)
                # construct the test set of the target data based on the source training data
                idxs_target = []
                idxs_train_source_ = []
                idxs_test_source_ = []
                for idx_train_source,idx_test_source in zip(idxs_train_source,idxs_test_source):
                    words_source = np.unique(df_data_source['labels'].values[idx_train_source])
                    words_target = [word for word in np.unique(df_data_target['labels'].values) if (word not in words_source)]
                    if len(words_target) > 1:
                        idx_words_target, = np.where(df_data_target['labels'].apply(lambda x: x in words_target) == True)
                        idxs_target.append(idx_words_target)
                        idxs_train_source_.append(idx_train_source)
                        idxs_test_source_.append(idx_test_source)
                # class check: sometime, the left-out 2 items belong to the same category
                idxs_target_temp = []
                idxs_train_source_temp = []
                idxs_test_source_temp = []
                for idx_,idx_source_train,idx_source_test in zip(
                                        idxs_target,
                                        idxs_train_source_,
                                        idxs_test_source_):
                    temp = df_data_target.iloc[idx_]
                    if len(np.unique(targets_target[idx_])) < 2:
                        print(pd.unique(temp['targets']),pd.unique(temp['labels']),targets_target[idx_],)#np.array([label_map[item] for item in temp['targets'].values])[:,-1])
                    else:
                        idxs_target_temp.append(idx_)
                        idxs_train_source_temp.append(idx_source_train)
                        idxs_test_source_temp.append(idx_source_test)
                
                idxs_target_test = idxs_target_temp
                idxs_train_source = idxs_train_source_temp
                idxs_test_source = idxs_test_source_temp
                
                n_splits = len(idxs_target_test)
            else:
                cv = StratifiedShuffleSplit(n_splits = n_splits,
                                            test_size = .1,
                                            random_state = 12345,
                                            )
                idxs_train_source,idxs_test_source,idxs_target_test = [],[],[]
                for idx_train_,idx_test_ in cv.split(BOLD_sc_source,targets_source):
                    idxs_train_source.append(idx_train_)
                    idxs_test_source.append(idx_test_)
                    idxs_target_test.append(np.random.choice(np.arange(BOLD_sc_target.shape[0]),
                                                             size = int(BOLD_sc_target.shape[0] * .8),
                                                             replace = False))
                    
            n_splits = len(idxs_target_test)
            print(f'{conscious_source} --> {conscious_target},perform {n_splits} cross validation')
            
            
            image_source = df_data_source['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
            image_target = df_data_target['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
            
            
            for encoding_model in ['DenseNet169']:
                saving_name = f"encoding model {conscious_source} {conscious_target} {roi_name} {encoding_model}.csv"
                processed   = glob(os.path.join(output_dir,"*.csv"))
                print(saving_name)
                if not os.path.join(output_dir,saving_name) in processed:
                    res,features_target,features_source = cross_validation(
                                                           feature_dir,
                                                           encoding_model,
                                                           custom_scorer,
                                                           BOLD_sc_source,
                                                           idxs_train_source,
                                                           idxs_test_source,
                                                           image_source,
                                                           image_target,)
                    regs = res['estimator']
                    preds   = [est.predict(features_target[idx_test]) for est,idx_test in zip(regs,idxs_target_test)]
                    scores  = np.array([metrics.r2_score(BOLD_sc_target[idx_test_target],pred,multioutput = 'raw_values') for idx_test_target,pred in zip(idxs_target_test,preds)])
                    corr    = [np.mean([np.corrcoef(a,b)[0, 1]**2 for a,b in zip(BOLD_norm_target[idx_test],pred)]) for idx_test,pred in zip(idxs_target_test,preds)]
                    results = OrderedDict()
                    try:
                        weights = [np.sum(np.abs(est.best_estimator_.coef_)) for est in regs]
                    except:
                        weights = [np.sum(np.abs(est.best_estimator_.steps[-1][-1].coef_)) for est in regs]
                    alphas = np.array([list(est.best_params_.values())[0] for est in regs])
                    results['weight_sum'] = weights
                    results['alphas'] = alphas
                    
                    mean_variance,results = fill_results(
                                                         scores,
                                                         results,
                                                         n_splits,
                                                         conscious_source,
                                                         conscious_target,
                                                         roi_name,
                                                         BOLD_sc_source,
                                                         features_source,
                                                         corr,)
                    print(f"{roi_name}, on images: {conscious_source} --> {conscious_target},{encoding_model},VE = {np.nanmean(mean_variance):.4f},corr = {np.mean(corr):.4f},||weights|| = {np.mean(weights):1.3e} with alpha = {stats.mode(alphas)[0][0]:1.0e},n posive voxels = {np.sum(scores.mean(0) > 0):.0f}/{scores.shape[1]}\n")
                    df_scores = pd.DataFrame(results)
                    df_scores['feature_type'] = 'image'
                    
                    res,features_target,features_source = cross_validation(
                                                           background_dir,
                                                           encoding_model,
                                                           custom_scorer,
                                                           BOLD_sc_source,
                                                           idxs_train_source,
                                                           idxs_test_source,
                                                           image_source,
                                                           image_target,)
                    regs = res['estimator']
                    preds   = [est.predict(features_target[idx_test]) for est,idx_test in zip(regs,idxs_target_test)]
                    scores  = np.array([metrics.r2_score(BOLD_sc_target[idx_test_target],pred,multioutput = 'raw_values') for idx_test_target,pred in zip(idxs_target_test,preds)])
                    corr    = [np.mean([np.corrcoef(a,b)[0, 1]**2 for a,b in zip(BOLD_norm_target[idx_test],pred)]) for idx_test,pred in zip(idxs_target_test,preds)]
                    results = OrderedDict()
                    try:
                        weights = [np.sum(np.abs(est.best_estimator_.coef_)) for est in regs]
                        n_params = [features_source.shape[1] * BOLD_sc_source.shape[1] for est in regs]
                    except:
                        weights = [np.sum(np.abs(est.best_estimator_.steps[-1][-1].coef_)) for est in regs]
                        n_params = [est.best_estimator_.steps[0][-1].n_components_ * BOLD_sc_source.shape[1] for est in regs]
                    alphas = np.array([list(est.best_params_.values())[0] for est in regs])
                    results['weight_sum'] = weights
                    results['alphas'] = alphas
                    mean_variance,results = fill_results(
                                                         scores,
                                                         results,
                                                         n_splits,
                                                         conscious_source,
                                                         conscious_target,
                                                         roi_name,
                                                         BOLD_sc_source,
                                                         features_source,
                                                         corr,)
                    print(f"{roi_name} on background: {conscious_source} --> {conscious_target},{encoding_model},VE = {np.nanmean(mean_variance):.4f},corr = {np.mean(corr):.4f},||weights|| = {np.mean(weights):1.3e} with alpha = {stats.mode(alphas)[0][0]:1.0e},n posive voxels = {np.sum(scores.mean(0) > 0):.0f}/{scores.shape[1]}\n")
                    df_background = pd.DataFrame(results)
                    df_background['feature_type'] = 'background'
                    
                    results_to_save = pd.concat([df_scores,df_background])
                    results_to_save.to_csv(os.path.join(output_dir,saving_name),index = False)
                else:
                    print(f'you have done {saving_name}')




















