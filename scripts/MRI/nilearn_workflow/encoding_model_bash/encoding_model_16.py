#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:54:57 2019

@author: nmei

encoding model that is interchangable btw 2 source folders: BOLD-average, BOLD-average-prestim

"""
import os
import numpy as np
import pandas as pd
print(os.getcwd())
from glob                      import glob
from tqdm                      import tqdm
from shutil                    import copyfile
copyfile('../../../utils.py','utils.py')
from utils                     import (LOO_partition
                                       )
from sklearn.model_selection   import LeaveOneGroupOut
from sklearn                   import metrics
from sklearn                   import linear_model
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing     import MinMaxScaler
from sklearn.model_selection   import GridSearchCV

from collections               import OrderedDict

sub                 = 'sub-01'
target_folder       = 'encoding_LOO'
data_folder         = 'BOLD_average' # BOLD_average_prepeak
stacked_data_dir    = '../../../../data/{}/{}/'.format(data_folder,sub)
feature_dir         = '../../../../data/computer_vision_features'
output_dir          = '../../../../results/MRI/nilearn/{}/{}'.format(sub,target_folder)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
BOLD_data           = glob(os.path.join(stacked_data_dir,'*BOLD.npy'))
event_data          = glob(os.path.join(stacked_data_dir,'*.csv'))
computer_models     = os.listdir(feature_dir)

label_map           = {'Nonliving_Things':[0,1],
                       'Living_Things':   [1,0]}
n_splits            = 1

def score_func(y, y_pred, **kwargs):
    temp        = metrics.r2_score(y,y_pred,multioutput = 'raw_values')
    temp        = np.array(temp)
    if np.sum(temp < 0) == len(y):
        score   = 0
    else:
        score   = temp[temp > 0].mean()
    return score
custom_scorer      = metrics.make_scorer(score_func,greater_is_better = True)

idx = 15
np.random.seed(12345)
BOLD_name,df_name   = BOLD_data[idx],event_data[idx]
BOLD                = np.load(BOLD_name)
df_event            = pd.read_csv(df_name)
roi_name            = df_name.split('/')[-1].split('_events')[0]
for conscious_state in ['unconscious','glimpse','conscious']:
    idx_unconscious = df_event['visibility'] == conscious_state
    data            = BOLD[idx_unconscious]
    VT              = VarianceThreshold()
    scaler          = MinMaxScaler()
    BOLD_norm       = VT.fit_transform(data)
    BOLD_sc         = scaler.fit_transform(BOLD_norm)
    df_data         = df_event[idx_unconscious].reset_index(drop=True)
    df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
    targets         = np.array([label_map[item] for item in df_data['targets'].values])[:,-1]
    groups          = df_data['labels'].values
    
    if n_splits <= 300:
        cv          = LeaveOneGroupOut()
        idxs_train,idxs_test = [],[]
        for idx_train,idx_test in cv.split(BOLD_sc,targets,groups = groups):
            idxs_train.append(idx_train)
            idxs_test.append(idx_test)
    else:
        idxs_train,idxs_test = LOO_partition(df_data)
    n_splits = len(idxs_train)
    
    image_names     = df_data['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
    
    for encoding_model in ['DenseNet169','MobileNetV2','VGG19','ResNet50']:
        saving_name = f"encoding model {conscious_state} {roi_name} {encoding_model}.csv"
        processed   = glob(os.path.join(output_dir,"*.csv"))
        
        if not os.path.join(output_dir,saving_name) in processed:
            
            features        = np.array([np.load(os.path.join(feature_dir,
                                                             encoding_model,
                                                             item)) for item in image_names])
            regs            = []
            # for some reason, sklearn cross-validate wrapper does not identify the best
            # alpha correctly even though the scoring function is correct.
            for idx_train,idx_test in tqdm(zip(idxs_train,idxs_test)):
                reg         = linear_model.Ridge(normalize      = True,
                                                 random_state   = 12345)
                
                reg         = GridSearchCV(reg,
                                           dict(alpha = np.logspace(-6,6,13)),
                                           scoring  = custom_scorer,
                                           n_jobs   = 32,
                                           cv       = LeaveOneGroupOut(),
                                           iid      = False,)
                reg.fit(features[idx_train],BOLD_norm[idx_train],groups = groups[idx_train])
                
                regs.append(reg)
            
            preds   = [est.predict(features[idx_test]) for est,idx_test in zip(regs,idxs_test)]
            scores  = np.array([metrics.r2_score(BOLD_norm[idx_test],pred,multioutput = 'raw_values') for idx_test,pred in zip(idxs_test,preds)])
            corr    = [np.mean([np.corrcoef(a,b).flatten()[1] for a,b in zip(BOLD_norm[idx_test],pred)]) for idx_test,pred in zip(idxs_test,preds)]
            
            mean_variance = np.array([score_func(BOLD_norm[idx_test],pred,) for idx_test,pred in zip(idxs_test,preds)])
            
            print(f"{conscious_state},VE = {np.nanmean(mean_variance):.4f},||weights|| = {np.mean([np.sum(np.abs(est.best_estimator_.coef_)) for est in regs]):.3}\ncorr = {np.mean(corr):.4f}")
            
            positive_voxels = np.array([np.sum(temp >= 0) for temp in scores])
            
            try:
                best_variance   = np.array([np.nanmax(temp) for temp in mean_variance])
            except:
                best_variance   = mean_variance.copy()
            
            results = OrderedDict()
            
            scores_to_save = mean_variance.copy()
            scores_to_save = np.nan_to_num(scores_to_save,)
            
            results['alphas'] = [list(est.best_params_.values())[0] for est in regs]
            results['mean_variance'] = scores_to_save
            results['best_variance'] = np.nan_to_num(best_variance,)
            results['fold'] = np.arange(n_splits) + 1
            results['conscious_state'] = [conscious_state] * n_splits
            results['roi_name'] = [roi_name] * n_splits
            results['positive voxels'  ]= positive_voxels
            results['weight_sum'] = [np.sum(np.abs(est.best_estimator_.coef_)) for est in regs]
            results['n_parameters'] = [BOLD_norm.shape[1] * features.shape[1]] * n_splits
            results['corr'] = corr
            
            results_to_save = pd.DataFrame(results)
            results_to_save.to_csv(os.path.join(output_dir,saving_name))
        else:
            print(saving_name)




















