#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:45:06 2019

@author: nmei
"""
import mne
import os
import re
import numpy as np
import pandas as pd
from glob                    import glob
from tqdm                    import tqdm
from datetime                import datetime
from shutil                  import copyfile
copyfile('../utils.py','utils.py')
from utils                   import get_frames,split_probe_path
from sklearn.model_selection import cross_validate
from sklearn.utils           import shuffle
from sklearn.linear_model    import Ridge
from sklearn.metrics         import r2_score,make_scorer
from sklearn.preprocessing   import StandardScaler
from functools               import partial
from matplotlib              import pyplot as plt



subject             = 'clara_5_22_2019' 
# there was a bug in the csv file, so the early behavioral is treated differently
date                = '/'.join(re.findall('\d+',subject))
date                = datetime.strptime(date,'%m/%d/%Y')
breakPoint          = datetime(2019,3,10)
if date > breakPoint:
    new             = True
else:
    new             = False

computer_vision_dir = '../../data/computer vision features'
working_dir         = f'../../data/clean EEG/{subject}'
working_data        = glob(os.path.join(working_dir,'*-epo.fif'))
behavioral_dir      = f'../../data/clean behavioral/{subject}'
df_behavioral       = pd.concat([pd.read_csv(f) for f in glob(os.path.join(behavioral_dir,'*.csv'))])
frames,_            = get_frames(directory = f'../../data/behavioral/{subject}',new = new)
# create the directories for figure and decoding results (numpy arrays)
figure_dir          = f'../../figures/EEG/encode/{subject}'
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
array_dir           = f'../../results/EEG/encode/{subject}'
if not os.path.exists(array_dir):
    os.makedirs(array_dir)


func   = partial(r2_score,multioutput = 'raw_values')
func.__name__ = 'R2'
scorer = make_scorer(func,needs_proba = False)
conscious_dict = {'1.0':'unconscious',
                  '2.0':'glimpse',
                  '3.0':'conscious',}
n_jobs = 8

for epoch_file in working_data:
    epochs  = mne.read_epochs(epoch_file)
    print('resampling')
    epochs  = epochs.resample(100)
    # get the corresponding image presented in that given trial
    df_behavioral['call_name'] = df_behavioral['probe_path'].apply(split_probe_path,idx = -1)
    df_behavioral['call_name'] = df_behavioral['call_name'].apply(lambda x:x.split('.')[0])
    df_behavioral['call_name'] = df_behavioral['call_name'].apply(lambda x:x + '.npy')
    df_behavioral = df_behavioral.reset_index()
    categorical_dict = {label:category for category,label in zip(df_behavioral['category'],df_behavioral['label'])}
    
    
    epoch_data = epochs.get_data()
    visibility = epochs.events[:,-1] % 10 - 6
    for conscious_state in np.unique(visibility):
        idx_picked, = np.where(visibility == conscious_state)
        df_picked = df_behavioral.iloc[idx_picked,:].reset_index()
        epoch_data_picked = epoch_data[idx_picked]
        for model_name in os.listdir(computer_vision_dir):
            array_saving_name = os.path.join(array_dir,
                                             f'{model_name} {conscious_dict[str(conscious_state)]} encoding CV.npy')
            all_arrays = glob(os.path.join(array_dir,
                                           "*.npy"))
            if array_saving_name in all_arrays:
                results = np.load(array_saving_name)
            else:
                all_features = glob(os.path.join(computer_vision_dir,model_name,'*.npy'))
            
                # prepare feature matrix
                X = []
                for array_name in df_picked['call_name'].values:
                    temp = [np.load(item) for item in all_features if (array_name == item.split('/')[-1])][0]
                    X.append(temp[np.newaxis,:])
                X = np.concatenate(X)
            
                # temporal encoding
                results = []
                for y in tqdm(np.rollaxis(epoch_data_picked,-1)):
                    scaler = StandardScaler()
                    y_norm = scaler.fit_transform(y)
                    features,channels = shuffle(X,y_norm)
                    scores_temp = []
                    # leave one object out -- leave one group out
                    idxs_train,idxs_test = [],[]
                    for (conscious_state,label),df_sub in df_picked.groupby(['visible.keys_raw','label']):
                        idx_test = np.array(df_sub.index)
                        idx_train = [ii for ii in range(df_picked.shape[0]) if (ii not in idx_test)]
                        idxs_train.append(idx_train)
                        idxs_test.append(idx_test)
                        
                        
                    regressor = Ridge(alpha = int(1e3),
                                      normalize = True,
                                      random_state = 12345,)
                    
                    res = cross_validate(regressor,features,channels,
                                         cv = zip(idxs_train,idxs_test),
                                         n_jobs = n_jobs,
                                         return_estimator = True)
                    
                    estimators = res['estimator']
                    scores_temp = np.array([scorer(est,features[idx_test],channels[idx_test]) for est,idx_test in zip(estimators,idxs_test)])
                    
                    
                    results.append(scores_temp)
                results = np.array(results)
                np.save(array_saving_name,results)
            n_splits = results.shape[1]
            
            scores = results.copy()
            scores[scores <= 0] = np.nan
            c = np.nanmean(np.nanmean(scores,-1),-1)


































