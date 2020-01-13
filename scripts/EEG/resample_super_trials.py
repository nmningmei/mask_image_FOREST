#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:04:58 2019

@author: nmei
"""

import mne
import os
print(os.getcwd())
import re
import numpy as np
import pandas as pd
from glob                    import glob
from datetime                import datetime
from collections             import Counter

from mne.decoding            import (
                                     Vectorizer,
                                     SlidingEstimator,
                                     cross_val_multiscore,
                                     GeneralizingEstimator
                                     )
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import (
                                     LogisticRegression,
                                     )
from sklearn.pipeline        import make_pipeline
from sklearn.model_selection import (
                                     StratifiedShuffleSplit,
                                     cross_val_score
                                     )
from sklearn.utils           import shuffle
from sklearn.base            import clone
from sklearn.metrics         import make_scorer,roc_auc_score
from matplotlib              import pyplot as plt
from scipy                   import stats
from functools               import partial
from shutil                  import copyfile
copyfile(os.path.abspath('../utils.py'),'utils.py')
from utils                   import (get_frames,
                                     plot_temporal_decoding,
                                     plot_temporal_generalization,
                                     plot_t_stats,
                                     plot_p_values)

n_jobs = -1
func   = partial(roc_auc_score,average = 'micro')
func.__name__ = 'micro_AUC'
scorer = make_scorer(func,needs_proba = True)
speed  = True

subject             = 'mattin_7_12_2019'
# there was a bug in the csv file, so the early behavioral is treated differently
date                = '/'.join(re.findall('\d+',subject))
date                = datetime.strptime(date,'%m/%d/%Y')
breakPoint          = datetime(2019,3,10)
if date > breakPoint:
    new             = True
else:
    new             = False
folder_name = "clean_EEG_premask_baseline"
target_name = 'resample_trials'
working_dir         = os.path.abspath(f'../../data/{folder_name}/{subject}')
working_data        = glob(os.path.join(working_dir,'*-epo.fif'))
frames,_            = get_frames(directory = os.path.abspath(f'../../data/behavioral/{subject}'),new = new)
# create the directories for figure and decoding results (numpy arrays)
figure_dir          = os.path.abspath(f'../../figures/EEG/{target_name}/{subject}')
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
array_dir           = os.path.abspath(f'../../results/EEG/{target_name}/{subject}')
if not os.path.exists(array_dir):
    os.makedirs(array_dir)
# define the number of cross validation we want to do.
n_splits            = 300
logistic = LogisticRegression(
                              solver        = 'lbfgs',
                              max_iter      = int(1e3),
                              random_state  = 12345
                              )

label_map = {'Living_Things':[0,1],
             'Nonliving_Things':[1,0]}

for epoch_file in working_data:
    epochs  = mne.read_epochs(epoch_file,preload=True)
    
    conscious   = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if (' conscious' in name)])
    see_maybe   = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('glimpse' in name)])
    unconscious = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('unconscious' in name)])
    del epochs
    behavioral_dir = os.path.join('/'.join(epoch_file.split('/')[:-3]),
                                  'clean behavioral',
                                  subject,
                                  'concat.csv')
    df = pd.read_csv(behavioral_dir)
    df['conscious_state'] = df['visible.keys_raw'].astype('int').map({1:'unconscious',
                                                                      2:'glimpse',
                                                                      3:'conscious'})
    for ii,(epochs,conscious_state) in enumerate(zip([unconscious.copy(),
                                                      see_maybe.copy(),
                                                      conscious.copy()],
                                                     ['unconscious',
                                                      'glimpse',
                                                      'conscious'])):
        epochs
        df_data = df[df['conscious_state'] == conscious_state].reset_index()
        
        counting = list(dict(Counter(df_data['subcategory'])).values())
        for_average = int(2/3 * (np.min(counting)))
        print(f"let's use {for_average} trials for averaging")
        temp = []
        y_ = []
        for subcategory,df_sub in df_data.groupby(['subcategory']):
            if pd.unique(df_sub['category'])[0] == 'Nonliving_Things':
                factor = 36
            else:
                factor = 54
            indices = list(df_sub.index)
            samples = np.random.choice(indices,
                                       size = (factor,for_average),
                                       replace = True)
            temp.append(samples)
            y_.append([pd.unique(df_sub['category'])[0]]*factor)
        temp = np.concatenate(temp)
        y_ = np.concatenate(y_)
        
        EEG_data = epochs.get_data()
        resampled = EEG_data[temp].mean(1)
        print('donwsampling...')
        resampled_epoch = mne.EpochsArray(resampled,epochs.info,tmin = epochs.tmin,baseline = epochs.baseline)
        resampled_epoch = resampled_epoch.resample(100)
        X = resampled_epoch.get_data().copy()
        y = np.array([label_map[item] for item in y_])[:,-1]
        
        cv = StratifiedShuffleSplit(n_splits = 20,
                                    test_size = 0.2,
                                    random_state = 12345,)
        clf         = make_pipeline(
                                    StandardScaler(),
                                    clone(logistic))
        time_gen    = GeneralizingEstimator(
                                    clf, 
                                    n_jobs              = 1, 
                                    scoring             = scorer,
                                    verbose             = True)
        scores_gen  = cross_val_multiscore(
                                        time_gen, 
                                        X,
                                        y,
                                        cv                  = cv, 
                                        n_jobs              = n_jobs
                                        )
        fig,ax = plot_temporal_generalization(scores_gen,epochs,ii,conscious_state,frames)
        fig,ax = plot_temporal_decoding(resampled_epoch.times,
                                        np.diagonal(scores_gen).T,
                                        frames,ii,conscious_state,0.5,n_splits)
        
        asdf
























