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
from tqdm                    import tqdm
from datetime                import datetime


from mne.decoding            import (
                                        Scaler,
                                        Vectorizer,
                                        SlidingEstimator,
                                        cross_val_multiscore,
                                        GeneralizingEstimator
                                        )
from sklearn.preprocessing   import StandardScaler
#from sklearn.calibration     import CalibratedClassifierCV
#from sklearn.svm             import LinearSVC
from sklearn.linear_model    import (
#                                        LogisticRegressionCV,
                                        LogisticRegression,
#                                        SGDClassifier
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools               import partial
from shutil                  import copyfile
copyfile(os.path.abspath('../utils.py'),'utils.py')
from utils                   import get_frames,bootstrap_behavioral_estimation

n_jobs = 8
func   = partial(roc_auc_score,average = 'micro')
func.__name__ = 'micro_AUC'
scorer = make_scorer(func,needs_proba = True)
speed  = True

subject             = 'clara_5_22_2019' 
# there was a bug in the csv file, so the early behavioral is treated differently
date                = '/'.join(re.findall('\d+',subject))
date                = datetime.strptime(date,'%m/%d/%Y')
breakPoint          = datetime(2019,3,10)
if date > breakPoint:
    new             = True
else:
    new             = False
working_dir         = os.path.abspath(f'../../data/clean EEG/{subject}')
working_data        = glob(os.path.join(working_dir,'*-epo.fif'))
frames,_            = get_frames(directory = os.path.abspath(f'../../data/behavioral/{subject}'),new = new)
# create the directories for figure and decoding results (numpy arrays)
figure_dir          = os.path.abspath(f'../../figures/EEG/decode_resample/{subject}')
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
array_dir           = os.path.abspath(f'../../results/EEG/decode_resample/{subject}')
if not os.path.exists(array_dir):
    os.makedirs(array_dir)
# define the number of cross validation we want to do.
n_splits            = 10

logistic = LogisticRegression(
                              solver        = 'lbfgs',
                              max_iter      = int(1e3),
                              random_state  = 12345
                              )

for epoch_file in working_data:
    epochs  = mne.read_epochs(epoch_file)
    df_all = pd.read_csv(f'../../data/clean behavioral/{subject}/concat.csv')
    unconscious = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('unconscious' in name)])
    
    df_temp = df_all[df_all['visible.keys_raw'] == 1].reset_index()
    
    saving_name_temporal_decoding = os.path.join(array_dir,
                                                 'temporal_decoding.npy')
    saving_name_temporal_generalization = os.path.join(array_dir,
                                                       'temporal_generalization.npy')
    if saving_name_temporal_generalization not in glob(os.path.join(array_dir,
                                                                    '*.npy')):
        total_permutations = 0
        temporal_decoding = []
        temporal_genralization = []
        for trying_many_times in range(int(1e3)):
            # per time
            sample_size = 0.5
            sample_index = np.arange(df_temp.shape[0])
            idx_sampled = np.random.choice(shuffle(sample_index),
                                           size = int(df_temp.shape[0]*sample_size),
                                           replace = True)
            df_sample = df_temp.loc[idx_sampled,:]
            pvals,scores,chance = bootstrap_behavioral_estimation(df_sample,200)
            for _ in range(int(1e6)):
                del idx_sampled
                sample_index = shuffle(sample_index)
                idx_sampled = np.random.choice(sample_index,
                                               size = int(df_temp.shape[0]*sample_size),
                                               replace = True)
                df_sample = df_temp.loc[idx_sampled,:]
                pvals,scores,chance = bootstrap_behavioral_estimation(df_sample,200)
                print(pvals.mean(),scores.mean(),chance.mean())
                if pvals.mean() > 0.05:
                    break
            epochs_sampled = unconscious.copy()[idx_sampled]
            
            # resample at 100 Hz to fasten the decoding process
            print('resampling...')
            epoch_temp = epochs_sampled.copy().resample(100)
            # temporal decoding
            cv          = StratifiedShuffleSplit(
                                        n_splits            = 30, 
                                        test_size           = 0.2, 
                                        random_state        = 12345)
            clf         = make_pipeline(
                                        StandardScaler(),
                                        clone(logistic))
            
            time_decod  = SlidingEstimator(
                                        clf, 
                                        n_jobs              = 1, 
                                        scoring             = scorer, 
                                        verbose             = True
                                        )
            X,y = epoch_temp.get_data(),epoch_temp.events[:,-1]
            y = y //100 - 2
            X,y = shuffle(X,y)
            times       = epoch_temp.times
            scores      = cross_val_multiscore(
                                            time_decod, 
                                            X,
                                            y,
                                            cv                  = cv, 
                                            n_jobs              = n_jobs,
                                            )
            scores_mean = scores.mean(0)
            scores_se   = scores.std(0) / np.sqrt(n_splits)
            temporal_decoding.append(scores_mean)
            # temporal generalization
            cv          = StratifiedShuffleSplit(
                                        n_splits            = 30, 
                                        test_size           = 0.2, 
                                        random_state        = 12345)
            clf         = make_pipeline(
                                        StandardScaler(),
                                        clone(logistic))
            time_gen    = GeneralizingEstimator(
                                        clf, 
                                        n_jobs              = n_jobs, 
                                        scoring             = scorer,
                                        verbose             = True)
            X,y = epoch_temp.get_data(),epoch_temp.events[:,-1]
            y = y //100 - 2
            X,y = shuffle(X,y)
            scores_gen  = cross_val_multiscore(
                                            time_gen, 
                                            X,
                                            y,
                                            cv                  = cv, 
                                            n_jobs              = n_jobs
                                            )
            scores_gen_ = []
            for s_gen,s in zip(scores_gen,scores):
                np.fill_diagonal(s_gen,s)
                scores_gen_.append(s_gen)
            scores_gen_ = np.array(scores_gen_)
            temporal_genralization.append(scores_gen_.mean(0))
            total_permutations += 1
            if total_permutations >= n_splits:
                break
        
        temporal_decoding = np.array(temporal_decoding)
        temporal_genralization = np.array(temporal_genralization)
        np.save(saving_name_temporal_decoding,temporal_decoding)
        np.save(saving_name_temporal_generalization,temporal_genralization)
    else:
        temporal_decoding = np.load(saving_name_temporal_decoding)
        temporal_genralization = np.load(saving_name_temporal_generalization)

























