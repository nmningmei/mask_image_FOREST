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


from mne.decoding            import (
                                     Vectorizer,
                                     SlidingEstimator,
                                     cross_val_multiscore,
                                     GeneralizingEstimator
                                     )
from sklearn.preprocessing   import Normalizer
from sklearn.dummy           import DummyClassifier
#from sklearn.calibration     import CalibratedClassifierCV
#from sklearn.svm             import LinearSVC
from sklearn.linear_model    import (
#                                        LogisticRegressionCV,
                                        LogisticRegression,
#                                        SGDClassifier
                                        )
from sklearn.pipeline        import make_pipeline
from sklearn.model_selection import (
                                        cross_val_score,
#                                        cross_validate
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
                                     LOO_partition,
                                     plot_temporal_decoding,
                                     plot_temporal_generalization,
                                     plot_t_stats,
                                     plot_p_values)

# use more than 1 CPU to parallize the training
n_jobs = -1
# customized scoring function
func                = partial(roc_auc_score,average = 'micro')
func.__name__       = 'micro_AUC'
scorer              = make_scorer(func,needs_proba = True)
speed               = True

subject             = 'aingere_5_16_2019' 
# there was a bug in the csv file, so the early behavioral is treated differently
date                = '/'.join(re.findall('\d+',subject))
date                = datetime.strptime(date,'%m/%d/%Y')
breakPoint          = datetime(2019,3,10)
if date > breakPoint:
    new             = True
else:
    new             = False
# define lots of path for data, outputs, etc
folder_name         = "clean_EEG_mask_baseline_with_session_info"
target_name         = 'decode_massive_CV'
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
n_splits            = -1

logistic = LogisticRegression(
                              solver        = 'lbfgs',
                              max_iter      = int(1e3),
                              random_state  = 12345
                              )
dummy = DummyClassifier(strategy = 'stratified',
                        random_state = 12345)

for epoch_file in working_data:
    epochs_loaded  = mne.read_epochs(epoch_file)
    behavioral_dir = os.path.join('/'.join(epoch_file.split('/')[:-3]),
                                  'clean behavioral',
                                  subject,
                                  'concat.csv')
    df = pd.read_csv(behavioral_dir)
    df['conscious_state'] = df['visible.keys_raw'].astype('int').map({1:'unconscious',
                                                                      2:'glimpse',
                                                                      3:'conscious'})
    events = epochs_loaded.events
    event_id = epochs_loaded.event_id
    for ii,conscious_state in enumerate(['unconscious','glimpse','conscious']):
        idx_ = df['conscious_state'] == conscious_state
        epochs = epochs_loaded.copy()[idx_]
        
        # resample at 100 Hz to fasten the decoding process
        print('resampling...')
        epoch_temp = epochs.copy().resample(100)
        df_data = df[df['conscious_state'] == conscious_state].reset_index()
        if speed:
            X,y = epoch_temp.get_data(),epoch_temp.events[:,-1]
            y = y //100 - 2
            times       = epoch_temp.times
        else:
            X,y = epochs.get_data(),epochs.events[:,-1]
            y = y //100 - 2
            times       = epochs.times
        np.random.seed(12345)
        idx_shuffle = shuffle(np.arange(X.shape[0]))
        X,y = X[idx_shuffle], y[idx_shuffle]
        df_data = df_data.iloc[idx_shuffle,:].reset_index()
        df_data['targets'] = df_data['category']
        print('preparing cv folds...')
        idxs_train,idxs_test = LOO_partition(df_data,target_column = 'label')
        cv = zip(idxs_train,idxs_test)
        print(f'{len(idxs_train)} folds')
#        n_splits = len(idxs_train)
#        for idx_test in idxs_test:
#            from collections import Counter
#            if len(Counter(y[idx_test])) < 2:
#                print(y[idx_test])
        ################ decode the whole segment ##################
        print('cross val scoring')
        saving_name     = os.path.join(array_dir,f'whole_segment_{conscious_state}.npy')
        if saving_name in glob(os.path.join(array_dir,'*.npy')):
            plscores    = np.load(saving_name)
        else:
            
            pipeline    = make_pipeline(
                                     Vectorizer(),
                                     Normalizer(),
                                     clone(logistic),
                                     )
            
            scores      = cross_val_score(
                                     pipeline,
                                     X,
                                     y,
                                     scoring                = scorer,
                                     cv                     = zip(idxs_train,idxs_test),
                                     n_jobs                 = n_jobs,
                                     )
            plscores = scores.copy()
            np.save(saving_name,plscores)
        print(plscores.shape)
        print(f'decode {conscious_state} = {plscores.mean():.4f}+/-{plscores.std():.4f}')
        
        ####################### temporal decoding ##########################
        print('temporal decoding')
        saving_name     = os.path.join(array_dir,f'temporal_decoding_{conscious_state}.npy')
        if saving_name in glob(os.path.join(array_dir,'*.npy')):
            scores = np.load(saving_name)
            
        else:
            
            clf         = make_pipeline(
                                        Normalizer(),
                                        clone(logistic))
            time_decod  = SlidingEstimator(
                                        clf, 
                                        n_jobs              = 1, 
                                        scoring             = scorer, 
                                        verbose             = True
                                        )
            scores      = cross_val_multiscore(
                                        time_decod, 
                                        X,
                                        y,
                                        cv                  = zip(idxs_train,idxs_test), 
                                        n_jobs              = n_jobs,
                                        )
            np.save(saving_name,scores)
        
        fig,ax = plot_temporal_decoding(times,scores,frames,ii,conscious_state,plscores,n_splits)
        fig.savefig(os.path.join(figure_dir,f'temporal decoding ({conscious_state}).png'),
                    bbox_inches = 'tight',
                    dpi         = 400)
        plt.close(fig)
        plt.clf()
        plt.close('all')
        
        ####################### temporal generalization ########################
        print('temporal generalization')
        saving_name     = os.path.join(array_dir,f'temporal_generalization_{conscious_state}.npy')
        if saving_name in glob(os.path.join(array_dir,'*.npy')):
            scores_gen = np.load(saving_name)
#            scores_chance = np.load(saving_name.replace('.npy','_chance.npy'))
        else:
            groups = epochs.events[:,1]
            clf         = make_pipeline(
                                        Normalizer(),
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
                                        cv                  = zip(idxs_train,idxs_test), 
                                        n_jobs              = n_jobs
                                        )
            np.save(saving_name,scores_gen)
            
            clf         = make_pipeline(
                                        Normalizer(),
                                        clone(dummy))
            time_gen    = GeneralizingEstimator(
                                        clf, 
                                        n_jobs              = 1, 
                                        scoring             = scorer,
                                        verbose             = True)
            
            scores_chance = cross_val_multiscore(
                                        time_gen,
                                        X,
                                        y,
                                        cv = zip(idxs_train,idxs_test),
                                        n_jobs = n_jobs,
                                        )
            np.save(saving_name.replace('.npy','_chance.npy'),scores_chance)
            
        scores_gen_ = scores_gen.copy()#[]
#        for s_gen,s in zip(scores_gen,scores):
#            np.fill_diagonal(s_gen,s)
#            scores_gen_.append(s_gen)
#        scores_gen_ = np.array(scores_gen_)
        
        fig,ax = plot_temporal_generalization(scores_gen_,epochs,ii,conscious_state,frames)
        fig.savefig(os.path.join(figure_dir,f'temporal generalization ({conscious_state}).png'),
                    bbox_inches = 'tight',
                    dpi         = 400)
        plt.close(fig)
        plt.clf()
        plt.close('all')
        
        ############################### permutation cluster test ##################################
        print('permutation cluster test')
        saving_name     = os.path.join(array_dir,f'permutation_cluster_test_T_obs_{conscious_state}.npy')
        if saving_name in glob(os.path.join(array_dir,'*.npy')):
            T_obs = np.load(saving_name)
            clusters = np.load(saving_name.replace('T_obs','clusters'))
            cluster_p_values = np.load(saving_name.replace('T_obs','cluster_p_values'))
        else:
            alpha           = 1e-5
            sigma           = 1e-3
            stat_fun_hat    = partial(mne.stats.ttest_1samp_no_p, sigma=sigma)
            ## perform nonparametric t test to find clusters in the conscious state
            # compute the threshold for t statistics
            t_threshold = stats.distributions.t.ppf(1 - alpha, scores.shape[0] - 1) 
            # apply the MNE python function of which I know less than nothing
            threshold_tfce = dict(start=0, step=0.1)
            T_obs, clusters, cluster_p_values, H0   = clu \
                        = mne.stats.permutation_cluster_1samp_test(
                                scores_gen_ - scores_chance,
                                threshold           = threshold_tfce,
                                stat_fun            = stat_fun_hat,
                                tail                = 1, # find clusters that are greater than the chance level
    #                            check_disjoint      = True, # not useful
                                seed                = 12345, # random seed
                                step_down_p         = 0.05,
                                buffer_size         = None, # stat_fun does not treat variables independently
                                n_jobs              = n_jobs,
                                )
            np.save(saving_name,T_obs)
            np.save(saving_name.replace("T_obs","clusters"),clusters)
            np.save(saving_name.replace('T_obs','cluster_p_values'),cluster_p_values)
        
        fig,ax = plot_t_stats(T_obs,clusters,cluster_p_values,times,ii,conscious_state,frames,)
        
        fig.savefig(os.path.join(figure_dir,f'stats ({conscious_state}).png'),
                    bbox_inches = 'tight',
                    dpi         = 400)
        plt.close(fig)
        plt.clf()
        plt.close('all')
        
        fig,ax = plot_p_values(times,clusters,cluster_p_values,ii,conscious_state,frames)
        
        fig.savefig(os.path.join(figure_dir,f'stats (p values, {conscious_state}).png'),
                    bbox_inches = 'tight',
                    dpi         = 400)
        plt.close(fig)
        plt.clf()
        plt.close('all')
