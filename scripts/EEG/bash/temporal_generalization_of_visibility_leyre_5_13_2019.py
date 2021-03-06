#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:38:52 2019

@author: nmei
"""

import mne
import os
print(os.getcwd())
import re
import numpy as np
from glob                    import glob
from datetime                import datetime


from mne.decoding            import (
                                     Vectorizer,
                                     SlidingEstimator,
                                     cross_val_multiscore,
                                     GeneralizingEstimator
                                     )
from sklearn.preprocessing   import StandardScaler,Binarizer
#from sklearn.calibration     import CalibratedClassifierCV
#from sklearn.svm             import LinearSVC
from sklearn.linear_model    import (
#                                        LogisticRegressionCV,
                                        LogisticRegression,
#                                        SGDClassifier
                                        )
from sklearn.multiclass      import (OneVsRestClassifier,
                                     OneVsOneClassifier)
from sklearn.pipeline        import make_pipeline
from sklearn.model_selection import (
                                        StratifiedShuffleSplit,
                                        cross_validate
                                        )
from sklearn.utils           import shuffle
from sklearn.base            import clone
from sklearn.metrics         import make_scorer,roc_auc_score
from matplotlib              import pyplot as plt
from scipy                   import stats
from functools               import partial
from shutil                  import copyfile
from itertools               import combinations
copyfile(os.path.abspath('../../utils.py'),'utils.py')
from utils                   import (get_frames,
                                     plot_temporal_decoding,
                                     plot_temporal_generalization,
                                     plot_t_stats,
                                     plot_p_values)

# use more than 1 CPU to parallize the training
n_jobs = -1
# customized scoring function
func                = partial(roc_auc_score,average = 'micro')
func.__name__       = 'micro_AUC'
func.__module__     = 'ranking'
scorer              = make_scorer(func,needs_proba = True)
speed               = True

subject = 'leyre_5_13_2019'
# there was a bug in the csv file, so the early behavioral is treated differently
date                = '/'.join(re.findall('\d+',subject))
date                = datetime.strptime(date,'%m/%d/%Y')
breakPoint          = datetime(2019,3,10)
if date > breakPoint:
    new             = True
else:
    new             = False
# define lots of path for data, outputs, etc
folder_name         = "clean_EEG_premask_baseline_ICA"
target_name         = 'decode_visibility_premask_baseline_ICA'
working_dir         = os.path.abspath(f'../../../data/{folder_name}/{subject}')
working_data        = glob(os.path.join(working_dir,'*-epo.fif'))
frames,_            = get_frames(directory = os.path.abspath(f'../../../data/behavioral/{subject}'),new = new)
# create the directories for figure and decoding results (numpy arrays)
figure_dir          = os.path.abspath(f'../../../figures/EEG/{target_name}/{subject}')
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
array_dir           = os.path.abspath(f'../../../results/EEG/{target_name}/{subject}')
if not os.path.exists(array_dir):
    os.makedirs(array_dir)
# define the number of cross validation we want to do.
n_splits            = 300

logistic = LogisticRegression(
                              solver        = 'lbfgs',
                              max_iter      = int(1e3),
                              random_state  = 12345
                              )
conscious_state_dict = {0:'unconscious',
                        1:'glimpse',
                        2:'conscious',}


for epoch_file in working_data:
    epochs  = mne.read_epochs(epoch_file)
    
    # resample at 100 Hz to fasten the decoding process
    print('resampling...')
    epoch_temp = epochs.copy().resample(100)
    
    ################ decode the whole segment ##################
    print('cross val scoring')
    
    cv          = StratifiedShuffleSplit(
                                     n_splits       = n_splits, 
                                     test_size      = 0.2, 
                                     random_state   = 12345)
    pipeline    = make_pipeline(
#                                     Scaler(epochs.info,),
                             Vectorizer(),
                             StandardScaler(),
                             OneVsRestClassifier(clone(logistic),),
                             )
    
    X,y = epoch_temp.get_data(),epoch_temp.events[:,-1]
    X = mne.decoding.Scaler(epochs.info).fit_transform(X)
    y = y % 10 - 6
    
    for combine_1,combine_2 in combinations([0,1,2],2):
        saving_name     = os.path.join(array_dir,f'whole_segment_{conscious_state_dict[combine_1]}_{conscious_state_dict[combine_2]}.npy')
        print(f'{conscious_state_dict[combine_1]} vs {conscious_state_dict[combine_2]}')
        
        if saving_name in glob(os.path.join(array_dir,'*.npy')):
            plscores    = np.load(saving_name)
        else:
            idx_1, = np.where(y == combine_1)
            idx_2, = np.where(y == combine_2)
            idx_ = np.concatenate([idx_1,idx_2])
            
            X_picked,y_picked = X[idx_],y[idx_]
            np.random.seed(12345)
            X_picked,y_picked = shuffle(X_picked,y_picked)
            y_picked_binarized = Binarizer(threshold=np.mean(np.unique(y_picked))).fit_transform(y_picked.reshape(-1,1)).ravel()
            res      = cross_validate(
                                     pipeline,
                                     X_picked,
                                     y_picked_binarized,
                                     cv                     = cv,
                                     return_estimator       = True,
                                     n_jobs                 = n_jobs,
                                     )
            regs = res['estimator']
            idxs_test = [idx_test for _,idx_test in cv.split(X_picked,y_picked)]
            preds = np.array([reg.predict_proba(X_picked[idx_test]) for idx_test,reg in zip(idxs_test,regs)])
            
            scores = np.array([roc_auc_score(y_picked[idx_test],pred[:,-1],average = 'micro') for idx_test,pred in zip(idxs_test,preds)])
            plscores = scores.copy()
            np.save(saving_name,plscores)
        print(f'decode {conscious_state_dict[combine_1]} vs {conscious_state_dict[combine_2]} = {plscores.mean():.4f}+/-{plscores.std():.4f}')
        
    ####################### temporal decoding ##########################
    print('temporal decoding')
    cv          = StratifiedShuffleSplit(
                                n_splits            = n_splits, 
                                test_size           = 0.2, 
                                random_state        = 12345)
    clf         = make_pipeline(
                                StandardScaler(),
                                OneVsOneClassifier(clone(logistic)),)
    
    time_decod  = SlidingEstimator(
                                clf, 
                                n_jobs              = 1, 
                                scoring             = 'accuracy', 
                                verbose             = False
                                )
    if speed:
        X,y = epoch_temp.get_data(),epoch_temp.events[:,-1]
        y = y % 10 - 6
        times       = epoch_temp.times
    else:
        X,y = epochs.get_data(),epochs.events[:,-1]
        y = y % 10 - 6
        times       = epochs.times
    for combine_1,combine_2 in combinations([0,1,2],2):
        saving_name     = os.path.join(array_dir,f'temporal_decoding_{conscious_state_dict[combine_1]}_{conscious_state_dict[combine_2]}.npy')
        print(f'{conscious_state_dict[combine_1]} vs {conscious_state_dict[combine_2]}')
        
        if saving_name in glob(os.path.join(array_dir,'*.npy')):
            scores    = np.load(saving_name)
        else:
            idx_1, = np.where(y == combine_1)
            idx_2, = np.where(y == combine_2)
            idx_ = np.concatenate([idx_1,idx_2])
            
            X_picked,y_picked = X[idx_],y[idx_]
            np.random.seed(12345)
            X_picked,y_picked = shuffle(X_picked,y_picked)
            y_picked_binarized = Binarizer(threshold=np.mean(np.unique(y_picked))).fit_transform(y_picked.reshape(-1,1)).ravel()
            scores      = cross_val_multiscore(
                                        time_decod, 
                                        X_picked,
                                        y_picked_binarized,
                                        cv                  = cv, 
                                        n_jobs              = n_jobs,
                                        )
            np.save(saving_name,scores)
        fig,ax = plot_temporal_decoding(times,
                                        scores,
                                        frames,
                                        None,
                                        f'{conscious_state_dict[combine_1]} vs {conscious_state_dict[combine_2]}',
                                        plscores,
                                        n_splits,
                                        ylim = (0.3,0.7))
        fig.savefig(os.path.join(figure_dir,f'temporal decoding ({conscious_state_dict[combine_1]}_{conscious_state_dict[combine_2]}).png'),
                    bbox_inches = 'tight',
                    dpi         = 400)
        plt.close(fig)
        plt.clf()
        plt.close('all')
    
    ####################### temporal generalization ########################
    print('temporal generalization')
    cv          = StratifiedShuffleSplit(
                                n_splits            = n_splits, 
                                test_size           = 0.2, 
                                random_state        = 12345)
    clf         = make_pipeline(
                                StandardScaler(),
                                clone(logistic))
    time_gen    = GeneralizingEstimator(
                                clf, 
                                n_jobs              = 1, 
                                scoring             = scorer,
                                verbose             = False)
    if speed:
        X,y = epoch_temp.get_data(),epoch_temp.events[:,-1]
        y = y % 10 - 6
        times       = epoch_temp.times
    else:
        X,y = epochs.get_data(),epochs.events[:,-1]
        y = y % 10 - 6
        times       = epochs.times
    for combine_1,combine_2 in combinations([0,1,2],2):
        saving_name     = os.path.join(array_dir,f'temporal_generalization_{conscious_state_dict[combine_1]}_{conscious_state_dict[combine_2]}.npy')
        print(f'{conscious_state_dict[combine_1]} vs {conscious_state_dict[combine_2]}')
        
        if saving_name in glob(os.path.join(array_dir,'*.npy')):
            scores_gen_ = np.load(saving_name)
            scores_chance = np.load(saving_name.replace('.npy','_chance.npy'))
        else:
            idx_1, = np.where(y == combine_1)
            idx_2, = np.where(y == combine_2)
            idx_ = np.concatenate([idx_1,idx_2])
            
            X_picked,y_picked = X[idx_],y[idx_]
            np.random.seed(12345)
            X_picked,y_picked = shuffle(X_picked,y_picked)
            y_picked_binarized = Binarizer(threshold=np.mean(np.unique(y_picked))).fit_transform(y_picked.reshape(-1,1)).ravel()
            scores_gen  = cross_val_multiscore(
                                    time_gen, 
                                    X_picked,
                                    y_picked_binarized,
                                    cv                  = cv, 
                                    n_jobs              = n_jobs
                                    )
            np.save(saving_name,scores_gen)
            
            scores_gen_ = scores_gen.copy()#[]
            #        for s_gen,s in zip(scores_gen,scores):
            #            np.fill_diagonal(s_gen,s)
            #            scores_gen_.append(s_gen)
            #        scores_gen_ = np.array(scores_gen_)
            # compute the chance level of the temporal generalization
            y_ = shuffle(y_picked_binarized)
            scores_chance = cross_val_multiscore(
                                        time_gen,
                                        X_picked,
                                        y_,
                                        cv = cv,
                                        n_jobs = n_jobs,
                                        )
            np.save(saving_name.replace('.npy','_chance.npy'),scores_chance)
        
        fig,ax = plot_temporal_generalization(scores_gen_.mean(0),# - scores_chance.mean(0),
                                              epochs,
                                              None,
                                              f'{conscious_state_dict[combine_1]} vs {conscious_state_dict[combine_2]}',
                                              frames,
#                                              vmin = None,
#                                              vmax = None,
                                              )
        fig.savefig(os.path.join(figure_dir,f'temporal generalization ({conscious_state_dict[combine_1]} vs {conscious_state_dict[combine_2]}).png'),
                    bbox_inches = 'tight',
                    dpi         = 400)
        plt.close(fig)
        plt.clf()
        plt.close('all')
    
    ############################### permutation cluster test ##################################
    print('permutation cluster test')
    for combine_1,combine_2 in combinations([0,1,2],2):
        saving_name     = os.path.join(array_dir,f'permutation_cluster_test_T_obs_{conscious_state_dict[combine_1]}_{conscious_state_dict[combine_2]}.npy')
        print(f'{conscious_state_dict[combine_1]} vs {conscious_state_dict[combine_2]}')
        
        if saving_name in glob(os.path.join(array_dir,'*.npy')):
            T_obs = np.load(saving_name)
            clusters = np.load(saving_name.replace('T_obs','clusters'))
            cluster_p_values = np.load(saving_name.replace('T_obs','cluster_p_values'))
        else:
            tg_name     = os.path.join(array_dir,f'temporal_generalization_{conscious_state_dict[combine_1]}_{conscious_state_dict[combine_2]}.npy')
            scores_gen_ = np.load(tg_name)
            scores_chance = np.load(tg_name.replace('.npy','_chance.npy'))
            alpha           = 0.0001
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
        
        fig,ax = plot_t_stats(T_obs,
                              clusters,
                              cluster_p_values,
                              times,
                              None,
                              f'{conscious_state_dict[combine_1]} vs {conscious_state_dict[combine_2]}',
                              frames,)
        
        fig.savefig(os.path.join(figure_dir,f'stats ({conscious_state_dict[combine_1]}_{conscious_state_dict[combine_2]}).png'),
                    bbox_inches = 'tight',
                    dpi         = 400)
        plt.close(fig)
        plt.clf()
        plt.close('all')
        
        fig,ax = plot_p_values(times,
                               clusters,
                               cluster_p_values,
                               None,
                               f'{conscious_state_dict[combine_1]} vs {conscious_state_dict[combine_2]}',
                               frames)
        
        fig.savefig(os.path.join(figure_dir,f'stats (p values, {conscious_state_dict[combine_1]}_{conscious_state_dict[combine_2]}).png'),
                    bbox_inches = 'tight',
                    dpi         = 400)
        plt.close(fig)
        plt.clf()
        plt.close('all')


































