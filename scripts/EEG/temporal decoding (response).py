#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:04:58 2019

@author: nmei
"""

import mne
import os
import re
import numpy as np
from glob                    import glob
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
from matplotlib              import pyplot as plt
from scipy                   import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools               import partial
from shutil                  import copyfile
copyfile('../utils.py','utils.py')
from utils                   import get_frames

n_jobs = 8

subject             = 'xabier_5_15_2019' 
# there was a bug in the csv file, so the early behavioral is treated differently
date                = '/'.join(re.findall('\d+',subject))
date                = datetime.strptime(date,'%m/%d/%Y')
breakPoint          = datetime(2019,3,10)
if date > breakPoint:
    new             = True
else:
    new             = False
working_dir         = f'../../data/clean EEG/{subject}'
working_data        = glob(os.path.join(working_dir,'*-epo.fif'))
frames,_            = get_frames(directory = f'../../data/behavioral/{subject}',
                                 new = new)
# create the directories for figure and decoding results (numpy arrays)
figure_dir          = f'../../figures/EEG/decode responses/{subject}'
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
array_dir           = f'../../results/EEG/decode decode responses/{subject}'
if not os.path.exists(array_dir):
    os.makedirs(array_dir)
# define the number of cross validation we want to do.
n_splits            = 500
#logistic = LogisticRegressionCV(
#                     Cs = np.logspace(-4,4,num = 9,),
#                     cv = 4,
#                     scoring = 'roc_auc',
#                     solver = 'lbfgs',
#                     max_iter = int(1e3),
#                     random_state = 12345)
logistic = LogisticRegression(
                              solver        = 'lbfgs',
                              max_iter      = int(1e3),
                              random_state  = 12345
                              )
#svm = LinearSVC(class_weight = 'balanced',
#                random_state = 12345,
#                tol = 1e-3)
#logistic = CalibratedClassifierCV(svm,cv=5)
#logistic = SGDClassifier(loss = 'modified_huber',
#                         penalty = "elasticnet",
#                         early_stopping = True,
#                         class_weight = 'balanced',
#                         random_state = 12345,
#                         max_iter = int(1e3),
#                         tol = 1e-3,)


for epoch_file in working_data:
    epochs  = mne.read_epochs(epoch_file)
    # resample at 100 Hz to fasten the decoding process
    print('resampling...')
    epochs.resample(100)
    
    conscious   = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if (' conscious' in name)])
    see_maybe   = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('glimpse' in name)])
    unconscious = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('unconscious' in name)])
    del epochs
    
    for ii,(epochs,conscious_state) in enumerate(zip([unconscious.copy(),
                                                      see_maybe.copy(),
                                                      conscious.copy()],
                                                     ['unconscious',
                                                      'glimpse',
                                                      'conscious'])):
        epochs
        
        # decode the whole segment
        print('cross val scoring')
        saving_name     = os.path.join(array_dir,f'whole_segment_{conscious_state}.npy')
        if saving_name in glob(os.path.join(array_dir,'*.npy')):
            plscores    = np.load(saving_name)
        else:
            cv          = StratifiedShuffleSplit(
                                             n_splits       = n_splits, 
                                             test_size      = 0.2, 
                                             random_state   = 12345)
            pipeline    = make_pipeline(
                                     Vectorizer(),
                                     StandardScaler(),
                                     clone(logistic),
                                     )
            X,y = epochs.get_data(),epochs.events[:,-1]
            y = y //10 % 10 - 4
            X,y = shuffle(X,y)
            
            scores      = cross_val_score(
                                     pipeline,
                                     X,
                                     y,
                                     scoring                = 'roc_auc',
                                     cv                     = cv,
                                     n_jobs                 = n_jobs,
                                     )
            plscores = scores.copy()
            np.save(saving_name,plscores)
        print(f'decode {conscious_state} = {plscores.mean():.4f}+/-{plscores.std():.4f}')
        
        # temporal decoding
        print('temporal decoding')
        saving_name     = os.path.join(array_dir,f'temporal_decoding_{conscious_state}.npy')
        if saving_name in glob(os.path.join(array_dir,'*.npy')):
            scores = np.load(saving_name)
        else:
            cv          = StratifiedShuffleSplit(
                                        n_splits            = n_splits, 
                                        test_size           = 0.2, 
                                        random_state        = 12345)
            clf         = make_pipeline(
                                        StandardScaler(),
                                        clone(logistic))
            
            time_decod  = SlidingEstimator(
                                        clf, 
                                        n_jobs              = 1, 
                                        scoring             ='roc_auc', 
                                        verbose             = True
                                        )
            X = epochs.get_data()
            y = epochs.events[:,-1]
            y = y //10 % 10 - 4
            X,y = shuffle(X,y)
            scores      = cross_val_multiscore(
                                        time_decod, 
                                        X,
                                        y,
                                        cv                  = cv, 
                                        n_jobs              = n_jobs,
                                        )
            np.save(saving_name,scores)
        
        times       = epochs.times
        scores_mean = scores.mean(0)
        scores_se   = scores.std(0) / np.sqrt(n_splits)
        
        fig,ax      = plt.subplots(figsize=(16,8))
        ax.plot(times,scores_mean,
                color = 'k',
                alpha = .9,
                label = f'Average across {n_splits} folds',
                )
        ax.fill_between(times,
                        scores_mean + scores_se,
                        scores_mean - scores_se,
                        color = 'red',
                        alpha = 0.4,
                        label = 'Standard Error',)
        ax.axhline(0.5,
                   linestyle    = '--',
                   color        = 'k',
                   alpha        = 0.7,
                   label        = 'Chance level')
        ax.axvline(0,
                   linestyle    = '--',
                   color        = 'blue',
                   alpha        = 0.7,
                   label        = 'Probe onset',)
        ax.axvspan(frames[ii][1] * .01 - frames[ii][2] * .01,
                   frames[ii][1] * .01 + frames[ii][2] * .01,
                   color        = 'blue',
                   alpha        = 0.3,
                   label        = 'probe offset ave +/- std',)
        ax.axvline(0.15 + 0.25 * 5,
                   linestyle    = '--',
                   color        = 'green',
                   alpha        = 0.7,
                   label        = 'Delay onset')
        ax.set(xlim     = (times.min(),
                           times.max()),
               ylim     = (0.4,0.6),
               title    = f'Temporal decoding of {conscious_state} = {plscores.mean():.3f}+/-{plscores.std():.3f}',
               )
        ax.legend()
        fig.savefig(os.path.join(figure_dir,f'temporal decoding ({conscious_state}).png'),
                    bbox_inches = 'tight',
                    dpi         = 400)
        plt.close(fig)
        plt.clf()
        plt.close('all')
        
        # temporal generalization
        print('temporal generalization')
        saving_name     = os.path.join(array_dir,f'temporal_generalization_{conscious_state}.npy')
        if saving_name in glob(os.path.join(array_dir,'*.npy')):
            scores_gen = np.load(saving_name)
        else:
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
                                        scoring             = 'roc_auc',
                                        verbose             = True)
            X,y = epochs.get_data(),epochs.events[:,-1]
            y = y //10 % 10 - 4
            X,y = shuffle(X,y)
            scores_gen  = cross_val_multiscore(
                                        time_gen, 
                                        X,
                                        y,
                                        cv                  = cv, 
                                        n_jobs              = n_jobs
                                        )
            np.save(saving_name,scores_gen)
        scores_gen_ = []
        for s_gen,s in zip(scores_gen,scores):
            np.fill_diagonal(s_gen,s)
            scores_gen_.append(s_gen)
        scores_gen_ = np.array(scores_gen_)
        
        fig, ax = plt.subplots(figsize = (10,10))
        im      = ax.imshow(
                            scores_gen_.mean(0), 
                            interpolation       = 'lanczos', 
                            origin              = 'lower', 
                            cmap                = 'RdBu_r',
                            extent              = epochs.times[[0, -1, 0, -1]], 
                            vmin                = 0.4, 
                            vmax                = 0.6,
                            )
        ax.set_xlabel('Testing Time (s)')
        ax.set_ylabel('Training Time (s)')
        ax.set_title(f'Temporal generalization of {conscious_state}')
        ax.axhline(0.,
                   linestyle                    = '--',
                   color                        = 'black',
                   alpha                        = 0.7,
                   label                        = 'Probe onset',)
        ax.axvline(0.,
                   linestyle                    = '--',
                   color                        = 'black',
                   alpha                        = 0.7,
    #               label                        = 'Probe onset',
                   )
        ax.axhspan(frames[ii][1] * .01 - frames[ii][2] * .01,
                   frames[ii][1] * .01 + frames[ii][2] * .01,
                   color                        = 'black',
                   alpha                        = 0.2,
                   label                        = 'probe offset ave +/- std',)
        ax.axvspan(frames[ii][1] * .01 - frames[ii][2] * .01,
                   frames[ii][1] * .01 + frames[ii][2] * .01,
                   color                        = 'black',
                   alpha                        = 0.2,
#                   label                        = 'probe offset ave +/- std',
                   )
        plt.colorbar(im, ax = ax)
        ax.legend()
        fig.savefig(os.path.join(figure_dir,f'temporal generalization ({conscious_state}).png'),
                    bbox_inches = 'tight',
                    dpi         = 400)
        plt.close(fig)
        plt.clf()
        plt.close('all')
        
        # permutation cluster test
        alpha           = 0.0001
        sigma           = 1e-3
        stat_fun_hat    = partial(mne.stats.ttest_1samp_no_p, sigma=sigma)
        ## perform nonparametric t test to find clusters in the conscious state
        # compute the threshold for t statistics
        t_threshold = stats.distributions.t.ppf(1 - alpha, scores.shape[0] - 1) 
        # apply the MNE python function of which I know less than nothing
        threshold_tfce = dict(start=0, step=0.2)
        T_obs, clusters, cluster_p_values, H0   = clu \
                    = mne.stats.permutation_cluster_1samp_test(
                            scores_gen_ - 0.5,#np.median(conscious),
                            threshold           = threshold_tfce,
                            stat_fun            = stat_fun_hat,
                            tail                = 1, # find clusters that are greater than the chance level
#                            check_disjoint      = True, # not useful
                            seed                = 12345, # random seed
                            step_down_p         = 0.005,
                            buffer_size         = None, # stat_fun does not treat variables independently
                            n_jobs              = n_jobs,
                            )
        # since the p values of each cluster is corrected for multiple comparison, 
        # we could directly use 0.05 as the threshold to filter clusters
        T_obs_plot              = 0 * np.ones_like(T_obs)
        k = np.array([np.sum(c) for c in clusters])
        from scipy.spatial import distance
        j = [distance.cdist(np.where(c ==  True)[0].reshape(1,-1),np.where(c ==  True)[1].reshape(1,-1))[0][0] for c in clusters]
        
        if np.max(k) > 1000:
            c_thresh = 1000
        elif 1000 > np.max(k) > 500:
            c_thresh = 500
        elif 500 > np.max(k) > 100:
            c_thresh = 100
        elif 100 > np.max(k) > 10:
            c_thresh = 10
        else:
            c_thresh = 0
        for c, p_val in zip(clusters, cluster_p_values):
            if (p_val <= 0.01) and (np.sum(c) >= c_thresh):# and (distance.cdist(np.where(c ==  True)[0].reshape(1,-1),np.where(c ==  True)[1].reshape(1,-1))[0][0] < 200):# and (np.sum(c) >= c_thresh):
                T_obs_plot[c]   = T_obs[c]
        # defind the range of the colorbar
        vmax = np.max(np.abs(T_obs))
        vmin = -vmax# - 2 * t_threshold
        plt.close('all')
        fig,ax = plt.subplots(figsize=(10,10))
        im      = ax.imshow(T_obs_plot,
                       origin                   = 'lower',
                       cmap                     = plt.cm.RdBu_r,# to emphasize the clusters
                       extent                   = epochs.times[[0, -1, 0, -1]],
                       vmin                     = vmin,
                       vmax                     = vmax,
                       interpolation            = 'lanczos',
                       )
        divider = make_axes_locatable(ax)
        cax     = divider.append_axes("right", 
                                      size      = "5%", 
                                      pad       = 0.2)
        cb      = plt.colorbar(im, 
                               cax              = cax,
                               ticks            = np.linspace(vmin,vmax,3))
        cb.ax.set(title = 'T Statistics')
        ax.plot([times[0],times[-1]],[times[0],times[-1]],
                linestyle                    = '--',
                color                        = 'black',
                alpha                        = 0.7,
                )
        ax.axhline(0.,
                   linestyle                    = '--',
                   color                        = 'black',
                   alpha                        = 0.7,
                   label                        = 'Probe onset',)
        ax.axvline(0.,
                   linestyle                    = '--',
                   color                        = 'black',
                   alpha                        = 0.7,
    #               label                        = 'Probe onset',
                   )
        ax.axhspan(frames[ii][1] * .01 - frames[ii][2] * .01,
                   frames[ii][1] * .01 + frames[ii][2] * .01,
                   color                        = 'black',
                   alpha                        = 0.2,
                   label                        = 'probe offset ave +/- std',)
        ax.axvspan(frames[ii][1] * .01 - frames[ii][2] * .01,
                   frames[ii][1] * .01 + frames[ii][2] * .01,
                   color                        = 'black',
                   alpha                        = 0.2,
#                   label                        = 'probe offset ave +/- std',
                   )
        ax.set(xlabel                           = 'Test time',
               ylabel                           = 'Train time',
               title                            = f'nonparametric t test of {conscious_state}')
        ax.legend()
        fig.savefig(os.path.join(figure_dir,f'stats ({conscious_state}).png'),
                    bbox_inches = 'tight',
                    dpi         = 400)
        plt.close(fig)
        plt.clf()
        plt.close('all')
        
        width = len(epochs.times)
        p_clust = np.ones((width, width))# * np.nan
        for c, p_val in zip(clusters, cluster_p_values):
#            print(np.sum(c))
#            if (distance.cdist(np.where(c ==  True)[0].reshape(1,-1),np.where(c ==  True)[1].reshape(1,-1))[0][0] < 200):
            if (np.sum(c) >= c_thresh):
                p_val_ = p_val.copy()
                if p_val_ > 0.05:
                    p_val_ = 1.
                p_clust[c] = p_val_
        
        # defind the range of the colorbar
        vmax = 1.
        vmin = 0.
        plt.close('all')
        fig,ax = plt.subplots(figsize = (10,10))
        im      = ax.imshow(p_clust,
                       origin                   = 'lower',
                       cmap                     = plt.cm.RdBu_r,# to emphasize the clusters
                       extent                   = epochs.times[[0, -1, 0, -1]],
                       vmin                     = vmin,
                       vmax                     = vmax,
                       interpolation            = 'hanning',
                       )
        divider = make_axes_locatable(ax)
        cax     = divider.append_axes("right", 
                                      size      = "5%", 
                                      pad       = 0.2)
        cb      = plt.colorbar(im, 
                               cax              = cax,
                               ticks            = [0,0.05,1])
        cb.ax.set(title = 'P values')
        ax.plot([times[0],times[-1]],[times[0],times[-1]],
                linestyle                       = '--',
                color                           = 'black',
                alpha                           = 0.7,
                )
        ax.axhline(0.,
                   linestyle                    = '--',
                   color                        = 'black',
                   alpha                        = 0.7,
                   label                        = 'Probe onset',)
        ax.axvline(0.,
                   linestyle                    = '--',
                   color                        = 'black',
                   alpha                        = 0.7,
    #               label                        = 'Probe onset',
                   )
        ax.axhspan(frames[ii][1] * .01 - frames[ii][2] * .01,
                   frames[ii][1] * .01 + frames[ii][2] * .01,
                   color                        = 'black',
                   alpha                        = 0.2,
                   label                        = 'probe offset ave +/- std',)
        ax.axvspan(frames[ii][1] * .01 - frames[ii][2] * .01,
                   frames[ii][1] * .01 + frames[ii][2] * .01,
                   color                        = 'black',
                   alpha                        = 0.2,
#                   label                        = 'probe offset ave +/- std',
                   )
        ax.set(xlabel                           = 'Test time',
               ylabel                           = 'Train time',
               title                            = f'p value map of {conscious_state}')
        ax.legend()
        fig.savefig(os.path.join(figure_dir,f'stats (p values, {conscious_state}).png'),
                    bbox_inches = 'tight',
                    dpi         = 400)
        plt.close(fig)
        plt.clf()
        plt.close('all')
