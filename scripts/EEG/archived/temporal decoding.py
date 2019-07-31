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
#from sklearn.svm             import LinearSVC
from sklearn.linear_model    import (
#                                        LogisticRegressionCV,
                                        LogisticRegression
                                        )
from sklearn.pipeline        import make_pipeline
from sklearn.model_selection import (
                                        StratifiedShuffleSplit,
                                        cross_val_score
                                        )
from sklearn.base 	     import clone
from matplotlib              import pyplot as plt
from scipy                   import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools               import partial
from utils                   import get_frames


subject             = 'marta_3_13_2019'
# there was a bug in the csv file, so the early behavioral is treated differently
date                = '/'.join(re.findall('\d+',subject))
date                = datetime.strptime(date,'%m/%d/%Y')
breakPoint          = datetime(2019,3,10)
if date > breakPoint:
    new             = True
else:
    new             = False
working_dir         = f'../data/clean EEG/{subject}'
working_data        = glob(os.path.join(working_dir,'*-epo.fif'))
frames              = get_frames(directory = f'../data/behavioral/{subject}',new = new)
# create the directories for figure and decoding results (numpy arrays)
figure_dir          = f'../figure/{subject}'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
array_dir           = f'../results/{subject}'
if not os.path.exists(array_dir):
    os.mkdir(array_dir)
# define the number of cross validation we want to do.
n_splits            = 100
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

for epoch_file in working_data:
    epochs  = mne.read_epochs(epoch_file)
    # resample at 100 Hz to fasten the decoding process
    print('resampling')
    epochs.resample(100)
    
    conscious = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if (' conscious' in name)])
    see_maybe = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('see_maybe' in name)])
    see_unknown = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('see_unknown' in name)])
    unconscious = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('unconscious' in name)])
    
    for ii,(epochs,conscious_state) in enumerate(zip([unconscious,see_unknown,see_maybe,conscious],
                                                     ['unconscious',
                                                      'see_unknown',
                                                      'see_maybe',
                                                      'conscious'])):
        epochs
        # decode the whole segment
        print('cross val scoring')
        saving_name = os.path.join(array_dir,f'whole_segment_{conscious_state}.npy')
        if saving_name in glob(os.path.join(array_dir,'*.npy')):
            plscores = np.load(saving_name)
        else:
            cv          = StratifiedShuffleSplit(
                                             n_splits       = n_splits, 
                                             test_size      = 0.2, 
                                             random_state   = 12345)
            
            pipeline    = make_pipeline(
                                     Scaler(epochs.info),
                                     Vectorizer(),
                                     StandardScaler(),
                                     clone(logistic),
                                     )
            
            scores      = cross_val_score(
                                     pipeline,
                                     epochs.get_data(),
                                     epochs.events[:,-1] //10 -1,
                                     scoring                = 'roc_auc',
                                     cv                     = cv,
                                     n_jobs                 = 6,
                                     )
            plscores = scores.copy()
            np.save(saving_name,plscores)
        print(f'decode {conscious_state} = {plscores.mean():.4f}+/-{plscores.std():.4f}')
        
        # temporal decoding
        print('temporal decoding')
        saving_name = os.path.join(array_dir,f'temporal_decoding_{conscious_state}.npy')
        if saving_name in glob(os.path.join(array_dir,'*.npy')):
            scores = np.load(saving_name)
        else:
            cv          = StratifiedShuffleSplit(
                                        n_splits            = n_splits, 
                                        test_size           = 0.2, 
                                        random_state        = 12345)
            clf         = make_pipeline(
                                        StandardScaler(),
                                        logistic)
            
            time_decod  = SlidingEstimator(
                                        clf, 
                                        n_jobs              = 1, 
                                        scoring             ='roc_auc', 
                                        verbose             = True
                                        )
            scores      = cross_val_multiscore(
                                        time_decod, 
                                        epochs.get_data(),
                                        epochs.events[:,-1]-1,
                                        cv                  = cv, 
                                        n_jobs              = 4
                                        )
            np.save(saving_name,scores)
        
        times       = epochs.times
        scores_mean = scores.mean(0)
        scores_se   = scores.std(0) / np.sqrt(100)
        
        fig,ax      = plt.subplots(figsize=(10,5))
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
        saving_name = os.path.join(array_dir,f'temporal_generalization_{conscious_state}.npy')
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
            
            scores_gen  = cross_val_multiscore(
                                        time_gen, 
                                        epochs.get_data(),
                                        epochs.events[:,-1]-1,
                                        cv                  = cv, 
                                        n_jobs              = 6
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
                            vmin                = 0.2, 
                            vmax                = 0.8,
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
        p_threshold     = 0.0001
        sigma           = 1e-3
        stat_fun_hat    = partial(mne.stats.ttest_1samp_no_p, sigma=sigma)
        ## perform nonparametric t test to find clusters in the conscious state
        # compute the threshold for t statistics
        t_threshold = -stats.distributions.t.ppf(p_threshold / 2., scores.shape[0] - 1) 
        # apply the MNE python function of which I know less than nothing
    #    threshold_tfce = dict(start=0, step=0.3)
        T_obs, clusters, cluster_p_values, H0   = clu \
                    = mne.stats.permutation_cluster_1samp_test(
                            scores_gen_ - 0.5,#np.median(conscious),
                            threshold           = t_threshold,
                            stat_fun            = stat_fun_hat,
                            tail                = 0,
                            check_disjoint      = True,
                            seed                = 12345,
                            step_down_p         = 0.05,
                            n_jobs              = 6
                            )
        # since the p values of each cluster is corrected for multiple comparison, 
        # we could directly use 0.05 as the threshold to filter clusters
        T_obs_plot              = 0 * np.ones_like(T_obs)
        for c, p_val in zip(clusters, cluster_p_values):
            if (p_val <= 0.05) and (np.sum(c) >= 10):
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
    #        print(np.sum(c))
            if np.sum(c) > 10:
                p_clust[c] = p_val
        
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
                               ticks            = np.linspace(vmin,vmax,3))
        cb.ax.set(title = 'P values')
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
