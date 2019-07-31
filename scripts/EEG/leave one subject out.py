#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:43:28 2019

@author: nmei
"""

import os
import mne
import numpy as np
from glob import glob
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score,make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import (GeneralizingEstimator,cross_val_multiscore,SlidingEstimator)
from functools import partial
from matplotlib import pyplot as plt
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable

func   = partial(roc_auc_score,average = 'micro')
func.__name__ = 'micro_AUC'
scorer = make_scorer(func,needs_proba = True)

logistic = LogisticRegression(
                              solver        = 'lbfgs',
                              max_iter      = int(1e3),
                              random_state  = 12345
                              )
n_jobs = 6

working_dir = '../../data/clean EEG'
working_data = glob(os.path.join(working_dir,'*','*.fif'))
array_dir = '../../results/EEG/LOO_CV'
if not os.path.exists(array_dir):
    os.makedirs(array_dir)
figure_dir = '../../figures/EEG/LOO_CV'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
def load_epochs(f,conscious_state,sub = 0):
    epochs_temp = mne.read_epochs(f)
    events = epochs_temp.events
    events[:,1] = sub + 1
    epochs_temp.events = events
    epochs_needed = mne.concatenate_epochs([epochs_temp[name] for name in epochs_temp.event_id.keys() if (conscious_state in name)])
    return epochs_needed

for conscious_state in ['unconscious','glimpse',' conscious']:
    conscious_state
    
    epochs = mne.concatenate_epochs([load_epochs(f,conscious_state,ii) for ii,f in enumerate(working_data)])
    
    print('resampling...')
    epochs.resample(100)
    
    X = epochs.get_data()
    y = epochs.events[:,-1] // 100 - 2
    np.random.seed(12345)
    X,y = shuffle(X,y)
    groups = epochs.events[:,1]
    
    cv = LeaveOneGroupOut()
    clf         = make_pipeline(
                                StandardScaler(),
                                clone(logistic))
    
    # temporal decoding
    print('temporal decoding')
    saving_name     = os.path.join(array_dir,f'temporal_decoding_{conscious_state}.npy')
    if saving_name in glob(os.path.join(array_dir,'*.npy')):
        scores = np.load(saving_name)
    else:
        time_decod  = SlidingEstimator(
                                            clf, 
                                            n_jobs              = 1, 
                                            scoring             = scorer, 
                                            verbose             = False
                                            )
        scores      = cross_val_multiscore(
                                            time_decod, 
                                            X,
                                            y,
                                            cv                  = cv, 
                                            n_jobs              = n_jobs,
                                            groups              = groups,
                                            )
        np.save(saving_name,scores)
    
    n_splits = scores.shape[0]
    times = epochs.times
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
    ax.set(xlim     = (times.min(),
                       times.max()),
           ylim     = (0.425,0.575),
           title    = f'Temporal decoding of {conscious_state}',
           )
    ax.legend()
    fig.savefig(os.path.join(figure_dir,
                             f'temporal decoding of {conscious_state}.png'),
    dpi = 400,
    bbox_inches = 'tight',)
    # temporal generalization
    print('temporal generalization')
    saving_name     = os.path.join(array_dir,f'temporal_generalization_{conscious_state}.npy')
    if saving_name in glob(os.path.join(array_dir,'*.npy')):
        scores_gen = np.load(saving_name)
    else:
        time_gen    = GeneralizingEstimator(
                                            clf, 
                                            n_jobs              = 1, 
                                            scoring             = scorer,
                                            verbose             = False)
        scores_gen  = cross_val_multiscore(
                                            time_gen, 
                                            X,
                                            y,
                                            cv                  = cv, 
                                            n_jobs              = n_jobs,
                                            groups              = groups,
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
                        vmin                = 0.425, 
                        vmax                = 0.575,
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
    plt.colorbar(im, ax = ax)
    ax.legend()
    fig.savefig(os.path.join(figure_dir,
                             f'temporal generalization of {conscious_state}.png'),
    dpi = 400,
    bbox_inches = 'tight')
    
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
#                        check_disjoint      = True, # not useful
                        seed                = 12345, # random seed
                        step_down_p         = 0.005,
                        buffer_size         = None, # stat_fun does not treat variables independently
                        n_jobs              = n_jobs,
                        )
    # since the p values of each cluster is corrected for multiple comparison, 
    # we could directly use 0.05 as the threshold to filter clusters
    T_obs_plot              = 0 * np.ones_like(T_obs)
    
    for c, p_val in zip(clusters, cluster_p_values):
        if (p_val <= 0.01):# and (distance.cdist(np.where(c ==  True)[0].reshape(1,-1),np.where(c ==  True)[1].reshape(1,-1))[0][0] < 200):# and (np.sum(c) >= c_thresh):
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
#        if (np.sum(c) >= c_thresh):
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













































