#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:04:58 2019

@author: nmei
"""

import mne
import os
import numpy as np
import pandas as pd
from glob import glob

from mne.decoding import (Scaler,
                          Vectorizer,
                          SlidingEstimator,
                          cross_val_multiscore,
                          GeneralizingEstimator)
from sklearn.preprocessing import StandardScaler
#from sklearn.svm import LinearSVC
from sklearn.linear_model import (
                                  LogisticRegressionCV,
                                  LogisticRegression
                                  )
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsOneClassifier
from sklearn.base import clone
from sklearn.model_selection import (StratifiedShuffleSplit,
                                     cross_val_score)
from matplotlib import pyplot as plt
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable

working_dir = '../data/clean EEG'
working_data = glob(os.path.join(working_dir,'*multi-epo.fif'))

figure_dir = '../figure'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
n_splits = 100
#logistic = LogisticRegressionCV(
#                     Cs = np.logspace(-4,4,num = 9,),
#                     cv = 4,
#                     scoring = 'roc_auc',
#                     solver = 'lbfgs',
#                     max_iter = int(1e3),
#                     random_state = 12345)
logistic = LogisticRegression(solver = 'lbfgs',
                              max_iter = int(1e3),
                              class_weight = 'balanced',
                              random_state = 12345)
logistic = OneVsOneClassifier(logistic)

for epoch_file in working_data:
    epochs = mne.read_epochs(epoch_file)
    # resample at 100 Hz to fasten the decoding process
    print('resampling')
    epochs.resample(100)
    # decode the whole segment
    cv = StratifiedShuffleSplit(n_splits = 10,#n_splits, 
                                test_size = 0.2, 
                                random_state = 12345)
    
    pipeline = make_pipeline(#Scaler(epochs.info),
                             Vectorizer(),
                             StandardScaler(),
                             clone(logistic),
                             )
    # generate the average 
    print('cross val scoring')
    scores = cross_val_score(pipeline,
                             epochs.get_data(),
                             epochs.events[:,-1]-1,
                             scoring = 'accuracy',
                             cv = cv,
                             n_jobs = 6,)
    print(f'{scores.mean():.4f}')
    
    # temporal decoding
    cv = StratifiedShuffleSplit(n_splits = n_splits, 
                                test_size = 0.2, 
                                random_state = 12345)
    clf = make_pipeline(StandardScaler(),
                        logistic)
    print('temporal decoding')
    time_decod = SlidingEstimator(clf, 
                                  n_jobs = 1, 
                                  scoring ='accuracy', 
                                  verbose=True)
    scores = cross_val_multiscore(time_decod, 
                                  epochs.get_data(),
                                  epochs.events[:,-1]-1,
                                  cv = cv, 
                                  n_jobs = 4)
    
    times = epochs.times
    scores_mean = scores.mean(0)
    scores_se = scores.std(0) / np.sqrt(100)
    
    fig,ax = plt.subplots(figsize=(10,5))
    ax.plot(times,scores_mean,color='k',alpha=.9,label='mean',)
    ax.fill_between(times,
                    scores_mean + scores_se,
                    scores_mean - scores_se,
                    color = 'red',
                    alpha = 0.4,
                    label = 'std',)
    ax.axhline(0.5,
               linestyle = '--',
               color = 'k',
               alpha = 0.7,
               label = 'chance')
    ax.axvline(0,
               linestyle = '--',
               color = 'blue',
               alpha = 0.7,
               label = 'onset',)
    ax.axvline(0.15,
               linestyle = '--',
               color = 'blue',
               alpha = 0.7,
               label = 'offset',)
    ax.set(xlim=(times.min(),
                 times.max()),
           )
    ax.legend()
    fig.savefig(os.path.join(figure_dir,'temporal decoding.png'),
                bbox_inches = 'tight',
                dpi = 400)
    plt.close(fig)
    plt.clf()
    plt.close('all')
    
    # temporal generalization
    print('temporal generalization')
    cv = StratifiedShuffleSplit(n_splits = 100, 
                                test_size = 0.2, 
                                random_state = 12345)
    clf = make_pipeline(StandardScaler(),
                        logistic)
    time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring='roc_auc',
                                 verbose=True)
    
    scores_gen = cross_val_multiscore(time_gen, 
                                      epochs.get_data(),
                                      epochs.events[:,-1]-1,
                                      cv = cv, 
                                      n_jobs = 4)
    scores_gen_ = []
    for s_gen,s in zip(scores_gen,scores):
        np.fill_diagonal(s_gen,s)
        scores_gen_.append(s_gen)
    scores_gen_ = np.array(scores_gen_)
    
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(scores_gen_.mean(0), 
                   interpolation='lanczos', 
                   origin='lower', 
                   cmap='RdBu_r',
                   extent=epochs.times[[0, -1, 0, -1]], 
                   vmin=0.2, vmax=0.8)
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Temporal generalization')
    ax.axhline(0.,linestyle='--',color='black',alpha=0.7,label='onset',)
    ax.axvline(0.,linestyle='--',color='black',alpha=0.7,label='onset',)
    ax.axhline(0.15,linestyle='--',color='blue',alpha=0.7,label='offset',)
    ax.axvline(0.15,linestyle='--',color='blue',alpha=0.7,label='offset',)
    plt.colorbar(im, ax=ax)
    ax.legend()
    fig.savefig(os.path.join(figure_dir,'temporal generalization.png'),
                bbox_inches = 'tight',
                dpi = 400)
    plt.close(fig)
    plt.clf()
    plt.close('all')
    
    # permutation cluster test
    p_threshold = 0.0001
    ## perform nonparametric t test to find clusters in the conscious state
    # compute the threshold for t statistics
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., scores.shape[0] - 1) 
    # apply the MNE python function of which I know less than nothing
    T_obs, clusters, cluster_p_values, H0 = clu \
        = mne.stats.permutation_cluster_1samp_test(scores_gen_ - 0.5,#np.median(conscious),
                                                   threshold = t_threshold,
                                                   tail = 1,
                                                   check_disjoint = True,
                                                   n_jobs = 4)
    # since the p values of each cluster is corrected for multiple comparison, 
    # we could directly use 0.05 as the threshold to filter clusters
    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if (p_val <= 0.05):# and (np.sum(c) >= 100):
            T_obs_plot[c] = T_obs[c]
    # defind the range of the colorbar
    vmax = np.max(np.abs(T_obs))
    vmin = -vmax# - 2 * t_threshold
    plt.close('all')
    fig,ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(T_obs_plot,
                   origin = 'lower',
                   cmap = plt.cm.RdBu_r,# to emphasize the clusters
                   extent = epochs.times[[0, -1, 0, -1]],
                   vmin = vmin,
                   vmax = vmax,
                   interpolation = 'hanning',
                   )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cb = plt.colorbar(im, cax=cax,ticks=np.linspace(vmin,vmax,3))
    cb.ax.set(title='t stats')
    ax.axhline(0.,linestyle='--',color='black',alpha=0.7,label='onset',)
    ax.axvline(0.,linestyle='--',color='black',alpha=0.7,label='onset',)
    ax.axhline(0.15,linestyle='--',color='blue',alpha=0.7,label='offset',)
    ax.axvline(0.15,linestyle='--',color='blue',alpha=0.7,label='offset',)
    ax.set(xlabel='Test time',ylabel='Train time',
           title='nonparametric t test')
    ax.legend()
    fig.savefig(os.path.join(figure_dir,'stats.png'),
                bbox_inches = 'tight',
                dpi = 400)
    plt.close(fig)
    plt.clf()
    plt.close('all')
