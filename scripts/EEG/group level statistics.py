#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:25:00 2019

@author: nmei
"""
import os
from glob import glob
from shutil import copyfile
copyfile('../utils.py','utils.py')
import utils
import numpy as np
from scipy import stats
import mne
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
n_jobs = 8

working_dir = '../../results/EEG/decode_7_19_logistic_no_pre'
numpy_arrays = glob(os.path.join(working_dir,'*',"*.npy"))
epochs = mne.read_epochs('../../data/clean EEG/aingere_5_16_2019/clean-epo.fif')
epochs = epochs.resample(100)
for conscious_state in ['unconscious','glimpse','conscious']:
    working_arrays = [item for item in numpy_arrays if ('temporal_generalization' in item)\
                      and (f'_{conscious_state}' in item)]
    
    td_arrays_ave = [np.load(f).mean(0)[np.newaxis,:] for f in working_arrays]
    scores = np.concatenate(td_arrays_ave)
    
    
    # permutation cluster test
    alpha           = 0.05
    sigma           = 1e-3
    stat_fun_hat    = partial(mne.stats.ttest_1samp_no_p, sigma=sigma)
    ## perform nonparametric t test to find clusters in the conscious state
    # compute the threshold for t statistics
    t_threshold = stats.distributions.t.ppf(1 - alpha, scores.shape[0] - 1) 
    # apply the MNE python function of which I know less than nothing
    threshold_tfce = dict(start=0, step=0.2)
    T_obs, clusters, cluster_p_values, H0   = clu \
                = mne.stats.permutation_cluster_1samp_test(
                        scores - 0.5,#np.median(conscious),
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
        if (p_val <= alpha):# and (np.sum(c) >= c_thresh):# and (distance.cdist(np.where(c ==  True)[0].reshape(1,-1),np.where(c ==  True)[1].reshape(1,-1))[0][0] < 200):# and (np.sum(c) >= c_thresh):
            T_obs_plot[c]   = T_obs[c]
    
    
    # defind the range of the colorbar
    vmax = np.max(np.abs(T_obs))
    vmin = -vmax# - 2 * t_threshold
    plt.close('all')
    fig,ax = plt.subplots(figsize=(10,10))
    times = epochs.times
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