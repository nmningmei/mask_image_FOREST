#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:02:22 2019

@author: nmei
"""

import os
import mne
from glob import glob
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

folder_name = 'decode_premask_baseline_ICA'
target_name = 'average_TG_across_subjects_mask_baseline_ICA'
working_dir = f'../../results/EEG/{folder_name}/'
figure_dir = f'../../figures/EEG/{target_name}'
saving_dir = f'../../results/EEG/{target_name}'
for d in [figure_dir,saving_dir]:
    if not os.path.exists(d):
        os.mkdir(d)
alpha_level = 0.01

for conscious_state in ['unconscious','glimpse','conscious']:
    working_data = glob(os.path.join(working_dir,
                                     f'*/temporal_generalization*_{conscious_state}.npy'))
    chance_data = glob(os.path.join(working_dir,
                                     f'*/temporal_generalization*_{conscious_state}_chance.npy'))
    
    TG_matrices = np.array([np.load(f) for f in working_data])
    TG_chances = np.array([np.load(f) for f in chance_data])
    TG_diff = np.mean(TG_matrices - TG_chances,0)
    
    sigma           = 1e-3
    stat_fun_hat    = partial(mne.stats.ttest_1samp_no_p, sigma=sigma)
    
    # apply the MNE python function of which I know less than nothing
    threshold_tfce = dict(start=0, step=0.1)
    T_obs, clusters, cluster_p_values, H0   = mne.stats.permutation_cluster_1samp_test(
                        TG_diff,
                        threshold           = threshold_tfce,
                        stat_fun            = stat_fun_hat,
                        tail                = 1, # find clusters that are greater than the chance level
                        seed                = 12345, # random seed
                        step_down_p         = 0.05,
                        buffer_size         = None, # stat_fun does not treat variables independently
                        n_jobs              = 6,
                        )
    times = [-0.2,1,-0.2,1]
    fig, ax = plt.subplots(figsize = (10,10))
    im      = ax.imshow(
                        TG_matrices.mean(1).mean(0), 
                        interpolation       = 'lanczos', 
                        origin              = 'lower', 
                        cmap                = 'RdBu_r',
                        extent              = times, 
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
                             f'temporal_generalization_{conscious_state}.png'),
                dpi = 400,
                bbox = 'tight')
    
    T_obs_plot              = 0 * np.ones_like(T_obs)
    
    for c, p_val in zip(clusters, cluster_p_values):
        if (p_val <= alpha_level):# and (distance.cdist(np.where(c ==  True)[0].reshape(1,-1),np.where(c ==  True)[1].reshape(1,-1))[0][0] < 200):# and (np.sum(c) >= c_thresh):
            T_obs_plot[c]   = T_obs[c]
    
    # defind the range of the colorbar
    vmax = np.max(np.abs(T_obs))
    vmin = -vmax# - 2 * t_threshold
    plt.close('all')
    fig,ax = plt.subplots(figsize=(10,10))
    im      = ax.imshow(T_obs_plot,
                   origin                   = 'lower',
                   cmap                     = plt.cm.RdBu_r,# to emphasize the clusters
                   extent                   = times,
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
    
    width = TG_matrices.shape[-1]
    p_clust = np.ones((width, width))# * np.nan
    for c, p_val in zip(clusters, cluster_p_values):
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
                   extent                   = times,
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































