#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:05:23 2020

@author: nmei
"""

import os
from glob import glob

import numpy as np

from nilearn.input_data import NiftiMasker
from nilearn.image      import new_img_like
from nibabel            import load as load_img
from nilearn.mass_univariate import permuted_ols
from nilearn.plotting import plot_stat_map

from itertools          import combinations
from matplotlib import pyplot as plt

from nipype.interfaces import fsl

from scipy import stats

for k in [1,2,3,4,5,6,7]:
    sub = f'sub-0{k}'
    working_dir = f'../../../../results/MRI/nilearn/RSA_searchlight/{sub}'
    working_data = glob(os.path.join(working_dir,'*.nii.gz'))
    mask_dir = f'/bcbl/home/home_n-z/nmei/MRI/uncon_feat/MRI/{sub}/anat/ROI_BOLD/'
    mask_file = glob(os.path.join(mask_dir,'combine_BOLD.nii.gz'))[0]
    univariate_test_dir = f'../../../../results/MRI/nilearn/univariate_test/{sub}'
    functional_brain = f'../../../../data/MRI/{sub}/func/example_func.nii.gz'
    standard_brain = '../../../../data/standard_brain/MNI152_T1_2mm_brain.nii.gz'
    transformation_dir_single = os.path.abspath(
                    f'../../../../data/MRI/{sub}/reg/example_func2standard.mat')
    standarded_dri = f'../../../../results/MRI/nilearn/RSA_searchlight_standarded/{sub}'
    if not os.path.exists(standarded_dri):
        os.makedirs(standarded_dri)
    if not os.path.exists(univariate_test_dir):
        os.makedirs(univariate_test_dir)
    figure_dir = f'../../../../figures/MRI/nilearn/RSA_searchlight/'
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    
    # figure = plt.figure(figsize = (16 * 2,5 * 3))
    
    conscious_state1 = 'conscious'
    conscious_state2 = 'unconscious'
    file_name1 = [item for item in working_data if (f'/{conscious_state1}.nii' in item)][0]
    file_name2 = [item for item in working_data if (f'/{conscious_state2}.nii' in item)][0]
    
    masker = NiftiMasker(mask_img = mask_file,
                         smoothing_fwhm = 6)
    temp1 = masker.fit_transform(file_name1)
    
    masker = NiftiMasker(mask_img = mask_file,
                         smoothing_fwhm = 6)
    temp2 = masker.fit_transform(file_name2)
    
    zscore1 = np.arctanh(temp1)
    zscore2 = np.arctanh(temp2)
    
    # speed things up
    zscore1 = zscore1.reshape(50,20,-1).mean(0)
    zscore2 = zscore2.reshape(50,20,-1).mean(0)
    
    # grouped_fmri = np.concatenate([zscore1,zscore2])
    grouped_fmri = np.zeros((zscore1.shape[0] + zscore2.shape[0],zscore2.shape[1]))
    grouped_fmri[1::2,:] = zscore1 # from the 2nd row
    grouped_fmri[::2,:] = zscore2
    
    condition1 = np.array([1] * zscore1.shape[0]).reshape(-1,1)
    condition2 = np.array([2] * zscore2.shape[0]).reshape(-1,1)
    
    # grouped_labels = np.concatenate([condition1,condition2])
    grouped_labels = np.hstack((condition1,condition2)).flatten().reshape(-1,1)
    
    print('calculating')
    neg_log_pvals,t_scores_original_data,_ = permuted_ols(
        grouped_labels,
        grouped_fmri,
        n_perm = int(1e5),
        n_jobs = -1,
        verbose = 2,
        two_sided_test = True,
        random_state = 12345,
        )
    signed_neg_log_pvals = neg_log_pvals.copy()
    signed_neg_log_pvals[t_scores_original_data <= 0] = 0
    signed_neg_log_pvals_unmasked = masker.inverse_transform(signed_neg_log_pvals)
    signed_neg_log_pvals_unmasked.to_filename(
        os.path.abspath(os.path.join(univariate_test_dir,
                                      f'{conscious_state1}_{conscious_state2}.nii.gz')))
    
    # unconscious against zero
    unconscious_zero = np.zeros((zscore1.shape[0] + zscore2.shape[0],zscore2.shape[1]))
    unconscious_zero[1::2,:] = zscore2
    
    nlp_un,tsord_un,_ = permuted_ols(
        grouped_labels,
        unconscious_zero,
        n_perm = int(1e5),
        n_jobs = -1,
        verbose = 2,
        two_sided_test = True,
        random_state = 12345,
        )
    snlp = nlp_un.copy()
    snlp[tsord_un <= 0] = 0
    snlp_unmasked = masker.inverse_transform(snlp)
    snlp_unmasked.to_filename(
        os.path.abspath(os.path.join(univariate_test_dir,
                                      'unconscious_zero.nii.gz')))
    
    # save consicous  zscores
    conscious_z = os.path.abspath(os.path.join(univariate_test_dir,
                                                'conscious_z.nii.gz'))
    temp_plot = zscore1.mean(0).copy()
    temp_plot[temp_plot <= 0] = 0
    masker.inverse_transform(temp_plot).to_filename(conscious_z)
    
    # save unconcsious zscores
    unconscious_z = os.path.abspath(os.path.join(univariate_test_dir,
                                                'unconscious_z.nii.gz'))
    temp_plot = zscore2.mean(0).copy()
    temp_plot[temp_plot <= 0] = 0
    masker.inverse_transform(temp_plot).to_filename(unconscious_z)
    
    # save mean(z(c) - z(u))
    diff_z = os.path.abspath(os.path.join(univariate_test_dir,
                                          'conscious_unconscious_z.nii.gz'))
    temp_plot = np.mean(zscore1 - zscore2,axis = 0)
    temp_plot[temp_plot <= 0] = 0
    masker.inverse_transform(temp_plot).to_filename(diff_z)
    
    # p of differences
    diff_p = os.path.abspath(os.path.join(univariate_test_dir,
                                      f'{conscious_state1}_{conscious_state2}.nii.gz'))
    
    # p of unconscious
    unconscious_p = os.path.abspath(os.path.join(univariate_test_dir,
                                      'unconscious_zero.nii.gz'))
    
    # transformation
    for maps in [conscious_z,unconscious_z,diff_z,diff_p,unconscious_p]:
        flt = fsl.FLIRT()
        flt.inputs.in_file = maps
        flt.inputs.reference = os.path.abspath(standard_brain)
        flt.inputs.output_type = 'NIFTI_GZ'
        flt.inputs.in_matrix_file = transformation_dir_single
        flt.inputs.out_matrix_file = os.path.abspath(
            os.path.join(standarded_dri,f'{maps.split("/")[-1].split(".")[0]}_flirt.mat'))
        flt.inputs.out_file = os.path.abspath(
            os.path.join(standarded_dri,maps.split('/')[-1])
            )
        flt.inputs.apply_xfm = True
        res = flt.run()
    
    conscious_z = os.path.abspath(os.path.join(standarded_dri,
                                                'conscious_z.nii.gz'))
    unconscious_z = os.path.abspath(os.path.join(standarded_dri,
                                                'unconscious_z.nii.gz'))
    diff_z = os.path.abspath(os.path.join(standarded_dri,
                                          'conscious_unconscious_z.nii.gz'))
    diff_p = os.path.abspath(os.path.join(standarded_dri,
                                      f'{conscious_state1}_{conscious_state2}.nii.gz'))
    unconscious_p = os.path.abspath(os.path.join(standarded_dri,
                                      'unconscious_zero.nii.gz'))
    
    print('plotting')
    figure = plt.figure(figsize = (16 * 2,5 * 3))
    
    cut_coords = 0,0,0
    threshold = 1e-3
    
    # conscious
    ax = figure.add_subplot(231)
    display = plot_stat_map(conscious_z,
                            standard_brain,
                            cmap = plt.cm.RdBu_r,
                            draw_cross = False,
                            threshold = threshold,
                            cut_coords = cut_coords,
                            axes = ax,
                            figure = figure,
                            title = f'{conscious_state1} positive z scores',)
    
    # unconscious
    ax = figure.add_subplot(232)
    display = plot_stat_map(unconscious_z,
                            standard_brain,
                            cmap = plt.cm.RdBu_r,
                            draw_cross = False,
                            threshold = threshold,
                            cut_coords = cut_coords,
                            axes = ax,
                            figure = figure,
                            title = f'{conscious_state2} positive z scores',)
    
    # conscious > unconscious
    ax = figure.add_subplot(233)
    display = plot_stat_map(diff_z,
                            standard_brain,
                            cmap = plt.cm.RdBu_r,
                            draw_cross = False,
                            cut_coords = cut_coords,
                            axes = ax,
                            threshold = threshold,
                            title = f'Positive difference of z scores between {conscious_state1} and {conscious_state2}')
    
    # p value: conscious > unconscious
    ax = figure.add_subplot(223)
    p_thres = 1e-3
    threshold = -np.log10(p_thres) # % corrected
    
    display = plot_stat_map(diff_p,
                            standard_brain,
                            threshold = threshold,
                            cmap = plt.cm.RdBu_r,
                            draw_cross = False,
                            cut_coords = cut_coords,
                            black_bg = True,
                            figure = figure,
                            axes = ax,
                            title = f'Negative P values, {sub} {conscious_state1} > {conscious_state2}\nthresholding by log p critical = {threshold:.1f}')
    
    # p value: unconscious v.s. zero
    ax = figure.add_subplot(224)
    display = plot_stat_map(unconscious_p,
                            standard_brain,
                            threshold = threshold,
                            cmap = plt.cm.RdBu_r,
                            draw_cross = False,
                            cut_coords = cut_coords,
                            black_bg = True,
                            figure = figure,
                            axes = ax,
                            title = f'P values of positive correlation, {sub}\nunconscious against chance level')
    
    figure.subplots_adjust(wspace = 0,hspace = 0)
    
    figure.savefig(os.path.join(
        '/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/figures',
        f'RSA_searchlight_{sub}.png'),
        dpi = 400,
        bbox_inches = 'tight')
    figure.savefig(os.path.join(
        figure_dir,
        f'RSA_searchlight_{sub}.png'),
        dpi = 400,
        bbox_inches = 'tight')























