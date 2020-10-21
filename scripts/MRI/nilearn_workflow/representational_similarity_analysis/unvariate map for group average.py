#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 03:29:34 2020

@author: nmei
"""
import os
import numpy as np
from glob import glob

from nilearn.input_data import NiftiMasker
from nilearn.image      import new_img_like
from nibabel            import load as load_fmri
from nilearn.mass_univariate import permuted_ols
from nilearn.plotting import plot_stat_map

from matplotlib import pyplot as plt

standard_brain = '../../../../data/standard_brain/MNI152_T1_2mm_brain.nii.gz'
standarded_dri = f'../../../../results/MRI/nilearn/RSA_searchlight_standarded_full/*'
figure_dir = f'../../../../figures/MRI/nilearn/RSA_searchlight_full/'

conscious_state1 = 'conscious'
conscious_state2 = 'unconscious'

conscious_z = glob(os.path.abspath(os.path.join(standarded_dri,
                                            'conscious_z.nii.gz')))
unconscious_z = glob(os.path.abspath(os.path.join(standarded_dri,
                                            'unconscious_z.nii.gz')))
diff_z = glob(os.path.abspath(os.path.join(standarded_dri,
                                      'conscious_unconscious_z.nii.gz')))


print('plotting')
figure,axes = plt.subplots(figsize = (16,5 * 3),
                           nrows = 3)

cut_coords = 0,0,0
threshold = 1e-3

# conscious
ax = axes[0]
temp = np.mean([np.asanyarray(load_fmri(item).dataobj) for item in conscious_z],0)
temp = new_img_like(load_fmri(standard_brain),temp)
display = plot_stat_map(temp,
                        standard_brain,
                        cmap = plt.cm.RdBu_r,
                        draw_cross = False,
                        threshold = threshold,
                        cut_coords = cut_coords,
                        axes = ax,
                        figure = figure,
                        title = f'{conscious_state1} positive z scores',)

# unconscious
ax = axes[1]
temp = np.mean([np.asanyarray(load_fmri(item).dataobj) for item in unconscious_z],0)
temp = new_img_like(load_fmri(standard_brain),temp)
display = plot_stat_map(temp,
                        standard_brain,
                        cmap = plt.cm.RdBu_r,
                        draw_cross = False,
                        threshold = threshold,
                        cut_coords = cut_coords,
                        axes = ax,
                        figure = figure,
                        title = f'{conscious_state2} positive z scores',)

# conscious > unconscious
ax = axes[2]
temp = np.mean([np.asanyarray(load_fmri(item).dataobj) for item in diff_z],0)
temp = new_img_like(load_fmri(standard_brain),temp)
display = plot_stat_map(temp,
                        standard_brain,
                        cmap = plt.cm.RdBu_r,
                        draw_cross = False,
                        cut_coords = cut_coords,
                        axes = ax,
                        threshold = threshold,
                        title = f'Positive difference of z scores between {conscious_state1} and {conscious_state2}')

figure.subplots_adjust(wspace = 0,hspace = 0)

figure.savefig(os.path.join(
    '/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/figures',
    f'RSA_searchlight_sub_ave_full.png'),
    dpi = 400,
    bbox_inches = 'tight')
figure.savefig(os.path.join(
    figure_dir,
    f'RSA_searchlight_sub_ave.png'),
    dpi = 400,
    bbox_inches = 'tight')