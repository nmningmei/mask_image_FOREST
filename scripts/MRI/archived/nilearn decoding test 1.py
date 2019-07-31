#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 21:12:49 2019

@author: nmei
"""

import os
from glob import glob
import numpy as np
import pandas as pd
from nilearn.input_data import NiftiMasker
from shutil import copyfile
copyfile('../utils.py','utils.py')
import utils

sub = 'sub-01'
parent_dir = '../../data/MRI/{}/func/session-*'.format(sub)

csvs = glob(os.path.join(parent_dir,
                         "*",
                         "*session*run*.csv"))
BOLDs = glob(os.path.join(parent_dir,
                          "*",
                          "*",
                          "ICA_AROMA",
                          "denoised_func_data_nonaggr.nii.gz"))
mask_img = '../../data/MRI/{}/func/session-02/sub-01_unfeat_run-01/FEAT.session2.run1.feat/mask.nii.gz'.format(sub)
csvs = np.sort(csvs)
BOLDs = np.sort(BOLDs)


df_events = []
fmri_data = []
for csv,BOLD in zip(csvs[:3],BOLDs):

    masker = NiftiMasker(
                         mask_img = mask_img,
                         t_r = 0.85,
                         smoothing_fwhm = None,
                         standardize = True,
                         detrend = True,
                         high_pass = 1)
    fmri_masked = masker.fit_transform(BOLD,)
    df = pd.read_csv(csv)
    
    idx_mask = df['volume_interest'] == 1
    
    fmri_masked_picked = fmri_masked[idx_mask]
    df_masked_picked = df[idx_mask].reset_index()
    
    
#    fmri = utils.groupy_average(fmri_masked_picked,
#                                df_masked_picked,
#                                groupby = ['trials'])
    
#    temp = []
#    for idx,df_sub in df_masked_picked.groupby(['trials']):
#        temp.append(df_sub.iloc[0,:].to_frame().T)
#    temp = pd.concat(temp)
    
    fmri_data.append(fmri_masked_picked)
    df_events.append(df_masked_picked)

fmri_data = np.concatenate(fmri_data)
df_events = pd.concat(df_events)




























