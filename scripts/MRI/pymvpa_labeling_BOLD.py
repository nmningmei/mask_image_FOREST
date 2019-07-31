#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:01:02 2019

@author: nmei
"""

import os
from glob import glob
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.datasets.miscfx import summary

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

print_summary = False

csvs = np.sort(csvs)
BOLDs = np.sort(BOLDs)

for ii,(csv,BOLD) in tqdm(enumerate(zip(csvs[:1],BOLDs))):
    mri_data = fmri_dataset(samples = BOLD,mask = mask_img,chunks=ii)
    vol_times = mri_data.sa.time_coords
    df = pd.read_csv(csv)
    for col_name in df.columns:
        mri_data.sa[col_name] = df[col_name].values
    mri_data = mri_data[df['volume_interest'] == 1]
    if print_summary:
        print(summary(mri_data))








































