#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:55:45 2019

@author: nmei

We are doing search light here
"""

import os
from glob import glob
import pandas as pd
import numpy as np
from shutil                  import copyfile
copyfile('../../../utils.py','utils.py')
from utils import LOO_partition,build_model_dictionary,plot_stat_map
from nibabel import load as load_fmri
from sklearn.metrics         import roc_auc_score
from nilearn.decoding        import SearchLight

from matplotlib                    import pyplot as plt


sub                 = 'sub-02'
first_session       = 1
stacked_data_dir    = '../../../../data/BOLD_whole_brain_averaged/{}/'.format(sub)
mask_dir            = '../../../../data/MRI/{}/anat/ROI_BOLD/combine_BOLD.nii.gz'.format(sub)
saving_dir          = '../../../../results/MRI/nilearn/searchlight/{}'.format(sub)
label_map           = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}
average             = True
n_splits            = 'LOO'
n_jobs              = -1

for conscious_state in ['unconscious','glimpse','conscious']:
    df_data                 = pd.read_csv(os.path.join(stacked_data_dir,
                                               f'whole_brain_{conscious_state}.csv'))
    df_data['id']           = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
    idxs_train,idxs_test    = LOO_partition(df_data,target_column = 'labels')
    
    
    BOLD_file       = os.path.join(stacked_data_dir,
                                   f'whole_brain_{conscious_state}.nii.gz')
    BOLD_image      = load_fmri(BOLD_file)
    targets         = np.array([label_map[item] for item in df_data['targets']])[:,-1]
    groups          = df_data['labels'].values
    
    decoder         = build_model_dictionary(remove_invariant = False)[
                                    'None + Linear-SVM']
    searchlight = SearchLight(mask_img = mask_dir,
                              process_mask_img = mask_dir,
                              radius = 6 * np.sqrt(3),
                              estimator = decoder,
                              n_jobs = n_jobs,
                              scoring = 'roc_auc',
                              cv = None,#zip(idxs_train[:3],idxs_test),
                              verbose = 1,
                              )
    searchlight.fit(BOLD_image,targets)

from nilearn.image import new_img_like
c = new_img_like(BOLD_image,searchlight.scores_)

example_func = glob(os.path.join(f'../../../../data/MRI/{sub}/func',
                                '*',
                                "*",
                                "*",
                                "*",
                                'example_func.nii.gz'))[0]
from scipy.stats import scoreatpercentile
fig,ax = plt.subplots(figsize = (10,6))
plot_stat_map(c,
               bg_img = example_func,
               threshold = 0.1,
               axes = ax,
               figure = fig,
               draw_cross = False,
               cmap = plt.cm.coolwarm,
               vmin_ = .5,
               vmax = scoreatpercentile(searchlight.scores_.flatten(),99),
               )









