#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 06:27:30 2020

@author: nmei
"""

import os
from glob import glob

from nipype.interfaces import fsl
from itertools import combinations
from nilearn.input_data import NiftiMasker
from nilearn.image import smooth_img

sub = 'sub-01'
working_dir = f'../../../../results/MRI/nilearn/RSA_searchlight_corrected/{sub}'
working_data = glob(os.path.join(working_dir,'*.nii.gz'))
mask_dir = f'/bcbl/home/home_n-z/nmei/MRI/uncon_feat/MRI/{sub}/anat/ROI_BOLD/'
mask_file = glob(os.path.join(mask_dir,'combine_BOLD.nii.gz'))[0]
tfce_dir = f'../../../../results/MRI/nilearn/tfce/{sub}'
if not os.path.exists(tfce_dir):
    os.makedirs(tfce_dir)


for file_name1,file_name2 in combinations(working_data, 2):
    conscious_state1 = file_name1.split('/')[-1].replace('.nii.gz','')
    conscious_state2 = file_name2.split('/')[-1].replace('.nii.gz','')
    print(conscious_state1,conscious_state2)
    
    masker = NiftiMasker(mask_img = mask_file)
    temp1 = masker.fit_transform(file_name1)
    temp2 = masker.fit_transform(file_name2)
    diff = temp1 - temp2
    del temp1,temp2
    
    difference = masker.inverse_transform(diff)
    print('smoothing')
    diff_smoothed = smooth_img(difference, 6)
    diff_smoothed.to_filename(os.path.join(tfce_dir,
                                        f'{conscious_state1}_{conscious_state2}_smoothed.nii.gz'))
    rand                        = fsl.Randomise()
    rand.inputs.in_file         = os.path.join(tfce_dir,
                                        f'{conscious_state1}_{conscious_state2}_smoothed.nii.gz')
    rand.inputs.mask            = os.path.abspath(mask_file)
    rand.inputs.tfce            = True
    rand.inputs.var_smooth      = 6
    rand.inputs.base_name       = os.path.abspath(os.path.join(tfce_dir,'{}_{}_tfce'.format(
                                    conscious_state1,conscious_state2)))
    rand.inputs.one_sample_group_mean = True
    rand.inputs.num_perm        = int(1e3)
    rand.inputs.seed            = 12345
    rand.cmdline
    rand.run()
    assdf
































