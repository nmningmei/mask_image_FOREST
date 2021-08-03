#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 09:48:07 2020

@author: nmei
"""

import os
import gc

from glob import glob
from tqdm import tqdm

import numpy as np

from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from nipype.interfaces import fsl

working_dir = '../../../../results/MRI/nilearn/decode_searchlight_LOSO'

for sub in [1,2,3,4,5,6,7]:
    mask_dir = f'../../../../data/MRI/sub-0{sub}/anat/ROI_BOLD/'
    mask_file = glob(os.path.join(mask_dir,'combine_BOLD.nii.gz'))[0]
    
    standard_brain = '../../../../data/standard_brain/MNI152_T1_2mm_brain.nii.gz'
    output_dir = f'../../../../results/MRI/nilearn/decoding_searchlight/sub-0{sub}'
    transformation_dir_single = os.path.abspath(
                    f'../../../../data/MRI/sub-0{sub}/reg/example_func2standard.mat')
    standarded_dir = f'../../../../results/MRI/nilearn/decoding_searchlight_standarded/sub-0{sub}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(standarded_dir):
        os.makedirs(standarded_dir)
    for conscious_source in ['unconscious','conscious']:
        for conscious_target in ['unconscious','conscious']:
            working_data = glob(os.path.join(
                working_dir,
                f'sub-0{sub}',
                f'{conscious_source}_{conscious_target}*.nii.gz'))
            svm = np.sort([item for item in working_data if ('SVM' in item)])
            chc = np.sort([item for item in working_data if ('SVM' not in item)])
            filename = os.path.join(
                output_dir,f'sub-0{sub}_{conscious_source}_{conscious_target}.nii.gz')
            if not os.path.exists(filename):
                masker = NiftiMasker(mask_img = mask_file,)
                temp1 = masker.fit_transform(svm)
                
                masker = NiftiMasker(mask_img = mask_file,)
                temp2 = masker.fit_transform(chc)
                
                zscore1 = temp1.copy()
                zscore2 = temp2.copy()
                
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
                    two_sided_test = False,
                    random_state = 12345,
                    )
                signed_neg_log_pvals = neg_log_pvals.copy()
                signed_neg_log_pvals[t_scores_original_data <= 0] = 0
                signed_neg_log_pvals_unmasked = masker.inverse_transform(signed_neg_log_pvals)
                signed_neg_log_pvals_unmasked.to_filename(filename)
                gc.collect()
                print('standardizing')
                flt = fsl.FLIRT()
                flt.inputs.in_file = os.path.abspath(filename)
                flt.inputs.reference = os.path.abspath(standard_brain)
                flt.inputs.output_type = 'NIFTI_GZ'
                flt.inputs.in_matrix_file = transformation_dir_single
                flt.inputs.out_matrix_file = os.path.abspath(
                    os.path.join(standarded_dir,
                                 f'sub-0{sub}_{conscious_source}_{conscious_target}_flirt.mat'))
                flt.inputs.out_file = os.path.abspath(
                    os.path.join(standarded_dir,f'sub-0{sub}_{conscious_source}_{conscious_target}.nii.gz')
                    )
                flt.inputs.apply_xfm = True
                res = flt.run()

























