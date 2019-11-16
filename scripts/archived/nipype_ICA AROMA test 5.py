#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:22:04 2019

@author: nmei
"""

import os
from glob import glob
from nipype.interfaces.fsl import ICA_AROMA
import re

sub = 'sub-01'
MRI_dir = '../../data/MRI/{}/func'.format(sub)
parent_dir = os.path.join(MRI_dir,'session-*','*','outputs','func')
preprocessed_BOLD_files = glob(os.path.join(parent_dir,
                                            'prefiltered_func.nii.gz'))
sample = preprocessed_BOLD_files[0]


AROMA_obj = ICA_AROMA()
parent_dir = '/'.join(sample.split('/')[:-2])
n_session = int(re.findall('\d+',re.findall('session-[\d+][\d+]',sample)[0])[0])
n_run = int(re.findall('\d+',re.findall('run-[\d+][\d+]',sample)[0])[0])
print(n_session,n_run)
func_to_struct = os.path.join(parent_dir,
                              'reg',
                              'example_func2highres.mat')
warpfield = os.path.join(parent_dir,
                         'reg',
                         'highres2standard_warp.nii.gz')
fsl_mcflirt_movpar = os.path.join(parent_dir,
                                  'func',
                                  'MC',
                                  'MCflirt.par')
mask = os.path.join(parent_dir,
                    'func',
                    'mask.nii.gz')
output_dir = os.path.join(parent_dir,
                          'func',
                          'ICA_AROMA')
AROMA_obj.inputs.in_file = os.path.abspath(sample)
AROMA_obj.inputs.mat_file = os.path.abspath(func_to_struct)
AROMA_obj.inputs.fnirt_warp_file = os.path.abspath(warpfield)
AROMA_obj.inputs.motion_parameters = os.path.abspath(fsl_mcflirt_movpar)
AROMA_obj.inputs.mask = os.path.abspath(mask)
AROMA_obj.inputs.denoise_type = 'nonaggr'
AROMA_obj.inputs.out_dir = os.path.abspath(output_dir)
cmdline = 'python ' + AROMA_obj.cmdline + ' -ow' # overwrite if exists ICA_AROMA results

os.system(cmdline)










