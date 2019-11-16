#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:24:55 2019

@author: nmei

A nipype pipeline: motion correction --> smooth --> masking

No highpass filer
"""

import os
from glob import glob
from shutil import copyfile
import numpy as np
copyfile('../utils.py','utils.py')
from utils import create_fsl_FEAT_workflow_func

functional_dir = '../../data/MRI/{}/func/session-{}/{}_unfeat_run-{}'

output_dir = '../../data/MRI/{}/func/session-{}/{}_unfeat_run-{}/outputs/func'
MC_dir = os.path.join(output_dir,'MC')
sub = 'sub-01'
session = '02'
run = '01'

output_dir = output_dir.format(sub,session,sub,run)
MC_dir = MC_dir.format(sub,session,sub,run)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(MC_dir):
    os.mkdir(MC_dir)
functional_data = glob(os.path.join(
                                    functional_dir.format(sub,session,sub,"*"),
                                    '*.nii'))
functional_data = [os.path.abspath(item) for item in np.sort(functional_data)]
# if the reference exists
ref_dir = '../../data/MRI/{}/func/session-{}/{}_unfeat_run-{}/outputs/func/example_func.nii.gz'
first_run = True # os.path.abspath(ref_dir.format(sub,'02',sub,'01'))
workflow_name = 'nipype_mimic_FEAT'
preproc,_,_ = create_fsl_FEAT_workflow_func(workflow_name = workflow_name,
                                        first_run = first_run)

# initialize some of the input files
preproc.inputs.inputspec.func = functional_data[int(run) - 1]
preproc.inputs.inputspec.fwhm = 3
preproc.base_dir = os.path.abspath(functional_dir.format(sub,session,sub,run))

# initialize all the output files
if first_run == True:
    preproc.inputs.extractref.roi_file = os.path.abspath(os.path.join(output_dir,
                                                                       'example_func.nii.gz'))
else:
    copyfile(first_run,output_dir,'example_func.nii.gz')

preproc.inputs.dilatemask.out_file = os.path.abspath(os.path.join(output_dir,
                                                              'mask.nii.gz'))
preproc.inputs.meanscale.out_file = os.path.abspath(os.path.join(output_dir,
                                                                        'prefiltered_func.nii.gz'))
preproc.inputs.gen_mean_func_img.out_file = os.path.abspath(os.path.join(output_dir,
                                                              'mean_func.nii.gz'))
preproc.write_graph()
res = preproc.run()

# moving MCflirt to MC folder in output directory
copyfile(glob(os.path.join(preproc.base_dir,
                      workflow_name,
                      'MCFlirt/mapflow/_MCFlirt0/',
                      '*.par'))[0],
        os.path.join(MC_dir,'MCflirt.par'))
copyfile(glob(os.path.join(preproc.base_dir,
                      workflow_name,
                      'MCFlirt/mapflow/_MCFlirt0/',
                      '*rot*'))[0],
        os.path.join(MC_dir,'rot.png'))
copyfile(glob(os.path.join(preproc.base_dir,
                      workflow_name,
                      'MCFlirt/mapflow/_MCFlirt0/',
                      '*trans*'))[0],
        os.path.join(MC_dir,'trans.png'))
copyfile(glob(os.path.join(preproc.base_dir,
                      workflow_name,
                      'MCFlirt/mapflow/_MCFlirt0/',
                      '*disp*'))[0],
        os.path.join(MC_dir,'disp.png'))



