#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:19:27 2019

@author: nmei

1. make a full processing pipeline for fmri data
2. if the fmri is the first run and being the reference one, extract the example_func and 
    coregistrate it to the structural scan

"""

import os
import numpy as np
import re
from glob           import glob
from shutil         import copyfile,rmtree
copyfile('../../utils.py',
         'utils.py')
from utils          import (create_fsl_FEAT_workflow_func,
                            create_registration_workflow,
                            registration_plotting)


sub             = 'sub-01' #specify subject name/code
first_session   = '02' # which session is the very first functional session, to which we align the rest
func_dir        = '../../../data/MRI/{}/func/'.format(sub) # define the parent directory of the functional data

# because for one subject we remove the first session, thus, there is "wrong" in the name of that session
# which is excluded
func_data       = [item for item in glob(os.path.join(func_dir,
                                                      '*',
                                                      '*',
                                                      '*.nii')) if ('wrong' not in item)]
func_data       = np.sort(func_data)

# pick the functional data
func_data_file = '../../../data/MRI/sub-01/func/session-06/sub-01_unfeat_run-08/sub-01_unfeat_run-08_bold.nii'
# get the number of session (1 to 6) and number of run (1 to 9)
temp            = re.findall('\d+',func_data_file)
n_session       = int(temp[1])
n_run           = int(temp[-1])


# create a nipype workflow for functional preprocessing
nipype_workflow_name = 'nipype_workflow'
if (n_session == int(first_session)) and (n_run == 1):
    # if it is the first run of the first session, we confirm this is the reference run
    first_run   = True
else:
    # otherwise, we specify the reference volume, to which we algin the rest of the runs
    first_run   = os.path.abspath(os.path.join(func_dir,
                                               'session-{}'.format(first_session),
                                               '{}_unfeat_run-01'.format(sub),
                                               'outputs',
                                               'func',
                                               'example_func.nii.gz'))
# initialize the workflow and specify the hyperparameters
# workflowname:     will create a folder that contain all the logs
# first_run:        specified above
# func_data_file:   the path to the functional data, .nii format
# fwhm:             spatial smoothing size
preproc,MC_dir,output_dir = create_fsl_FEAT_workflow_func(
        workflow_name       = nipype_workflow_name,
        first_run           = first_run,
        func_data_file      = os.path.abspath(func_data_file),
        fwhm                = 3,
        )
# make a figure of the workflow
preproc.write_graph()
# run the workflow
res             = preproc.run()

# moving MCflirt results to MC folder in output directory
copyfile(glob(os.path.join(preproc.base_dir,
                           nipype_workflow_name,
                           'MCFlirt/mapflow/_MCFlirt0/',
                           '*.par'))[0],
         os.path.join(MC_dir,'MCflirt.par'))
copyfile(glob(os.path.join(preproc.base_dir,
                           nipype_workflow_name,
                           'MCFlirt/mapflow/_MCFlirt0/',
                           '*rot*'))[0],
         os.path.join(MC_dir,'rot.png'))
copyfile(glob(os.path.join(preproc.base_dir,
                           nipype_workflow_name,
                           'MCFlirt/mapflow/_MCFlirt0/',
                           '*trans*'))[0],
         os.path.join(MC_dir,'trans.png'))
copyfile(glob(os.path.join(preproc.base_dir,
                           nipype_workflow_name,
                           'MCFlirt/mapflow/_MCFlirt0/',
                           '*disp*'))[0],
         os.path.join(MC_dir,'disp.png'))
copyfile(glob(os.path.join(preproc.base_dir,
                           nipype_workflow_name,
                           'graph.png'))[0],
         os.path.join(output_dir,'session {} run {}.png'.format(n_session,n_run)))

for log_file in glob(os.path.join(preproc.base_dir,"*","*","*","*","*",'report.rst')):
    log_name = log_file.split('/')[-5]
    copyfile(log_file,os.path.join(output_dir,'log_{}.rst'.format(log_name)))

# remove the logs
rmtree(os.path.join(preproc.base_dir,
                    nipype_workflow_name))
# registration if only the first sesssion first run
if first_run == True:
    # define the parent path of the structural scan and standard brain scans
    anat_dir            = '../../../data/MRI/{}/anat/'
    standard_brain      = '/opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
    standard_head       = '/opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm.nii.gz'
    standard_mask       = '/opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz'
    # specify the path of the structural scan with and without the skull
    anat_brain          = os.path.abspath(glob(os.path.join(anat_dir.format(sub),'*brain*'))[0]) # BET
    anat_head           = os.path.abspath(glob(os.path.join(anat_dir.format(sub),'*6.nii'))[0])
    # the so-called "example_func.nii.gz"
    func_ref            = os.path.join(preproc.base_dir,
                                             'outputs',
                                             'func',
                                             'example_func.nii.gz')
    # define the output path for saving the coregistration results
    output_dir          = '../../../data/MRI/{}/func/session-{}/{}_unfeat_run-{}/outputs/reg'.format(
                                sub,
                                first_session,
                                sub,
                                '01')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # create the registration workflow
    # anat_brain    : path of the structural scan after BET
    # anat_head     : path of the structural scan before BET
    # func_ref      : the so-called "example_func.nii.gz'
    # standard_brain: MNI brain after BET
    # standard_head : MNI brain before BET
    # standard_mask : mask of BET for MNI brain
    registration        = create_registration_workflow(
                                anat_brain,
                                anat_head,
                                func_ref,
                                standard_brain,
                                standard_head,
                                standard_mask,
                                workflow_name = 'registration',
                                output_dir = output_dir)
    registration.write_graph()
    registration.run()
    
    
    # plot the results
    registration_plotting(output_dir,
                          anat_brain,
                          standard_brain)











































