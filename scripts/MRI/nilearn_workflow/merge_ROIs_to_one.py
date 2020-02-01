#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:10:24 2019

@author: nmei
"""

import os
import numpy as np
import pandas as pd
from glob import glob
from shutil import copyfile
copyfile('../../utils.py','utils.py')
from nipype.interfaces import freesurfer,fsl
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
from nibabel import load as load_mri
from nilearn.image import new_img_like

freesurfer_list = pd.read_csv('../../../FreesurferLTU.csv')

sub = 'sub-06'
first_session = 1
os.environ['SUBJECTS_DIR'] = os.path.abspath('../../../data/MRI/{}/'.format(sub))

working_dir = f'../../../data/MRI/{sub}/'
ROI_dir = os.path.join(working_dir,'anat/ROI_BOLD/')
ROIs_in_fsl_space = glob(os.path.join(ROI_dir,
                                      "*BOLD.nii.gz"))
ROIs_in_fsl_space = [item for item in ROIs_in_fsl_space if ("combine" not in item)]

data = []
for f in ROIs_in_fsl_space:
    temp = load_mri(f).get_data()
    data.append(temp)
data = np.array(data)

data = data.sum(0)
data[data > 0] = 1
temp = load_mri(glob(os.path.abspath(
                    os.path.join(working_dir,
                                 "func",
                                 f"session-0{first_session}",
                                 f"{sub}*run-01",
                                 "outputs",
                                 "func",
                                 "mask.nii.gz")))[0]).get_data()
data[temp < 1] = 0
new_mask = new_img_like(load_mri(ROIs_in_fsl_space[0]),
                        data = data,
                        affine = load_mri(ROIs_in_fsl_space[0]).affine)
new_mask.to_filename(os.path.join(ROI_dir,'combine_BOLD.nii.gz'))





























#merger = fsl.ImageMaths(in_file = ROIs_in_fsl_space[0],
#                        in_file2 = ROIs_in_fsl_space[1],
#                        op_string = '-add',
#                        out_file = os.path.abspath(
#                                    os.path.join(ROI_dir,
#                                    'TEMP_BOLD.nii.gz')))
#merger.run()
#binarize = fsl.ImageMaths(op_string = '-bin')
#binarize.inputs.in_file = merger.inputs.out_file
#binarize.inputs.out_file = os.path.abspath(
#                        os.path.join(ROI_dir,
#                        'TEMP_BOLD.nii.gz'))
#binarize.run()
#for file in ROIs_in_fsl_space[2:]:
#    merger = fsl.ImageMaths(in_file = os.path.abspath(
#                                    os.path.join(ROI_dir,
#                                    'TEMP_BOLD.nii.gz')),
#                            in_file2 = file,
#                            op_string = '-add',
#                            out_file = os.path.abspath(
#                                        os.path.join(ROI_dir,
#                                        'TEMP_BOLD.nii.gz')))
#    merger.run()
#    binarize = fsl.ImageMaths(op_string = '-bin')
#    binarize.inputs.in_file = merger.inputs.out_file
#    binarize.inputs.out_file = os.path.abspath(
#                            os.path.join(ROI_dir,
#                            'TEMP_BOLD.nii.gz'))
#    binarize.run()

#merger = fsl.ImageMaths(in_file = os.path.abspath(
#                        os.path.join(ROI_dir,
#                        'ctx-combined_ROIs_BOLD.nii.gz'),
#                                    ),
#                        in_file2 = glob(os.path.abspath(
#                                os.path.join(working_dir,
#                                             "func",
#                                             f"session-0{first_session}",
#                                             f"{sub}*run-01",
#                                             "outputs",
#                                             "func",
#                                             "mask.nii.gz")))[0],
#                        out_file = os.path.abspath(
#                        os.path.join(ROI_dir,
#                        'ctx-combined_ROIs_BOLD.nii.gz')),
#                        op_string = '-add')
#merger.run()
#binarize = fsl.ImageMaths(op_string = '-bin')
#binarize.inputs.in_file = merger.inputs.out_file
#binarize.inputs.out_file = os.path.abspath(
#                        os.path.join(ROI_dir,
#                        'ctx-combined_ROIs_BOLD.nii.gz'))














