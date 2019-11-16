#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:01:54 2019

@author: nmei

get rois

"""
import os
import pandas as pd
from nipype.interfaces import freesurfer,fsl
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

freesurfer_list = pd.read_csv('../../FreesurferLTU.csv')

sub = 'sub-01'
os.environ['SUBJECTS_DIR'] = os.path.abspath('../../data/MRI/{}/'.format(sub))
in_file = os.path.abspath('../../data/MRI/{}/anat/{}/mri/aparc+aseg.mgz'.format(sub,sub))
original = os.path.abspath('../../data/MRI/{}/anat/{}/mri/orig/001.mgz'.format(sub,sub))
ROI_anat_dir = '../../data/MRI/{}/anat/ROIs'.format(sub)
if not os.path.exists(ROI_anat_dir):
    os.mkdir(ROI_anat_dir)

roi_names = """fusiform
inferiorparietal
superiorparietal
inferiortemporal
lateraloccipital
lingual
parahippocampal
pericalcarine
precuneus
superiorfrontal
parsopercularis
parsorbitalis
parstriangularis
rostralmiddlefrontal"""
idx_label,label_names = [],[]
for name in roi_names.split('\n'):
    for label_name in freesurfer_list['Label Name']:
        if (name in label_name) and ('ctx' in label_name):
            idx = freesurfer_list[freesurfer_list['Label Name'] == label_name]['#No.'].values[0]
            if str(idx)[1] == '0':
                idx_label.append(idx)
                label_names.append(label_name)

for idx,label_name in zip(idx_label,label_names):
    binary_file = os.path.abspath(os.path.join(ROI_anat_dir,'{}.nii.gz'.format(label_name)))
    binarizer = freesurfer.Binarize(in_file = in_file,
                                    match = [idx],
                                    binary_file = binary_file)
    print(binarizer.cmdline)
    binarizer.run()
    fsl_swapdim = fsl.SwapDimensions(new_dims = ('x', 'z', '-y'),)
    fsl_swapdim.inputs.in_file = binarizer.inputs.binary_file
    fsl_swapdim.inputs.out_file = binarizer.inputs.binary_file.replace('.nii.gz','_fsl.nii.gz')
    print(fsl_swapdim.cmdline)
    fsl_swapdim.run()
    mc = freesurfer.MRIConvert()
    mc.inputs.in_file = fsl_swapdim.inputs.out_file
    mc.inputs.reslice_like = original
    mc.inputs.out_file = fsl_swapdim.inputs.out_file
    print(mc.cmdline)
    mc.run()
    
    

















