#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:55:34 2019

@author: nmei

transform ROIs from structual space to BOLD space


"""
import os
from glob import glob
from nipype.interfaces import fsl
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as util
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')


sub = 'sub-01'
anat_dir = '../../data/MRI/{}/anat'.format(sub)
ROI_in_structural = glob(os.path.join(anat_dir,'ROIs','*fsl.nii.gz'))

first_session = 2
preprocessed_functional_dir = '../../data/MRI/{}/func/session-0{}/{}_unfeat_run-01/outputs'.format(
        sub,first_session,sub)

output_dir = os.path.join(anat_dir,'ROI_BOLD')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for roi in ROI_in_structural:
    roi = os.path.abspath(roi)
    roi_name = roi.split('/')[-1]
    simple_workflow = pe.Workflow(name = 'struc2BOLD')
    
    inputnode = pe.Node(interface = util.IdentityInterface(
                    fields=['flt_in_file',
                            'flt_in_matrix',
                            'flt_reference',
                            'mask']),
                    name = 'inputspec')
    outputnode = pe.Node(interface = util.IdentityInterface(
                    fields=['BODL_mask']),
                    name = 'outputspec')
    """
     flirt 
 -in /export/home/dsoto/dsoto/fmri/$s/sess2/label/$i 
 -ref /export/home/dsoto/dsoto/fmri/$s/sess2/run1_prepro1.feat/example_func.nii.gz  
 -applyxfm 
 -init /export/home/dsoto/dsoto/fmri/$s/sess2/run1_prepro1.feat/reg/highres2example_func.mat 
 -out  /export/home/dsoto/dsoto/fmri/$s/label/BOLD${i}
    """
    flirt_convert = pe.MapNode(
            interface = fsl.FLIRT(apply_xfm = True),
            iterfield = ['in_file','reference','in_matrix_file'],
            name = 'flirt_convert')
    simple_workflow.connect(inputnode,'flt_in_file',flirt_convert,'in_file')
    simple_workflow.connect(inputnode,'flt_reference',flirt_convert,'reference')
    simple_workflow.connect(inputnode,'flt_in_matrix',flirt_convert,'in_matrix_file')
    
    """
     fslmaths /export/home/dsoto/dsoto/fmri/$s/label/BOLD${i} -mul 2 
     -thr `fslstats /export/home/dsoto/dsoto/fmri/$s/label/BOLD${i} -p 99.6` 
    -bin /export/home/dsoto/dsoto/fmri/$s/label/BOLD${i}
    """
    def getthreshop(thresh):
        return ['-mul 2 -thr %.10f -bin' % (val) for val in thresh]
    getthreshold = pe.MapNode(
            interface=fsl.ImageStats(op_string='-p 99.6'),
            iterfield = ['in_file','mask_file'],
            name='getthreshold')
    simple_workflow.connect(flirt_convert,'out_file',getthreshold,'in_file')
    simple_workflow.connect(inputnode,'mask',getthreshold,'mask_file')
    
    threshold = pe.MapNode(
            interface=fsl.ImageMaths(suffix='_thresh',
                                     op_string = '-mul 2 -bin'),
            iterfield=['in_file','op_string'],
            name='thresholding')
    simple_workflow.connect(flirt_convert,'out_file',threshold,'in_file')
    simple_workflow.connect(getthreshold,('out_stat',getthreshop),threshold,'op_string')
#    simple_workflow.connect(threshold,'out_file',outputnode,'BOLD_mask')
    
    bound_by_mask = pe.MapNode(
            interface = fsl.ImageMaths(suffix='_mask',op_string='-mas'),
            iterfield=['in_file','in_file2'],
            name = 'bound_by_mask')
    simple_workflow.connect(threshold,'out_file',bound_by_mask,'in_file')
    simple_workflow.connect(inputnode,'mask',bound_by_mask,'in_file2')
    simple_workflow.connect(bound_by_mask,'out_file',outputnode,'BOLD_mask')
    
    # setup inputspecs 
    simple_workflow.inputs.inputspec.flt_in_file = roi
    simple_workflow.inputs.inputspec.flt_in_matrix = os.path.abspath(os.path.join(preprocessed_functional_dir,
                                                        'reg',
                                                        'highres2example_func.mat'))
    simple_workflow.inputs.inputspec.flt_reference = os.path.abspath(os.path.join(preprocessed_functional_dir,
                                                        'func',
                                                        'example_func.nii.gz'))
    simple_workflow.inputs.inputspec.mask = os.path.abspath(os.path.join(preprocessed_functional_dir,
                                                        'func',
                                                        'mask.nii.gz'))
    simple_workflow.inputs.bound_by_mask.out_file = os.path.abspath(os.path.join(output_dir,
                                                             roi_name.replace('_fsl.nii.gz',
                                                                              '_BOLD.nii.gz')))
    simple_workflow.base_dir = os.path.abspath(output_dir)
    simple_workflow.write_graph(dotfilename='{}.dot'.format(roi_name.split('.')[0]))
    simple_workflow.run()
    





















#    flt = fsl.FLIRT()
#    flt.inputs.in_file = roi
#    flt.inputs.reference = os.path.abspath(os.path.join(preprocessed_functional_dir,
#                                                        'func',
#                                                        'example_func.nii.gz'))
#    flt.inputs.output_type = 'NIFTI_GZ'
#    flt.inputs.in_matrix_file = os.path.abspath(os.path.join(preprocessed_functional_dir,
#                                                        'reg',
#                                                        'highres2example_func.mat'))
#    flt.inputs.out_file = os.path.abspath(os.path.join(output_dir,
#                                                       roi.split('/')[-1].replace(
#                                                               '_fsl.nii.gz','_BOLD.nii.gz')))
#    flt.inputs.apply_xfm = True
#    print(flt.cmdline)
#    flt.run()
