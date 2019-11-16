#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:24:55 2019

@author: nmei
"""

import os
from glob import glob
functional_dir = '../../data/MRI/sub-{}/func/session-{}/sub-01_unfeat_run-{}'

sub = '01'
session = '01'
run = '01'

os.path.exists(functional_dir.format(sub,session,run))
functional_data = glob(os.path.join(
                                    functional_dir.format(sub,session,"*"),
                                    '*.nii'))
functional_data = [os.path.abspath(item) for item in functional_data]
from nipype.workflows.fmri.fsl.preprocess import create_featreg_preproc
preproc = create_featreg_preproc(name='preprocessing',
                                 highpass = True,
                                 whichvol = 'middle',
                                 whichrun = 0)
preproc.inputs.inputspec.func = functional_data
preproc.inputs.inputspec.fwhm = 3
preproc.inputs.inputspec.highpass = 60
preproc.inputs.meanfuncmask.robust = True
preproc.base_dir = 'tmp/'
preproc.write_graph()
res = preproc.run()




from nipype.interfaces import fsl
from nipype.interfaces import utility as util
from nipype.pipeline import engine as pe
#fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
#preproc = pe.Workflow(name = 'fMRI_preprocessing')
#
#inputnode = pe.Node(
#        interface = util.IdentityInterface(
#                fields=['func','fwhm','anat']),
#                name = 'inputspec')
#outputnode = pe.Node(
#        interface = util.IdentityInterface(
#                fields = [
#                        'reference',
#                        'motion_parameters',
#                        'realigned_files',
#                        'motion_plots',
#                        'mask',
#                        'smoothed_files',
#                        'mean']),
#                name = 'outputspec')
## first step
#img2float = pe.MapNode(
#        interface=fsl.ImageMaths(
#                out_data_type='float',op_string='',suffix='_dtype'),
#                iterfield=['in_file'],
#                name = 'img2float')
#preproc.connect(inputnode,'func',img2float,'in_file')
## extract example fMRI volume
#extract_ref = pe.Node(interface=fsl.ExtractROI(t_size=1),
#                      iterfield=['in_file'],
#                      name = 'extractref')
#preproc.connect(img2float,('out_file',pickrun,whichrun),
#                extract_ref,'in_file')
#preproc.connect(img2float,('out_file',pickvol,0,whichvol),
#                extract_ref,'t_min')
#preproc.connect(extract_ref,'roi_file',
#                outputnode,'reference')
## motion correction
#motion_correct = pe.MapNode(
#        interface = fsl.MCFLIRT(
#                save_mats = True,
#                save_plots = True,
#                interpolation = 'spline'),
#                name = 'realign',
#                iterfield = ['in_file'])
#preproc.connect(img2float,'out_file',
#                motion_correct,'in_file',)
#preproc.connect(extract_ref,'roi_file',
#                motion_correct,'ref_file',)
#preproc.connect(motion_correct,'par_file',
#                outputnode,'motiion_parameters')
#preproc.connect(motion_correct,'out_file',
#                outputnode,'realigned_files')
## plot the estimated motion parameters
#plot_motion = pe.MapNode(
#        interface=fsl.PlotMotionParams(in_source='fsl'),
#        name = 'plot_motion',
#        iterfield = ['in_file'])
#plot_motion.iterables = ('plot_type',['rotations','translations'])
#preproc.connect(motion_correct,'par_file',
#                plot_motion,'in_file')
#preproc.connect(plot_motion,'out_file',
#                outputnode,'motion_plots')
## extract the mean volume of the first functional run
#meanfunc = pe.Node(
#        interface=fsl.ImageMaths(op_string = '-Tmean',suffix='_mean',
#                                 ),
#        name = 'meanfunc')
#preproc.connect(motion_correct,('out_file',pickrun,whichrun),
#                meanfunc,'in_file')
## strip the skull from the mean functional to generate a mask
#meanfuncmask = pe.Node(
#        interface=fsl.BET(mask=True,no_output=True,frac=0.3,robust=True,),
#        name='meanfuncmask')
#preproc.connect(meanfunc,'out_file',
#                meanfuncmask,'in_file')
## mask the functional runs with the extracted mask
#maskfunc = pe.MapNode(
#        interface=fsl.ImageMaths(suffix='_bet',op_string='-mas'),
#        iterfield=['in_file'],
#        name='maskfunc')
#preproc.connect(motion_correct,'out_file',
#                maskfunc,'in_file')
#preproc.connect(meanfuncmask,'mask_file',
#                maskfunc,'in_file2')























