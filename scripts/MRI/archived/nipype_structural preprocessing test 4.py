#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:40:03 2019

@author: nmei

a nipype pipeline to coregistration among example_func.nii.gz, structual.nii.gz, and standard MNI

"""
import os
from glob import glob
from shutil import copyfile
copyfile('../utils.py','utils.py')


sub = 'sub-01'
# define the parent path of the structural scan and standard brain scans
anat_dir            = '../../../data/MRI/{}/anat/'
standard_brain      = '/opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
standard_head       = '/opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm.nii.gz'
standard_mask       = '/opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz'
# specify the path of the structural scan with and without the skull
anat_brain          = os.path.abspath(glob(os.path.join(anat_dir.format(sub),'*brain*'))[0]) # BET
anat_head           = os.path.abspath(glob(os.path.join(anat_dir.format(sub),'*t1*.nii*'))[0])
#
session = '02'
run = '01'
example_func            = os.path.abspath(os.path.join(
                        '../../../data/MRI',
                        sub,
                        'func',
                        'session-{}'.format(session),
                        '{}_unfeat_run-{}'.format(sub,run),
                        'outputs',
                        'func',
                        'example_func.nii.gz'))


output_dir = '../../../data/MRI/{}/func/session-{}/sub-01_unfeat_run-{}/test_flow/reg'.format(
        sub,session,run)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

from nipype.interfaces          import fsl
from nipype.interfaces         import utility as util
from nipype.pipeline           import engine as pe
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
registration                    = pe.Workflow(name = 'registration')
inputnode                       = pe.Node(
                                    interface   = util.IdentityInterface(
                                    fields      = [
                                            'highres', # anat_brain
                                            'highres_head', # anat_head
                                            'example_func',
                                            'standard', # standard_brain
                                            'standard_head',
                                            'standard_mask'
                                            ]),
                                    name        = 'inputspec')
outputnode                      = pe.Node(
                                interface   = util.IdentityInterface(
                                fields      = ['example_func2highres_nii_gz',
                                               'example_func2highres_mat',
                                               'linear_example_func2highres_log',
                                               'highres2example_func_mat',
                                               'highres2standard_linear_nii_gz',
                                               'highres2standard_mat',
                                               'linear_highres2standard_log',
                                               'highres2standard_nii_gz',
                                               'highres2standard_warp_nii_gz',
                                               'highres2standard_head_nii_gz',
#                                               'highres2standard_apply_warp_nii_gz',
                                               'highres2highres_jac_nii_gz',
                                               'nonlinear_highres2standard_log',
                                               'highres2standard_nii_gz',
                                               'standard2highres_mat',
                                               'example_func2standard_mat',
                                               'example_func2standard_warp_nii_gz',
                                               'example_func2standard_nii_gz',
                                               'standard2example_func_mat',
                                               ]),
                                name        = 'outputspec')
"""
fslmaths /bcbl/home/public/Consciousness/uncon_feat/data/MRI/sub-01/anat/sub-01-T1W_mprage_sag_p2_1iso_MGH_day_6_nipy_brain highres
fslmaths /bcbl/home/public/Consciousness/uncon_feat/data/MRI/sub-01/anat/sub-01-T1W_mprage_sag_p2_1iso_MGH_day_6_nipy_brain  highres_head
fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain standard
fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm standard_head
fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil standard_mask
"""
# skip

"""
/opt/fsl/fsl-5.0.10/fsl/bin/flirt 
    -in example_func 
    -ref highres 
    -out example_func2highres 
    -omat example_func2highres.mat 
    -cost corratio 
    -dof 7 
    -searchrx -180 180 
    -searchry -180 180 
    -searchrz -180 180 
    -interp trilinear 
"""
linear_example_func2highres = pe.MapNode(
        interface   = fsl.FLIRT(cost = 'corratio',
                                interp = 'trilinear',
                                dof = 7,
                                save_log = True,
                                searchr_x = [-180, 180],
                                searchr_y = [-180, 180],
                                searchr_z = [-180, 180],),
        iterfield   = ['in_file','reference'],
        name        = 'linear_example_func2highres')
registration.connect(inputnode, 'example_func',
                     linear_example_func2highres, 'in_file')
registration.connect(inputnode, 'highres',
                     linear_example_func2highres, 'reference')
registration.connect(linear_example_func2highres, 'out_file',
                     outputnode, 'example_func2highres_nii_gz')
registration.connect(linear_example_func2highres, 'out_matrix_file',
                     outputnode, 'example_func2highres_mat')
registration.connect(linear_example_func2highres, 'out_log',
                     outputnode, 'linear_example_func2highres_log')

"""
/opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
    -inverse -omat highres2example_func.mat example_func2highres.mat
"""
get_highres2example_func = pe.MapNode(
        interface = fsl.ConvertXFM(invert_xfm = True),
        iterfield = ['in_file'],
        name = 'get_highres2example_func')
registration.connect(linear_example_func2highres,'out_matrix_file',
                     get_highres2example_func,'in_file')
registration.connect(get_highres2example_func,'out_file',
                     outputnode,'highres2example_func_mat')

"""
/opt/fsl/fsl-5.0.10/fsl/bin/flirt 
    -in highres 
    -ref standard 
    -out highres2standard 
    -omat highres2standard.mat 
    -cost corratio 
    -dof 12 
    -searchrx -180 180 
    -searchry -180 180 
    -searchrz -180 180 
    -interp trilinear 
"""
linear_highres2standard = pe.MapNode(
        interface = fsl.FLIRT(cost = 'corratio',
                            interp = 'trilinear',
                            dof = 12,
                            save_log = True,
                            searchr_x = [-180, 180],
                            searchr_y = [-180, 180],
                            searchr_z = [-180, 180],),
        iterfield = ['in_file','reference'],
        name = 'linear_highres2standard')
registration.connect(inputnode,'highres',
                     linear_highres2standard,'in_file')
registration.connect(inputnode,'standard',
                     linear_highres2standard,'reference',)
registration.connect(linear_highres2standard,'out_file',
                     outputnode,'highres2standard_linear_nii_gz')
registration.connect(linear_highres2standard,'out_matrix_file',
                     outputnode,'highres2standard_mat')
registration.connect(linear_highres2standard,'out_log',
                     outputnode,'linear_highres2standard_log')
"""
/opt/fsl/fsl-5.0.10/fsl/bin/fnirt 
    --iout=highres2standard_head 
    --in=highres_head 
    --aff=highres2standard.mat 
    --cout=highres2standard_warp 
    --iout=highres2standard 
    --jout=highres2highres_jac 
    --config=T1_2_MNI152_2mm 
    --ref=standard_head 
    --refmask=standard_mask 
    --warpres=10,10,10
"""
nonlinear_highres2standard = pe.MapNode(
        interface = fsl.FNIRT(warp_resolution = (10,10,10),
                              config_file = "T1_2_MNI152_2mm"),
        iterfield = ['in_file','ref_file','affine_file','refmask_file'],
        name = 'nonlinear_highres2standard')
# -- iout
registration.connect(nonlinear_highres2standard,'warped_file',
                     outputnode,'highres2standard_head_nii_gz')
# --in
registration.connect(inputnode,'highres',
                     nonlinear_highres2standard,'in_file')
# --aff
registration.connect(linear_highres2standard,'out_matrix_file',
                     nonlinear_highres2standard,'affine_file')
# --cout
registration.connect(nonlinear_highres2standard,'fieldcoeff_file',
                     outputnode,'highres2standard_warp_nii_gz')
# --jout
registration.connect(nonlinear_highres2standard,'jacobian_file',
                     outputnode,'highres2highres_jac_nii_gz')
# --ref
registration.connect(inputnode,'standard_head',
                     nonlinear_highres2standard,'ref_file',)
# --refmask
registration.connect(inputnode,'standard_mask',
                     nonlinear_highres2standard,'refmask_file')
# log
registration.connect(nonlinear_highres2standard,'log_file',
                     outputnode,'nonlinear_highres2standard_log')
"""
/opt/fsl/fsl-5.0.10/fsl/bin/applywarp 
    -i highres 
    -r standard 
    -o highres2standard 
    -w highres2standard_warp
"""
warp_highres2standard = pe.MapNode(
        interface = fsl.ApplyWarp(),
        iterfield = ['in_file','ref_file','field_file'],
        name = 'warp_highres2standard')
registration.connect(inputnode,'highres',
                     warp_highres2standard,'in_file')
registration.connect(inputnode,'standard',
                     warp_highres2standard,'ref_file')
registration.connect(warp_highres2standard,'out_file',
                     outputnode,'highres2standard_nii_gz')
registration.connect(nonlinear_highres2standard,'fieldcoeff_file',
                     warp_highres2standard,'field_file')
"""
/opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
    -inverse -omat standard2highres.mat highres2standard.mat
"""
get_standard2highres = pe.MapNode(
        interface = fsl.ConvertXFM(invert_xfm = True),
        iterfield = ['in_file'],
        name = 'get_standard2highres')
registration.connect(linear_highres2standard,'out_matrix_file',
                     get_standard2highres,'in_file')
registration.connect(get_standard2highres,'out_file',
                     outputnode,'standard2highres_mat')
"""
/opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
    -omat example_func2standard.mat -concat highres2standard.mat example_func2highres.mat
"""
get_exmaple_func2standard = pe.MapNode(
        interface = fsl.ConvertXFM(concat_xfm = True),
        iterfield = ['in_file','in_file2'],
        name = 'get_exmaple_func2standard')
registration.connect(linear_example_func2highres, 'out_matrix_file',
                     get_exmaple_func2standard,'in_file')
registration.connect(linear_highres2standard,'out_matrix_file',
                     get_exmaple_func2standard,'in_file2')
registration.connect(get_exmaple_func2standard,'out_file',
                     outputnode,'example_func2standard_mat')
"""
/opt/fsl/fsl-5.0.10/fsl/bin/convertwarp 
    --ref=standard 
    --premat=example_func2highres.mat 
    --warp1=highres2standard_warp 
    --out=example_func2standard_warp
"""
convertwarp_example2standard = pe.MapNode(
        interface = fsl.ConvertWarp(),
        iterfield = ['reference','premat','warp1'],
        name = 'convertwarp_example2standard')
registration.connect(inputnode,'standard',
                     convertwarp_example2standard,'reference')
registration.connect(linear_example_func2highres,'out_matrix_file',
                     convertwarp_example2standard,'premat')
registration.connect(nonlinear_highres2standard,'fieldcoeff_file',
                     convertwarp_example2standard,'warp1')
registration.connect(convertwarp_example2standard,'out_file',
                     outputnode,'example_func2standard_warp_nii_gz')
"""
/opt/fsl/fsl-5.0.10/fsl/bin/applywarp 
    --ref=standard 
    --in=example_func 
    --out=example_func2standard 
    --warp=example_func2standard_warp
"""
warp_example2stand = pe.MapNode(
        interface = fsl.ApplyWarp(),
        iterfield = ['ref_file','in_file','field_file'],
        name = 'warp_example2stand')
registration.connect(inputnode,'standard',
                     warp_example2stand,'ref_file')
registration.connect(inputnode,'example_func',
                     warp_example2stand,'in_file')
registration.connect(warp_example2stand,'out_file',
                     outputnode,'example_func2standard_nii_gz')
registration.connect(convertwarp_example2standard,'out_file',
                     warp_example2stand,'field_file')
"""
/opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
    -inverse -omat standard2example_func.mat example_func2standard.mat
"""
get_standard2example_func = pe.MapNode(
        interface = fsl.ConvertXFM(invert_xfm = True),
        iterfield = ['in_file'],
        name = 'get_standard2example_func')
registration.connect(get_exmaple_func2standard,'out_file',
                     get_standard2example_func,'in_file')
registration.connect(get_standard2example_func,'out_file',
                     outputnode,'standard2example_func_mat')

registration.base_dir = output_dir

registration.inputs.inputspec.highres = anat_brain
registration.inputs.inputspec.highres_head= anat_head
registration.inputs.inputspec.example_func = example_func
registration.inputs.inputspec.standard = standard_brain
registration.inputs.inputspec.standard_head = standard_head
registration.inputs.inputspec.standard_mask = standard_mask

# define all the oupput file names with the directory
registration.inputs.linear_example_func2highres.out_file          = os.path.abspath(os.path.join(output_dir,
                        'example_func2highres.nii.gz'))
registration.inputs.linear_example_func2highres.out_matrix_file   = os.path.abspath(os.path.join(output_dir,
                        'example_func2highres.mat'))
registration.inputs.linear_example_func2highres.out_log           = os.path.abspath(os.path.join(output_dir,
                        'linear_example_func2highres.log'))
registration.inputs.get_highres2example_func.out_file        = os.path.abspath(os.path.join(output_dir,
                        'highres2example_func.mat'))
registration.inputs.linear_highres2standard.out_file         = os.path.abspath(os.path.join(output_dir,
                        'highres2standard_linear.nii.gz'))
registration.inputs.linear_highres2standard.out_matrix_file  = os.path.abspath(os.path.join(output_dir,
                        'highres2standard.mat'))
registration.inputs.linear_highres2standard.out_log          = os.path.abspath(os.path.join(output_dir,
                        'linear_highres2standard.log'))
# --iout
registration.inputs.nonlinear_highres2standard.warped_file  = os.path.abspath(os.path.join(output_dir,
                        'highres2standard.nii.gz'))
# --cout
registration.inputs.nonlinear_highres2standard.fieldcoeff_file    = os.path.abspath(os.path.join(output_dir,
                        'highres2standard_warp.nii.gz'))
# --jout
registration.inputs.nonlinear_highres2standard.jacobian_file      = os.path.abspath(os.path.join(output_dir,
                        'highres2highres_jac.nii.gz'))
registration.inputs.nonlinear_highres2standard.log_file           = os.path.abspath(os.path.join(output_dir,
                        'nonlinear_highres2standard.log'))
registration.inputs.warp_highres2standard.out_file                = os.path.abspath(os.path.join(output_dir,
                        'highres2standard.nii.gz'))
registration.inputs.get_standard2highres.out_file       = os.path.abspath(os.path.join(output_dir,
                        'standard2highres.mat'))
registration.inputs.get_exmaple_func2standard.out_file               = os.path.abspath(os.path.join(output_dir,
                        'example_func2standard.mat'))
registration.inputs.convertwarp_example2standard.out_file     = os.path.abspath(os.path.join(output_dir,
                        'example_func2standard_warp.nii.gz'))
registration.inputs.warp_example2stand.out_file       = os.path.abspath(os.path.join(output_dir,
                        'example_func2standard.nii.gz'))
registration.inputs.get_standard2example_func.out_file       = os.path.abspath(os.path.join(output_dir,
                        'standard2example_func.mat'))


registration.run()






























