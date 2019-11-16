#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:00:03 2019

@author: nmei

This is not set up as a workflow because we want all the intermediate outputs

"""

import os
from glob import glob
from nipype.interfaces import fsl

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
func_ref            = os.path.abspath(os.path.join(
                        '../../../data/MRI',
                        sub,
                        'func',
                        'session-{}'.format(session),
                        '{}_unfeat_run-{}'.format(sub,run),
                        'outputs',
                        'func',
                        'example_func.nii.gz'))


output_dir = '../../../data/MRI/{}/func/session-{}/sub-01_unfeat_run-{}/test/reg'.format(
        sub,session,run)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

"""
fslmaths /bcbl/home/public/Consciousness/uncon_feat/data/MRI/sub-01/anat/sub-01-T1W_mprage_sag_p2_1iso_MGH_day_6_nipy_brain highres
fslmaths /bcbl/home/public/Consciousness/uncon_feat/data/MRI/sub-01/anat/sub-01-T1W_mprage_sag_p2_1iso_MGH_day_6_nipy_brain  highres_head
fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain standard
fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm standard_head
fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil standard_mask

"""
fslmaths = fsl.ImageMaths()
fslmaths.inputs.in_file = anat_brain
fslmaths.inputs.out_file = os.path.abspath(os.path.join(output_dir,'highres.nii.gz'))
fslmaths.cmdline
fslmaths.run()

fslmaths = fsl.ImageMaths()
fslmaths.inputs.in_file = anat_head
fslmaths.inputs.out_file = os.path.abspath(os.path.join(output_dir,'highres_head.nii.gz'))
fslmaths.cmdline
fslmaths.run()

fslmaths = fsl.ImageMaths()
fslmaths.inputs.in_file = standard_brain
fslmaths.inputs.out_file = os.path.abspath(os.path.join(output_dir,'standard.nii.gz'))
fslmaths.cmdline
fslmaths.run()

fslmaths = fsl.ImageMaths()
fslmaths.inputs.in_file = standard_head
fslmaths.inputs.out_file = os.path.abspath(os.path.join(output_dir,'standard_head.nii.gz'))
fslmaths.cmdline
fslmaths.run()

fslmaths = fsl.ImageMaths()
fslmaths.inputs.in_file = standard_mask
fslmaths.inputs.out_file = os.path.abspath(os.path.join(output_dir,'standard_mask.nii.gz'))
fslmaths.cmdline
fslmaths.run()

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
flt = fsl.FLIRT()
flt.inputs.in_file = func_ref
flt.inputs.reference = anat_brain
flt.inputs.out_file = os.path.abspath(os.path.join(output_dir,'example_func2highres.nii.gz'))
flt.inputs.out_matrix_file = os.path.abspath(os.path.join(output_dir,'example_func2highres.mat'))
flt.inputs.out_log = os.path.abspath(os.path.join(output_dir,'example_func2highres.log'))
flt.inputs.cost = 'corratio'
flt.inputs.interp = 'trilinear'
flt.inputs.searchr_x = [-180, 180]
flt.inputs.searchr_y = [-180, 180]
flt.inputs.searchr_z = [-180, 180]
flt.inputs.dof = 7
flt.inputs.save_log = True
flt.cmdline
flt.run()

"""
/opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
    -inverse -omat highres2example_func.mat example_func2highres.mat
"""
inverse_transformer = fsl.ConvertXFM()
inverse_transformer.inputs.in_file = os.path.abspath(os.path.join(output_dir,"example_func2highres.mat"))
inverse_transformer.inputs.invert_xfm = True
inverse_transformer.inputs.out_file = os.path.abspath(os.path.join(output_dir,'highres2example_func.mat'))
inverse_transformer.cmdline
inverse_transformer.run()

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
flt = fsl.FLIRT()
flt.inputs.in_file = anat_brain
flt.inputs.reference = standard_brain
flt.inputs.out_file = os.path.abspath(os.path.join(output_dir,'highres2standard_linear.nii.gz'))
flt.inputs.out_matrix_file = os.path.abspath(os.path.join(output_dir,'highres2standard.mat'))
flt.inputs.out_log = os.path.abspath(os.path.join(output_dir,'highres2standard.log'))
flt.inputs.cost = 'corratio'
flt.inputs.interp = 'trilinear'
flt.inputs.searchr_x = [-180, 180]
flt.inputs.searchr_y = [-180, 180]
flt.inputs.searchr_z = [-180, 180]
flt.inputs.dof = 12
flt.inputs.save_log = True
flt.cmdline
flt.run()

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

fnirt_mprage = fsl.FNIRT()
# --iout name of output image
fnirt_mprage.inputs.warped_file = os.path.abspath(os.path.join(output_dir,
                                                             'highres2standard.nii.gz'))
# --in input image
fnirt_mprage.inputs.in_file = anat_head
# --aff affine transform
fnirt_mprage.inputs.affine_file = os.path.abspath(os.path.join(output_dir,
                                                               'highres2standard.mat'))
# --cout output file with field coefficients
fnirt_mprage.inputs.fieldcoeff_file = os.path.abspath(os.path.join(output_dir,
                                                                   'highres2standard_warp.nii.gz'))
# --jout
fnirt_mprage.inputs.jacobian_file = os.path.abspath(os.path.join(output_dir,
                                                                 'highres2highres_jac.nii.gz'))
# --config
fnirt_mprage.inputs.config_file = 'T1_2_MNI152_2mm'
# --ref
fnirt_mprage.inputs.ref_file = os.path.abspath(standard_head)
# --refmask
fnirt_mprage.inputs.refmask_file = os.path.abspath(standard_mask)
# --warpres
fnirt_mprage.inputs.warp_resolution = (10, 10, 10)
# log
fnirt_mprage.inputs.log_file = os.path.abspath(os.path.join(output_dir,
                                                            'highres2standard.log'))
## --subsamp = 4,2,1,1
#fnirt_mprage.inputs.subsampling_scheme = [4,2,1,1]
## --miter = 5,5,5,5
#fnirt_mprage.inputs.max_nonlin_iter = [5,5,5,5]
## infwhm = 6,4,2,2
#fnirt_mprage.inputs.in_fwhm = [6,4,2,2]
## reffwhm = 4,2,0,0
#fnirt_mprage.inputs.ref_fwhm = [4,2,0,0]
## --lambda
#fnirt_mprage.inputs.regularization_lambda = [300,150,100,50,40,30]
## --regmod
#fnirt_mprage.inputs.regularization_model = 'bending_energy'
## --ssqlambda
#fnirt_mprage.inputs.skip_lambda_ssq = True
## --estint
#fnirt_mprage.inputs.apply_intensity_mapping = 1
fnirt_mprage.cmdline
fnirt_mprage.run()

"""
/opt/fsl/fsl-5.0.10/fsl/bin/applywarp 
    -i highres 
    -r standard 
    -o highres2standard 
    -w highres2standard_warp
"""
aw = fsl.ApplyWarp()
aw.inputs.in_file = anat_brain
aw.inputs.ref_file = os.path.abspath(standard_brain)
aw.inputs.out_file = os.path.abspath(os.path.join(output_dir,
                                                  'highres2standard.nii.gz'))
aw.inputs.field_file = os.path.abspath(os.path.join(output_dir,
                                                    'highres2standard_warp.nii.gz'))
aw.cmdline
aw.run()

"""
/opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
    -inverse -omat standard2highres.mat highres2standard.mat
"""
inverse_transformer = fsl.ConvertXFM()
inverse_transformer.inputs.in_file = os.path.abspath(os.path.join(output_dir,"highres2standard.mat"))
inverse_transformer.inputs.invert_xfm = True
inverse_transformer.inputs.out_file = os.path.abspath(os.path.join(output_dir,'standard2highres.mat'))
inverse_transformer.cmdline
inverse_transformer.run()

"""
/opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
    -omat example_func2standard.mat -concat highres2standard.mat example_func2highres.mat
"""
inverse_transformer = fsl.ConvertXFM()
inverse_transformer.inputs.in_file2 = os.path.abspath(os.path.join(output_dir,"highres2standard.mat"))
inverse_transformer.inputs.in_file = os.path.abspath(os.path.join(output_dir,
                                                                   "example_func2highres.mat"))
inverse_transformer.inputs.concat_xfm = True
inverse_transformer.inputs.out_file = os.path.abspath(os.path.join(output_dir,'example_func2standard.mat'))
inverse_transformer.cmdline
inverse_transformer.run()

"""
/opt/fsl/fsl-5.0.10/fsl/bin/convertwarp 
    --ref=standard 
    --premat=example_func2highres.mat 
    --warp1=highres2standard_warp 
    --out=example_func2standard_warp
"""
warputils = fsl.ConvertWarp()
warputils.inputs.reference = os.path.abspath(standard_brain)
warputils.inputs.premat = os.path.abspath(os.path.join(output_dir,
                                                       "example_func2highres.mat"))
warputils.inputs.warp1 = os.path.abspath(os.path.join(output_dir,
                                                      "highres2standard_warp.nii.gz"))
warputils.inputs.out_file = os.path.abspath(os.path.join(output_dir,
                                                         "example_func2standard_warp.nii.gz"))
warputils.cmdline
warputils.run()

"""
/opt/fsl/fsl-5.0.10/fsl/bin/applywarp 
    --ref=standard 
    --in=example_func 
    --out=example_func2standard 
    --warp=example_func2standard_warp
"""
aw = fsl.ApplyWarp()
aw.inputs.ref_file = os.path.abspath(standard_brain)
aw.inputs.in_file = os.path.abspath(func_ref)
aw.inputs.out_file = os.path.abspath(os.path.join(output_dir,
                                                  "example_func2standard.nii.gz"))
aw.inputs.field_file = os.path.abspath(os.path.join(output_dir,
                                                    "example_func2standard_warp.nii.gz"))
aw.run()
"""
/opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
    -inverse -omat standard2example_func.mat example_func2standard.mat
"""
inverse_transformer = fsl.ConvertXFM()
inverse_transformer.inputs.in_file = os.path.abspath(os.path.join(output_dir,
                                                           "example_func2standard.mat"))
inverse_transformer.inputs.out_file = os.path.abspath(os.path.join(output_dir,
                                                            "standard2example_func.mat"))
inverse_transformer.inputs.invert_xfm = True
inverse_transformer.cmdline
inverse_transformer.run()

######################
###### plotting ######
example_func2highres = os.path.abspath(os.path.join(output_dir,
                                                    'example_func2highres'))
example_func2standard = os.path.abspath(os.path.join(output_dir,
                                                     "example_func2standard"))
highres2standard = os.path.abspath(os.path.join(output_dir,
                                                'highres2standard'))
highres = os.path.abspath(anat_brain)
standard = os.path.abspath(standard_brain)

plot_example_func2highres = f"""
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {example_func2highres} {highres} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2highres}1.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {highres} {example_func2highres} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2highres}2.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend {example_func2highres}1.png - {example_func2highres}2.png {example_func2highres}.png; 
/bin/rm -f sl?.png {example_func2highres}2.png
/bin/rm {example_func2highres}1.png
""".replace("\n"," ")

plot_highres2standard = f"""
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {highres2standard} {standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {highres2standard}1.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {standard} {highres2standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {highres2standard}2.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend {highres2standard}1.png - {highres2standard}2.png {highres2standard}.png; 
/bin/rm -f sl?.png {highres2standard}2.png
/bin/rm {highres2standard}1.png
""".replace("\n"," ")

plot_example_func2standard = f"""
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {example_func2standard} {standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2standard}1.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {standard} {example_func2standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2standard}2.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend {example_func2standard}1.png - {example_func2standard}2.png {example_func2standard}.png; 
/bin/rm -f sl?.png {example_func2standard}2.png
""".replace("\n"," ")
for cmdline in [plot_example_func2highres,plot_example_func2standard,plot_highres2standard]:
    os.system(cmdline)


