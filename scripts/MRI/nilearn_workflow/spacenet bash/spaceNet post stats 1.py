#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:32:45 2019

@author: nmei

This script takes the coefficients fit by space net classifier from each of the 30 folds
and compute the so-called "pattern", cov(BOLD.T).dot(coef) and plot the average pattern
in the strcutural space and standard space

"""

import os

import pandas   as pd
import numpy    as np
import seaborn  as sns

from glob                  import glob
from nilearn.input_data    import NiftiMasker
from matplotlib            import pyplot as plt
from nilearn.plotting      import plot_stat_map
from nipype.interfaces     import fsl
from sklearn.preprocessing import MinMaxScaler
from shutil                import copyfile

copyfile('../../../../scripts/utils.py','utils.py')
sns.set_style('white')
sns.set_context('poster')
from utils              import standard_MNI_coordinate_for_plot

# define working path, mask, brain, transformation matrices and saving paths
working_dir             = '../../../../results/MRI/nilearn/spacenet'
sub                     = 'sub-01'
masks                   = glob(
        os.path.join('../../../../data/MRI/{}/anat/ROI_BOLD/*.nii.gz'.format(sub))
                                )
anat_brain              = glob(
        os.path.join('../../../../data/MRI/{}/anat/*brain*'.format(sub))
                                )
standard_brain          = '/opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
transformation_matrix   = glob(os.path.join(f'../../../../data/MRI/{sub}/func/session-*/*/outputs/reg/example_func2highres.mat'))
working_data            = glob(os.path.join(working_dir,sub,"*.nii.gz"))
# this is the file that saves decoding scores of the space net classifier
working_df              = glob(os.path.join(working_dir,sub,"*.csv"))
df                      = pd.read_csv(working_df[0])
figure_dir              = '../../../../figures/MRI/nilearn/spanceNet'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
# get plotting coordinates in standard space
coordinates = standard_MNI_coordinate_for_plot()

for conscious_state,df_sub in df.groupby(['conscious_state']):
    working_item        = [item for item in working_data if (f"spaceNet_coef_{conscious_state}" in item)]
    scores              = df_sub['score'].values
    whole_brain_data    = glob(os.path.join('../../../../data',
                                            'BOLD_whole_brain_averaged',
                                            sub,
                                            f'whole_brain_{conscious_state}.nii.gz')
                                )
    for mask in masks:
        roi_name        = mask.split('/')[-1].split('_BOLD')[0].replace('ctx-','')
        masker          = NiftiMasker(mask_img      = mask,
                                      standardize   = True,
                                      detrend       = True,
                                      )
        masker.fit(whole_brain_data[0])
        BOLD            = masker.transform(whole_brain_data[0])
        
        weights         = np.array([masker.transform(item)[0] for item in working_item])
        
        # too much memory consuming if use whole brain - 17 G data
        X_cov           = np.cov(BOLD.T)
        # formula for computing patterns
        patterns            = np.array([X_cov.dot(w.T.dot(1)).T for w in weights])
        # normalize per fold
        patterns_standard   = patterns / patterns.mean(0).std()
        patterns_to_plot    = masker.inverse_transform(patterns_standard.mean(0))
        
        patterns_to_plot.to_filename(os.path.join(working_dir,sub,f'patterns_{conscious_state}_{roi_name}.nii.gz'))
        
        if not os.path.exists(os.path.join(figure_dir,sub)):
            os.mkdir(os.path.join(figure_dir,sub))
        
        # conver to structural space
        flt                         = fsl.FLIRT()
        flt.inputs.in_file          = os.path.abspath(os.path.join(working_dir,
                                                                   sub,
                                                                   f'patterns_{conscious_state}_{roi_name}.nii.gz'))
        flt.inputs.reference        = anat_brain[0]
        flt.inputs.output_type      = 'NIFTI_GZ'
        flt.inputs.in_matrix_file   = transformation_matrix[0]
        flt.inputs.out_file         = os.path.abspath(os.path.join(working_dir,
                                                                   sub,
                                                                   f'patterns_{conscious_state}_{roi_name}_highres.nii.gz'))
        flt.inputs.apply_xfm        = True
        res                         = flt.run()
        
        saving_name = os.path.join(figure_dir,sub,f'patterns_{conscious_state}_{roi_name}.png')
        
        fig,ax = plt.subplots(figsize = (12,6))
        plot_stat_map(res.outputs.out_file,
                      bg_img        = anat_brain[0],
                      colorbar      = True,
                      figure        = fig,
                      axes          = ax,
                      title         = f'patterns_{conscious_state}_{roi_name}\nred = nonliving,blue = living, standardized scale',
                      threshold     = 5e-2,
#                      vmax          = 1.,
                      draw_cross    = False,
                      )
        fig.savefig(saving_name,
                    dpi = 400,
                    bbox_inches = 'tight')
        
        # convert to standard space
        flt                         = fsl.FLIRT()
        flt.inputs.in_file          = os.path.abspath(os.path.join(working_dir,
                                                                   sub,
                                                                   f'patterns_{conscious_state}_{roi_name}.nii.gz'))
        flt.inputs.reference        = standard_brain
        flt.inputs.output_type      = 'NIFTI_GZ'
        flt.inputs.in_matrix_file   = transformation_matrix[0].replace("example_func2highres","example_func2standard")
        flt.inputs.out_file         = os.path.abspath(os.path.join(working_dir,
                                                                   sub,
                                                                   f'patterns_standard_{conscious_state}_{roi_name}_standard.nii.gz'))
        flt.inputs.apply_xfm        = True
        res                         = flt.run()
        
        saving_name = os.path.join(figure_dir,sub,f'patterns_standard_{conscious_state}_{roi_name}.png')
        
        fig,ax = plt.subplots(figsize = (12,6))
        plot_stat_map(res.outputs.out_file,
                      bg_img        = standard_brain,
                      colorbar      = True,
                      cut_coords    = coordinates[roi_name],
                      figure        = fig,
                      axes          = ax,
                      title         = f'patterns_{conscious_state}_{roi_name}\nred = nonliving,blue = living, standardized scale',
                      threshold     = 5e-2,
#                      vmax          = 1.,
                      draw_cross    = False,
                      )
        fig.savefig(saving_name,
                    dpi = 400,
                    bbox_inches = 'tight')
        
        plt.close('all')




