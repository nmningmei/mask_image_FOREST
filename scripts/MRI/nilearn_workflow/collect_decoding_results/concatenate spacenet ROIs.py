#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:14:46 2019

@author: nmei

must load fsl first


"""

import os
from glob import glob

from nilearn.plotting import plot_stat_map
from nipype.interfaces import freesurfer,fsl
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

working_dir = '../../../../results/MRI/nilearn/spacenet/'
standard_brain = '../../../../data/standard_brain/MNI152_T1_2mm_brain.nii.gz'
figure_dir = '../../../../figures/MRI/nilearn/collection_of_results'
working_data = glob(os.path.join(working_dir,'*','*standard*nii.gz'))

df_temp = dict(filename = [],
               conscious_state = [],
               roi_name = [],
               side = [],
               sub_name = [],
               )
for f in working_data:
    sub_name = f.split('/')[-2]
    conscious_state = f.split('/')[-1].split('_')[2]
    (side,roi_name) = f.split('/')[-1].split('_')[3].split('-')
    df_temp['filename'].append(f)
    df_temp['conscious_state'].append(conscious_state)
    df_temp['roi_name'].append(roi_name)
    df_temp['side'].append(side)
    df_temp['sub_name'].append(sub_name)
df = pd.DataFrame(df_temp)

for (sub_name,conscious_state),df_sub in df.groupby(['sub_name','conscious_state']):
    first_img = df_sub['filename'].values[0]
    output_dir = os.path.join(*first_img.split('/')[:-1])
    for second_img in df_sub['filename'].values[1:]:
        merger = fsl.ImageMaths(in_file = os.path.abspath(first_img),
                                in_file2 = os.path.abspath(second_img),
                                op_string = '-add')
        merger.inputs.out_file = os.path.abspath(
                                    os.path.join(output_dir,
                                                 f'patterns_concate_{conscious_state}.nii.gz'))
        merger.run()
        
        first_img = os.path.join(output_dir,f'patterns_concate_{conscious_state}.nii.gz')
    
    plt.close('all')
    fig,ax = plt.subplots(figsize = (36,6))
    plot_stat_map(first_img,
                  bg_img        = standard_brain,
                  colorbar      = True,
                  display_mode  = 'z',
                  cut_coords    = np.arange(-30,67,6),
                  figure        = fig,
                  axes          = ax,
                  title         = f'patterns on standard MNI brain,{sub_name} {conscious_state}\nred = nonliving,blue = living',
                  threshold     = .05,
                  vmax          = 1.5,
                  draw_cross    = False,
                  )
    fig.savefig(os.path.join(figure_dir,
                             f'concatenated_patterns_{sub_name}_{conscious_state}.png'),
    dpi = 400,
    bbox_inches = 'tight')
    plt.close('all')























