#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:08:19 2019

@author: nmei
"""

import os
from glob import glob

import gc
gc.collect()

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from shutil import copyfile
from joblib import Parallel,delayed
copyfile('../../../utils.py','utils.py')
import utils

sns.set_style('whitegrid')
sns.set_context('poster')

sub = 'sub-01'
folder = 'encoding_LOO'

working_dir = f'../../../../results/MRI/nilearn/{sub}/{folder}'
working_data = glob(os.path.join(working_dir,'*.csv'))

figure_dir = f'../../../../figures/MRI/nilearn/{sub}/{folder}'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

def rename_roi(x):
    return x.split('-')[-1] + '-' + x.split('-')[1]

df = []
target_col,ylabel = 'corr','Correlation'
def read_csv(f):
    df_temp = pd.read_csv(f)
    df_img = df_temp[df_temp['feature_type'] == 'image']
    df_bkg = df_temp[df_temp['feature_type'] == 'background']
    tol_img = df_img[target_col].values# * df_img['positive voxels'].values
    tol_bkg = df_bkg[target_col].values# * df_bkg['positive voxels'].values
    
    df_img.loc[:,target_col] = -(tol_img - tol_bkg)
    return df_img

df = Parallel(n_jobs = -1, verbose = 1)(delayed(read_csv)(**{'f':f}) for f in working_data)
df = pd.concat(df)
#df = df.groupby(['conscious_state','roi_name','model_name']).mean().reset_index()
temp = np.array([item.split('-') for item in df['roi_name'].values])
df.loc[:,'roi_name'] = temp[:,1]
df.loc[:,'side'] = temp[:,0]

df_plot = pd.concat(df[df['conscious_source'] == state].sort_values(
                            ['conscious_target','roi_name','side']
                                ) for state in ['unconscious','glimpse','conscious'])
df_plot.loc[:,'x'] = 0
df_plot.loc[:,'region'] = df_plot['roi_name'].map(utils.define_roi_category())
df_plot = pd.concat([df_sub for ii,df_sub in df_plot.groupby(['region','roi_name','conscious_source','conscious_target',
                                                  'side'])])

g = sns.catplot(x = 'roi_name',
                y = target_col,
                hue = 'side',
                row = 'conscious_source',
                row_order = ['unconscious','glimpse','conscious'],
                col = 'conscious_target',
                col_order = ['unconscious','glimpse','conscious'],
                data = df_plot,
                kind = 'bar',
                aspect = 3,
                )
(g.set_axis_labels('ROIs',ylabel)
  .set_titles('{row_name} --> {col_name}')
  )
[ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 90, ha = 'center') for ax in g.axes[-1]]
xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
[ax.axvline(6.5,linestyle='-' ,color='k',alpha=0.5) for ax in g.axes.flatten()]
g.savefig(os.path.join(figure_dir,'Encoding model with Ridge Regression.png'),
          dpi = 500,
          bbox_inches = 'tight')
g.savefig(os.path.join(figure_dir,'Encoding model with Ridge Regression (light).png'),
#          dpi = 500,
          bbox_inches = 'tight')