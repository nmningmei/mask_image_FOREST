#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:08:19 2019

@author: nmei
"""

import os
from glob import glob

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from shutil import copyfile
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

def read_csv(f):
    temp = pd.read_csv(f).iloc[:,1:]
    temp['model_name'] = f.split(' ')[-1].split('.')[0]
    return temp

def rename_roi(x):
    return x.split('-')[-1] + '-' + x.split('-')[1]

df = pd.concat([read_csv(f) for f in working_data])
#df = df.groupby(['conscious_state','roi_name','model_name']).mean().reset_index()
temp = np.array([item.split('-') for item in df['roi_name'].values])
df['roi_name'] = temp[:,1]
df['side'] = temp[:,0]

df_plot = pd.concat(df[df['conscious_state'] == state].sort_values(['roi_name','side']) for state in ['unconscious','glimpse','conscious'])
df_plot['x'] = 0
df_plot['region'] = df_plot['roi_name'].map(utils.define_roi_category())
df_plot = pd.concat([df_sub for ii,df_sub in df_plot.groupby(['region','roi_name','conscious_state',
                                                  'side'])])
g = sns.catplot(x = 'roi_name',
                y = 'corr',
                hue = 'conscious_state',
                hue_order = ['unconscious','glimpse','conscious'],
                row = 'model_name',
                col = 'side',
                data = df_plot,
                kind = 'bar',
                aspect = 3,
                )
(g.set_axis_labels('ROIs','Correlations')
  .set_titles('{row_name} | {col_name}')
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