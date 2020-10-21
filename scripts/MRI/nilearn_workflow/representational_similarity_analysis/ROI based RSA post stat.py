#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 04:57:44 2020

@author: nmei
"""

import os
import utils

from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')
sns.set_context('poster',font_scale = 1.5)

working_dir = '../../../../results/MRI/nilearn/ROI_RSA'
figure_dir = '../../../../figures/MRI/nilearn/ROI_RSA'
paper_dir = '/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/figures'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
working_data = glob(os.path.join(working_dir,'*','*.csv'))

df = pd.concat([pd.read_csv(f) for f in tqdm(working_data)])

df['region'] = df['roi_name'].map(utils.define_roi_category())
df['roi_name'] = df['roi_name'].map(utils.rename_ROI_for_plotting())
df = df.sort_values(['sub_name','roi_name','region','conscious_state'])
df['Condition'] = df['conscious_state']


g = sns.catplot(x = 'roi_name',
                y = 'corr',
                hue = 'Condition',
                row = 'sub_name',
                data = df,
                seed = 12345,
                aspect = 3,
                kind = 'bar',
                hue_order = ['random','unconscious','glimpse','conscious'],
                )
(g.set_axis_labels('ROIs','$\\rho$',)
  .set_titles('{row_name}'))
xticklabels = pd.unique(df['roi_name'])
[ax.axvline(6.5,linestyle = '-',color = 'black',alpha = .5) for ax in g.axes.flatten()]
for ax in g.axes[-1]:
    ax.set_xticklabels(xticklabels,rotation = 90)
g.savefig(os.path.join(figure_dir,'ROI RSA correlation.jpeg'),
          dpi = 400,
          bbox_inches = 'tight')
g.savefig(os.path.join(paper_dir,'ROI_RSA_correlation.jpeg'),
          dpi = 400,
          bbox_inches = 'tight')

df['z'] = df['corr'].apply(np.arctanh)
g = sns.catplot(x = 'roi_name',
                y = 'z',
                hue = 'Condition',
                row = 'sub_name',
                data = df,
                seed = 12345,
                aspect = 3,
                kind = 'bar',
                hue_order = ['random','unconscious','glimpse','conscious'],
                )
(g.set_axis_labels('ROIs','$Z$',)
  .set_titles('{row_name}'))
xticklabels = pd.unique(df['roi_name'])
[ax.axvline(6.5,linestyle = '-',color = 'black',alpha = .5) for ax in g.axes.flatten()]
for ax in g.axes[-1]:
    ax.set_xticklabels(xticklabels,rotation = 90)
g.savefig(os.path.join(figure_dir,'ROI RSA z score.jpeg'),
          dpi = 400,
          bbox_inches = 'tight')
g.savefig(os.path.join(paper_dir,'ROI_RSA_z_score.jpeg'),
          dpi = 400,
          bbox_inches = 'tight')






















