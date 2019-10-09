#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:03:13 2019

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

def rename_roi(x):
    return x.split('-')[-1] + '-' + x.split('-')[1]

sub = 'sub-01'
target_folder_temporal = 'temporal_generalization'
target_folder_original = 'LOO'
working_dir_temporal = f'../../../../results/MRI/nilearn/{sub}/{target_folder_temporal}'
working_data_temporal = np.sort(glob(os.path.join(working_dir_temporal,'*None +*.csv')))
working_dir_original = f'../../../../results/MRI/nilearn/{sub}/{target_folder_original}'
working_data_original = np.sort(glob(os.path.join(working_dir_original,'*None +*.csv')))

figure_dir = f'../../../../figures/MRI/nilearn/{sub}/comparing_on_post'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

df_temporal = pd.concat([pd.read_csv(f) for f in working_data_temporal])
df_original = pd.concat([pd.read_csv(f) for f in working_data_original])
df_temporal['type'] = 'Post Response'
df_original['type'] = 'Probe Onset'

df = pd.concat([df_original,df_temporal])

n_splits = np.max(df['fold'])

temp = np.array([item.split('-') for item in df['roi'].values])
df['roi_name'] = temp[:,1]
df['side'] = temp[:,0]

idx = df['model'].apply(lambda x: 'Dummy' not in x)
df_plot = df[idx]

g = sns.catplot(x = 'roi_name',
                y = 'roc_auc',
                hue = 'type',
                row = 'condition_source',
                row_order = ['unconscious','glimpse','conscious'],
                col = 'side',
                data = df_plot,
                aspect = 6,
                kind = 'violin',
                **{'cut': 0,
                   'split': True,
                   'inner': 'quartile'})
(g.set_axis_labels('ROIs','ROC AUC')
  .set_titles('{row_name} | side {col_name}')
  .set(ylim = (-.1,1.1))
  )
[ax.axhline(0.5, linestyle = '--', color = 'black', alpha = 1.) for ax in g.axes.flatten()]
[ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 90, ha = 'center') for ax in g.axes[-1]]
xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
g.fig.suptitle('Comparison between Probe Onset (4 - 7 secs after probe) and Post Response (2 - 5 secs after response)',y = 1.05)
g.savefig(os.path.join(figure_dir,
                       'comparison violinplot.png'),
        dpi = 400,
        bbox_inches = 'tight')
plt.close('all')


g = sns.catplot(x = 'roi_name',
                y = 'roc_auc',
                hue = 'type',
                row = 'condition_source',
                row_order = ['unconscious','glimpse','conscious'],
                col = 'side',
                data = df_plot,
                aspect = 4,
                kind = 'bar',
                )
(g.set_axis_labels('ROIs','ROC AUC')
  .set_titles('{row_name} | side {col_name}')
  .set(ylim = (-.1,1.1))
  )
[ax.axhline(0.5, linestyle = '--', color = 'black', alpha = 1.) for ax in g.axes.flatten()]
[ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 90, ha = 'center') for ax in g.axes[-1]]
xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
g.fig.suptitle('Comparison between Probe Onset (4 - 7 secs after probe) and Post Response (2 - 5 secs after response)',y = 1.05)





































