#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:28:15 2019

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
target_folder = 'temporal_generalization'
working_dir = f'../../../../results/MRI/nilearn/{sub}/{target_folder}'
working_data = glob(os.path.join(working_dir,'*.csv'))

figure_dir = f'../../../../figures/MRI/nilearn/{sub}/{target_folder}'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

df = pd.concat([pd.read_csv(f) for f in working_data])

n_splits = np.max(df['fold'])

temp = np.array([item.split('-') for item in df['roi'].values])
df['roi_name'] = temp[:,1]
df['side'] = temp[:,0]
idx = df['model'].apply(lambda x: 'PCA' not in x)
df_plot = df[idx]

g = sns.catplot(x = 'roi_name',
                y = 'roc_auc',
                hue = 'condition_target',
                col = 'model',
                col_order = ['None + Linear-SVM'],
                row = 'side',
                data = df_plot,
                kind = 'point',
                aspect = 3,
                **{'dodge': True},
                )
(g.set_axis_labels('ROIs','ROC AUC')
  .set_titles('{row_name}')
  )
[ax.axhline(0.5, linestyle = '--', color = 'black', alpha = 1.) for ax in g.axes.flatten()]
[ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 90, ha = 'center') for ax in g.axes[-1]]
xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
g.fig.suptitle(f'Fit at 4 - 7 secs after onset of probe, test at 2 - 5 secs after responses\n {n_splits} folds, GroupShuffleSplit',y = 1.02)
g.savefig(os.path.join(figure_dir,
                       'Temporal generalization.png'),
          dpi = 500,
          bbox_inches = 'tight')
g.savefig(os.path.join(figure_dir,
                       'Temporal generalization (light).png'),
          bbox_inches = 'tight')