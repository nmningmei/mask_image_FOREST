#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:29:40 2020

@author: nmei
"""

import os
from glob import glob

import numpy as np
import pandas as pd

import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')

sub = 'sub-01'
working_dir = '../../../../results/MRI/nilearn/{}/RSA'.format(sub)
figure_dir = '../../../../figures/MRI/nilearn/{}/RSA'.format(sub)
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

metric = 'euclidean'
working_data = glob(os.path.join(working_dir,f'*{metric}*.csv'))
df = pd.concat([pd.read_csv(f) for f in working_data])
temp = np.array([item.split('-') for item in df['roi_name'].values])
df.loc[:,'roi_name'] = temp[:,1]
df.loc[:,'side'] = temp[:,0]

g = sns.catplot(x = 'roi_name',
                y = 'spearmanr',
                hue = 'side',
                row = 'conscious',
                row_order = ['unconscious','glimpse','conscious'],
                data = df,
                kind = 'bar',
                aspect = 3,
                **{'ci':'sd'}
                )
(g.set_axis_labels('ROIs',"Correlation")
  .set_titles('{row_name}')
  )
[ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 90, ha = 'center') for ax in g.axes[-1]]
g.savefig(os.path.join(figure_dir,
                       'RSA (minkowski).png'),
        dpi = 400,
        bbox_inches = 'tight')