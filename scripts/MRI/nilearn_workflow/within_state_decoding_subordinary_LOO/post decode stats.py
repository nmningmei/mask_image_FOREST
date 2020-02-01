#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 12:06:18 2019

@author: nmei
"""

import os
import re
from glob import glob

import pandas as pd
import numpy as np
import seaborn as sns

from shutil import copyfile
copyfile('../../../utils.py','utils.py')
import utils

from matplotlib import pyplot as plt
sns.set_style('whitegrid')
sns.set_context('poster')

sub = 'sub-07'
experiment = 'subordinary_LOO'
working_dir = f'../../../../results/MRI/nilearn/{sub}/{experiment}'
working_data = glob(os.path.join(working_dir,'*.csv'))
saving_dir = f'../../../../results/MRI/nilearn/{sub}/{experiment}_poststats'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
figure_dir = f'../../../../figures/MRI/nilearn/{sub}/{experiment}_poststats'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

results = dict(
        sub_name = [],
        roi_name = [],
        side = [],
        roc_auc = [],
        pval = [],
        conscious_state = [],
        )
for f in working_data:
    temp = re.findall('\(.*?\)',f)[0].replace("(",'').replace(")","")
    sub_name,roi_name,conscious_state,_,_,estimater = temp.split(' ')
    side,roi_name = roi_name.split('-')
    df_temp = pd.read_csv(f)
    chance_mean,chance_std = df_temp['chance_mean'].mean(),df_temp['chance_std'].mean()
    sampled_chance = np.random.normal(chance_mean,chance_std,size = int(1e5))
    pval = (np.sum(sampled_chance > df_temp['roc_auc'].mean()) + 1) / (sampled_chance.shape[0] + 1)
    results['sub_name'].append(sub_name)
    results['roi_name'].append(roi_name)
    results['side'].append(side)
    results['roc_auc'].append(df_temp['roc_auc'].mean())
    results['pval'].append(pval)
    results['conscious_state'].append(conscious_state)
    n_folds = len(df_temp)

results = pd.DataFrame(results)
results = results.sort_values(['pval'])
converter = utils.MCPConverter(pvals = results['pval'].values)
d = converter.adjust_many()
results['p_corrected'] = d['bonferroni'].values
results['stars'] = results['p_corrected'].apply(utils.stars)

df = utils.load_same_same(sub,experiment)

df = df.sort_values(['roi_name','side','condition_source',])
results = results.sort_values(['roi_name','side','conscious_state'])
results.to_csv(os.path.join(saving_dir,'post stats.csv'),index = False)

g = sns.catplot(x = 'roi_name',
                y = 'roc_auc',
                hue = 'side',
                row = 'condition_source',
                row_order = ['unconscious','glimpse','conscious'],
                data = df,
                kind = 'bar',
                aspect = 3,)
(g.set_axis_labels('ROIs','ROC AUC')
  .set_titles('{row_name}')
  .set(ylim=(0.,1.)))
[ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 45, ha = 'right') for ax in g.axes[-1]]
xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
[ax.axhline(0.5,linestyle='--',color='k',alpha=0.5) for ax in g.axes.flatten()]
[ax.axvline(6.5,linestyle='-' ,color='k',alpha=0.5) for ax in g.axes.flatten()]
# add stars
for n_row, conscious_state in enumerate(['unconscious','glimpse','conscious']):
    ax = g.axes[n_row][0]
    
    for ii,text_obj in enumerate(xtick_order):
        position = text_obj.get_position()
        xtick_label = text_obj.get_text()
        rows = results[results['conscious_state'] == conscious_state]
        rows = rows[rows['roi_name'] == xtick_label]
        for (ii,temp_row),adjustment in zip(rows.iterrows(),[-0.35,0.05]):
            if '*' in temp_row['stars']:
                ax.annotate(temp_row['stars'],
                            xy = (position[0] + adjustment,0.9))
estimator = 'Linear-SVM'
title = 'within subject decoding of {}, fold = {}\n nilearn pipeline, estimator = {}\nBoferroni corrected within consciousness state\n*:<0.05,**<0.01,***<0.001'.format(
        sub,n_folds,estimator)
g.fig.suptitle(title,y = 1.1)
g.savefig(os.path.join(figure_dir,
                       'decode subordinary.jpeg'),
        dpi = 300,
        bbox_inches = 'tight')













