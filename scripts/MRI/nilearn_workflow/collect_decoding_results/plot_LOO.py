#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:33:10 2019

@author: nmei
"""

import os
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd

import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')

from shutil import copyfile
copyfile('../../../utils.py','utils.py')
import utils

def get_fs(x):
    return x.split(' + ')[0]
def get_clf(x):
    return x.split(' + ')[1]
def rename_roi(x):
    return x.split('-')[-1] + '-' + x.split('-')[1]

figure_dir = '../../../../figures/MRI/nilearn/collection_of_results'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

raw_folder = 'LOO'
raw_file = '*None*csv'
working_dir = '../../../../results/MRI/nilearn/sub-*/{}'.format(raw_folder)
working_data = glob(os.path.join(working_dir,raw_file))

df_plot = []
for f in tqdm(working_data):
    df_sub = pd.read_csv(f)
    df_sub['sub'] = f.split('/')[-3]
    df_plot.append(df_sub)
df_plot = pd.concat(df_plot)

if 'model_name' not in df_plot.columns:
    df_plot['model_name'] = df_plot['model']
df_plot['feature_selector']  = df_plot['model_name'].apply(get_fs)
df_plot['estimator']         = df_plot['model_name'].apply(get_clf)
if 'score' in df_plot.columns:
    df_plot['roc_auc'] = df_plot['score']

temp            = np.array([item.split('-') for item in df_plot['roi'].values])
df_plot['roi_name']  = temp[:,1]
df_plot['side']      = temp[:,0]

df_plot['region'] = df_plot['roi_name'].map(utils.define_roi_category())
df_plot = df_plot.sort_values([
                               'region',
                               'roi_name',
                               'condition_source',
                               'side',])
df_plot = pd.concat([df_sub for ii,df_sub in df_plot.groupby(['region','roi_name','condition_source',
                                                  'side'])])


# add stars
target_folder = 'LOO_stats'
target_file = 'LOO 4 models.csv'
working_dir = '../../../../results/MRI/nilearn/sub-*/{}'.format(target_folder)
working_data = glob(os.path.join(working_dir,target_file))

df = []
for f in working_data:
    df_temp = pd.read_csv(f)
    df_sub = df_temp[df_temp['feature_selector'] == 'None']
    df_sub['sub'] = f.split('/')[-3]
    df.append(df_sub)
df = pd.concat(df)
df['region'] = df['roi_name'].map(utils.define_roi_category())
df = df.sort_values([
                   'region',
                   'roi_name',
                   'conscious_state',
                   'sub',
                   'side',])
df = pd.concat([df[df['conscious_state'] == conscious_state] for conscious_state in ['unconscious','glimpse','conscious']])


g = sns.catplot(x = 'roi_name',
                y = 'roc_auc',
                hue = 'side',
                row = 'sub',
                col = 'condition_source',
                col_order = ['unconscious','glimpse','conscious'],
                data = df_plot,
                kind = 'bar',
                aspect = 3)
(g.set_axis_labels('ROIs','ROC AUC')
  .set_titles('{col_name} | {row_name}')
  .set(ylim=(0.35,.75)))
[ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 90, ha = 'center') for ax in g.axes[-1]]
[ax.axhline(0.5,linestyle='--',color='k',alpha=0.5) for ax in g.axes.flatten()]
[ax.axvline(6.5,linestyle='-' ,color='k',alpha=0.5) for ax in g.axes.flatten()]
xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())

for n_row,sub in enumerate(pd.unique(df['sub'])):
    for n_col, conscious_state in enumerate(['unconscious','glimpse','conscious']):
        ax = g.axes[n_row][n_col]
        
        for ii,text_obj in enumerate(xtick_order):
            position = text_obj.get_position()
            xtick_label = text_obj.get_text()
            
            rows = df[np.logical_and(
                    df['sub'] == sub,
                    df['conscious_state'] == conscious_state)]
            rows = rows[rows['roi_name'] == xtick_label]
            for (ii,temp_row),adjustment in zip(rows.iterrows(),[-0.45,0.055]):
                if '*' in temp_row['stars']:
                    ax.annotate(temp_row['stars'],
                                xy = (position[0] + adjustment,0.70))

g.savefig(os.path.join(figure_dir,
                       'Decoding_LOO.png'),
                    dpi = 400,
                    bbox_inches = 'tight')
g.savefig(os.path.join(figure_dir,
                       'Decoding_LOO_light.png'),
#                    dpi = 400,
                    bbox_inches = 'tight')





