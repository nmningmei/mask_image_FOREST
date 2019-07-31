#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:13:12 2019

@author: nmei
"""

import os
from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')
import utils

working_dir = '../../../results/pymvpa'
saving_dir = '../../../results/MRI/pymvpa/decoding stats 4 models'
figure_dir = '../../../figures/MRI/pymvpa/decoding 4 models'
for d in [saving_dir,figure_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

def get_fs(x):
    return x.split(' + ')[0]
def get_clf(x):
    return x.split(' + ')[1]
def rename_roi(x):
    return x.split('-')[-1] + '-' + x.split('-')[1]

working_data = glob(os.path.join(working_dir,'*.csv'))
df = pd.concat([pd.read_csv(f) for f in working_data])
if 'model_name' not in df.columns:
    df['model_name'] = df['model']
if 'conscious_state' not in df.columns:
    df['conscious_state'] = df['condition_source']
#df = df.groupby(['conscious_state','model_name','roi','sub']).mean().reset_index()
df['feature_selector'] = df['model_name'].apply(get_fs)
df['estimator'] = df['model_name'].apply(get_clf)
if 'score' in df.columns:
    df['roc_auc'] = df['score']
df['roi'] = df['roi'].map(utils.get_roi_dict())
temp = np.array([item.split('-') for item in df['roi'].values])
df['roi_name'] = temp[:,1]
df['side'] = temp[:,0]

df_plot = pd.concat(df[df['conscious_state'] == state].sort_values(['roi_name','side']) for state in ['unconscious','glimpse','conscious'])
df_plot['x'] = 0

#results = dict(
#        roi_name = [],
#        side = [],
#        feature_selector = [],
#        conscious_state = [],
#        ps_mean = [],
#        ps_std = [],
#        diff = [])
#for (roi,side,feature_selector,conscious_state),df_sub in df_plot.copy().groupby([
#        'roi_name','side','feature_selector','conscious_state']):
#    
##    df_baseline = df_sub[df_sub['estimator'] == 'Dummy'].sort_values(['roi','feature_selector','conscious_state'])
#    df_est = df_sub[df_sub['estimator'] != 'Dummy'].sort_values(['roi','feature_selector','conscious_state'])
#    
#    a = df_est['roc_auc'].values
##    b = df_baseline['roc_auc'].values
#    
#    ps = utils.resample_ttest(a,0.5,n_ps=100,n_permutation=10000,one_tail=True)
#    
#    results['roi_name'].append(roi)
#    results['side'].append(side)
#    results['feature_selector'].append(feature_selector)
#    results['conscious_state'].append(conscious_state)
#    results['ps_mean'].append(ps.mean())
#    results['ps_std'].append(ps.std())
#    results['diff'].append(a.mean() - 0.5)
#
#results = pd.DataFrame(results)
#
#temp = []
#for (conscious_state),df_sub in results.groupby(['conscious_state']):
#    df_sub = df_sub.sort_values(['ps_mean'])
#    converter = utils.MCPConverter(pvals = df_sub['ps_mean'].values)
#    d = converter.adjust_many()
#    df_sub['ps_corrected'] = d['bonferroni'].values
#    temp.append(df_sub)
#results = pd.concat(temp)
#
#
#results = results.sort_values(['conscious_state','roi_name','side','feature_selector','ps_corrected'])
#results.to_csv(os.path.join(saving_dir,'decoding 4 models.csv'),index=False)
#
#results_trim = results[results['ps_corrected'] < 0.05]
#for cs,df_sub in results_trim.groupby(['conscious_state']):
#    from collections import Counter
#    df_sub['roi'] = df_sub['side'] + '-' + df_sub['roi_name']
#    print(cs,Counter(df_sub['roi']))
#    print()

df_plot['region'] = df_plot['roi_name'].map(utils.define_roi_category())
df_plot = pd.concat([df_sub for ii,df_sub in df_plot.groupby(['region','roi_name','conscious_state',
                                                  'side'])])
violin = dict(split = True,cut = 0, inner = 'quartile')
g = sns.catplot(x = 'roi_name',
                y = 'roc_auc',
                hue = 'side',
                row = 'feature_selector',
#                row_order = ['lh','rh'],
                row_order = ['None','PCA','Mutual','RandomForest'],
                col = 'conscious_state',
                col_order = ['unconscious','glimpse','conscious'],
                data = df_plot,#[df_plot['feature_selector'] == 'RandomForest'],
                kind = 'violin',
                aspect = 3,
                **violin
                )
(g.set_axis_labels('ROIs','ROC AUC')
  .set_titles('{row_name} | {col_name}')
  .set(ylim=(0.2,1.)))
[ax.set_xticklabels(pd.unique(df_plot['roi_name']),rotation = 45, ha = 'center') for ax in g.axes[-1]]
[ax.axhline(0.5,linestyle='--',color='k',alpha=0.5) for ax in g.axes.flatten()]
[ax.axvline(4.5,linestyle='-',color='k',alpha=0.5) for ax in g.axes.flatten()]
title = 'within subject decoding, fold = 100\nPymvpa pipeline'
g.fig.suptitle(title,y = 1.05)
g.savefig(os.path.join(figure_dir,'decoding as a function of roi,feature selector, conscious state (light).png'),
#          dpi = 400,
          bbox_inches = 'tight')
g.savefig(os.path.join(figure_dir,'decoding as a function of roi,feature selector, conscious state.png'),
          dpi = 400,
          bbox_inches = 'tight')


