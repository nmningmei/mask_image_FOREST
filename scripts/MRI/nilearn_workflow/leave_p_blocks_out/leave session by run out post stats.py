#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:58:17 2020

@author: nmei
"""

import os
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')

from matplotlib import pyplot as plt
from shutil import copyfile
copyfile('../../../utils.py','utils.py')
import utils

sub = 'sub-01'
here = 'LSRO'
working_dir = f'../../../../results/MRI/nilearn/{sub}/{here}'
working_data = glob(os.path.join(working_dir,'*.csv'))

saving_dir = '../../../../results/MRI/nilearn/{}/{}_stats'.format(sub,here)
figure_dir = '../../../../figures/MRI/nilearn/{}/{}'.format(sub,here)

for d in [saving_dir,figure_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

def get_fs(x):
    return x.split(' + ')[0]
def get_clf(x):
    return x.split(' + ')[1]
def rename_roi(x):
    return x.split('-')[-1] + '-' + x.split('-')[1]

df = pd.concat([pd.read_csv(f) for f in working_data])

n_folds = df['fold'].max()
if 'model_name' not in df.columns:
    df['model_name'] = df['model']
if 'conscious_state' not in df.columns:
    df['conscious_state'] = df['condition_source']
#df = df.groupby(['conscious_state','model_name','roi','sub']).mean().reset_index()
df['feature_selector'] = df['model_name'].apply(get_fs)
df['estimator'] = df['model_name'].apply(get_clf)
if 'score' in df.columns:
    df['roc_auc'] = df['score']

temp = np.array([item.split('-') for item in df['roi'].values])
df['roi_name'] = temp[:,1]
df['side'] = temp[:,0]

df_plot = pd.concat(df[df['conscious_state'] == state].sort_values(['roi_name','side']) for state in ['unconscious','glimpse','conscious'])
df_plot['x'] = 0
if not os.path.exists(os.path.join(saving_dir,f'{here}.csv')):
    results = dict(
            roi_name = [],
            side = [],
            feature_selector = [],
            conscious_state = [],
            ps_mean = [],
            ps_std = [],
            p = [],
            t = [],
            diff = [])
    for (roi,side,feature_selector,conscious_state),df_sub in tqdm(df_plot.copy().groupby([
            'roi_name','side','feature_selector','conscious_state'])):
        from scipy import stats
        df_baseline = df_sub[df_sub['estimator'] == 'Dummy'].sort_values(['roi','feature_selector','conscious_state'])
        df_est = df_sub[df_sub['estimator'] != 'Dummy'].sort_values(['roi','feature_selector','conscious_state'])
        
        a = df_est['roc_auc'].values
        b = df_baseline['roc_auc'].values
        
        t,p = stats.ttest_ind(a,b,)
        if a.mean() <= b.mean():
            p = 1-p/2
        else:
            p = p/2
        np.random.seed(12345)
        ps = utils.resample_ttest_2sample(a,b,n_ps = 10,
                                          n_permutation=int(1e5),
                                          one_tail = True,
                                          match_sample_size = False,
                                          n_jobs = -1,
                                          verbose = 1,
                                          )
        
        results['roi_name'].append(roi)
        results['side'].append(side)
        results['feature_selector'].append(feature_selector)
        results['conscious_state'].append(conscious_state)
        results['ps_mean'].append(ps.mean())
        results['ps_std'].append(ps.std())
        results['p'].append(p)
        results['t'].append(t)
        results['diff'].append(np.mean(a - 0.5))
    
    results = pd.DataFrame(results)
    
    temp = []
    for (conscious_state),df_sub in results.groupby(['conscious_state']):
        df_sub = df_sub.sort_values(['ps_mean'])
        converter = utils.MCPConverter(pvals = df_sub['ps_mean'].values)
        d = converter.adjust_many()
        df_sub['ps_corrected'] = d['bonferroni'].values
        temp.append(df_sub)
    results = pd.concat(temp)
    
    results['stars'] = results['ps_corrected'].apply(utils.stars)
    results = results.sort_values(['conscious_state','roi_name','side','feature_selector','ps_corrected'])
    results.to_csv(os.path.join(saving_dir,f'{here}.csv'),index=False)
else:
    results = pd.read_csv(os.path.join(saving_dir,f'{here}.csv'))



df_plot['region'] = df_plot['roi_name'].map(utils.define_roi_category())
df_plot = pd.concat([df_sub for ii,df_sub in df_plot.groupby(['region','roi_name','conscious_state',
                                                  'side'])])
violin = dict(split = True,cut = 0, inner = 'quartile')
df_plot = df_plot[df_plot['estimator'] != 'Dummy']
for estimator in pd.unique(df_plot['estimator']):
    df_plot_ = df_plot[df_plot['estimator'] == estimator]
    
    g = sns.catplot(x = 'roi_name',
                    y = 'roc_auc',
                    hue = 'side',
                    col = 'feature_selector',
                    col_order = ['None'],#,'PCA','Mutual','RandomForest'],
                    row = 'conscious_state',
                    row_order = ['unconscious','glimpse','conscious'],
                    data = df_plot_,
                    kind = 'violin',
                    aspect = 3,
                    **violin
                    )
    (g.set_axis_labels('ROIs','ROC AUC')
      .set_titles('{row_name}')
      .set(ylim=(0.,1.2)))
    [ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 45, ha = 'center') for ax in g.axes[-1]]
    xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
    [ax.axhline(0.5,linestyle='--',color='k',alpha=0.5) for ax in g.axes.flatten()]
    [ax.axvline(6.5,linestyle='-' ,color='k',alpha=0.5) for ax in g.axes.flatten()]
    
    # add stars
    for n_col,feature_selector in enumerate(['None']):#,'PCA','Mutual','RandomForest']):
        for n_row, conscious_state in enumerate(['unconscious','glimpse','conscious']):
            ax = g.axes[n_row][n_col]
            
            for ii,text_obj in enumerate(xtick_order):
                position = text_obj.get_position()
                xtick_label = text_obj.get_text()
                rows = results[np.logical_and(results['feature_selector'] == feature_selector,
                                         results['conscious_state'] == conscious_state)]
                rows = rows[rows['roi_name'] == xtick_label]
                for (ii,temp_row),adjustment in zip(rows.iterrows(),[-0.2,0.2]):
                    if '*' in temp_row['stars']:
                        ax.annotate(temp_row['stars'],
                                    xy = (position[0] + adjustment,1.1),
                                    ha='center')
    
    title = 'within subject decoding, leave 3 runs out, fold = {}\n nilearn pipeline, estimator = {}\nBoferroni corrected within consciousness state\n*:<0.05,**<0.01,***<0.001'.format(n_folds,estimator)
    g.fig.suptitle(title,y = 1.1)