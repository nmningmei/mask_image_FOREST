#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:13:12 2019

@author: nmei
"""

import os
from glob import glob
from tqdm import tqdm
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')

from shutil import copyfile
copyfile('../../../utils.py','utils.py')
import utils

sub = 'sub-07'
target_folder = 'LOO_cross_state'
target_file = '*None*csv'
working_dir = '../../../../results/MRI/nilearn/{}/{}'.format(sub,target_folder)
working_data = glob(os.path.join(working_dir,target_file))
beha_dir = '../../../../data/behavioral/{}'.format(sub)
results,summary_ = utils.get_frames(beha_dir,EEG = False)
saving_dir = '../../../../results/MRI/nilearn/{}/{}_stats'.format(sub,target_folder)
figure_dir = '../../../../figures/MRI/nilearn/{}/{}'.format(sub,target_folder)

for d in [saving_dir,figure_dir]:
    if not os.path.exists(d):
        os.makedirs(d)
with open(os.path.join(saving_dir,'report.txt'),'w') as f:
    f.write(summary_)
    f.close()

df = pd.concat([pd.read_csv(f) for f in working_data])
n_folds = df['fold'].max()
if 'model_name' not in df.columns:
    df['model_name'] = df['model']
#df = df.groupby(['conscious_state','model_name','roi','sub']).mean().reset_index()
df['feature_selector'] = df['model_name'].apply(utils.get_fs)
df['estimator'] = df['model_name'].apply(utils.get_clf)
if 'score' in df.columns:
    df['roc_auc'] = df['score']

temp = np.array([item.split('-') for item in df['roi'].values])
df['roi_name'] = temp[:,1]
df['side'] = temp[:,0]

df_plot = []
for state_source in ['unconscious','glimpse','conscious']:
    for state_target in ['unconscious','glimpse','conscious']:
        if state_source != state_target:
            temp = np.logical_and(df['condition_source'] == state_source,
                                  df['condition_target'] == state_target,)
            df_plot.append(df[temp].sort_values(['roi_name','side']))
df_plot = pd.concat(df_plot)

df_same_same = utils.load_same_same(sub)

sort_by = ['sub', 'roi', 'model', 'condition_target', 'condition_source', 'flip',
       'language', 'transfer', 'model_name',
       'feature_selector', 'estimator', 'roi_name', 'side']
df_plot = df_plot.sort_values(sort_by)
df_same_same = df_same_same.sort_values(sort_by)
df_plot = pd.concat([df_plot,df_same_same])

results = dict(
        roi_name = [],
        side = [],
        feature_selector = [],
        condition_target = [],
        condition_source = [],
        ps_mean = [],
        ps_std = [],
        diff = [],
        p = [],
        t = [],)
for (roi,side,feature_selector,condition_target,condition_source),df_sub in tqdm(df_plot.copy().groupby([
        'roi_name','side','feature_selector','condition_target','condition_source'])):
    
    grouping = ['roi','feature_selector',]
    df_baseline = df_sub[df_sub['estimator'] == 'Dummy'].sort_values(grouping)
    df_estimate = df_sub[df_sub['estimator'] != 'Dummy'].sort_values(grouping)
    
    a = df_estimate['roc_auc'].values
    b = df_baseline['roc_auc'].values
    
    t,p = stats.ttest_ind(a,b,)
    if a.mean() <= b.mean():
        p = 1-p/2
    else:
        p = p/2
        
    ps = utils.resample_ttest_2sample(a,b,
                                      n_ps=100,
                                      n_permutation=int(1e2),
                                      one_tail=True,
                                      n_jobs = -1,)
    
    results['roi_name'].append(roi)
    results['side'].append(side)
    results['feature_selector'].append(feature_selector)
    results['condition_source'].append(condition_source)
    results['condition_target'].append(condition_target)
    results['ps_mean'].append(ps.mean())
    results['ps_std'].append(ps.std())
    results['diff'].append(np.mean(a - b))
    results['t'].append(t)
    results['p'].append(p)

results = pd.DataFrame(results)

col = 'p'
results = results.sort_values([col])
converter = utils.MCPConverter(pvals = results[col].values)
d = converter.adjust_many()
results['ps_corrected'] = d['bonferroni'].values


results = results.sort_values(['condition_source','condition_target','roi_name','side','feature_selector','ps_corrected'])
results.to_csv(os.path.join(saving_dir,'decoding 4 models.csv'),index=False)

results['stars'] = results['ps_corrected'].apply(utils.stars)

#results_trim = results[results['ps_corrected'] < 0.05]
#for cs,df_sub in results_trim.groupby(['conscious_state']):
#    from collections import Counter
#    df_sub['roi'] = df_sub['side'] + '-' + df_sub['roi_name']
#    print(cs,Counter(df_sub['roi']))
#    print()

df_plot['region'] = df_plot['roi_name'].map(utils.define_roi_category())
df_plot = pd.concat([df_sub for ii,df_sub in df_plot.groupby(['region','roi_name','condition_target','condition_source',
                                                  'side'])])
violin = dict(split = True,cut = 0, inner = 'quartile')
df_plot = df_plot[df_plot['estimator'] != 'Dummy']

estimator = 'Linear-SVM'
g = sns.catplot(x = 'roi_name',
                y = 'roc_auc',
                hue = 'side',
                row = 'condition_source',
                row_order = ['unconscious','glimpse','conscious'],
                col = 'condition_target',
                col_order = ['unconscious','glimpse','conscious'],
                data = df_plot,
                kind = 'bar',
                aspect = 3,
#                **violin
                )
(g.set_axis_labels('ROIs','ROC AUC')
  .set_titles('{row_name} --> {col_name}')
  .set(ylim=(0.35,1.)))
[ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 90, ha = 'center') for ax in g.axes[-1]]
xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
[ax.axhline(0.5,linestyle='--',color='k',alpha=0.5) for ax in g.axes.flatten()]
[ax.axvline(6.5,linestyle='-' ,color='k',alpha=0.5) for ax in g.axes.flatten()]

# add stars
for n_row, condition_source in enumerate(['unconscious','glimpse','conscious']):
    for n_col, condition_target in enumerate(['unconscious','glimpse','conscious']):
        ax = g.axes[n_row][n_col]
        
        for ii,text_obj in enumerate(xtick_order):
            position = text_obj.get_position()
            xtick_label = text_obj.get_text()
            rows = results[np.logical_and(results['condition_source'] == condition_source,
                                          results['condition_target'] == condition_target)]
            rows = rows[rows['roi_name'] == xtick_label]
            for (ii,temp_row),adjustment in zip(rows.iterrows(),[-0.45,0.055]):
                if '*' in temp_row['stars']:
                    ax.annotate(temp_row['stars'],
                                xy = (position[0] + adjustment,0.9))

title = 'within subject - out-of-sample, fold varies\n nilearn pipeline, estimator = {}\nBoferroni corrected\n*:<0.05,**<0.01,***<0.001'.format(n_folds,estimator)
g.fig.suptitle(title,y = 1.1)
g.savefig(os.path.join(figure_dir,f'decoding ({estimator}) as a function of roi,feature selector, conscious state (light).png'),
#          dpi = 400,
          bbox_inches = 'tight')
g.savefig(os.path.join(figure_dir,f'decoding ({estimator}) as a function of roi,feature selector, conscious state.png'),
          dpi = 400,
          bbox_inches = 'tight')


