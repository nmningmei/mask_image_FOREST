#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:13:12 2019

@author: nmei
"""

import os
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')

from shutil import copyfile
copyfile('../../../utils.py','utils.py')
import utils

sub = 'sub-01'
target_folder = 'LOO_cross_state'
working_dir = '../../../../results/MRI/nilearn/{}/{}'.format(sub,target_folder)
beha_dir = '../../../../data/behavioral/{}'.format(sub)
results,summary_ = utils.get_frames(beha_dir,EEG = False)
saving_dir = '../../../../results/MRI/nilearn/{}/{}_stats'.format(sub,target_folder)
figure_dir = '../../../../figures/MRI/nilearn/{}/{}'.format(sub,target_folder)
n_jobs = 8

for d in [saving_dir,figure_dir]:
    if not os.path.exists(d):
        os.makedirs(d)
with open(os.path.join(saving_dir,'report.txt'),'w') as f:
    f.write(summary_)
    f.close()

def get_fs(x):
    return x.split(' + ')[0]
def get_clf(x):
    return x.split(' + ')[1]
def rename_roi(x):
    return x.split('-')[-1] + '-' + x.split('-')[1]

working_data    = glob(os.path.join(working_dir,'4 models decoding*.csv'))
df              = pd.concat([pd.read_csv(f) for f in working_data])
n_folds         = 'varies'

if 'model_name' not in df.columns:
    df['model_name'] = df['model']


df['feature_selector']  = df['model_name'].apply(get_fs)
df['estimator']         = df['model_name'].apply(get_clf)
if 'score' in df.columns:
    df['roc_auc'] = df['score']

temp            = np.array([item.split('-') for item in df['roi'].values])
df['roi_name']  = temp[:,1]
df['side']      = temp[:,0]

df_plot         = pd.concat(df[df['condition_source'] == state].sort_values(['roi_name','side']) for state in ['unconscious','glimpse','conscious'])
df_plot['x']    = 0

if not os.path.exists(os.path.join(saving_dir,'LOO 4 models.csv')):
    results = dict(
            roi_name            = [],
            side                = [],
            feature_selector    = [],
            conscious_source    = [],
            conscious_target    = [],
            ps_mean             = [],
            ps_std              = [],
            p                   = [],
            t                   = [],
            diff                = [],
            )
    for (roi,
         side,
         feature_selector,
         conscious_source,
         conscious_target,),df_sub in tqdm(df_plot.copy().groupby([
            'roi_name',
            'side',
            'feature_selector',
            'condition_source',
            'condition_target',])):
        
        grouping = ['roi','feature_selector','condition_source','condition_target']
        df_baseline = df_sub[df_sub['estimator'] == 'Dummy'].sort_values(grouping)
        df_estimate = df_sub[df_sub['estimator'] != 'Dummy'].sort_values(grouping)
        
        a = df_estimate['roc_auc'].values
        b = df_baseline['roc_auc'].values
        
        t,p = stats.ttest_ind(a,b,)
        if a.mean() <= b.mean():
            p = 1-p/2
        else:
            p = p/2
        
        ps = utils.resample_ttest_2sample(a,b,n_ps = 100,
                                          n_permutation=int(1e5),
                                          one_tail = True,
                                          n_jobs = n_jobs,)
        
        results['roi_name'].append(roi)
        results['side'].append(side)
        results['feature_selector'].append(feature_selector)
        results['conscious_source'].append(conscious_source)
        results['conscious_target'].append(conscious_target)
        results['ps_mean'].append(ps.mean())
        results['ps_std'].append(ps.std())
        results['p'].append(p)
        results['t'].append(t)
        results['diff'].append(np.mean(a - b))
    
    results = pd.DataFrame(results)
    
    temp = []
    for (conscious_source,conscious_target,),df_sub in results.groupby(['conscious_source',
            'conscious_target']):
        df_sub = df_sub.sort_values(['ps_mean'])
        converter = utils.MCPConverter(pvals = df_sub['ps_mean'].values)
        d = converter.adjust_many()
        df_sub['ps_corrected'] = d['bonferroni'].values
        temp.append(df_sub)
    results = pd.concat(temp)
    
    results['stars'] = results['ps_corrected'].apply(utils.stars)
    results = results.sort_values(['conscious_source','conscious_target','roi_name','side','feature_selector','ps_corrected'])
    results.to_csv(os.path.join(saving_dir,'LOO 4 models.csv'),index=False)
else:
    results = pd.read_csv(os.path.join(saving_dir,'LOO 4 models.csv'))


#results_trim = results[results['ps_corrected'] < 0.05]
#for cs,df_sub in results_trim.groupby(['conscious_state']):
#    from collections import Counter
#    df_sub['roi'] = df_sub['side'] + '-' + df_sub['roi_name']
#    print(cs,Counter(df_sub['roi']))
#    print()



df_plot['region'] = df_plot['roi_name'].map(utils.define_roi_category())
df_plot = pd.concat([df_sub for ii,df_sub in df_plot.groupby([
        'region','roi_name','condition_source','condition_target','side'])])
violin = dict(split = True,cut = 0, inner = 'quartile')
df_plot = df_plot[df_plot['estimator'] != 'Dummy']

for estimator in pd.unique(df_plot['estimator']):
    df_plot_ = df_plot[df_plot['estimator'] == estimator]
    
    g = sns.catplot(x = 'roi_name',
                    y = 'roc_auc',
                    hue = 'side',
                    col = 'condition_source',
                    col_order = ['unconscious','glimpse','conscious'],
                    row = 'condition_target',
                    row_order = ['unconscious','glimpse','conscious'],
                    data = df_plot_,
                    kind = 'bar',
                    aspect = 3,
                    )
    (g.set_axis_labels('ROIs','ROC AUC')
      .set_titles('{row_name}')
      .set(ylim=(0.,1.)))
    [ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 90, ha = 'center') for ax in g.axes[-1]]
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
                for (ii,temp_row),adjustment in zip(rows.iterrows(),[-0.3,0.1]):
                    if '*' in temp_row['stars']:
                        ax.annotate(temp_row['stars'],
                                    xy = (position[0] + adjustment,0.9))
    
    title = 'within subject decoding, fold = {}\n nilearn pipeline, estimator = {}\nBoferroni corrected within consciousness state\n*:<0.05,**<0.01,***<0.001'.format(n_folds,estimator)
    g.fig.suptitle(title,y = 1.1)
    g.savefig(os.path.join(figure_dir,f'decoding ({estimator}) as a function of roi,feature selector, conscious state (light) _temp.png'),
    #          dpi = 400,
              bbox_inches = 'tight')
    g.savefig(os.path.join(figure_dir,f'decoding ({estimator}) as a function of roi,feature selector, conscious state _temp.png'),
              dpi = 400,
              bbox_inches = 'tight')


