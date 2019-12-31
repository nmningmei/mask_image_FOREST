#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:36:38 2019

@author: nmei

this script is to access the behavioral statistics of the uncon_feat
fMRI experiment for each of the subjects
The statistical significance is estimated by a permutation test


"""

import os
import gc
from glob import glob

import numpy  as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.utils   import shuffle
from matplotlib      import pyplot as plt
from joblib          import Parallel,delayed
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('paper')

from shutil import copyfile
copyfile('../../utils.py','utils.py')
import utils

figure_dir      = '../../../figures/MRI/nilearn/behavioral'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
working_dir     = '../../../data/behavioral'
working_data    = glob(os.path.join(working_dir,'sub-0*','*','*.csv'))

df              = []
for f in working_data:
#    frames,_ = utils.get_frames(directory = os.path.abspath(
#            os.path.join(working_dir,f)),new = True,EEG = 'fMRI')
    
    df_temp         = pd.read_csv(f).dropna()
    df_temp['sub']  = f.split('/')[-3]
    df.append(df_temp)
df                  = pd.concat(df)

n_sim = int(1e5)
n_sample = int(2e3)
results = dict(pval         = [],
               sub          = [],
               accuracy     = [],
               chance_mean  = [],
               chance_std   = [],
               visibility   = [])
for sub,df_sub in df.groupby(['sub']):
    for col in ['probe_Frames_raw',
                'response.keys_raw',
                'visible.keys_raw']:
        df_sub[col] = df_sub[col].apply(utils.str2int)
    for visibility,df_sub_vis in df_sub.groupby(['visible.keys_raw']):
        correct_ans     = df_sub_vis['correctAns_raw'].values.astype('int32')
        responses       = df_sub_vis['response.keys_raw'].values.astype('int32')
        score           = roc_auc_score(correct_ans,responses)
        
        experiment      = score
        np.random.seed(12345)
        gc.collect()
        def _chance(responses,correct_ans):
            idx_        = np.random.choice(np.arange(len(responses)),
                                           n_sample,
                                           replace = True)
            random_responses = shuffle(responses)
            return roc_auc_score(correct_ans[idx_],random_responses[idx_])
        
        chance_level          = Parallel(n_jobs = -1,verbose = 1)(delayed(_chance)(**{
                            'responses':responses,
                            'correct_ans':correct_ans}) for i in range(n_sim))
        chance_level    = np.array(chance_level)
        pval            = (np.sum(chance_level > experiment) + 1) / (n_sim + 1)
        results['sub'           ].append(sub)
        results['pval'          ].append(pval)
        results['accuracy'      ].append(experiment)
        results['chance_mean'   ].append(chance_level.mean())
        results['chance_std'    ].append(chance_level.std() / np.sqrt(n_sim))
        results['visibility'    ].append(visibility)

results = pd.DataFrame(results)
#df_plot = pd.melt(results,
#                  id_vars = ['sub','visibility','pval','chance_std'],
#                  value_vars = ['accuracy','chance_mean'],
#                  )
#df_plot.columns = ['sub','visibility','pval','chance_std','kind','score']
df_plot     = results.copy()
df_plot     = df_plot.sort_values(['pval'])
converter   = utils.MCPConverter(pvals = df_plot['pval'].values)
d           = converter.adjust_many()
df_plot['p_corrected'] = d['bonferroni'].values
df_plot['Behavioral' ] = df_plot['p_corrected'] < 0.05
df_plot['Behavioral' ] = df_plot['Behavioral'].map({True:'Above Chance',False:'At Chance'})
df_plot['visibility' ] = df_plot['visibility'].map({
        1:'Unconscious',
        2:'Glimpse',
        3:'Conscious',})
df_plot = df_plot.sort_values(['visibility'],ascending = False)

fig,ax = plt.subplots(figsize = (8,6))
ax = sns.swarmplot(
                 x      = 'visibility',
                 y      = 'accuracy',
                 hue    = 'Behavioral',
                 size   = 12,
                 data   = df_plot,
                 ax     = ax,)
ax.set(xlabel = 'Conscious State',
       ylabel = 'ROC AUC')
fig.savefig(os.path.join(figure_dir,
                         'behaviroal.png'),
            dpi = 400,
            bbox_inches = 'tight')
