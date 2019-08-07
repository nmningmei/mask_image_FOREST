#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:50:46 2019

@author: nmei
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')

from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from shutil import copyfile
copyfile('../utils.py','utils.py')
import utils

all_subjects = ['aingere_5_16_2019',
                'alba_6_10_2019',
                'alvaro_5_16_2019',
                'clara_5_22_2019',
                'ana_5_21_2019',
                'inaki_5_9_2019',
                'jesica_6_7_2019',
                'leyre_5_13_2019',
                'lierni_5_20_2019',
                'maria_6_5_2019',
                'matie_5_23_2019',
                'out_7_19_2019',
                'mattin_7_12_2019',
                'pedro_5_14_2019',
                'xabier_5_15_2019',
                ]
all_subjects = np.sort(all_subjects)

working_dir = '../../data/behavioral'
figure_dir = '../../figures/EEG/behaviorals'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
saving_dir = '../../results/EEG/behaviorals'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

for sub_name in all_subjects:
    df = pd.concat([pd.read_csv(f).dropna() for f in glob(os.path.join(working_dir,
                                                    sub_name,
                                                    '*trials.csv'))])
    for col in ['probeFrames_raw',
                'response.keys_raw',
                'visible.keys_raw']:
        df[col] = df[col].apply(utils.str2int)
    df['visibility'] = df['visible.keys_raw'].map({1.:'unconscious',2.:'glimpse',3.:'conscious'})
    df = df.sort_values(['visible.keys_raw'])
    
    # RT as a function of visibility
    fig,ax = plt.subplots(figsize = (8,6))
    ax = sns.barplot(x = 'visibility',
                     y = 'response.rt_raw',
                     data = df,
                     ax = ax)
    ax.set(xlabel = 'Visibility Ratings',ylabel = 'Reaction Time (sec)',
           title = f'{sub_name}')
    fig.savefig(os.path.join(figure_dir,
                             f'{sub_name}_RT_visibility.png'),
                dpi = 400,
                bbox_inches = 'tight')
    
    # distribution of accuracy as a function of visibility
    n_bootstrap = int(5e3)
    df_acc_name = os.path.join(saving_dir,
                               f'{sub_name}_bootstrapping.csv')
    if os.path.exists(df_acc_name):
        df_acc = pd.read_csv(df_acc_name)
        df_p_val = pd.read_csv(df_acc_name.replace('bootstrapping','stats'))
    else:
        df_acc = dict(visibility = [],
                      score = [])
        for visibility,df_sub in df.groupby(['visibility']):
            responses = df_sub['response.keys_raw'].values - 1
            answers = df_sub['correctAns_raw'].values - 1
            np.random.seed(12345)
            for n_ in tqdm(range(n_bootstrap)):
                idx = np.random.choice(np.arange(responses.shape[0]),size = int(responses.shape[0]*0.8),
                                       replace = True)
                
                response_ = responses[idx]
                answer_ = answers[idx]
                score_ = roc_auc_score(answer_,response_)
                df_acc['visibility'].append(visibility)
                df_acc['score'].append(score_)
        df_acc = pd.DataFrame(df_acc)
        df_acc.to_csv(df_acc_name,index = False)
        df_p_val = dict(visibility = [],
                        ps_mean = [],)
        for visibility,df_sub in df_acc.groupby(['visibility']):
            scores = df_sub['score'].values
            df_temp = df[df['visibility'] == visibility]
            # by keeping the answers in order but shuffle the response, we can estimate the chance
            # level accuracy
            chance = np.array([roc_auc_score(df_temp['correctAns_raw'].values - 1,
                                             shuffle(df_temp['response.keys_raw'].values - 1))\
                        for _ in tqdm(range(n_bootstrap))])
            pvals = utils.resample_ttest_2sample(scores,chance,one_tail = True,match_sample_size = True)
            df_p_val['visibility'].append(visibility)
            df_p_val['ps_mean'].append(pvals.mean())
        df_p_val = pd.DataFrame(df_p_val)
        df_p_val = df_p_val.sort_values(['ps_mean'])
        pvals = df_p_val['ps_mean'].values
        converter = utils.MCPConverter(pvals = pvals)
        d = converter.adjust_many()
        df_p_val['ps_corrected'] = d['bonferroni'].values
        df_p_val['star'] = df_p_val['ps_corrected'].apply(utils.stars)
        df_p_val.to_csv(df_acc_name.replace('bootstrapping','stats'),index = False)
    fig,ax = plt.subplots(figsize = (8,6))
    ax = sns.barplot(x = 'visibility',
                     y = 'score',
                     ci = 'sd',
                     data = df_acc,
                     )
    ax.set(xlabel = 'Visibility Ratings', ylabel = 'Accuracy',
           title = f'{sub_name}_bootstrapped_acc',
           ylim = (0.4,1.05))
    
    for ii,(visibility,row) in enumerate(df_p_val.groupby(['visibility'])):
        ax.annotate(row['star'].values[0],xy = (ii - 0.1,1.01))
    fig.savefig(os.path.join(figure_dir,
                             f'{sub_name}_bootstrapped_acc.png'),
    dpi = 300,
    bbox_inches = 'tight')
    
    






































