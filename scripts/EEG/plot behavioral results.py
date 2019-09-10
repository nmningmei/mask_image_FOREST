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
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
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
np.random.seed(12345)
for sub_name in all_subjects:
    df = pd.concat([pd.read_csv(f).dropna() for f in glob(os.path.join(working_dir,
                                                    sub_name,
                                                    '*trials.csv'))])
    for col in ['probeFrames_raw',
                'response.keys_raw',
                'visible.keys_raw']:
        df[col] = df[col].apply(utils.str2int)
    df['visibility'] = df['visible.keys_raw'].map({1.:'unconscious',2.:'glimpse',
      3.:'conscious'})
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
    n_bootstrap = int(30)
    if n_bootstrap < df.shape[0] / 10:
        cv = StratifiedKFold(n_splits = n_bootstrap,shuffle = True,random_state = 12345)
    else:
        cv = StratifiedShuffleSplit(n_splits = n_bootstrap,
                                    test_size = 0.2,
                                    random_state = 12345)
    df_acc_name = os.path.join(saving_dir,
                               f'{sub_name}_bootstrapping.csv')
    if os.path.exists(df_acc_name.replace('bootstrapping','stats')):
        df_acc = pd.read_csv(df_acc_name)
        df_acc_chance = pd.read_csv(df_acc_name.replace('bootstrapping','chance'))
        df_p_val = pd.read_csv(df_acc_name.replace('bootstrapping','stats'))
    else:
        df_acc = dict(visibility = [],
                      score = [])
        for visibility,df_sub in df.groupby(['visibility']):
            responses = df_sub['response.keys_raw'].values - 1
            answers = df_sub['correctAns_raw'].values - 1
            for idx,_ in cv.split(responses,answers):
                response_ = responses[idx]
                answer_ = answers[idx]
                score_ = roc_auc_score(answer_,response_,average='micro')
                df_acc['visibility'].append(visibility)
                df_acc['score'].append(score_)
        df_acc = pd.DataFrame(df_acc)
        df_acc.to_csv(df_acc_name,index = False)


        df_chance = dict(visibility = [],
                         score = [])
        for visibility,df_sub in df.groupby(['visibility']):
            responses = df_sub['response.keys_raw'].values - 1
            answers = df_sub['correctAns_raw'].values - 1
            for idx,_ in cv.split(responses,answers):
                response_ = responses[idx]
                answer_ = answers[idx]
                score_ = roc_auc_score(answer_,shuffle(response_),average='micro')
                df_chance['visibility'].append(visibility)
                df_chance['score'].append(score_)
        df_chance = pd.DataFrame(df_chance)
        df_chance.to_csv(df_acc_name.replace('bootstrapping','chance'))
        
        df_p_val = dict(diff_mean = [],
                        diff_std = [],
                        visibility = [],
                        ps_mean = [],
                        ps_std = [],
                        p = [],
                        t = [],)
        for visibility,df_acc_sub, in df_acc.groupby('visibility'):
            df_chance_sub = df_chance[df_chance['visibility'] == visibility]
            ps = utils.resample_ttest_2sample(df_acc_sub['score'].values,
                                              df_chance_sub['score'].values,
                                              one_tail = True,
                                              match_sample_size = False)
            from scipy import stats
            t,p = stats.ttest_rel(df_acc_sub['score'].values,
                                  df_chance_sub['score'].values,)
            if df_acc_sub['score'].values.mean() <= df_chance_sub['score'].values.mean():
                p = 1-p/2
            else:
                p = p/2
            df_p_val['diff_mean'].append(np.mean(df_acc_sub['score'].values - df_chance_sub['score'].values)
            )
            df_p_val['diff_std'].append(np.std(df_acc_sub['score'].values - df_chance_sub['score'].values)
            )
            df_p_val['visibility'].append(visibility)
            df_p_val['ps_mean'].append(ps.mean())
            df_p_val['ps_std'].append(ps.std())
            df_p_val['p'].append(p)
            df_p_val['t'].append(t)
        
        df_p_val = pd.DataFrame(df_p_val)
        df_p_val = df_p_val.sort_values(['ps_mean'])
        pvals = df_p_val['ps_mean'].values
        converter = utils.MCPConverter(pvals = pvals)
        d = converter.adjust_many()
        df_p_val['ps_corrected'] = d['bonferroni'].values
        df_p_val['star'] = df_p_val['ps_corrected'].apply(utils.stars)
        df_p_val.to_csv(df_acc_name.replace('bootstrapping','stats'),index = False)
        df_acc['type'] = 'score'
        df_chance['type'] = 'chance'
    df_acc_plot = pd.concat([df_acc,df_chance])
    fig,ax = plt.subplots(figsize = (8,6))
    ax = sns.barplot(x = 'visibility',
                     y = 'score',
                     hue = 'type',
                     ci = 'sd',
                     data = df_acc_plot,
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
    
    plt.close('all')






































