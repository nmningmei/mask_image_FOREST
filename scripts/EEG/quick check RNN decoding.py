#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:30:41 2019

@author: nmei
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')
from glob import glob
from matplotlib import pyplot as plt

folder_name1 = "RNN"
working_dir1 = f'../../results/EEG/{folder_name1}'
working_data1 = glob(os.path.join(working_dir1,'*.csv'))

folder_name2 = 'decode_premask_baseline_ICA'
working_dir2 = f'../../results/EEG/{folder_name2}/*/'
working_data2 = glob(os.path.join(working_dir2,'whole_seg*.npy'))

df = []
for ii,f in enumerate(working_data1):
    temp = pd.read_csv(f)
    temp['sub'] = ii + 1
    df.append(temp)
df = pd.concat(df)
df['chance'] = df['initial']
df['RNN'] = df['score']
df_plot = df.groupby(['sub','conscious_state']).mean().reset_index()

temp = []
for conscious_state in pd.unique(df_plot['conscious_state']):
    picked = [item for item in working_data2 if(f'_{conscious_state}' in item)]
    data = np.array([np.load(item) for item in picked])
    temp.append(data.mean(1))
temp = np.concatenate(temp)

df_plot['logistic'] = temp

df_plot_ = pd.melt(df_plot,id_vars = ['sub','conscious_state'],value_vars = ['RNN','logistic'])

fig,ax = plt.subplots(figsize=(8,6))
ax = sns.violinplot(x = 'conscious_state',
                 y = 'value',
                 hue = 'variable',
                 data = df_plot_,
                 split = True,
                 cut = 0,
                 inner = 'quartile',
                 ax = ax,
                 )
ax.axhline(0.5,linestyle = '--',color = 'black',alpha = 0.5)


