#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:00:10 2019

@author: nmei
"""

import pandas as pd
import os
from glob import glob
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')
from matplotlib import pyplot as plt

working_dir = '../data/behavioral/'
working_data = glob(os.path.join(working_dir,'*2019','*trials.csv'))
df = []
for f in working_data:
    temp = pd.read_csv(f).dropna()
    temp['sub'] = f.split('/')[3]
    df.append(temp)
df = pd.concat(df)

visible_map = {"'1'":'unconscious',"'2'":'see_maybe',"'3'":'see_some',"'4'":'conscious'}
visible_order = ['unconscious','see_maybe','see_some','conscious']
df['visible.keys_raw'] = df['visible.keys_raw'].map(visible_map)


fig, axes = plt.subplots(figsize = (10,15),nrows = 2)
for (sub,df_sub),ax in zip(df.groupby(['sub']),axes.flatten()):
    sns.countplot(x = 'response.keys_raw',
                  hue = 'visible.keys_raw',
                  hue_order = visible_order,
                  data = df_sub,
                  ax = ax)
    ax.set_title(sub)
fig.tight_layout()
fig.savefig('count.png')


working_data = glob(os.path.join(working_dir,'*2019','*log'))
for f in working_data:
    with open(os.path.join(working_dir,'{}.txt'.format(f.split('/')[3])),'a') as input_file:
        with open(f,'r') as log_file:
            for line in log_file:
                if 'trigger' in line:
                    input_file.write(line)






































