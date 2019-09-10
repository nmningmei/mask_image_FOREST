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

df = []
for ii,f in enumerate(os.listdir(os.getcwd())):
    temp = pd.read_csv(f)
    temp['sub'] = ii + 1
    df.append(temp)
df = pd.concat(df)

df_plot = df.groupby(['sub','conscious_state']).mean().reset_index()

g = sns.factorplot(x = 'conscious_state',
                   y = 'score',
                   kind = 'violin',
                   data = df_plot,
                   aspect = 1.5,
                   **dict(cut = 0,
                          inner = 'quartile',))
g.set(ylim=(0.4,0.7))
g.axes[0][0].axhline(0.5)
g.savefig('temp.png',dpi = 400)

a = df_plot[df_plot['conscious_state'] == 'unconscious']['score'].values
null = a - a.mean() + 0.5
null_dist = np.random.choice(null,size = (a.shape[0],10000),replace = True).mean(0)
p = (np.sum(null_dist >= a.mean()) + 1) / (10000 + 1)
