#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:27:10 2019

@author: nmei
"""

import pandas as pd
import numpy as np

from tqdm import tqdm

df = pd.read_csv('../../../../results/MRI/nilearn/sub-01/L2O/4 models decoding (sub-01 lh-fusiform unconscious PCA + Linear-SVM).csv')
df_plot = pd.DataFrame(index = pd.unique(df['test1']),columns = pd.unique(df['test2']))

for (a,b), df_sub in tqdm(df.groupby(['test1','test2'])):
    df_plot.loc[a,b] = df[np.logical_and(df['test1'] == a,
                                         df['test2'] == b)
                         ]['accuracy'].values[0]
    df_plot.loc[b,a] = df[np.logical_and(df['test1'] == a,
                                         df['test2'] == b)
                         ]['accuracy'].values[0]
df_plot = df_plot[df_plot.index]
ddf = df_plot.fillna(0)
















