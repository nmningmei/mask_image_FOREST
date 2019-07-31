#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:01:10 2019

@author: nmei
"""

import os
from glob import glob
import pandas as pd
import numpy as np
from collections import Counter

working_dir = '../../data/behavioral/Patxi_3_22_2019'
working_data = glob(os.path.join(working_dir,'*trials.csv'))
mapping = {"'1.0'":1,"'2.0'":2,"'3.0'":3,"'4.0'":4,
           "'5.0'":5,"'6.0'":6,"'7.0'":7,
           "'1'":1,"'2'":2,"'3'":3,"'4'":4,
           "'5'":5,"'6'":6,"'7'":7,
           '1':1,'2':2,'3':3,'4':4,'5':5,
           '6':6,'7':7,
           'nan':np.nan}
df_full = []
for f in working_data:
    df = pd.read_csv(f).iloc[:32,:]
    try:
        df['probeFrames_raw'] = np.array(df['probeFrames_raw'].values,dtype=np.float32)
    except:
        df['probeFrames_raw'] = df['probeFrames_raw'].astype(str)
        df['probeFrames_raw'] = df['probeFrames_raw'].map(mapping)
    df['probeFrames_raw'] = df['probeFrames_raw'].fillna(99)
    print(Counter(df['probeFrames_raw']))
    CC = dict(Counter(df['probeFrames_raw']))
    if CC[1.0] > 16:
        print(f)
        temp = df.copy()
        temp['response.keys_raw'] = temp['response.keys_raw'].fillna("'3'")
        temp['visible.keys_raw'] = temp['visible.keys_raw'].fillna("'2'")
        
        temp['response.keys_raw'] = temp['response.keys_raw'].map(mapping)
        temp['response.keys_raw'] = temp['response.keys_raw'] - 1
        
        temp['visible.keys_raw'] = temp['visible.keys_raw'].map(mapping)
        temp['visible.keys_raw'] = temp['visible.keys_raw'] - 1
        df_full.append(temp)
    else:
        df['response.keys_raw'] = df['response.keys_raw'].map(mapping)
        
        df['visible.keys_raw'] = df['visible.keys_raw'].map(mapping)
        
        df_full.append(df)
df_full = pd.concat(df_full)
with open(os.path.join(working_dir,'behavioral report.txt'),'a') as f:
    for (vis),df_sub in df_full.groupby(['visible.keys_raw']):
        print(f'visible = {vis}')
        print(f'# of trials = {df_sub.shape[0]}')
        print(f'p(correct) = {df_sub["response.corr_raw"].mean():.4f}')
        print(f'response balance: {Counter(df_sub["response.keys_raw"])}')
        print(f'class balance: {Counter(df_sub["category"])}')
        print()
        
        f.write(f'visible = {vis}\n')
        f.write(f'# of trials = {df_sub.shape[0]}\n')
        f.write(f'p(correct) = {df_sub["response.corr_raw"].mean():.4f}\n')
        f.write(f'response balance: {Counter(df_sub["response.keys_raw"])}\n')
        f.write(f'class balance: {Counter(df_sub["category"])}\n\n')
    f.close()



















