#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:28:31 2019

@author: nmei
"""

import os
import re
from glob import glob
import pandas as pd
import numpy as np


sub = 'ning_4_6_2019'
working_dir = '../../data/behavioral/{}'.format(sub)
working_data = glob(os.path.join(working_dir,'*trials.csv'))

visible_map = {1:'unconscious',
               2:'glimpse',
               3:'conscious',
               99:'missing data'}

def read(f):
    temp = pd.read_csv(f).iloc[:-12,:]
    return temp
def extract(x):
    try:
        return int(re.findall('\d',x)[0])
    except:
        return int(99)
df = pd.concat([read(f) for f in working_data])
numerical_columns = ['probe_Frames_raw',
                     'response.keys_raw',
                     'visible.keys_raw',]
for col_name in numerical_columns:
    df[col_name] = df[col_name].apply(extract)

for visible,df_sub in df.groupby(['visible.keys_raw']):
    probe_frame = df_sub['probe_Frames_raw'].values
    correct = df_sub['response.corr_raw'].values
    print('{:12},probe presented for {:.3f}+/-{:.3f}, p(correct) = {:.3f} for {:3} trials'.format(
            visible_map[visible],probe_frame.mean(),probe_frame.std(),
            correct.mean(),df_sub.shape[0]))