#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:02:51 2019

@author: nmei
"""

import os
from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')

roi_dict ={'lh_fusif': 'ctx-lh-fusiform',
           'lh_infpar': 'ctx-lh-inferiorparietal',
           'lh_inftemp': 'ctx-lh-inferiortemporal',
           'lh_latoccip': 'ctx-lh-lateraloccipital',
           'lh_lingual': 'ctx-lh-lingual',
           'lh_middlefrontal': 'ctx-lh-rostralmiddlefrontal',
           'lh_phipp':' ctx-lh-parahippocampal',
           'lh_precun': 'ctx-lh-precuneus',
           'lh_sfrontal':'ctx-lh-superiorfrontal',
           'lh_superiorparietal': 'ctx-lh-superiorparietal',
           'lh_ventrollateralPFC': 'ctx-lh-ventrolateralPFC',
           'lh_pericalc':'ctx-lh-pericalcarine',
           'rh_fusif': 'ctx-rh-fusiform',
           'rh_infpar': 'ctx-rh-inferiorparietal',
           'rh_inftemp': 'ctx-rh-inferiortemporal',
           'rh_latoccip': 'ctx-rh-lateraloccipital',
           'rh_lingual': 'ctx-rh-lingual',
           'rh_middlefrontal': 'ctx-rh-rostralmiddlefrontal',
           'rh_phipp': 'ctx-rh-parahippocampal',
           'rh_precun': 'ctx-rh-precuneus',
           'rh_sfrontal':'ctx-rh-superiorfrontal',
           'rh_superiorparietal':'ctx-rh-superiorparietal',
           'rh_ventrollateralPFC': 'ctx-rh-ventrolateralPFC',
           'rh_pericalc':'ctx-rh-pericalcarine'}


pymvpa_dir = '../../results/pymvpa'
nilearn_dir = '../../results/MRI/decoding'

pymvpa_data = glob(os.path.join(pymvpa_dir,'*.csv'))
nilearn_data = glob(os.path.join(nilearn_dir,'*.csv'))
df_pymvpa = pd.concat([pd.read_csv(f) for f in pymvpa_data])
df_nilearn = pd.concat([pd.read_csv(f) for f in nilearn_data])
df_pymvpa['conscious_state'] = df_pymvpa['condition_source']
df_pymvpa['model_name'] = df_pymvpa['model']
df_pymvpa['roi'] = df_pymvpa['roi'].map(roi_dict)
df_pymvpa['score'] = df_pymvpa['roc_auc']
idx = np.array(['Dummy' not in row for row in df_nilearn['model_name'].values])
df_nilearn = df_nilearn.iloc[idx,:]

df_nilearn = df_nilearn.groupby(['sub',
                                 'roi',
                                 'conscious_state',
                                 'model_name',]).mean().reset_index()
df_pymvpa = df_pymvpa.groupby(['sub',
                               'roi',
                               'conscious_state',
                               'model_name']).mean().reset_index()

df_pymvpa['pipeline'] = 'pymvpa'
df_nilearn['pipeline'] = 'nilearn'
cols = ['sub','roi','conscious_state',
        'model_name','score','pipeline']
df_pymvpa = df_pymvpa[cols]
df_nilearn = df_nilearn[cols]

df = pd.concat([df_pymvpa,df_nilearn])
#df = df_nilearn.subtract(df_pymvpa)

g = sns.catplot(x = 'conscious_state',
                y = 'score',
                hue = 'pipeline',
#                row = 'roi',
#                col = 'model_name',
                data = df,
                kind = 'violin',
                aspect = 2,
                **dict(split=True,
                       cut = 0))






























































