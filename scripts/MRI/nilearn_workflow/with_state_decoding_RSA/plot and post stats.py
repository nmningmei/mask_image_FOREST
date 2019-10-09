#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:05:02 2019

@author: nmei
"""

import os
from glob import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')

from shutil import copyfile
copyfile('../../utils.py','utils.py')
import utils

sub = 'sub-02'
saving_dir = '../../../../figures/MRI/nilearn/{}/L2O'.format(sub)
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
working_dir = '../../../../results/MRI/nilearn/{}/L2O'.format(sub)
working_data = glob(os.path.join(working_dir,'*.csv'))
df = pd.concat([pd.read_csv(f) for f in working_data])

for (conscious_state,roi,),df_sub in df.groupby(['condition_source',
                                                      'roi',]):
    df_chance = df_sub[df_sub['model'] == 'None + Dummy']
    df_sub = df_sub[df_sub['model'] == 'None + Linear-SVM']
    
    df_chance = df_chance.sort_values(['test1','test2'])
    df_sub = df_sub.sort_values(['test1','test2'])
    
    a = df_sub['accuracy'].values
    b = df_chance['accuracy'].values
    
    ps = utils.resample_ttest_2sample(a,b,one_tail = True)
    
#    df_sub['scores'] = df_sub['accuracy'] - df_sub['accuracy'].mean()
    unique_objects = pd.unique(df_sub['test1'])
    temp = np.ones((len(unique_objects),len(unique_objects))) * np.nan
    temp = pd.DataFrame(temp,
                        index = unique_objects,
                        columns = unique_objects)
    for ii,row in df_sub.iterrows():
        temp.loc[row['test1'],row['test2']] = row['accuracy']
        temp.loc[row['test2'],row['test1']] = row['accuracy']
    df_plot = temp.fillna(0)
    g = sns.clustermap(df_plot,
#                       metric = 'cosine',
                       figsize = (30,30),
                       cmap = plt.cm.coolwarm,
                       yticklabels = True,
                       xticklabels = True,)
    g.fig.suptitle('RSA with decoding scores, Leave 2 groups out\nscores = {:.4f} +/- {:.4f},p = {:.4f} +/- {:.4f}\n{} {}'.format(
            df_sub['accuracy'].mean(),df_sub['accuracy'].std(),
            ps.mean(),ps.std(),
            conscious_state,roi))
    g.savefig(os.path.join(saving_dir,
                           'RDM_map_{}_{}.png'.format(conscious_state,roi)))
    plt.close('all')