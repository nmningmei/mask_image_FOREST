#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:47:08 2019

@author: nmei

This script is to test if the computer vision model feature representations could form clusters


"""
import os
from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')
from shutil import copyfile
copyfile('../utils.py','utils.py')
from utils import LOO_partition
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.spatial.distance import pdist,squareform
from matplotlib import pyplot as plt



figure_dir = '../../figures/computer vision features'
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
computer_vision_dir = '../../data/computer vision features'
df_temp = pd.read_csv('../../data/behavioral/aingere_5_16_2019/aingere_Experiment_EEG_version_2019_may_16_1608trials.csv')
df_temp = df_temp.dropna()
n_splits = 48
categorical_dict = {a:b for a,b in zip(df_temp['label'].values,df_temp['category'].values)}
for model_name in os.listdir(computer_vision_dir):
    features = np.array([np.load(item) for item in glob(os.path.join(computer_vision_dir,
                                                 model_name,
                                                 '*.npy'))])
    groups = np.array([item.split('/')[-1].split('_')[0] for item in glob(os.path.join(computer_vision_dir,
                                                 model_name,
                                                 '*.npy'))])
    unique_names = np.unique(groups)
    labels = np.array([categorical_dict[item] for item in groups])
    
    df_cv = pd.DataFrame(np.vstack([groups,labels]).T,
                         columns = ['labels','targets'])
    idxs_train,idxs_test = LOO_partition(features,df_cv)
    
    clf = LinearSVC(random_state = 12345)
    pipeline = make_pipeline(StandardScaler(),clf)
    res = cross_val_score(pipeline,features,labels,
                          cv = zip(idxs_train,idxs_test),
                          groups = groups,
                          scoring = 'roc_auc',
                          n_jobs = 6,
                          verbose = 1,)
    
    # prepare for RDM
    df = pd.DataFrame(np.vstack([groups,labels]).T,columns = ['groups','labels'])
    df = df.sort_values(['groups','labels']).reset_index()
    features = features[df['index'].values]
    
    feature_ave = np.array([features[np.array(df_sub.index)].mean(0) for item,df_sub in df.groupby(['groups'])])
    names = np.array([item for item,df_sub in df.groupby(['groups'])])
    
    feature_plot = feature_ave - np.mean(feature_ave,1).reshape(-1,1)
    
    RDM = squareform(pdist(feature_plot,'cosine'))
    np.fill_diagonal(RDM,np.nan)
    
    df_RDM = pd.DataFrame(RDM,columns = names,index = names)
    df_RDM = df_RDM.fillna(0)
    method = 'complete'
    metric = 'cosine'
    g = sns.clustermap(df_RDM,
                       method = method,
                       metric = metric,
                       figsize = (30,30),
                       yticklabels = True,
                       cmap = plt.cm.coolwarm)
    g.fig.suptitle(f'RDM of {model_name} feature representations\nmethod = "{method}", metric = "{metric}"\ndescriminative = {res.mean():.4f}')
    g.fig.axes[2].axhline(n_splits,linestyle = '--',color = 'black',)
    g.fig.axes[2].axvline(n_splits,linestyle = '--',color = 'black',)
    g.savefig(os.path.join(figure_dir,
                           f'{model_name} RMD.png'),
    dpi = 400,
    bbox_inches = 'tight')






























