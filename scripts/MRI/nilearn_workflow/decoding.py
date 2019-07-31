#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:59:34 2019

@author: nmei
"""
import os
import re
from glob import glob
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from shutil import copyfile
copyfile('../../utils.py','utils.py')
from utils import groupy_average

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit,LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier



sub = 'sub-01'
stacked_data_dir = '../../../data/BOLD_no_average/{}/'.format(sub)
BOLD_data = glob(os.path.join(stacked_data_dir,'*BOLD*.npy'))
event_data = glob(os.path.join(stacked_data_dir,'*.csv'))
np.random.seed(12345)
for BOLD_name,df_name in zip(BOLD_data,event_data):
    BOLD = np.load(BOLD_name)
    df_event = pd.read_csv(df_name)
    roi_name = df_name.split('/')[-1].split('_events')[0]
    conscious_state = 'unconscious'
    idx_unconscious = df_event['visibility'] == conscious_state
    label_map = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}
    data = BOLD[idx_unconscious]
    df_data = df_event[idx_unconscious]
    targets = np.array([label_map[item] for item in df_data['targets'].values])[:,-1]

make_class = {name:[] for name in pd.unique(df_data['targets'])}
for ii,df_sub in df_data.groupby(['labels']):
    target = pd.unique(df_sub['targets'])
    label = pd.unique(df_sub['labels'])
    make_class[target[0]].append(label[0])

pick_test_classes = [[label1,label2] for label1 in make_class['Living_Things'] for label2 in make_class['Nonliving_Things']]


cv = LeaveOneGroupOut()
s = []
for train,test in cv.split(data,targets,df_data['session'].values):
    
    X,y = data[train],targets[train]
    
    X_test,y_test = data[test],targets[test]
    df_test = df_data.iloc[test].reset_index()
    X_test_ave,temp = groupy_average(X_test,df_test,groupby=['trials'])
    y_test = np.array([label_map[item] for item in temp['targets']])[:,-1]
    
    pipeline = make_pipeline(MinMaxScaler(),PCA(),CalibratedClassifierCV(LinearSVC(),cv=5))
    pipeline.fit(X,y)
    preds = pipeline.predict_proba(X_test_ave)[:,-1]
#    print(roc_auc_score(y,pipeline.predict(X)))
    print(roc_auc_score(y_test,preds))
    s.append(roc_auc_score(y_test,preds))
print(np.mean(s),np.std(s))

