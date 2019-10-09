#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:41:30 2019

@author: nmei
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')

from glob import glob
from tqdm import tqdm
from shutil import copyfile
copyfile('../../utils.py','utils.py')
from utils import get_label_category_mapping,get_label_subcategory_mapping,LOO_partition

from sklearn.model_selection import StratifiedKFold,cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

from matplotlib import pyplot as plt

from scipy.spatial import distance

working_dir = '../../../data/computer_vision_features'
report_dir = '../../../results/copmuter_vision_features'
figure_dir = '../../../figures/computer_vision_features'
for d in [report_dir,figure_dir]:
    if not os.path.exists(d):
        os.mkdir(d)

for computer_vision_model in os.listdir(working_dir):
    working_data = glob(os.path.join(working_dir,computer_vision_model,'*.npy'))
    labels = []
    features = []
    for ii,item in tqdm(enumerate(working_data)):
        feature_ = np.load(item)
        features.append(feature_)
        labels.append(item.split('/')[-1].split('.')[0].split('_')[0])
    features = np.array(features)
    labels = np.array(labels)
    targets = np.array([get_label_category_mapping()[item] for item in labels])
    
    features,labels,targets = shuffle(features,labels,targets)
    
    df_data = pd.DataFrame()
    df_data['labels'] = labels
    df_data['targets'] = targets
    
    idxs_train,idxs_test = LOO_partition(df_data)
    
    targets_ = np.array([{'Living_Things':0,'Nonliving_Things':1}[item] for item in targets])
    
    cv = zip(idxs_train,idxs_test)#StratifiedKFold(20,shuffle = True,random_state = 12345)
    clf = CalibratedClassifierCV(LinearSVC(class_weight = 'balanced',random_state = 12345),
                                 cv = 3,)
    clf = make_pipeline(StandardScaler(),clf)
    res = cross_validate(clf,features,targets_,scoring = 'roc_auc',cv = cv,
                         n_jobs = 10,return_estimator = True,verbose = 2)
    scores = np.array([roc_auc_score(targets_[idx_test],est.predict_proba(features[idx_test])[:,-1]) for est,idx_test in tqdm(zip(res['estimator'],idxs_test))])#.split(features,labels))])
    
    idx_wrong, = np.where(scores < 1)
    a = f"""{computer_vision_model}, leave 2 objects out cross validation (folds = {len(idxs_train)}),
scores = {scores.mean():.4f} +/- {scores.std():.4f}
pairs of living-nonliving objects incorrectly decoded:
"""
    for ii,idx_test,iii in zip([np.unique(item) for item in labels[np.array(idxs_test)[idx_wrong]]],
                            np.array(idxs_test)[idx_wrong],
                            idx_wrong):
        item1,item2 = ii
        temp_score = roc_auc_score(targets_[idx_test],res['estimator'][iii].predict_proba(features[idx_test])[:,-1]) 
        a += f'{str(item1):25s},{str(item2):25} ROC_AUC = {temp_score:.4f}\n'
    if not os.path.exists(os.path.join(report_dir,computer_vision_model)):
        os.mkdir(os.path.join(report_dir,computer_vision_model))
    with open(os.path.join(report_dir,computer_vision_model,'report.txt'),'w') as f:
        f.write(a)
        f.close()
    ####################
    temp = pd.DataFrame(labels,columns = ['labels'])
    temp['category'] = temp['labels'].map(get_label_category_mapping())
    temp['subcategory'] = temp['labels'].map(get_label_subcategory_mapping())
    
    X = []
    y = []
    for label_,df_sub in temp.groupby(['labels']):
        idx_pick = df_sub.index
        X.append(features[idx_pick].mean(0))
        y.append(label_)
    X = np.array(X)
    y = np.array(y)
    
    temp = pd.DataFrame(y,columns = ['labels'])
    temp['category'] = temp['labels'].map(get_label_category_mapping())
    temp['subcategory'] = temp['labels'].map(get_label_subcategory_mapping())
    temp = temp.sort_values(['category','subcategory','labels'])
    idx_sort = temp.index
    
    RDM_ = distance.squareform(distance.pdist(X - X.mean(1).reshape(-1,1),'cosine'))
    np.fill_diagonal(RDM_,np.nan)
    RDM_ = pd.DataFrame(RDM_,columns = y,index = y)
    RDM = RDM_.loc[y[idx_sort],y[idx_sort]]
    plt.close('all')
    fig,ax = plt.subplots(figsize = (30,25))
    ax = sns.heatmap(RDM,
                     cmap = plt.cm.RdBu_r,
#                     vmin = 0.,
#                     vmax = 1.,
                     xticklabels = True,
                     yticklabels = True,
                     ax = ax,
                     )
    if not os.path.exists(os.path.join(figure_dir,computer_vision_model)):
        os.mkdir(os.path.join(figure_dir,computer_vision_model))
    fig.savefig(os.path.join(figure_dir,computer_vision_model,'RDM.png'),
                dpi = 500,
                bbox_inches = 'tight')
    plt.close('all')


"""
    cv = StratifiedKFold(n_splits=int(20), 
                         random_state=12345, shuffle=True)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1, solver='liblinear',
                                           multi_class='auto'))
    # Compute confusion matrix for each cross-validation fold
    res = cross_validate(clf,
                         features,
                         labels,
                         scoring = 'accuracy',
                         cv = cv,
                         n_jobs = 8,
                         verbose = 1,
                         return_estimator = True)
    
    classes = set(labels)
    y_pred = np.zeros((len(labels), len(classes)))
    for clf,(train, test) in zip(res['estimator'],cv.split(features, labels)):
        # Probabilistic prediction (necessary for ROC-AUC scoring metric)
        y_pred[test] = clf.predict_proba(features[test])
    
    temp = np.array([softmax(row) for row in y_pred])
    
    ylabels = []
    confusion = np.zeros((len(classes), len(classes)))
    for ii, train_class in tqdm(enumerate(classes)):
        ylabels.append(train_class)
        for jj in range(ii, len(classes)):
            confusion[ii, jj] = roc_auc_score(labels == train_class, y_pred[:, jj])
            confusion[jj, ii] = confusion[ii, jj]
    
    ylabels = np.array(ylabels)
    
    ###########################
    temp = pd.DataFrame(ylabels,columns = ['labels'])
    temp['category'] = temp['labels'].map(get_label_category_mapping())
    temp['subcategory'] = temp['labels'].map(get_label_subcategory_mapping())
    temp = temp.sort_values(['category','subcategory','labels'])
    idx_sort = temp.index
    
    df = pd.DataFrame(confusion,columns = ylabels,index = ylabels)
    
    df_plot = df.loc[ylabels[idx_sort],ylabels[idx_sort]]
    
    fig,ax = plt.subplots(figsize = (16,16))
    ax = sns.heatmap(df_plot,
                     cmap = plt.cm.RdBu_r,
                     vmin = 0.2,
                     vmax = 0.8,
                     xticklabels = True,
                     yticklabels = True,
                     ax = ax,
                     )
"""




















