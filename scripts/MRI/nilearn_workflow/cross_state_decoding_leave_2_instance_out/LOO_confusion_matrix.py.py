#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:29:41 2021

@author: nmei
"""

import os

import pandas as pd
import numpy as np
import seaborn as sns

from glob import glob
from tqdm import tqdm
from sklearn import metrics
from collections import Counter
from shutil import copyfile
copyfile('../../../utils.py','utils.py')
import utils

from matplotlib import pyplot as plt

folder_name = 'decoding_13_11_2020'
working_dir = f"../../../../results/MRI/nilearn/{folder_name}/sub*"
figure_dir = '../../../../figures/MRI/nilearn/decoding_confusion_matrix/binary'
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

for condition in ['conscious','unconscious']:
    working_data = glob(os.path.join(working_dir,f'*_{condition}_{condition}_None_Linear-SVM*csv'))
    for f in tqdm(working_data):
        df_temp = pd.read_csv(f)
        if 0.5 > df_temp['roc_auc'].values.mean():
            df_data = pd.read_csv(os.path.join('../../../../data/BOLD_average_BOLD_average_lr',
                                               pd.unique(df_temp['sub'])[0],
                                               f'{pd.unique(df_temp["roi"])[0]}_events.csv'))
            df_data = df_data[df_data['visibility'] == condition]
            df_temp['roi_name'] = df_temp['roi']
            df_temp['roi_name'] = df_temp['roi_name'].map(utils.rename_ROI_for_plotting())
            df_temp['region'] = df_temp['roi_name'].map(utils.define_roi_category())
            
            sub_name = pd.unique(df_temp['sub'])[0]
            roi_name = pd.unique(df_temp['roi_name'])[0]
            score = df_temp['roc_auc'].mean()
            
            p_sum = df_temp['tp'] + df_temp['fp']
            n_sum = df_temp['fn'] + df_temp['tn']
            df_temp['tpr'] = df_temp['tp'] / (df_temp['tp'] + df_temp['fn'])
            df_temp['fpr'] = df_temp['tn'] / (df_temp['tn'] + df_temp['fp'])
            df_temp['tp'] = df_temp['tp'] / p_sum
            df_temp['fp'] = df_temp['fp'] / p_sum
            df_temp['fn'] = df_temp['fn'] / n_sum
            df_temp['tn'] = df_temp['tn'] / n_sum
            cm_mean = df_temp.mean()[['tp','fp','fn','tn']].values.reshape((2,2))
            cm_std = df_temp.std()[['tp','fp','fn','tn']].values.reshape((2,2))
            disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm_mean,display_labels = ['Living','NonLiving'])
            disp.plot()
            title = f'{sub_name} {roi_name} = {score:.4f}'
            disp.ax_.set(title = title)
            disp.figure_.savefig(os.path.join(figure_dir,f'{sub_name} {condition} {roi_name}.jpg'))
            plt.close('all')
            
            idxs_train,idxs_test = utils.LOO_partition(df_data,'labels')
            pairs = [pd.unique(df_data.iloc[idx_test]['labels']) for idx_test in idxs_test]
            
            df_for_sort = [[t,s,l] for (t,s,l),df_sub in df_data.groupby(['targets','subcategory','labels'])]
            df_for_sort = pd.DataFrame(df_for_sort,columns = ['targets','subcategory','labels'])
            df_for_sort = df_for_sort.sort_values(['targets','subcategory','labels'])
            df_roc_auc = pd.DataFrame(np.ones((int(96/2),int(96/2))) * np.nan,
                                      index = df_for_sort['labels'][:48],
                                      columns = df_for_sort['labels'][48:],
                                      )
            for (idx_row,row),pair in zip(df_temp.iterrows(),pairs):
                df_roc_auc.loc[pair[0],pair[1]] = row['roc_auc']
#                df_roc_auc.loc[pair[1],pair[0]] = row['roc_auc']

# get distribution of objects
df_plot = dict(sub_name = [],
               conscious = [],
               cv_type = [],
               target_type = [],
               count = [],
               )
for condition in ['conscious','unconscious']:
    working_data = glob(os.path.join(
            '../../../../data/BOLD_whole_brain_averaged/sub-*',
            f'whole_brain_{condition}.csv'))
    for f in working_data:
        df_data = pd.read_csv(f)
        idxs_train,idxs_test = utils.LOO_partition(df_data,'labels')
        for idx_train,idx_test in zip(idxs_train,idxs_test):
            df_roll = df_data.iloc[idx_train]
            for key,item in dict(Counter(df_roll['targets'].values)).items():
                df_plot['sub_name'].append(f.split('/')[-2])
                df_plot['conscious'].append(condition)
                df_plot['cv_type'].append('train')
                df_plot['target_type'].append(key)
                df_plot['count'].append(item/len(idx_train))
            
            df_roll = df_data.iloc[idx_test]
            for key,item in dict(Counter(df_roll['targets'].values)).items():
                df_plot['sub_name'].append(f.split('/')[-2])
                df_plot['conscious'].append(condition)
                df_plot['cv_type'].append('test')
                df_plot['target_type'].append(key)
                df_plot['count'].append(item/len(idx_test))
df_plot = pd.DataFrame(df_plot)
df_plot['sub'] = df_plot['sub_name'].map(utils.subj_map())

g = sns.catplot(x = 'cv_type',
                y = 'count',
                hue = 'target_type',
                row = 'sub',
                col = 'conscious',
                data = df_plot,
                kind = 'violin',
                **{'cut':0,
                   'inner':'quartile',
                   'split':True,})
(g.set_axis_labels('','proportion')
  .set_titles('{row_name} | {col_name}')
  .set(xticklabels = ['Training set','Testing set']))
g.savefig('../../../../figures/MRI/nilearn/collection_of_results/cross-validation counts.jpg',
          bbox_inches = 'tight')

# count items
df_count = pd.DataFrame(index = df_for_sort['labels'].values,
                        columns = [f'sub-0{ii+1}_{condition}' for condition in ['conscious','unconscious'] for ii in range(7)])

for condition in ['conscious','unconscious']:
    working_data = glob(os.path.join(
            '../../../../data/BOLD_whole_brain_averaged/sub-*',
            f'whole_brain_{condition}.csv'))
    for f in working_data:
        df_data = pd.read_csv(f)
        sub_name = f.split('/')[-2]
        for key,item in dict(Counter(df_data['labels'])).items():
            df_count.loc[key,f'{sub_name}_{condition}'] = item
df_count['labels'] = list(df_count.index)
df_count['subcategory'] = df_count['labels'].map(utils.get_label_subcategory_mapping())
df_count['category'] = df_count['labels'].map(utils.get_label_category_mapping())
df_count = df_count.sort_values(['category','subcategory','labels'])
df_count.to_csv('trial counts.csv',index = False)



















