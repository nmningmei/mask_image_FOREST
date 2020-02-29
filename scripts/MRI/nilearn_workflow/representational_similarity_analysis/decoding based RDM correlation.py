#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:08:15 2020

@author: nmei
"""

import os
import gc
import warnings
warnings.filterwarnings('ignore') 
import pandas  as pd
import numpy   as np
import seaborn as sns

from glob                      import glob
from tqdm                      import tqdm
from sklearn.utils             import shuffle
from shutil                    import copyfile
copyfile('../../../utils.py','utils.py')
from utils                     import (LOO_partition,
                                       get_label_category_mapping,
                                       get_label_subcategory_mapping,
                                       make_df_axis)
from sklearn.model_selection   import cross_validate,LeavePGroupsOut
from sklearn.preprocessing     import MinMaxScaler
from sklearn                   import metrics
from sklearn.exceptions        import ConvergenceWarning
from sklearn.utils.testing     import ignore_warnings
from collections               import OrderedDict
from matplotlib                import pyplot as plt
from scipy.spatial             import distance
from scipy.stats               import spearmanr
from joblib                    import Parallel,delayed


sub                 = 'sub-01'
stacked_data_dir    = '../../../../data/BOLD_average/{}/'.format(sub)
output_dir          = '../../../../results/MRI/nilearn/{}/RSA'.format(sub)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
array_dir           = '../../../../results/MRI/nilearn/{}/RSA_RDM_arrays'.format(sub)
if not os.path.exists(array_dir):
    os.makedirs(array_dir)
figure_saving_dir   = '../../../../figures/MRI/nilearn/{}/RSA'.format(sub) # <- save the csv for making the figures
if not os.path.exists(figure_saving_dir):
    os.mkdir(figure_saving_dir)
feature_dir         = '../../../../data/computer_vision_features'
BOLD_data           = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))
event_data          = np.sort(glob(os.path.join(stacked_data_dir,'*.csv')))
label_map           = {'Nonliving_Things':[0,1],
                       'Living_Things':   [1,0]}
average             = True
n_jobs              = -1
pdist_metric        = 'minkowski'

model_names = ['PCA + Linear-SVM']

idx = 0
np.random.seed(12345)
BOLD_name,df_name   = BOLD_data[idx],event_data[idx]
BOLD                = np.load(BOLD_name)
df_event            = pd.read_csv(df_name)
roi_name            = df_name.split('/')[-1].split('_events')[0]
for conscious_state in ['unconscious','glimpse','conscious']:
    idx_unconscious = df_event['visibility'] == conscious_state
    data            = BOLD[idx_unconscious]
    df_data         = df_event[idx_unconscious].reset_index(drop=True)
    df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
    targets         = np.array([label_map[item] for item in df_data['targets'].values])[:,-1]
    groups          = df_data['labels'].values
    images          = df_data['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
    EVC_feature        = np.array([np.load(os.path.join(feature_dir,
                                                     'DenseNet169',
                                                     item)) for item in images])
    idxs_train,idxs_test  = LOO_partition(df_data)
    n_splits = len(idxs_train)
    
    for model_name in model_names:
        file_name   = f'RSA ({sub} {roi_name} {conscious_state} {pdist_metric}).csv'
        if not os.path.exists(os.path.join(output_dir,file_name)):
            np.random.seed(12345)
            idx_train,idx_test = idxs_train[0],idxs_test[0]
            
            def add_track(df_sub):
                n_rows = df_sub.shape[0]
                if len(df_sub.index.values) > 1:
                    temp = '+'.join(str(item + 10) for item in df_sub.index.values)
                else:
                    temp = str(df_sub.index.values[0])
                df_sub = df_sub.iloc[0,:].to_frame().T # why did I use 1 instead of 0?
                df_sub['n_volume'] = n_rows
                df_sub['time_indices'] = temp
                return df_sub
            def compute_RDM(idx_train,RDM_measure = pdist_metric):
                BOLD_data,EVC_data,df_picked = data.copy()[idx_train],EVC_feature.copy()[idx_train],df_data.copy().iloc[idx_train,:].reset_index(drop=True)
                
                BOLD_features = np.array([np.mean(BOLD_data[df_sub.index],0) for _,df_sub in df_picked.groupby('labels')])
                df_averaged = pd.concat([add_track(df_sub) for ii,df_sub in df_picked.groupby('labels')])
                image_features = np.array([np.mean(EVC_data[df_sub.index],0) for _,df_sub in df_picked.groupby('labels')])
                df_averaged = df_averaged.reset_index(drop=True).drop('index',axis = 1)
                idx_sort = list(df_averaged.sort_values(['targets','subcategory','labels']).index)
                df_averaged = df_averaged.sort_values(['targets','subcategory','labels'])
                BOLD_features = BOLD_features[idx_sort]
                image_features = image_features[idx_sort]
            
                RDM_BOLD = distance.pdist(BOLD_features - BOLD_features.mean(1).reshape(-1,1),
                                          metric = RDM_measure)
                RDM_image = distance.pdist(image_features - image_features.mean(1).reshape(-1,1),
                                           metric = RDM_measure)
            
                RDM_corr,pval = spearmanr(RDM_BOLD,RDM_image,)
                return RDM_corr,pval,RDM_BOLD,RDM_image,df_averaged
            
            gc.collect()
            RDMs_corr,pvals,RDMs_BOLD,RDMs_image,dfs_averaged = zip(*Parallel(n_jobs = -1,verbose = 1,)(delayed(compute_RDM)(**{
                    'idx_train':idx_train}) for idx_train in idxs_train))
            
            results = OrderedDict()
            results['folds'] = np.arange(n_splits) + 1
            results['spearmanr'] = RDMs_corr
            results['pvals'] = pvals
            results['conscious'] = [conscious_state] * n_splits
            results['roi_name'] = [roi_name] * n_splits
            results_to_save = pd.DataFrame(results)
            results_to_save.to_csv(os.path.join(output_dir,file_name),index = False)
            
            for RDMs,excel_file_name in zip([RDMs_BOLD,RDMs_image],['BOLD_RDMs','CNN_RDMs']):
                RDMs,excel_file_name
                with pd.ExcelWriter(os.path.join(figure_saving_dir,f'{excel_file_name}.xlsx')) as writer:
                    for ii,(RDM,df_averaged) in tqdm(enumerate(zip(RDMs,dfs_averaged))):
                        array_data = distance.squareform(RDM)
                        df_temp = pd.DataFrame(array_data,columns = df_averaged['labels'].values,
                                               index = df_averaged['labels'].values)
                        df_temp.to_excel(writer,sheet_name = f'fold_{ii+1}')
            
                        fig,axes = plt.subplots(ncols = 2,
                                                figsize = (30,15))
#            for fold,(RDM_BOLD,RDM_image,corr,pval) in enumerate(zip(RDMs_BOLD,RDMs_image,RDMs_corr,pvals)):
#                RDM_BOLD,RDM_image,corr,pval
#                fig,axes = plt.subplots(figsize = (30,15),
#                                        ncols = 2,)
#                for ii,(ax, RDM) in enumerate(zip(axes,[RDM_BOLD,RDM_image])):
#                    array_data = distance.squareform(RDM)
#                    np.fill_diagonal(array_data,np.nan)
#                    im = ax.imshow(array_data,
#                                   origin = 'lower',
#                                   cmap = plt.cm.coolwarm,
#                                   vmin = 5,
#                                   vmax = 30,
#                                   )
#                    ax.set(xticks = np.arange(array_data.shape[0]),
#                           yticks = np.arange(array_data.shape[0]),
#                           xticklabels = df_averaged['labels'].values,
#                           yticklabels = df_averaged['labels'].values,)
#                    ax.axhline(array_data.shape[0] / 2 - .5,linestyle = '--',color = 'black',alpha = 1.)
#                    ax.axvline(array_data.shape[0] / 2 - .5,linestyle = '--',color = 'black',alpha = 1.)
#                    ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 90, ha = 'center')
#                fig.subplots_adjust(right=0.8)
#                cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
#                fig.colorbar(im, cax=cbar_ax)
#                fig.subtitle(f'fold {fold + 1}correlation = {corr:.4f}, p = {pval:.2e}')
#                fig.savefig(os.path.join(figure_saving_dir,''))





















