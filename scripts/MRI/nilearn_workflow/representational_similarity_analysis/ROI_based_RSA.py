#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 06:53:45 2020

@author: nmei

An RSA on ROI-based format

"""

import os
import gc

import pandas as pd
import numpy  as np
import multiprocessing

print(f'availabel cpus = {multiprocessing.cpu_count()}')
from glob                    import glob
from tqdm                    import tqdm
from shutil                  import copyfile
copyfile('../../../utils.py','utils.py')

from sklearn.feature_selection import SelectPercentile,mutual_info_classif
from sklearn                   import decomposition
from sklearn.utils             import shuffle as sk_shuffle
from functools                 import partial
from scipy.spatial             import distance
from scipy.stats               import spearmanr
from collections               import OrderedDict,Counter
from joblib                    import Parallel,delayed

# interchangable part:
sub                     = 'sub-01'

stacked_data_dir        = '../../../../data/BOLD_average_BOLD_average_lr/{}/'.format(sub)
feature_dir             = '../../../../data/computer_vision_features_no_background'
output_dir              = '../../../../results/MRI/nilearn/ROI_RSA/{}'.format(sub)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
BOLD_data               = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))
event_data              = np.sort(glob(os.path.join(stacked_data_dir,'*.csv')))


label_map               = {'Nonliving_Things':[0,1],
                           'Living_Things':   [1,0]}
average                 = True
n_jobs                  = -1
n_splits                = int(1e4)


np.random.seed(12345)
for BOLD_name,df_name in zip(BOLD_data,event_data):
    BOLD                    = np.load(BOLD_name)
    df_event                = pd.read_csv(df_name)
    roi_name                = df_name.split('/')[-1].split('_events')[0]
    print(roi_name)
    
    BOLD_random             = np.array([sk_shuffle(row) for row in BOLD])
    images                  = df_event['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
    CNN_feature             = np.array([np.load(os.path.join(feature_dir,
                                                     'VGG19',
                                                     item)) for item in images])
    groups                  = df_event['labels'].values
    n_components            = int(pd.unique(groups).shape[0])
    
    def _proc(df_data):
        df_picked = df_data.groupby('labels').apply(lambda x: x.sample(n = 1).drop('labels',axis = 1)).reset_index()
        df_picked = df_picked.sort_values(['targets','subcategory','labels'])
        idx_test  = df_picked['level_1'].values
        return idx_test
    print(f'partitioning data for {n_splits} folds')
    idxs_sample   = Parallel(n_jobs = -1, verbose = 1)(delayed(_proc)(**{
                'df_data':df_event,}) for _ in range(n_splits))
    
    
    gc.collect()
    
    def _RSA(idx_sample):
        BOLD_sampled            = BOLD_random[idx_sample]
        feature_sampled         = CNN_feature[idx_sample]
        BOLD_sampled_selected   = decomposition.FastICA(n_components,random_state = 12345).fit_transform(BOLD_sampled,)
        CNN_sampled_selected    = decomposition.FastICA(n_components,random_state = 12345).fit_transform(feature_sampled)
        RDM_BOLD                = distance.pdist(BOLD_sampled_selected,'correlation')
        RDM_feature             = distance.pdist(CNN_sampled_selected,'correlation')
        r,p                     = spearmanr(RDM_BOLD,RDM_feature)
        # print(r,p)
        return r
    
    correlations = Parallel(n_jobs = n_jobs, verbose = 1,
                            )(delayed(_RSA)(**{
                                'idx_sample':idx_sample}) for idx_sample in idxs_sample)

    correlations = np.array(correlations)
    # correlations = np.array([_RSA(idx) for idx in tqdm(idxs_sample,
    #             desc = f'{sub}_{roi_name}_random_RSA')])
    gc.collect()
    
    df = pd.DataFrame(correlations.reshape(-1,1),columns = ['corr'])
    df['roi_name'] = roi_name
    df['sub_name'] = sub
    df['conscious_state'] = 'random'
    
    df.to_csv(os.path.join(output_dir,f'{sub}_{roi_name}_random_RSA.csv'),
              index = False)
    
    for conscious_state_source in ['unconscious','glimpse','conscious']:
        idx_unconscious_source  = df_event['visibility'] == conscious_state_source
        data_source             = BOLD[idx_unconscious_source]
        df_data_source          = df_event[idx_unconscious_source].reset_index(drop=True)
        df_data_source['id']    = df_data_source['session'] * 1000 + df_data_source['run'] * 100 + df_data_source['trials']
        targets_source          = np.array([label_map[item] for item in df_data_source['targets'].values])[:,-1]
        images                  = df_data_source['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
        CNN_feature             = np.array([np.load(os.path.join(feature_dir,
                                                         'VGG19',
                                                         item)) for item in images])
        groups                  = df_data_source['labels'].values
        
        def _proc(df_data):
            df_picked = df_data.groupby('labels').apply(lambda x: x.sample(n = 1).drop('labels',axis = 1)).reset_index()
            df_picked = df_picked.sort_values(['targets','subcategory','labels'])
            idx_test  = df_picked['level_1'].values
            return idx_test
        print(f'partitioning data for {n_splits} folds')
        idxs_sample   = Parallel(n_jobs = -1, verbose = 1)(delayed(_proc)(**{
                    'df_data':df_data_source,}) for _ in range(n_splits))
        
        gc.collect()
        
        def _RSA(idx_sample):
            BOLD_sampled            = data_source[idx_sample]
            feature_sampled         = CNN_feature[idx_sample]
            BOLD_sampled_selected   = decomposition.PCA(.95,random_state = 12345).fit_transform(BOLD_sampled,)
            CNN_sampled_selected    = decomposition.PCA(.95,random_state = 12345).fit_transform(feature_sampled)
            RDM_BOLD                = distance.pdist(BOLD_sampled_selected,'correlation')
            RDM_feature             = distance.pdist(CNN_sampled_selected,'correlation')
            r,p                     = spearmanr(RDM_BOLD,RDM_feature)
            # print(r,p)
            return r
        
        correlations = Parallel(n_jobs = n_jobs, verbose = 1,
                                )(delayed(_RSA)(**{
                                    'idx_sample':idx_sample}) for idx_sample in idxs_sample)
    
        correlations = np.array(correlations)
        
        # correlations = np.array([_RSA(idx) for idx in tqdm(idxs_sample,
        #                                                    desc = f'{sub}_{roi_name}_{conscious_state_source}_RSA')])
        
        df = pd.DataFrame(correlations.reshape(-1,1),columns = ['corr'])
        df['roi_name'] = roi_name
        df['sub_name'] = sub
        df['conscious_state'] = conscious_state_source
        
        df.to_csv(os.path.join(output_dir,f'{sub}_{roi_name}_{conscious_state_source}_RSA.csv'),
                  index = False)















