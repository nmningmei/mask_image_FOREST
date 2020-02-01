#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 12:53:16 2019

@author: nmei
"""

import os
import gc
import utils_deep
from glob import glob

import numpy  as np
import pandas as pd

n_splits            = 50
n_permutations      = int(1e3)
verbose             = 1
working_dir         = '../../results/agent_models'
print('gathering ...')
working_data        = np.sort(glob(os.path.join(working_dir,
                                 '*',
                                 '*',
                                 '*.npy')))
working_data        = working_data.reshape(-1,2)

#idx                 = 0 # change index
for features_,labels_   in working_data:#[idx]
#features_,labels_ = working_data[idx]
#if True:
#    print(os.path.join(
#            '/'.join(features_.split('/')[:-2])))
    cnn_csv_name        = os.path.join(
            '/'.join(features_.split('/')[:-2]),
            'scores as a function of decoder and noise.csv')
    temp                = pd.read_csv(cnn_csv_name)
    df_temp             = {col_name:[] for col_name in temp.columns}
    var                 = features_.split('/')[-2]
    saving_name = os.path.join(
            '/'.join(features_.split('/')[:-1]),
            'scores as a function of decoder and noise.csv')
    
    if not os.path.exists(saving_name):
        print(saving_name)
        for decoder in ['linear-svm','rbf-svm','randomforest','knn']:
            res,permu_scores,ps_decode = utils_deep.decode_hidden(decoder,
                                                                  hidden_features   = np.load(features_),
                                                                  labels            = np.load(labels_),
                                                                  n_splits          = n_splits,
                                                                  n_permutations    = n_permutations,
                                                                  verbose           = verbose,)
            print(f"{decoder},{res.mean():.3f} vs {permu_scores.mean():.3f},p = {ps_decode.mean():.4f}")
            df_temp['noise_level'       ].append(var)
            df_temp['decoder'           ].append(decoder)
            df_temp['performance_mean'  ].append(res.mean())
            df_temp['performance_std'   ].append(res.std())
            df_temp['chance_mean'       ].append(permu_scores.mean())
            df_temp['chance_std'        ].append(permu_scores.std())
            df_temp['pval'              ].append(ps_decode)
            df_temp['n_permute'         ].append(n_permutations)
        df_to_save = pd.DataFrame(df_temp)
        df_to_save.to_csv(saving_name,index = False)
        gc.collect()
        print('done')
    else:
        print(f'you have done {saving_name}')
