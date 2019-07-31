#!/usr/bin/env python3 -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:19:34 2019

@author: nmei
"""

import os
import pandas as pd
import numpy  as np

from glob                    import glob
from tqdm                    import tqdm
from sklearn.utils           import shuffle
from nibabel                 import load as load_fmri
from nilearn.image           import index_img
from shutil                  import copyfile
copyfile('../../utils.py','utils.py')
from utils                   import (groupby_average,
                                     check_train_balance,
                                     check_train_test_splits,
                                     customized_partition,
                                     build_model_dictionary)
from sklearn.metrics         import roc_auc_score
from sklearn.preprocessing   import MinMaxScaler
from nilearn.decoding        import SpaceNetClassifier
from nilearn.input_data      import NiftiMasker
from sklearn.model_selection import GroupShuffleSplit#StratifiedKFold

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

sub                 = 'sub-04'
first_session       = 1
stacked_data_dir    = '../../../data/BOLD_whole_brain_averaged/{}/'.format(sub)
mask_dir            = '../../../data/MRI/{}/anat/ROI_BOLD'.format(sub)
output_dir          = '../../../results/MRI/nilearn/spacenet/{}'.format(sub)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
masks               = glob(os.path.join(mask_dir,'*.nii.gz'))
whole_brain_mask    = f'../../../data/MRI/{sub}/func/session-0{first_session}/{sub}_unfeat_run-01/outputs/func/mask.nii.gz'
label_map           = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}
average             = True
n_splits            = 30

s = dict(
        conscious_state     = [],
        score               = [],
        fold                = [],)
for conscious_state in ['unconscious','glimpse','conscious']:
    df_data         = pd.read_csv(os.path.join(stacked_data_dir,
                                               f'whole_brain_{conscious_state}.csv'))
    df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
    idxs_test       = customized_partition(df_data,n_splits = n_splits)
    
    
    BOLD_file       = os.path.join(stacked_data_dir,
                                   f'whole_brain_{conscious_state}.nii.gz')
    BOLD_image      = load_fmri(BOLD_file)
    targets         = np.array([label_map[item] for item in df_data['targets']])[:,-1]
    groups          = df_data['labels'].values
    
    masker          = NiftiMasker(mask_img = whole_brain_mask,standardize = True)
    cv              = GroupShuffleSplit(n_splits=n_splits,test_size=0.2,random_state=12345)
    for fold,(idx_train,idx_test) in enumerate(cv.split(df_data.values,targets,groups)):
        clf         = SpaceNetClassifier(
                                 mask                   = whole_brain_mask, 
                                 penalty                = "graph-net",
                                 l1_ratios              = np.linspace(0.2,0.8,10),
                                 standardize            = True,
                                 eps                    = 1e-3,
                                 n_jobs                 = 10,
                                 tol                    = 1e-3,
                                 cv                     = 8,
                                 screening_percentile   = 20.0,
                                 verbose                = 1,
                                 )
        
        BOLD_train,y_train  = index_img(BOLD_image,idx_train),targets[idx_train]
        BOLD_test,y_test    = index_img(BOLD_image,idx_test),targets[idx_test]
        
        clf.fit(BOLD_train,y_train)
        preds = clf.decision_function(masker.fit_transform(BOLD_test))
        score = roc_auc_score(y_test,preds)
        print(conscious_state,fold,score)
        s['conscious_state' ].append(conscious_state)
        s['score'           ].append(score)
        s['fold'            ].append(fold + 1)
        
        clf.coef_img_.to_filename(os.path.join(output_dir,f'spaceNet_coef_{conscious_state}_{fold+1}.nii.gz'))
        
        df = pd.DataFrame(s)
        df.to_csv(os.path.join(output_dir,'spaceNet_temp_results.csv'),index=False)

