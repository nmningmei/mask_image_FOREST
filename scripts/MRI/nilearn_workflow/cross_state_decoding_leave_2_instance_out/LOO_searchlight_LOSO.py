#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 05:43:31 2020

@author: nmei
"""

import os
import gc

from glob    import glob
from tqdm    import tqdm
from nibabel import load as load_fmri

import numpy  as np
import pandas as pd

from shutil                  import copyfile
copyfile('../../../utils.py','utils.py')
from utils                   import (
#                                     customized_partition,
#                                     check_train_test_splits,
#                                     check_train_balance,
                                      build_model_dictionary,
                                     # Find_Optimal_Cutoff,
                                     LOO_partition
                                     )
from sklearn.model_selection import check_cv,LeaveOneGroupOut
from sklearn.utils import shuffle as sk_shuffle
from nilearn.decoding import SearchLight
from nilearn.image import concat_imgs,new_img_like
from joblib import Parallel,delayed

if __name__ == "__main__":

    sub                 = 'sub-01'
    stacked_data_dir    = '../../../../data/BOLD_whole_brain_averaged/{}/'.format(sub)
    mask_dir            = '../../../../data/MRI/{}/anat/ROI_BOLD'.format(sub)
    output_dir          = '../../../../results/MRI/nilearn/decode_searchlight_LOSO/{}'.format(sub)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    masks               = glob(os.path.join(mask_dir,'*.nii.gz'))
    whole_brain_mask    = f'../../../../data/MRI/{sub}/func/mask.nii.gz'
    combined_mask       = f'../../../../data/MRI/{sub}/anat/ROI_BOLD/combine_BOLD.nii.gz'
    func_brain          = f'../../../../data/MRI/{sub}/func/example_func.nii.gz'
    feature_dir         = '../../../../data/computer_vision_features_no_background'
    label_map           = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}
    average             = True
    n_jobs              = -1
    
    conscious_source    = 'unconscious'
    conscious_target    = 'unconscious'
    model_names         = ['None + Linear-SVM','None + Dummy']
    np.random.seed(12345)
    
    df_data_source         = pd.read_csv(os.path.join(stacked_data_dir,
                                               f'whole_brain_{conscious_source}.csv'))
    df_data_source['id']   = df_data_source['session'] * 1000 +\
                             df_data_source['run'] * 100 +\
                             df_data_source['trials']
    df_data_source         = df_data_source[df_data_source.columns[1:]]
    targets_source         = np.array([label_map[item] for item in df_data_source['targets'].values])[:,-1]

    BOLD_file_source       = os.path.join(stacked_data_dir,
                                   f'whole_brain_{conscious_source}.nii.gz')
    BOLD_image_source      = load_fmri(BOLD_file_source)
    
    df_data_target         = pd.read_csv(os.path.join(stacked_data_dir,
                                               f'whole_brain_{conscious_target}.csv'))
    df_data_target['id']   = df_data_target['session'] * 1000 +\
                             df_data_target['run'] * 100 +\
                             df_data_target['trials']
    df_data_target         = df_data_target[df_data_target.columns[1:]]
    targets_target         = np.array([label_map[item] for item in df_data_target['targets'].values])[:,-1]

    BOLD_file_target       = os.path.join(stacked_data_dir,
                                   f'whole_brain_{conscious_target}.nii.gz')
    BOLD_image_target      = load_fmri(BOLD_file_target)
    

    print(f'partitioning source: {conscious_source}::{conscious_target}')
    cv = LeaveOneGroupOut()
    groups = df_data_source['session'].values *10 + df_data_source['run'].values
    idxs_train_source,idxs_test_target = [],[]
    for train,test in cv.split(df_data_source['session'].values,groups = groups):
        idxs_train_source.append(sk_shuffle(train))
        idxs_test_target.append(test)
    if conscious_source != conscious_target:
        idxs_test_target = [df_data_target.index.to_list() for _ in idxs_test_target]
        
    # concate imgs
    BOLD_concat = concat_imgs([BOLD_image_source,BOLD_image_target])
    labels_concat = np.concatenate([targets_source,targets_target])
    # modify cv index
    idxs_trains = idxs_train_source.copy()
    
    idxs_tests = np.array([np.array(item) + BOLD_image_source.shape[-1] for item in idxs_test_target])
    for fold,_ in enumerate(idxs_tests):#print(fold)
        idx_train,idx_test = idxs_trains[fold],idxs_tests[fold]
        for model_name in model_names:
            file_name           = f'{conscious_source}_{conscious_target}_{model_name}_{fold}.nii.gz'
            print(f'{model_name} fold-{fold} {conscious_source}::{len(idx_train)} --> {conscious_target}::{len(idx_test)}')
            if not os.path.exists(os.path.join(output_dir,file_name)):
                np.random.seed(12345)
                
                gc.collect()
                
                def _fit(whole_brain_mask = whole_brain_mask,
                         combined_mask = combined_mask,
                         idx_train = None,
                         idx_test = None,
                         ):
                    gc.collect()
                    pipeline        = build_model_dictionary(n_jobs            = 1,
                                                             remove_invariant  = True,
                                                             l1                = True,)[model_name]
                    clf = SearchLight(whole_brain_mask,
                                      combined_mask,
                                      radius = 10,
                                      estimator = pipeline,
                                      n_jobs = -1,
                                      verbose = 0,
                                      cv = check_cv(zip([idx_train],[idx_test])),
                                      scoring = 'roc_auc',
                                      )
                    clf.fit(BOLD_concat,labels_concat,)
                    gc.collect()
                    return clf.scores_
                
                temp = _fit(whole_brain_mask,
                            combined_mask,
                            idx_train = idx_train,
                            idx_test = idx_test,
                            )
                    
                
                results_to_save = new_img_like(whole_brain_mask,temp,)
                results_to_save.to_filename(os.path.join(output_dir,file_name))

