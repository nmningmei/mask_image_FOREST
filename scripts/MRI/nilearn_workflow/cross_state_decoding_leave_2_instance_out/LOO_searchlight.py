#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 05:43:31 2020

@author: nmei
"""

import os
import gc
import multiprocessing

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
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from nilearn.decoding import SearchLight
from nilearn.image import concat_imgs,index_img
from brainiak.searchlight.searchlight import Searchlight
from brainiak.searchlight.searchlight import Ball
from joblib                    import Parallel,delayed

if __name__ == "__main__":

    sub                 = 'sub-01'
    stacked_data_dir    = '../../../../data/BOLD_whole_brain_averaged/{}/'.format(sub)
    mask_dir            = '../../../../data/MRI/{}/anat/ROI_BOLD'.format(sub)
    output_dir          = '../../../../results/MRI/nilearn/LOO_searchlight/{}'.format(sub)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    masks               = glob(os.path.join(mask_dir,'*.nii.gz'))
    whole_brain_mask    = f'../../../../data/MRI/{sub}/func/mask.nii.gz'
    func_brain          = f'../../../../data/MRI/{sub}/func/example_func.nii.gz'
    feature_dir         = '../../../../data/computer_vision_features_no_background'
    label_map           = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}
    average             = True
    n_jobs              = -1
    
    conscious_source    = 'unconscious'
    conscious_target    = 'unconscious'
    model_names         = ['None + Linear-SVM','None + Dummy',]
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
    
    print(f'partitioning target: {conscious_target}')
    idxs_train_target,idxs_test_target  = LOO_partition(df_data_target)
    n_splits                            = len(idxs_test_target)
    print(f'{n_splits} folds of testing')

    print(f'partitioning source: {conscious_source}')
    # for each fold of the train-test, we throw away the subcategories that exist in the target
    cv_warning                  = False
    idxs_train_source           = []
    idxs_test_source            = []
    for idx_test_target in tqdm(idxs_test_target):
        df_data_target_sub      = df_data_target.iloc[idx_test_target]
        unique_subcategories    = pd.unique(df_data_target_sub['labels'])
        # category check:
        # print(Counter(df_data_target_sub['targets']))
        idx_train_source        = []
        idx_test_source         = []
        for subcategory,df_data_source_sub in df_data_source.groupby(['labels']):
            if subcategory not in unique_subcategories:
                idx_train_source.append(list(df_data_source_sub.index))
            else:
                idx_test_source.append(list(df_data_source_sub.index))
        idx_train_source        = np.concatenate(idx_train_source)
        idx_test_source         = idx_train_source.copy()
        
        # check if the training and testing have subcategory overlapping
        target_set              = set(pd.unique(df_data_target.iloc[idx_test_target]['labels']))
        source_set              = set(pd.unique(df_data_source.iloc[idx_train_source]['labels']))
        overlapping             = target_set.intersection(source_set)
        # print(f'overlapped subcategories: {overlapping}')
        if len(overlapping) > 0:
            cv_warning          = True
        idxs_train_source.append(idx_train_source)
        # the testing set for the source does NOT matter since we don't care its performance
        idxs_test_source.append(idx_test_source)
    
    
    
    
    # modify cv index
    idxs_trains = idxs_train_source.copy()
    
    idxs_tests = np.array([item + BOLD_image_source.shape[-1] for item in idxs_test_target]
                              )
    if not cv_warning:
        for model_name in model_names:
            file_name           = f'({sub}_{conscious_source}_{conscious_target}_{model_name}).csv'.replace(' + ','_')
            print(f'{model_name} {conscious_source} --> {conscious_target}')
            if not os.path.exists(os.path.join(output_dir,file_name)):
                np.random.seed(12345)
                
                targets         = targets_source.copy()
                
                
                # Define voxel function
                def sfn(l, msk, myrad, bcast_var):
                    """
                    l: BOLD
                    msk: mask array
                    myrad: not use
                    bcast_var: label zip()cv
                    """
                    X = l[0].copy()
                    X = X.reshape(msk.shape[0]*msk.shape[1]*msk.shape[2],l[0].shape[3]).T
                    y = bcast_var[0].copy()
                    idx_train,idx_test = bcast_var[1]
                    pipeline        = build_model_dictionary(n_jobs            = 1,
                                                             remove_invariant  = True,
                                                             l1                = True,)[model_name]
                    # print(X.shape,y.shape,cv)
                    pipeline.fit(X[idx_train],y[idx_train])
                    y_true = y[idx_test]
                    y_pred = pipeline.predict_proba(X[idx_test])[:,-1]
                    res = roc_auc_score(y_true,y_pred)
                    # print(res)
                    return res
                
                gc.collect()
                
                def _searchlight_decode(BOLD_concat,
                                        labels_concat,
                                    sl_rad = 6, 
                                    max_blk_edge = 5, 
                                    shape = Ball,
                                    min_active_voxels_proportion = 0,
                                    idx_train_source = None,
                                    idx_test_target = None,
                                    n_jobs = 1,
                                    ):
                    sl = Searchlight(sl_rad = sl_rad, 
                                     max_blk_edge = max_blk_edge, 
                                     shape = shape,
                                     min_active_voxels_proportion = min_active_voxels_proportion,
                                     )
                    sl.distribute([np.asanyarray(BOLD_concat.dataobj)], 
                                   np.asanyarray(load_fmri(whole_brain_mask).dataobj) == 1)
                    sl.broadcast([labels_concat,(idx_train_source,idx_test_target)])
                    # run searchlight algorithm
                    global_outputs = sl.run_searchlight(sfn,pool_size = n_jobs)
                    return global_outputs
                for _ in range(10):
                    gc.collect()
                
                print(f'working ')
                for idx_train_source,idx_test_target in tqdm(zip(idxs_train_source[:5],idxs_test_target)):
                    BOLD_concat = concat_imgs([index_img(BOLD_image_target,idx_train_source),
                                               index_img(BOLD_image_target,idx_test_target)])
                    labels_concat = np.concatenate([targets_source[idx_train_source],
                                                    targets_target[idx_test_target]])
                    global_outputs = _searchlight_decode(BOLD_concat,
                                                         labels_concat,
                                                        idx_train_source = np.arange(idx_train_source.shape[0]),
                                                        idx_test_target  = np.arange(idx_test_target.shape[0]) + idx_train_source.shape[0],
                                                        n_jobs = multiprocessing.cpu_count())
                    adf
                    print(f'# voxel = {np.sum(global_outputs > 0.5)}, max roc = {np.max(global_outputs):.4f}, ave roc = {np.mean(global_outputs):.4f}')
                    gc.collect()
                
                
                # clf = SearchLight(whole_brain_mask,
                #                   radius = 6,
                #                   estimator = 'svc',
                #                   n_jobs = -1,
                #                   verbose = 1,
                #                   cv = 2,#zip(idxs_trains,idxs_tests),
                #                   scoring = 'roc_auc',
                #                   )
                # clf.fit(BOLD_concat,labels_concat,)















