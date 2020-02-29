#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:04:11 2020

@author: nmei
"""

import os
import gc
import warnings
warnings.filterwarnings('ignore') 
import pandas  as pd
import numpy   as np
np.set_printoptions(suppress=True)
import seaborn as sns

from glob                      import glob
from tqdm                      import tqdm
from copy                      import copy
from sklearn.utils             import shuffle
from shutil                    import copyfile
copyfile('../../../utils.py','utils.py')
from sklearn.model_selection   import StratifiedShuffleSplit
from sklearn.preprocessing     import MinMaxScaler
from sklearn                   import metrics,linear_model
from sklearn.decomposition     import PCA
from sklearn.pipeline          import make_pipeline
from sklearn.exceptions        import ConvergenceWarning
from sklearn.utils.testing     import ignore_warnings
from collections               import OrderedDict
from matplotlib                import pyplot as plt
from scipy.spatial             import distance
from scipy.stats               import spearmanr
from joblib                    import Parallel,delayed
from nibabel                   import load as load_fmri
from nilearn.image             import new_img_like
from nilearn.input_data        import NiftiMasker
from nilearn.decoding          import SearchLight
from sklearn.metrics           import make_scorer
from brainiak.searchlight.searchlight import Searchlight
from brainiak.searchlight.searchlight import Ball

def score_func(y, y_pred,):
    temp        = metrics.r2_score(y,y_pred,multioutput = 'raw_values')
    if np.sum(temp > 0):
        return temp[temp > 0].mean()
    else:
        return 0
custom_scorer      = metrics.make_scorer(metrics.r2_score,greater_is_better = True)

def sfn(l, msk, myrad, bcast_var):
    """
    l: BOLD
    msk: mask array
    myrad: not use
    bcast_var: label -- CNN features
    """
    BOLD = l[0][msk,:].T.copy()
    features = bcast_var[0].copy()
    idx_train = bcast_var[1]
    idx_test = bcast_var[2]
    
    reg = linear_model.RidgeCV(alphas = np.logspace(0,10,11),
                               scoring = custom_scorer,
                               normalize = True, 
#                               random_state = 12345,
                               )
    reg = make_pipeline(PCA(n_components = .99, random_state = 12345),
                        reg)
    reg.fit(features[idx_train],BOLD[idx_train])
    score = metrics.r2_score(reg.predict(features[idx_test]),BOLD[idx_test],multioutput = 'raw_values')
    print(score.round(1))
    return score.mean()

if __name__ == "__main__":

    sub                 = 'sub-01'
    first_session       = 2
    stacked_data_dir    = '../../../../data/BOLD_whole_brain_averaged/{}/'.format(sub)
    mask_dir            = '../../../../data/MRI/{}/anat/ROI_BOLD'.format(sub)
    output_dir          = '../../../../results/MRI/nilearn/encoding_searchlight/{}'.format(sub)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    masks               = glob(os.path.join(mask_dir,'*.nii.gz'))
    whole_brain_mask    = f'../../../../data/MRI/{sub}/anat/ROI_BOLD/combine_BOLD.nii.gz'
    func_brain          = f'../../../../data/MRI/{sub}/func/example_func.nii.gz'
    feature_dir         = '../../../../data/computer_vision_features'
    label_map           = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}
    average             = True
    n_splits            = 2
    test_size           = 0.1
    n_jobs              = -1
    
    for conscious_state in ['unconscious','glimpse','conscious']:
        df_data         = pd.read_csv(os.path.join(stacked_data_dir,
                                                   f'whole_brain_{conscious_state}.csv'))
        df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
        df_data         = df_data[df_data.columns[1:]]
        BOLD_file       = os.path.join(stacked_data_dir,
                                       f'whole_brain_{conscious_state}.nii.gz')
        targets         = np.array([label_map[item] for item in df_data['targets']])[:,-1]
        images          = df_data['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
        CNN_feature     = np.array([np.load(os.path.join(feature_dir,
                                                         'DenseNet169',
                                                         item)) for item in images])
        groups          = df_data['labels'].values
        
        masker          = NiftiMasker(mask_img = whole_brain_mask,standardize = False)
        BOLD_vecs       = masker.fit_transform(BOLD_file)
        scaler                  = MinMaxScaler((-1,1))
        BOLD_sc          = scaler.fit_transform(BOLD_vecs)
        BOLD_image = masker.inverse_transform(BOLD_sc)
        
        cv = StratifiedShuffleSplit(n_splits = n_splits,test_size = test_size,random_state = 12345)
        idxs_train,idxs_test = [],[]
        for idx_train,idx_test in cv.split(BOLD_vecs,targets):
            idxs_train.append(idx_train)
            idxs_test.append(idx_test)
        
        def _searchligh_RSA(idx_train,
                            idx_test,
                            sl_rad = 6, 
                            max_blk_edge = 5, 
                            shape = Ball,
                            min_active_voxels_proportion = 0,
                            ):
            sl = Searchlight(sl_rad = sl_rad, 
                             max_blk_edge = max_blk_edge, 
                             shape = shape,
                             min_active_voxels_proportion = min_active_voxels_proportion,
                             )
            sl.distribute([BOLD_image.get_data()], load_fmri(whole_brain_mask).get_data() == 1)
            sl.broadcast([CNN_feature,idx_train,idx_test])
            # run searchlight algorithm
            global_outputs = sl.run_searchlight(sfn,pool_size = 1)
            return global_outputs
        print(f'{conscious_state}')
        res = Parallel(n_jobs = -1,verbose = 1,)(delayed(_searchligh_RSA)(**{
                'idx_train':idx_train,
                'idx_test':idx_test}) for idx_train,idx_test in zip(idxs_train,idxs_test))
    
        results_to_save = np.zeros(np.concatenate([BOLD_image.shape[:3],[n_splits]]))
        for ii,item in enumerate(res):
            results_to_save[:,:,:,ii] = np.array(item, dtype=np.float)
        results_to_save = new_img_like(BOLD_image,results_to_save,)
        results_to_save.to_filename(os.path.join(output_dir,f'{conscious_state}.nii.gz'))






















