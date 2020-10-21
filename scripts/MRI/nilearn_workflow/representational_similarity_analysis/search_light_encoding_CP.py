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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from glob                      import glob
from tqdm                      import tqdm
from copy                      import copy
from sklearn.utils             import shuffle
from shutil                    import copyfile
copyfile('../../../utils.py','utils.py')
from utils                     import make_ridge_model_CV
from sklearn.model_selection   import StratifiedShuffleSplit
from sklearn.preprocessing     import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn                   import metrics,linear_model
from sklearn.decomposition     import PCA
from sklearn.pipeline          import make_pipeline
from sklearn.exceptions        import ConvergenceWarning
from sklearn.model_selection   import cross_validate
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
#from brainiak.searchlight.searchlight import Searchlight
#from brainiak.searchlight.searchlight import Ball

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
    scaler = MinMaxScaler((-1,1))
    y_train = scaler.fit_transform(BOLD[idx_train])
    y_test = scaler.transform(BOLD[idx_test])
    reg = make_ridge_model_CV(perform_pca = False)
    reg.fit(features[idx_train],y_train)
    score = metrics.r2_score(reg.predict(features[idx_test]),y_test,multioutput = 'raw_values')
    print(score.round(1))
    return score.mean()

if __name__ == "__main__":

    sub                 = 'sub-01'
    first_session       = 2
    stacked_data_dir    = '../../../../data/BOLD_whole_brain_averaged/{}/'.format(sub)
    mask_dir            = '../../../../data/MRI/{}/anat/ROI_BOLD'.format(sub)
    output_dir          = '../../../../results/MRI/nilearn/encoding_searchlight/{}'.format(sub)
    if not os.path.exists(output_dir):
        os.mkirs(output_dir)
    masks               = glob(os.path.join(mask_dir,'*.nii.gz'))
    whole_brain_mask    = f'../../../../data/MRI/{sub}/anat/ROI_BOLD/combine_BOLD.nii.gz'
    func_brain          = f'../../../../data/MRI/{sub}/func/example_func.nii.gz'
    feature_dir         = '../../../../data/computer_vision_features'
    label_map           = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}
    average             = True
    n_splits            = 10
    test_size           = 0.2
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
        data = masker.fit_transform(load_fmri(BOLD_file))
#        BOLD_image      = load_fmri(BOLD_file)
        VT                      = VarianceThreshold()
        scaler                  = MinMaxScaler((-1,1))
        BOLD_norm        = VT.fit_transform(data)
        BOLD_sc          = scaler.fit_transform(BOLD_norm)

        
        
        cv = StratifiedShuffleSplit(n_splits = n_splits,test_size = test_size,random_state = 12345)
        idxs_train,idxs_test = [],[]
        for idx_train,idx_test in cv.split(df_data,targets):
            idxs_train.append(idx_train)
            idxs_test.append(idx_test)
        
        
        reg = make_ridge_model_CV(perform_pca = True,alpha_space = [1,27])
        res = cross_validate(reg,CNN_feature,BOLD_sc,cv = zip(idxs_train,idxs_test),return_estimator = True,n_jobs = -1,verbose = 1)
        regs = res['estimator']
        preds   = [est.predict(CNN_feature[idx_test]) for est,idx_test in zip(regs,idxs_test)]
        scores  = np.array([metrics.r2_score(BOLD_sc[idx_test],pred,multioutput = 'raw_values') for idx_test_target,pred in zip(idxs_test,preds)])
        corr    = [np.mean([np.corrcoef(a,b)[0, 1]**2 for a,b in zip(BOLD_sc[idx_test],pred)]) for idx_test,pred in zip(idxs_test,preds)]
        score = np.mean(scores,0)
        print(score[score > 0])
        
#        def _searchlight_RSA(idx_train,
#                            idx_test,
#                            sl_rad = 1, 
#                            max_blk_edge = 5, 
#                            shape = Ball,
#                            min_active_voxels_proportion = 0,
#                            ):
#            sl = Searchlight(sl_rad = sl_rad, 
#                             max_blk_edge = max_blk_edge, 
#                             shape = shape,
#                             min_active_voxels_proportion = min_active_voxels_proportion,
#                             )
#            sl.distribute([BOLD_image.get_data()], load_fmri(whole_brain_mask).get_data() == 1)
#            sl.broadcast([CNN_feature,idx_train,idx_test])
#            # run searchlight algorithm
#            global_outputs = sl.run_searchlight(sfn,pool_size = 1)
#            return global_outputs
#        print(f'{conscious_state}')
#        k = _searchlight_RSA(idxs_train[0],idxs_test[0])
#        results_to_save = new_img_like(BOLD_image,k,)
#        results_to_save.to_filename(os.path.join(output_dir,f'{conscious_state}.nii.gz'))
        asdf
#        res = Parallel(n_jobs = -1,verbose = 1,)(delayed(_searchlight_RSA)(**{
#                'idx_train':idx_train,
#                'idx_test':idx_test}) for idx_train,idx_test in zip(idxs_train,idxs_test))
#    
#        results_to_save = np.zeros(np.concatenate([BOLD_image.shape[:3],[n_splits]]))
#        for ii,item in enumerate(res):
#            results_to_save[:,:,:,ii] = np.array(item, dtype=np.float)
#        results_to_save = new_img_like(BOLD_image,results_to_save,)
#        results_to_save.to_filename(os.path.join(output_dir,f'{conscious_state}.nii.gz'))






















