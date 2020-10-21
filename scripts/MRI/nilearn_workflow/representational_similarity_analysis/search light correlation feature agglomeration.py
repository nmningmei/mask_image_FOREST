#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 06:27:30 2020

@author: nmei
"""

import os
from glob import glob

import numpy as np

from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.image import grid_to_graph
from skopt import BayesSearchCV
from skopt.space import Integer
from nilearn.input_data import NiftiMasker
from nilearn.image      import new_img_like
from nibabel            import load as load_img

sub = 'sub-01'
working_dir = f'../../../../results/MRI/nilearn/RSA_searchlight/{sub}'
working_data = glob(os.path.join(working_dir,'*.nii.gz'))
BOLD_mask   = glob('/bcbl/home/home_n-z/nmei/MRI/uncon_feat/MRI/{}/func/*/*/*/*/mask.nii.gz'.format(
                sub))[0]
functional_mask = load_img(BOLD_mask)
functional_brain = f'../../../../data/MRI/{sub}/func/example_func.nii.gz'
combined_mask_dir = f'/bcbl/home/home_n-z/nmei/MRI/uncon_feat/MRI/{sub}/anat/ROI_BOLD/'
combined_mask_file = glob(os.path.join(combined_mask_dir,'combine_BOLD.nii.gz'))[0]
mask_dir    = '/bcbl/home/home_n-z/nmei/MRI/uncon_feat/MRI/{}/anat/ROI_BOLD'.format(
                sub)
masks       = glob(os.path.join(mask_dir,'*.nii.gz'))
left = np.sort([item for item in masks if 'lh' in item])
right = np.sort([item for item in masks if 'rh' in item])
unique_mask_names = [item.split('/')[-1].split('-')[2].split('_BOLD')[0] for item in left]

mask_name   = [item for item in masks if unique_mask_names[0] in item]
# combine the left and right ROIs
array_1 = np.asanyarray(load_img(mask_name[0]).dataobj)
array_2 = np.asanyarray(load_img(mask_name[1]).dataobj)
array_combined = array_1 + array_2
array_combined[array_combined > 0] = 1
array_combined[np.asanyarray(functional_mask.dataobj) == 0] = 0
roi_mask_combined = new_img_like(functional_mask,array_combined)

lable_map = dict(unconscious = 0,
                 glimpse = 1,
                 conscious = 2,)

# full-size combined ROI
features = []
labels = []
for file_name in working_data:
    file_name
    conscious_state = file_name.split('/')[-1].replace('.nii.gz','')
    masker = NiftiMasker(mask_img = roi_mask_combined,smoothing_fwhm = 6)
    
    features_ = masker.fit_transform(file_name)
    # features_ = features_.reshape(-1,5,features_.shape[-1]).mean(1)
    labels_ = np.array([lable_map[conscious_state] for _ in range(features_.shape[0])])
    
    features.append(features_)
    labels.append(labels_)
    
    del features_

features = np.concatenate(features)
labels = np.concatenate(labels)

print(f'combined ROI, features shape = {features.shape}')
cv = StratifiedShuffleSplit(n_splits = 10,test_size = .2,random_state = 12345)
mask_array = np.asanyarray(roi_mask_combined.dataobj)
connectivity = grid_to_graph(*mask_array.shape,mask = mask_array.astype('bool'))
feature_selection = FeatureAgglomeration(connectivity = connectivity,
                                          # affinity = 'cosine',
                                          linkage = 'single',
                                         )
rf = RandomForestClassifier(random_state = 12345,n_jobs = 1)

raw_scores = cross_val_score(rf,
                             features,
                             labels,
                             cv = cv,
                             scoring = 'accuracy',
                             n_jobs = -1,
                             verbose = 1)

pipeline = make_pipeline(feature_selection,rf)
params = dict(featureagglomeration__n_clusters = Integer(2, 100))#features.shape[1]))
search = BayesSearchCV(pipeline, params,
                       cv = cv,
                       scoring = 'accuracy',
                       n_jobs = -1,
                       verbose = 1,
                       )
search.fit(features,labels)

from nilearn.plotting import plot_stat_map
from matplotlib import pyplot as plt
feature_cluster_labels = search.best_estimator_.steps[0][-1].labels_
plot_stat_map(masker.inverse_transform(feature_cluster_labels),
              functional_brain,
              cmap = plt.cm.Set1,
              threshold = .9,
              draw_cross = False,)
asdf





for mask_pick in unique_mask_names:
    mask_name   = [item for item in masks if mask_pick in item]
    # combine the left and right ROIs
    array_1 = np.asanyarray(load_img(mask_name[0]).dataobj)
    array_2 = np.asanyarray(load_img(mask_name[1]).dataobj)
    array_combined = array_1 + array_2
    array_combined[array_combined > 0] = 1
    array_combined[np.asanyarray(functional_mask.dataobj) == 0] = 0
    roi_mask_combined = new_img_like(functional_mask,array_combined)
    
    features = []
    labels = []
    for file_name in working_data:
        file_name
        conscious_state = file_name.split('/')[-1].replace('.nii.gz','')
        masker = NiftiMasker(mask_img = roi_mask_combined,)
        
        features_ = masker.fit_transform(file_name)
        # features_ = features_.reshape(-1,5,features_.shape[-1]).mean(1)
        labels_ = np.array([lable_map[conscious_state] for _ in range(features_.shape[0])])
        
        features.append(features_)
        labels.append(labels_)
        
        del features_

    features = np.concatenate(features)
    labels = np.concatenate(labels)
    print(f'{mask_pick} features shape = {features.shape}')
    cv = StratifiedShuffleSplit(n_splits = 10,test_size = .2,random_state = 12345)
    params = dict(featureagglomeration__n_clusters =Integer(2, 100))
    
    feature_selection = FeatureAgglomeration()
    rf = RandomForestClassifier(random_state = 12345,n_jobs = 1)
    pipeline = make_pipeline(feature_selection,rf)
    search = BayesSearchCV(pipeline, params,
                           cv = cv,
                           scoring = 'accuracy',
                           n_jobs = -1,
                           verbose = 1,
                           )
    search.fit(features,labels)
    
    




























