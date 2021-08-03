#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:47:13 2020

@author: nmei

semisupervised learning clusterin
train a feature extractor for whole brain

"""

import os
import torch
import numpy as np
from glob import glob
from nilearn.input_data import NiftiMasker
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import OPTICS
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from torch.utils.data import DataLoader,TensorDataset
from torch.autograd import Variable
from torch import nn
from shutil import copyfile
copyfile('../../../utils_deep.py','utils_deep.py')
from utils_deep import CNN2D_model,rnn_classifier

wholebrain_folder = 'BOLD_whole_brain_averaged'
conscious_state = 'unconscious'
batch_size = 16
shuffle = True

working_dir = f'../../../../data/{wholebrain_folder}'
working_data = np.sort(glob(os.path.join(working_dir,'*',f'whole_brain_{conscious_state}.nii.gz')))
working_csv = np.sort(glob(os.path.join(working_dir,'*',f'whole_brain_{conscious_state}.csv')))
working_masks = np.sort(glob(
        os.path.join(*working_dir.split('/')[:-1],
        'MRI',
        '*',
        'func',
        'mask.nii.gz'),
        ))

# pick one
idx = 0
masker = NiftiMasker(mask_img = working_masks[idx],
                     standardize = True,
                     detrend = True,
                     verbose = 1,
                     )
masker.fit()
VT = VarianceThreshold()
scaler = MinMaxScaler()
cluster = OPTICS(n_jobs = -1)
pipeline = make_pipeline(VT,scaler,cluster)
# load and preprocess the data (i.e., standarization, detrending)
temp = masker.transform(working_data[idx])
# initial cluster of the data, for psuedo-labeling
#pipeline.fit(temp)
# scaler the data for later use
temp = scaler.fit_transform(temp)
# reshape back to 4D
BOLD = np.asanyarray(masker.inverse_transform(temp).dataobj)
# convert numpyarray to tensors, here, Tensor.T is a way of swapping axes (permute)
data = TensorDataset(torch.from_numpy(BOLD).T)
# create data loader
dataloader = DataLoader(data,
                        batch_size = batch_size,
                        shuffle = shuffle,
                        num_workers = 2,
                        )
k = next(iter(dataloader))[0]


FeatureExtractor = CNN2D_model(batch_size = batch_size)
classifier = rnn_classifier(batch_size = batch_size,feature_extractor = FeatureExtractor)
a,b = classifier(k)

print(a.shape,b.shape)

from torchvision import models
mobilenet = models.mobilenet_v2(pretrained = True)
mobilenet.features[0][0] = nn.Conv2d(66,32,kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
j = mobilenet(k)

