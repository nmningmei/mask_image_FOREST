#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:13:27 2019

@author: nmei
"""

import keras
from keras import layers,applications,models,optimizers,losses,metrics
from nibabel import load as load_fmri
import numpy as np

base_model = applications.nasnet.NASNetLarge(
                                                   include_top = False,
                                                   weights = None,
                                                   input_shape = (88,88,66),
                                                   pooling = 'max',)

BOLD = load_fmri('filtered.nii.gz')
BOLD_data = BOLD.get_data()
BOLD_data = np.swapaxes(BOLD_data,-1,-2)
BOLD_data = np.swapaxes(BOLD_data,-2,-3)
BOLD_data = np.swapaxes(BOLD_data,-3,-4)

labels = np.random.choice(2,size = 508)
labels = np.vstack([labels,1-labels]).T

inputs = base_model.inputs[0]
base_outputs = base_model.outputs[0]
dense1 = layers.Dense(100,name = 'dense1')(base_outputs)
outputs = layers.Dense(2,activation='softmax',name = 'outputs')(dense1)
model = models.Model(inputs,outputs,name = 'classifier')
model.compile(optimizer = optimizers.Adam(),loss = losses.categorical_crossentropy,metrics=[metrics.categorical_accuracy])

model.fit(BOLD_data,labels,epochs = 10,batch_size=2)
