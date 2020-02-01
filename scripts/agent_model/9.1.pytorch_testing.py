#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:55:30 2020

@author: nmei

require:
    pip install cnn_finetune
"""

from cnn_finetune import make_model
import torch.nn as nn
from tensorflow.keras.preprocessing.image import ImageDataGenerator

working_dir         = '../../data'
# define some hyperparameters for training
batch_size          = 16
image_resize        = 128
drop_rate           = 0
hidden_activation   = 'relu'
output_activation   = 'sigmoid'
model_name          = 'DenseNet169'
preprocess_input    = None
patience            = 5
device              = 'cpu'
n_splits            = 50 # n_split for decoding the hidden layer
n_permutations      = int(1e3) # for computation speed
n_sessions          = int(2e2) # n_permutations for CNN performance
loss_func           = None
hidden_units        = 10
verbose             = 1
max_epochs          = int(1e4) # arbitrary choice
model_dir           = f'../../results/agent_models/{model_name}_{drop_rate}_{hidden_units}_{hidden_activation}_{output_activation}'



#def make_classifier(in_features,hidden_features, num_classes,actiation_func):
#    return nn.Sequential(
#        nn.Linear(in_features, hidden_features),
#        actiation_func(inplace=True),
#        nn.Linear(hidden_features, num_classes),
#    )
#
#model = make_model('vgg16', num_classes=10, pretrained=True, input_size=(256, 256), classifier_factory=make_classifier)





























































