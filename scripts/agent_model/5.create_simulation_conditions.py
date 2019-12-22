#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:21:28 2019

@author: nmei
"""
import itertools
import numpy as np
import pandas as pd

units = [5,10,20,50,100,300]
dropouts = [0, 0.25,0.5,0.75]
models = ['DenseNet169',           # 1664
          'InceptionV3',           # 2048
          'MobileNetV2',           # 1280
          'ResNet50',              # 1536
          'VGG19',                 # 2048
          'Xception',              # 1280
          ]
activations = ['elu',
               'relu',
               'selu',
               'sigmoid',
               'tanh',
               'linear',
               ]
output_activations = ['sofmax','sigmoid',]


temp = np.array(list(itertools.product(*[units,dropouts,models,activations,output_activations])))
df = pd.DataFrame(temp,columns = ['units','dropouts','model_names','hidden_activations','output_activations'])
df['units'] = df['units'].astype(int)
df['dropouts'] = df['dropouts'].astype(float)

preproc_func = {'DenseNet169':'densenet',
                'InceptionV3':'inception_v3',
                'MobileNetV2':'mobilenet_v2',
                'ResNet50':'resnet50',
                'VGG19':'vgg19',
                'Xception':'xception',
                }
df['preprocess_input'] = df['model_names'].map(preproc_func)