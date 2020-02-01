#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:07:53 2020

@author: nmei
"""

units = [5,10,20,50,100,300]
dropouts = [0,0.1,0.2]
models = ['DenseNet169',           # 1664
          'InceptionV3',           # 2048
          'MobileNetV2',           # 1280
          'ResNet50',              # 1536
          'VGG19',                 # 2048
          'Xception',              # 1280
          ]
activations = {'elu':nn.ELU,
               'relu':nn.ReLU,
               'selu':nn.SELU,
               'sigmoid':nn.Sigmoid,
               'tanh':nn.Tanh,
               'linear':nn.Linear,
               }
output_activations = {'softmax':nn.Softmax,'sigmoid':nn.Sigmoid,}
output_dim = {'softmax':2,'sigmoid':1}