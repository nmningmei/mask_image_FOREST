#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:50:12 2019

@author: nmei
"""

import os
from glob import glob
import numpy as np
from shutil import copyfile

working_dir = ['../data/train','../data/validate']
saving_dir = ['../temp/train','../temp/validate']
for d in saving_dir:
    if not os.path.exists(d):
        os.makedirs(d)

for wd,sd in zip(working_dir,saving_dir):
    classes = os.listdir(wd)
    
    for cate in classes:
        all_images  = glob(os.path.join(wd,cate,'*.jpg'))
        
        sampled = np.random.choice(all_images,size= 100,replace=False)
        for sample in sampled:
            sample_saving_dir = sample.replace('data/','temp/')
            if not os.path.exists(os.path.join(*sample_saving_dir.split('/')[:-1])):
                os.makedirs(os.path.join(*sample_saving_dir.split('/')[:-1]))
            copyfile(sample,sample_saving_dir)
