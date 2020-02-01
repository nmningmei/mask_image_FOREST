#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 13:14:45 2019

@author: nmei
"""

import os
from PIL import Image
from glob import glob
from tqdm import tqdm

working_dir = '../../../data/101_ObjectCategories/'
saving_dir = '../../../data/101_ObjectCategories_grayscaled/'

if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

images = glob(os.path.join(working_dir,
                           "*",
                           "*"))

for file_name in tqdm(images):
    img = Image.open(file_name)
    
    saving_name = file_name.split('/')
    saving_name[4] = '101_ObjectCategories_grayscaled'
    if not os.path.exists(os.path.join(*saving_name[:-1])):
        os.makedirs(os.path.join(*saving_name[:-1]))
    
    img_save = img.convert('LA')
    
    img_save.save(os.path.join(*saving_name).replace('jpg','png'))





































