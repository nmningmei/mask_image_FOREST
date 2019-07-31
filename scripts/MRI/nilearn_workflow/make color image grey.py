#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:29:03 2019

@author: nmei
"""
import os
from glob import glob
from PIL import Image
from tqdm import tqdm

working_dir = '../../../experiment_images_tilted_2'
working_data = glob(os.path.join(working_dir,
                                 "*",
                                 "*",
                                 "*.jpg"))
saving_dir = '../../../grayscaled'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
for image in tqdm(working_data):
    saving_name = image.split('/')
    saving_name[3] = 'grayscaled'
    
    saving_path = '/'.join(saving_name[:-1])
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    img = Image.open(image).convert("L")
    img.save('/'.join(saving_name))
