#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:40:30 2019

@author: nmei
"""

import os
from glob import glob
import numpy as np
fMRI_dir = '../../data/MRI/sub-01/func'
from shutil import copyfile

all_files = np.array(glob(os.path.join(fMRI_dir,'*','*')))
all_files = np.sort(all_files)
all_files = all_files.reshape(-1,9,3)

for session in all_files:
    for single_run in session:
        name = single_run[0].split('.tsv')[0]
        folder_to_recreate = '{}'.format(name)
        if not os.path.exists(folder_to_recreate):
            os.mkdir(folder_to_recreate)
        for item in single_run:
            copyfile(item,
                     os.path.join(folder_to_recreate,
                                  item.split('/')[-1]))
        
