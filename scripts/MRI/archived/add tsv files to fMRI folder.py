#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 21:01:23 2019

@author: nmei
"""

import os
from glob import glob
import numpy as np
behavior_dir = '../data/behavioral/ning_4_6_2019'
fMRI_dir = '../data/MRIunconsfeats/sub-01/func'
from shutil import copyfile

behavorial_files = glob(os.path.join(behavior_dir,'*trials.tsv'))
behavorial_files = np.sort(behavorial_files)
for files,fMRI_folder in zip(behavorial_files.reshape((-1,9)),
                             os.listdir(fMRI_dir)):
    for f in list(files):
        copyfile(f,
                 os.path.join(fMRI_dir,fMRI_folder,f.split('/')[-1]))
    for fMRI_file,f in zip(glob(os.path.join(fMRI_dir,
                                             fMRI_folder,
                                             '*.nii')),
                           glob(os.path.join(fMRI_dir,
                                             fMRI_folder,
                                             '*.tsv'))):
        file_name = fMRI_file.split('/')[-1].split('.')[0].replace('_bold','')
        pre = f.split('/')[:-1]
        os.rename(f,'{}/{}.tsv'.format('/'.join(pre),file_name))