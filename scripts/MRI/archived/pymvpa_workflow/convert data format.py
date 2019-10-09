#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:53:01 2019

@author: nmei
"""
import os
from glob import glob
import numpy as np
import pandas as pd
import utils


working_dir = '../../../data/ds_STACKED/'
working_data = glob(os.path.join(working_dir,'ning','*.pkl'))
saving_dir = '../../../data/BOLD_stacked'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
for f in working_data:
    dataset = utils.load_preprocessed(f)
    roi_name = f.split("BOLD")[-1].split('_fsl')[0]
    dataset.sa['labels'] = np.array([utils.map_labels()[item] for item in dataset.sa['labels']])
    
    data = dataset.samples.astype('float32')
    df_data = pd.DataFrame({name:np.array(value) for name,value in dataset.sa.items()})
    
    np.save(os.path.join(saving_dir,'{}.npy'.format(roi_name)),
            data)
    df_data.to_csv(os.path.join(saving_dir,
                                '{}.csv'.format(roi_name)),
                    index = False)
