#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 19:06:31 2019

@author: nmei
"""
import os
import pickle
from glob import glob
from shutil                  import copyfile
copyfile('../../utils.py','utils.py')
from utils                   import (partioning_preload)
sub                 = 'sub-01'
stacked_data_dir    = '../../../data/BOLD/{}/'.format(sub)
BOLD_data           = glob(os.path.join(stacked_data_dir,'*BOLD*.npy'))
event_data          = glob(os.path.join(stacked_data_dir,'*.csv'))
label_map           = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}
n_splits = 500
conscious_state = 'unconscious'#['unconscious','glimpse','conscious']:

train_test_split = partioning_preload(
                                      event_data[0],
                                      label_map,
                                      conscious_state,
                                      n_splits = n_splits)
output_dir = '../../../results/MRI/customized_partition'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir,f'{conscious_state}.pkl'),'wb') as fp:
    pickle.dump(train_test_split,fp,protocol = 2)
    fp.close()













