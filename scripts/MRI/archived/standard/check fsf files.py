#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:08:47 2019

@author: nmei
"""

import os
from glob import glob

working_dir = ''


fsf_files = glob(os.path.join(working_dir,'*','*.fsf'))

for fsf_file in fsf_files:
    with open(fsf_file,'r') as f:
        for line in f:
            if "set feat_files(1)" in line:
                print(line)
        f.close()