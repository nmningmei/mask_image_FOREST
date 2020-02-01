#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:25:16 2019

@author: nmei
"""

import os
from glob import glob
from tqdm import tqdm

working_dir = ''
working_data = glob(os.path.join(working_dir,'*.csv'))