#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:35:18 2020

@author: nmei
"""

import os
import utils_deep
from glob import glob

import numpy  as np
import pandas as pd

working_dir         = '../../results/agent_models'
CNN_data            = glob(os.path.join(working_dir,'*','*.csv'))
decoding_data       = glob(os.path.join(working_dir,'*','*','*.csv'))