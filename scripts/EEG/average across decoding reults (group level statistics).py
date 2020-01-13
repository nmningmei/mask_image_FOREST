#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 12:42:11 2020

@author: nmei
"""

import os
from glob import glob
from shutil import copyfile
copyfile(os.path.abspath('../utils.py'),'utils.py')
import numpy as np
from utils import (get_frames,
                   LOO_partition,
                   plot_temporal_decoding,
                   plot_temporal_generalization,
                   plot_t_stats,
                   plot_p_values)

folder_name = 'decode_massive_CV'
working_dir = f'../../results/EEG/{folder_name}'
for ii,conscious_state in enumerate(['unconscious','glimpse','conscious']):
    working_data = glob(os.path.join(working_dir,'*',f'temporal_generalization_{conscious_state}.npy'))
    
    data = np.array([np.load(f).mean(0) for f in working_data])
    times = [-.2,1,-.2,1]
    fig,ax = plot_temporal_generalization(data - 0.5,
                                          times,
                                          None,
                                          conscious_state,
                                          None,
                                          vmin = 0,
                                          vmax = .01,)
    
