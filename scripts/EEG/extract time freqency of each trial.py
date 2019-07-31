#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:33:58 2019

@author: nmei
"""

import mne
import os
import re
import numpy as np
from glob                    import glob
from datetime                import datetime


from mne.decoding            import (
                                        Scaler,
                                        Vectorizer,
                                        SlidingEstimator,
                                        cross_val_multiscore,
                                        GeneralizingEstimator
                                        )
from sklearn.preprocessing   import StandardScaler
#from sklearn.calbration      import CalibratedClassifierCV
#from sklearn.svm             import LinearSVC
from sklearn.linear_model    import (
#                                        LogisticRegressionCV,
                                        LogisticRegression
                                        )
from sklearn.pipeline        import make_pipeline
from sklearn.model_selection import (
                                        StratifiedShuffleSplit,
                                        cross_val_score
                                        )
from sklearn.base            import clone
from matplotlib              import pyplot as plt
from scipy                   import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools               import partial
from shutil                  import copyfile
copyfile('../utils.py','utils.py')
from utils                   import get_frames

n_jobs = 4
subject             = 'matie_5_23_2019' # pedro_5_14_2019 iÃ±aki_5_9_2019 clara_5_22_2019 ana_5_21_2019 leyre_5_13_2019 lierni_5_20_2019
# there was a bug in the csv file, so the early behavioral is treated differently
date                = '/'.join(re.findall('\d+',subject))
date                = datetime.strptime(date,'%m/%d/%Y')
breakPoint          = datetime(2019,3,10)
if date > breakPoint:
    new             = True
else:
    new             = False
working_dir         = f'../../data/clean EEG/{subject}'
working_data        = glob(os.path.join(working_dir,'*-epo.fif'))
frames              = get_frames(directory = f'../../data/behavioral/{subject}',new = new)
# create the directories for figure and decoding results (numpy arrays)
figure_dir          = f'../../figure/TF/{subject}'
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
array_dir           = f'../../data/TF/{subject}'
if not os.path.exists(array_dir):
    os.makedirs(array_dir)