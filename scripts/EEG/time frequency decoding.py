#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:10:31 2019

@author: nmei
"""

import mne
import os
import re
import numpy as np
from glob                      import glob
from datetime                  import datetime


from mne.time_frequency        import tfr_morlet
from mne                       import decoding
from sklearn.preprocessing     import StandardScaler
from xgboost                   import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model      import LogisticRegressionCV
from sklearn.pipeline          import make_pipeline
from sklearn.model_selection   import StratifiedShuffleSplit,cross_val_score
from sklearn.utils             import shuffle
from shutil                    import copyfile
copyfile('../utils.py','utils.py')
from utils                     import get_frames

all_subjects = [
                'matie_5_23_2019',
                'pedro_5_14_2019',
                'aingere_5_16_2019',
                'inaki_5_9_2019',
                'clara_5_22_2019',
                'ana_5_21_2019',
                'leyre_5_13_2019',
                'lierni_5_20_2019',]
for subject in all_subjects:
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
    frames,res          = get_frames(directory = f'../../data/behavioral/{subject}',new = new)
    
    saving_dir          = f'../../data/TF/{subject}'
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    
    
    for epoch_file in working_data:
        epochs  = mne.read_epochs(epoch_file)
        # resample at 100 Hz to fasten the decoding process
        print('resampling')
        epochs.resample(100)
        
        conscious   = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if (' conscious' in name)])
        see_maybe   = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('glimpse' in name)])
        unconscious = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('unconscious' in name)])
        
        for ii,(epochs,conscious_state) in enumerate(zip([unconscious,see_maybe,conscious],
                                                         ['unconscious',
                                                          'glimpse',
                                                          'conscious'])):
            epochs
            freqs = np.arange(epochs.info['highpass'],epochs.info['lowpass'],1.)
            n_cycles = freqs / 2.
            tfr = tfr_morlet(epochs,freqs=freqs, n_cycles=n_cycles,
                           return_itc=False,
                           average=False,n_jobs = 2)
            tfr.events = epochs.events
            tfr.event_id = epochs.event_id
            xgb = XGBClassifier(
                            learning_rate                           = 1e-3, # not default
                            max_depth                               = 10, # not default
                            n_estimators                            = 200, # not default
                            objective                               = 'binary:logistic', # default
                            booster                                 = 'gbtree', # default
                            subsample                               = 0.9, # not default
                            colsample_bytree                        = 0.9, # not default
                            reg_alpha                               = 0, # default
                            reg_lambda                              = 1, # default
                            random_state                            = 12345, # not default
                            importance_type                         = 'gain', # default
                            n_jobs                                  = 4,# default to be 1
                                                  )
            rf = SelectFromModel(xgb,
                            prefit                                  = False,
                            threshold                               = 'mean' # induce sparsity
                            )
            logistic = LogisticRegressionCV(Cs = np.logspace(-3,3,7),
                                          penalty = 'l2',
                                          solver        = 'lbfgs',
                                          tol           = 1e-3,
                                          max_iter      = int(1e3),
                                          scoring       = 'roc_auc',
                                          cv            = StratifiedShuffleSplit(),
                                          random_state  = 12345,
                                          n_jobs        = 4,
                                          )
    
            clf = make_pipeline(decoding.Vectorizer(),StandardScaler(),rf,logistic)
            features = tfr.data.astype('float32')
            targets = tfr.events[:,-1]//10 -2
            features,targets = shuffle(features,targets)
            scores = cross_val_score(clf,features,targets,scoring='roc_auc',
                                     cv=3,n_jobs=6,verbose=2)
            sdfa