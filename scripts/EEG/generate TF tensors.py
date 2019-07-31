#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:04:58 2019

@author: nmei

Generate time frequency power spectral wit morlet cycles


"""

import mne
import os
import re
import numpy as np
from glob                    import glob
from datetime                import datetime


from mne.time_frequency      import tfr_morlet
from shutil                  import copyfile
copyfile('../utils.py','utils.py')
from utils                   import get_frames

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
            tfr.save(os.path.join(saving_dir,f'TFR_morlet_{conscious_state}-tfr.h5'),overwrite=True)
