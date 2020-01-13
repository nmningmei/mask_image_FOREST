#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:22:06 2019

@author: nmei
"""
import os
import mne
import numpy as np
from mne.time_frequency import tfr_morlet

all_subjects = [
#                'aingere_5_16_2019',
#                'alba_6_10_2019',
#                'alvaro_5_16_2019',
#                'clara_5_22_2019',
                'ana_5_21_2019',
                'inaki_5_9_2019',
                'jesica_6_7_2019',
                'leyre_5_13_2019',
                'lierni_5_20_2019',
                'maria_6_5_2019',
                'matie_5_23_2019',
                'out_7_19_2019',
                'mattin_7_12_2019',
                'pedro_5_14_2019',
                'xabier_5_15_2019',
                ]
for subject in all_subjects:#= 'matie_5_23_2019'
    working_dir = f'../../data/EEG/{subject}'
    folder_name = "clean_EEG_premask_baseline"
    epochs = mne.read_epochs(f'../../data/{folder_name}/{subject}/clean-epo.fif')
    print('downsampling')
    epochs = epochs.resample(100)
    
    freqs = np.arange(8,20.5,0.5)
    n_cycles = freqs / 2.
    conscious   = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if (' conscious' in name)])
    see_maybe   = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('glimpse' in name)])
    unconscious = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('unconscious' in name)])
    del epochs
    for ii,(epochs,conscious_state) in enumerate(zip([unconscious,see_maybe,conscious],
                                                         ['unconscious',
                                                          'glimpse',
                                                          'conscious'])):
        tfr = tfr_morlet(epochs,
                         freqs=freqs,
                         n_cycles=n_cycles,
                         return_itc=False,
                         average=False,
                         n_jobs = -1)
        avgpower = tfr.average()
        avgpower.plot([0], baseline=(-.2, 0), mode='mean',
                      title='Using Morlet wavelets and EpochsTFR', show=False)
    
    
    
    
    
    
    
    
    
    
    

