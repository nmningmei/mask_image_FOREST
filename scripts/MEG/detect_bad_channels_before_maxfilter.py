#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:32:49 2019

@author: nmei

detect bad channels before maxfilter
"""

import os
import re
import mne
from glob import glob
import numpy as np
from shutil import copyfile
copyfile('../utils.py','utils.py')
from autoreject import Ransac


sub = 'pilot'
working_dir = f'../../data/MEG/{sub}'
working_data = glob(os.path.join(working_dir,'*.fif'))
bad_working_data = []#glob(os.path.join(working_dir,'60 Hz','*.fif'))

temp_ch_type_mapping = {"EEG 061":"eog",
                        "EEG 062":"ecg",
                        "EEG 063":"eog",}

for idx,a in enumerate(np.concatenate([working_data,bad_working_data])):
    session = int(re.findall(r'\d+',a)[-1])
    raw = mne.io.read_raw_fif(a,preload = True)
    ch_names = raw.ch_names
    new_ch_names = {item:item[:3] + ' ' + item[3:] for item in ch_names}
    raw.rename_channels(new_ch_names)
    
    if any(['EEG' in item for item in raw.ch_names]):
        raw.set_channel_types(temp_ch_type_mapping)
        eog_chs = ["EEG 061", "EEG 063"]
        ecg_chs = ["EEG 063"]
    else:
        eog_chs = ["EOG 061", "EOG 062"]
        ecg_chs = ["ECG 064"]
    print([item for item in raw.ch_names if ('EEG' in item) or ('EOG' in item) or ('ECG' in item)])
    events = mne.find_events(raw,'STI 101',min_duration = 0.01)
    
    decode_events = events[0::3]
    respond_events = events[1::3]
    visible_events = events[2::3]
    
    recode_events = decode_events.copy()
    recode_events[:,-1] = decode_events[:,-1] * 100 + respond_events[:,-1] * 10 + visible_events[:,-1]
    
    event_ids = {}
    decode_ids = {'living':2,'nonliving':3}
    respond_ids = {'living':4,'nonliving':5}
    visible_ids = {'unconscious':6,
                   'glimpse':7,
                   'conscious':8,}
    unique_event_id = {f'{a_key} {b_key} {c_key}':100*a+10*b+c for a_key,a in decode_ids.items() for b_key,b in respond_ids.items() for c_key,c in visible_ids.items()}
    unique_recode_events = np.unique(recode_events[:,-1])
    unique_event_id = {a:b for a,b in unique_event_id.items() if (b in unique_recode_events)}
    print()
    hpi_meas = raw.info['hpi_meas'][0]
    coil_freq = [hpi_meas['hpi_coils'][ii]['coil_freq'][0] for ii in range(5)]
    
    # everytime before filtering, explicitly pick the type of channels you want
    # to perform the filters
    picks = mne.pick_types(raw.info,
                           meg = True,  # No MEG
                           eeg = False, # NO EEG
                           eog = True,  # YES EOG
                           ecg = True,  # YES ECG
                           )
    # regardless the bandpass filtering later, we should always filter
    # for wire artifacts and their oscillations
    notch_filter = list(np.concatenate([np.arange(50,241,50),
                                        coil_freq]).astype('int'))
    raw.notch_filter(notch_filter,picks = picks)
    print()
    # epoch the data
    picks = mne.pick_types(raw.info,
                           meg = True,
                           eog = True,
                           ecg = True,
                           )
    epochs      = mne.Epochs(raw,
                             recode_events,    # numpy array
                             unique_event_id,  # dictionary
                             tmin        = -.7,
                             tmax        = 1.,
                             baseline    = (-.7,-.2), # range of time for computing the mean references for each channel and subtract these values from all the time points per channel
                             picks       = picks,
                             detrend     = 1, # detrend
                             preload     = True # must be true if we want to do further processing
                             )
    # go through gradiometers
    picks = mne.pick_types(epochs.info, meg='grad', eeg=False,
                       stim=False, eog=False,
                       include=[], exclude=[])
    ransac = Ransac(verbose = 'tqdm', 
                    min_corr = 0.5,
                    picks = picks, 
                    n_jobs = 8,
                    random_state = 12345)
    epochs_clean = ransac.fit_transform(epochs)
    print(session,'\n','\n'.join(ransac.bad_chs_))
    raw.info['bads'] += ransac.bad_chs_
    with open(os.path.join(working_dir,'bad_grad_report.txt'),'a') as f:
        if len(ransac.bad_chs_) < 1:
            f.write(f'session {session}, no bad channels\n')
        else:
            for ch_name in ransac.bad_chs_:
                f.write(f'session {session},{ch_name}\n')
        f.close()
    # go through MAG
    picks = mne.pick_types(epochs.info, meg='mag', eeg=False,
                       stim=False, eog=False,
                       include=[], exclude=[])
    ransac = Ransac(verbose = 'tqdm', 
                    min_corr = 0.5,
                    picks = picks, 
                    n_jobs = 8,
                    random_state = 12345)
    epochs_clean = ransac.fit_transform(epochs)
    print(session,'\n'.join(ransac.bad_chs_))
    raw.info['bads'] += ransac.bad_chs_
    with open(os.path.join(working_dir,'bad_mag_report.txt'),'a') as f:
        for ch_name in ransac.bad_chs_:
            f.write(f'session {session},{ch_name}\n')
        f.close()
    
