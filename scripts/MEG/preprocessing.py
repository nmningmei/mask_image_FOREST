#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:56:45 2019

@author: nmei
"""
import os
import re
import mne
from glob import glob
import numpy as np
from shutil import copyfile
copyfile('../utils.py','utils.py')
from utils import preprocessing_unconscious,get_frames



sub = 'pilot_maxfilterd'
working_dir = f'../../data/MEG/{sub}'
working_data = glob(os.path.join(working_dir,'*.fif'))

behavioral_dir = '../../data/behavioral/ning_MEG_pilot'
frames,temp = get_frames(directory = behavioral_dir,new = True)

saving_dir = f'../../data/clean_MEG_epochs/{sub}'
concate_dir = f'../../data/concat_MEG_epochs/{sub}'
for d in [saving_dir,concate_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

temp_ch_type_mapping = {"EEG 061":"eog",
                        "EEG 062":"ecg",
                        "EEG 063":"eog",}

for idx,a in enumerate(working_data):
    session = int(re.findall(r'\d+',a)[0])
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
    
    epochs = preprocessing_unconscious(raw,recode_events,session,
                                       event_id = unique_event_id,
                                       tmin = -.7,
                                       tmax = 1,
                                       baseline = (-.7,-.2),
                                       perform_ICA = True,
                                       eog_chs = eog_chs,
                                       ecg_chs = ecg_chs,)
    epochs.save(os.path.join(saving_dir,f'session{session}-epo.fif'))

epochs_concat = []
for ep in glob(f'../../data/clean_MEG_epochs/{subject}/*-epo.fif'):
    epochs_concat.append(mne.read_epochs(ep,))
epochs_concat = mne.concatenate_epochs(epochs_concat)
epochs_concat.save(f'../../data/concat_MEG_epochs/{subject}/clean-epo.fif')


for key in unique_event_id.keys():
    evoked = epochs.copy().pick_types(meg=True,eeg=False)[key].average()
    _=evoked.plot_joint(title = key)





















