#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:47:49 2019

@author: nmei
"""

import mne
import os
import numpy as np
import pandas as pd
from utils import preprocessing_conscious
from glob import glob
subject = 'marta_3_13_2019'
working_dir = f'../data/EEG/{subject}'
behavioral_dir = f'../data/behavioral/{subject}'
working_data = glob(os.path.join(working_dir,'*.vhdr'))
behavioral_data = glob(os.path.join(behavioral_dir,'*trials.csv'))
visible_map = {"'1'":5,"'2'":6,"'3'":7,"'4'":8}
response_map = {"'1'":3,"'2'":4}
probe_map = {'Living_Things':1,'Nonliving_Things':2}

epochs_concat = []
for idx,eeg_file in enumerate(working_data):
    # load the data with vhdr, aka header file
    raw = mne.io.read_raw_brainvision(eeg_file,preload=True,stim_channel=True)
    # if we work with only 32 channels, we need to rename them in a specific order
#    channel_names = '''Fp1 Fz F3 F7 FT9 FC5 FC1 C3 T7 TP9 CP5 CP1 Pz P3 P7 O1 Oz O2 P4 P8 TP10 CP6 CP2 Cz C4 T8 FT10 FC6 FC2 F4 F8 Fp2 AF7 AF3 AFz F1 F5 FT7 FC3 C1 C5 TP7 CP3 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2 CPz CP4 TP8 C6 C2 FC4 FT8 F6 AF8 AF4 F2 Iz'''
#    channel_names = channel_names.split(' ')
#    channel_names.insert(65,'STI 014')
#    channel_map = {a:b for a,b in zip(raw.ch_names,channel_names)}
#    # change the existed channel names
#    raw.rename_channels(channel_map)
    # read standard montage - montage is important for visualization
    montage = mne.channels.read_montage('standard_1020',ch_names=raw.ch_names)
    raw.set_montage(montage)
    # set the EOG channels
    channel_types = {'FT9':'eog','FT10':'eog','TP9':'eog','TP10':'eog'}
    raw.set_channel_types(channel_types)
    # get events
    events = mne.find_events(raw,stim_channel='STI 014')
    decode_events = events[events[:,-1] < 3]
    try:
        df_temp = pd.read_csv(behavioral_data[idx]).dropna()
        df_temp['response.keys_raw'] = df_temp['response.keys_raw'].map(response_map)
        df_temp['visible.keys_raw'] = df_temp['visible.keys_raw'].map(visible_map)
        df_temp['probe_trigger'] = df_temp['category'].map(probe_map)
        respond_triggers = df_temp['response.keys_raw'].values
        visible_triggers = df_temp['visible.keys_raw'].values
        
        respond_events = decode_events.copy()
        respond_events[:,-1] = respond_triggers
        visible_events = decode_events.copy()
        visible_events[:,-1] = visible_triggers
        
        recode_events = decode_events.copy()
        recode_events[:,-1] = decode_events[:,-1] * 10 + visible_events[:,-1]
    except:
        df_temp = pd.read_csv(behavioral_data[idx]).dropna().iloc[1:,:]
        df_temp['response.keys_raw'] = df_temp['response.keys_raw'].map(response_map)
        df_temp['visible.keys_raw'] = df_temp['visible.keys_raw'].map(visible_map)
        df_temp['probe_trigger'] = df_temp['category'].map(probe_map)
        respond_triggers = df_temp['response.keys_raw'].values
        visible_triggers = df_temp['visible.keys_raw'].values
        
        respond_events = decode_events.copy()
        respond_events[:,-1] = respond_triggers
        visible_events = decode_events.copy()
        visible_events[:,-1] = visible_triggers
        
        recode_events = decode_events.copy()
        recode_events[:,-1] = decode_events[:,-1] * 10 + visible_events[:,-1]
    

    event_ids = {}
    decode_ids = {'living':1,'nonliving':2}
    respond_ids = {'living':3,'nonliving':4}
    visible_ids = {'unconscious':5,
                   'see_unknown':6,
                   'see_maybe':7,
                   'conscious':8,}
    unique_event_id = {f'{a_key} {b_key}':10*a+b for a_key,a in decode_ids.items() for b_key,b in visible_ids.items()}
    
    epochs = preprocessing_conscious(raw,recode_events,
                                     event_id = unique_event_id,
                                     n_interpolates = np.arange(1,32,4),
                                     consensus_pers = np.linspace(0,1.0,11),
                                     tmin = -0.1 * 2,
                                     tmax = 0.1 + 0.1 * 2 + 1.,
                                     high_pass = 0.1,
                                     low_pass = 50)
    
    epochs_concat.append(epochs)

epochs = mne.concatenate_epochs(epochs_concat)

for key in epochs.event_id.keys():
    evoked = epochs[key].average()
    evoked.plot_joint(title = key)

if not os.path.exists(f'../data/clean EEG/{subject}'):
    os.mkdir(f'../data/clean EEG/{subject}')
epochs.save(f'../data/clean EEG/{subject}/clean-epo.fif')





























