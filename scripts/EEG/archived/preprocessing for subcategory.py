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
working_dir = '../data/EEG/'
working_data = glob(os.path.join(working_dir,'*.vhdr'))
behaviroal_dir = '../data/behavioral'
behavioral_data = glob(os.path.join(behaviroal_dir,'*trials.csv'))
subcategory = {'Animals':0,'Birds':1,'Marine_creatures':2,'Buildings':4,
               'Clothing':5,'Furniture':6,'Kitchen_Uten':7,'Vehicles':8}


epochs_concat = []
for eeg_file,behav_file in zip(np.sort(working_data),behavioral_data):
    # load the data with vhdr, aka header file
    raw = mne.io.read_raw_brainvision(eeg_file,preload=True,stim_channel=True)
    # if we work with only 32 channels, we need to rename them in a specific order
    channel_names = '''Fp1 Fz F3 F7 FT9 FC5 FC1 C3 T7 TP9 CP5 CP1 Pz P3 P7 O1 Oz O2 P4 P8 TP10 CP6 CP2 Cz C4 T8 FT10 FC6 FC2 F4 F8 Fp2'''
    channel_names = channel_names.split(' ')
    channel_names.insert(33,'STI 014')
    channel_map = {a:b for a,b in zip(raw.ch_names,channel_names)}
    # change the existed channel names
    raw.rename_channels(channel_map)
    # read standard montage - montage is important for visualization
    montage = mne.channels.read_montage('standard_1020',ch_names=raw.ch_names);montage.plot()
    raw.set_montage(montage)
    # set the EOG channels
    channel_types = {'FT9':'eog','FT10':'eog'}
    raw.set_channel_types(channel_types)
    # get events
    events = mne.find_events(raw,stim_channel='STI 014')
    events = events[events[:,-1] < 3]
    
    df = pd.read_csv(behav_file).dropna()
    df = df.sort_values(['order']).reset_index()
    if df.shape[0] != events.shape[0]:
        df = df.iloc[1:,:].reset_index()
    
    events[:,-1] = df['subcategory'].map(subcategory)
    
    
    epochs = preprocessing_conscious(raw,events,
                                     n_interpolates = np.arange(1,32,4),
                                     consensus_pers = np.linspace(0,1.0,11),
                                     event_id = subcategory,
                                     tmin = -0.25,
                                     tmax = 0.15 + 0.25 * 5,
                                     high_pass = 0.1,
                                     low_pass = 50)
    
    epochs_concat.append(epochs)

epochs = mne.concatenate_epochs(epochs_concat)
for key,value in subcategory.items():
    evoked = epochs[key].average()
    evoked.plot_joint(title=key)

epochs.save('../data/clean EEG/clean-multi-epo.fif')





























