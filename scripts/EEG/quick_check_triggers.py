#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:39:26 2019

@author: nmei
"""

import mne
import numpy as np


raw = mne.io.read_raw_brainvision("Ander_session1.vhdr",preload=True,stim_channel=True)
montage = mne.channels.read_montage('standard_1020',ch_names=raw.ch_names);montage.plot()
raw.set_montage(montage)
# set the EOG channels
channel_types = {'FT9':'eog','FT10':'eog','TP9':'eog','TP10':'eog'}
raw.set_channel_types(channel_types)
# get events
events = mne.find_events(raw,stim_channel='STI 014')
decode_events = events[events[:,-1] < 4]
respond_events = events[np.logical_and(3 < events[:,-1], events[:,-1] < 6)]
visible_events = events[5 < events[:,-1]]
recode_events = decode_events.copy()
recode_events[:,-1] = decode_events[:,-1] * 10 + visible_events[:,-1]

print("{} images, {} responses, {} visibility responses".format(
        decode_events.shape[0],
        respond_events.shape[0],
        visible_events.shape[0]))