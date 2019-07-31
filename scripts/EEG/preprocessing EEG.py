#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:47:49 2019

@author: nmei
"""

import mne
import os
import numpy as np
#import pandas as pd
from shutil import copyfile
copyfile('../utils.py','utils.py')
from utils import preprocessing_conscious,get_frames
from glob import glob
from matplotlib import pyplot as plt


subject = 'inaki_5_9_2019'

working_dir = f'../../data/EEG/{subject}'
log_dir = f'../../figures/EEG/preprocessing logs/{subject}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(f'../../data/clean EEG/{subject}'):
    os.makedirs(f'../../data/clean EEG/{subject}')
if not os.path.exists(f'../../data/clean epochs/{subject}'):
    os.makedirs(f'../../data/clean epochs/{subject}')
behavioral_dir = f'../../data/behavioral/{subject}'
working_data = glob(os.path.join(working_dir,'*.vhdr'))
behavioral_data = glob(os.path.join(behavioral_dir,'*trials.csv'))
frames,temp = get_frames(directory = f'../../data/behavioral/{subject}',new = True)
with open(f'../../results/EEG/report/{subject}.txt','w') as f:
    f.write(temp)
    f.close()
visible_map = {"'1'":6,"'2'":7,"'3'":8,}


for idx,eeg_file in enumerate(working_data):
    # load the data with vhdr, aka header file
    raw = mne.io.read_raw_brainvision(eeg_file,preload=True,stim_channel=True)
    raw.info['subject'] = subject
    # read standard montage - montage is important for visualization
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
    if eeg_file == '../../../data/EEG/inaki_5_9_2019/iaki_session5.vhdr':
        recode_events[:,-1] = decode_events[:,-1] * 100 + respond_events[1:,-1] * 10 + visible_events[1:,-1]
    elif eeg_file == '../../../data/EEG/ana_5_21_2019/Ana_session8.vhdr':
        recode_events[:,-1] = decode_events[:,-1] * 100 + respond_events[1:,-1] * 10 + visible_events[1:,-1]
    elif eeg_file == '../../../data/EEG/jesica_6_7_2019/Jesica_session10.vhdr':
        recode_events[:,-1] = decode_events[:,-1] * 100 + respond_events[1:,-1] * 10 + visible_events[2:,-1]
    else:
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
    
    epochs = preprocessing_conscious(raw,recode_events,idx,
                                     event_id = unique_event_id,
                                     n_interpolates = np.arange(1,30,4),
                                     consensus_pers = np.linspace(0,1.0,11),
                                     tmin = -0.5,
                                     tmax = 1,#0.05 + 0.1 * 2 + 1.,
                                     high_pass = 0.001,
                                     low_pass = 30,
                                     logging = os.path.join(log_dir,f'{idx}.png'),
                                     )
    epochs.save(f'../../data/clean epochs/{subject}/{idx}-epo.fif')

epochs_concat = []
for ep in glob(f'../../data/clean epochs/{subject}/*-epo.fif'):
    epochs_concat.append(mne.read_epochs(ep,))

epochs_concat = mne.concatenate_epochs(epochs_concat)
print('finished cleaning, not plotting')

for key in epochs.event_id.keys():
    evoked = epochs[key].average()
    evoked.plot_joint(title = key)
plt.close('all')


epochs_concat.save(f'../../data/clean EEG/{subject}/clean-epo.fif')






























