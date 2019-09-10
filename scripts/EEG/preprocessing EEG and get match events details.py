#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:47:49 2019

@author: nmei
"""

import mne
import os
import re
import numpy as np
import pandas as pd
from shutil import copyfile
copyfile('../utils.py','utils.py')
from utils import preprocessing_conscious,get_frames,preprocess_behavioral_file
from glob import glob
from matplotlib import pyplot as plt


subject = 'alba_6_10_2019'

working_dir = f'../../data/EEG/{subject}'
log_dir = f'../../figures/EEG/preprocessing logs/{subject}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(f'../../data/clean EEG detrend only/{subject}'):
    os.makedirs(f'../../data/clean EEG detrend only/{subject}')
if not os.path.exists(f'../../data/clean epochs/{subject}'):
    os.makedirs(f'../../data/clean epochs/{subject}')
if not os.path.exists(f'../../data/clean behavioral/{subject}'):
    os.makedirs(f'../../data/clean behavioral/{subject}')
behavioral_dir = f'../../data/behavioral/{subject}'
working_data = glob(os.path.join(working_dir,'*.vhdr'))
behavioral_data = glob(os.path.join(behavioral_dir,'*trials.csv'))
frames,temp = get_frames(directory = f'../../data/behavioral/{subject}',new = True)
with open(f'../../results/EEG/report/{subject}.txt','w') as f:
    f.write(temp)
    f.close()
visible_map = {"'1'":6,"'2'":7,"'3'":8,}


for idx,(behaviroal_file,eeg_file) in enumerate(zip(
                np.sort(behavioral_data),
                working_data)):
    behaviroal_file,eeg_file
    df_temp = preprocess_behavioral_file(behaviroal_file)
    session = re.findall(r'\d',eeg_file)[-1]
    # load the data with vhdr, aka header file
    raw = mne.io.read_raw_brainvision(eeg_file,preload=True,stim_channel=True)
    raw.info['subject'] = subject
    # read standard montage - montage is important for visualization
    montage = mne.channels.read_montage('standard_1020',ch_names=raw.ch_names);#montage.plot()
    raw.set_montage(montage)
    # set the EOG channels
    channel_types = {'FT9':'eog','FT10':'eog','TP9':'eog','TP10':'eog'}
    raw.set_channel_types(channel_types)
    # get events
    events = mne.find_events(raw,stim_channel='STI 014')
    decode_events = events[events[:,-1] < 4] # 2 and 3
    respond_events = events[np.logical_and(3 < events[:,-1], events[:,-1] < 6)] # 4 and 5
    visible_events = events[5 < events[:,-1]] # 6, 7, 8
    recode_events = decode_events.copy()
    if eeg_file == '../../data/EEG/inaki_5_9_2019/iaki_session5.vhdr':
        recode_events[:,-1] = decode_events[:,-1] * 100 + respond_events[1:,-1] * 10 + visible_events[1:,-1]
        df_temp = df_temp.iloc[1:,:]
    elif eeg_file == '../../data/EEG/ana_5_21_2019/Ana_session8.vhdr':
        recode_events[:,-1] = decode_events[:,-1] * 100 + respond_events[1:,-1] * 10 + visible_events[1:,-1]
        df_temp = df_temp.iloc[1:,:]
    elif eeg_file == '../../data/EEG/jesica_6_7_2019/Jesica_session10.vhdr':
        recode_events[:,-1] = decode_events[:,-1] * 100 + respond_events[1:,-1] * 10 + visible_events[2:,-1]
        df_temp = df_temp.iloc[2:,:]
    elif eeg_file == '../../data/EEG/alba_6_10_2019/Alba_session10.vhdr':
        recode_events[:,-1] = decode_events[:,-1] * 100 + respond_events[1:,-1] * 10 + visible_events[2:,-1]
        df_temp = df_temp.iloc[2:,:]
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
    print()
    epochs = preprocessing_conscious(raw,recode_events,idx,
                                     event_id = unique_event_id,
                                     tmin = - (0.3 + (1 / 60 * 20)),
                                     tmax = 1,
                                     baseline = (round(- (0.3 + (1 / 60 * 20)),3), round(-(1 / 60 * 20),3))
                                     )
    
    epochs.save(f'../../data/clean epochs/{subject}/{idx}-epo.fif')
    print()
    df_temp.to_csv(f'../../data/clean behavioral/{subject}/{idx}-epo.csv',index=False)

epochs_concat = []
for ep in glob(f'../../data/clean epochs/{subject}/*-epo.fif'):
    epochs_concat.append(mne.read_epochs(ep,))
df_concat = []
for f in glob(f'../../data/clean behavioral/{subject}/*-epo.csv'):
    df_concat.append(pd.read_csv(f))

epochs_concat = mne.concatenate_epochs(epochs_concat)
df_concat = pd.concat(df_concat)

print('finished cleaning, not plotting')
for key in epochs_concat.event_id.keys():
    evoked = epochs[key].average()
    fig = evoked.plot_joint(title = key)
    fig.savefig(os.path.join(log_dir,f'{key}.png'))
plt.close('all')


epochs_concat.save(f'../../data/clean EEG detrend only/{subject}/clean-epo.fif')
df_concat.to_csv(f'../../data/clean behavioral/{subject}/concat.csv')






























