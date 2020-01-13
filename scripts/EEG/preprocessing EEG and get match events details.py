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

label_map = {'Living_Things':1,'Nonliving_Things':2}
all_subjects = [
                'aingere_5_16_2019',
                'alba_6_10_2019',
                'alvaro_5_16_2019',
                'clara_5_22_2019',
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
    log_dir = f'../../figures/EEG/preprocessing logs/{subject}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    folder_name = "clean_EEG_mask_baseline_with_session_info"
    if not os.path.exists(f'../../data/{folder_name}/{subject}'):
        os.makedirs(f'../../data/{folder_name}/{subject}')
    if not os.path.exists(f'../../data/clean epochs/{subject}'):
        os.makedirs(f'../../data/clean epochs/{subject}')
    if not os.path.exists(f'../../data/clean behavioral/{subject}'):
        os.makedirs(f'../../data/clean behavioral/{subject}')
    behavioral_dir = f'../../data/behavioral/{subject}'
    working_data = np.sort(glob(os.path.join(working_dir,'*.vhdr')))
    behavioral_data = np.sort(glob(os.path.join(behavioral_dir,'*trials.csv')))
    frames,temp = get_frames(directory = f'../../data/behavioral/{subject}',new = True)
    with open(f'../../results/EEG/report/{subject}.txt','w') as f:
        f.write(temp)
        f.close()
    visible_map = {"'1'":6,"'2'":7,"'3'":8,}
    df_behavior = {}
    for f in behavioral_data:
        df_temp = pd.read_csv(f).iloc[96:,:2]
        session = int(df_temp[df_temp['category'] == 'session']['probe_path'].values[0])
        print(f,session)
        df_behavior[session] = preprocess_behavioral_file(f).dropna()
    for idx,(behaviroal_file,eeg_file) in enumerate(zip(
                    behavioral_data,
                    working_data)):
        behaviroal_file,eeg_file
        session = int(re.findall(r'\d+',eeg_file)[-1])
        df_temp = df_behavior[session]
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
            subtract = 96 - recode_events.shape[0]
            df_temp = df_temp.iloc[subtract:,:]
        elif eeg_file == '../../data/EEG/ana_5_21_2019/Ana_session8.vhdr':
            recode_events[:,-1] = decode_events[:,-1] * 100 + respond_events[1:,-1] * 10 + visible_events[1:,-1]
            subtract = 96 - recode_events.shape[0]
            df_temp = df_temp.iloc[subtract:,:]
        elif eeg_file == '../../data/EEG/jesica_6_7_2019/Jesica_session10.vhdr':
            recode_events[:,-1] = decode_events[:,-1] * 100 + respond_events[1:,-1] * 10 + visible_events[2:,-1]
            subtract = 96 - recode_events.shape[0]
            df_temp = df_temp.iloc[subtract:,:]
        elif eeg_file == '../../data/EEG/alba_6_10_2019/Alba_session10.vhdr':
            recode_events[:,-1] = decode_events[:,-1] * 100 + respond_events[1:,-1] * 10 + visible_events[2:,-1]
            subtract = 96 - recode_events.shape[0]
            df_temp = df_temp.iloc[subtract:,:]
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
        #################################################################################################
        #################################################################################################
        #################################################################################################

        epochs = preprocessing_conscious(raw,recode_events,idx,
                                         event_id = unique_event_id,
                                         tmin = -.2,
                                         tmax = 1,
                                         baseline = (-.2,0.),
                                         lowpass = 40,
                                         perform_ICA = True,)
        epochs.events[:,1] = [idx] * epochs.events.shape[0]
        epochs.filter(None,30,)
        matches = df_temp['category'].map(label_map).values == epochs.events[:,-1] // 100 - 1
        if np.sum(matches) < df_temp.shape[0]:
            asdf
        #################################################################################################
        #################################################################################################
        #################################################################################################
        epochs.save(f'../../data/clean epochs/{subject}/{idx}-epo.fif')
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
        try:
            evoked = epochs[key].average()
            fig = evoked.plot_joint(title = key)
            fig.savefig(os.path.join(log_dir,f'{key}.png'))
        except:
            pass
    plt.close('all')
    
    
    epochs_concat.save(f'../../data/{folder_name}/{subject}/clean-epo.fif')
    df_concat.to_csv(f'../../data/clean behavioral/{subject}/concat.csv')






























