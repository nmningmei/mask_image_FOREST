#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:30:43 2019

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


subject = 'matie_5_23_2019'
folder_name = "clean_EEG_premask_baseline_ICA"

frames,temp = get_frames(directory = f'../../data/behavioral/{subject}',new = True)
with open(f'../../results/EEG/report/{subject}.txt','w') as f:
    f.write(temp)
    f.close()


working_dir = f'../../data/EEG/{subject}'
behavioral_dir = f'../../data/behavioral/{subject}'

log_dir = f'../../figures/EEG/preprocessing logs/{subject}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(f'../../data/{folder_name}/{subject}'):
    os.makedirs(f'../../data/{folder_name}/{subject}')
if not os.path.exists(f'../../data/clean epochs/{subject}'):
    os.makedirs(f'../../data/clean epochs/{subject}')
if not os.path.exists(f'../../data/clean behavioral/{subject}'):
    os.makedirs(f'../../data/clean behavioral/{subject}')


working_data = glob(os.path.join(working_dir,'*.vhdr'))
working_data_sessions = [int(re.findall(r'\d+', item)[-1]) for item in working_data]
idx_sort = np.argsort(working_data_sessions)
working_data = np.array(working_data)[idx_sort]
behavioral_data = np.sort(glob(os.path.join(behavioral_dir,'*trials.csv')))
log_data = np.sort([item.replace('trials.csv','.log') for item in behavioral_data])


visible_map = {"'1'":6,"'2'":7,"'3'":8,}


for idx,(behaviroal_file,eeg_file,log_file) in enumerate(zip(
                behavioral_data,
                working_data,
                log_data)):
    print(behaviroal_file,eeg_file,log_file)
    print()
    
    temp = dict(time = [],
                desp = [],)
    with open(log_file,'r') as logs:
        pre_mask_count = 1
        for line in logs:
            if ("premask_1" in line) and\
            ("premask_1: autoDraw = False" not in line) and\
            ("premask_1: phase" not in line) and\
            ("premask_1: autoDraw = True" not in line) and\
            ("Created" not in line):
                present_time = re.findall(r'\d+.\d+',line)
                temp['time'].append(float(present_time[0]))
                temp['desp'].append(f'premask_{pre_mask_count}')
                pre_mask_count += 1
            elif "probe: autoDraw = True" in line:
                present_time = re.findall(r'\d+.\d+',line)
                temp['time'].append(float(present_time[0]))
                temp['desp'].append(f'probe')
                pre_mask_count = 1
            elif ("trigger 3" in line) or ("trigger 2" in line):
                persent_time = re.findall(r'\d+.\d+',line)
                temp['time'].append(float(present_time[0]))
                temp['desp'].append(f'trigger')
                pre_mask_count = 1
    logs.close()
    temp = pd.DataFrame(temp)
    temp['pick'] = temp['desp'].apply(lambda x: ("probe" in x) or ("_20" in x) or ("trigger" in x))
    temp = temp[temp['pick'] == True]
    last_mask = temp[temp['desp'] == "premask_20"]
    probe = temp[temp['desp'] == 'probe']
    trigger = temp[temp['desp'] == 'trigger']
    time_diff = - (last_mask['time'].values - probe['time'].values)
    
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
        time_diff = time_diff[1:]
    elif eeg_file == '../../data/EEG/ana_5_21_2019/Ana_session8.vhdr':
        recode_events[:,-1] = decode_events[:,-1] * 100 + respond_events[1:,-1] * 10 + visible_events[1:,-1]
        df_temp = df_temp.iloc[1:,:]
        time_diff = time_diff[1:]
    elif eeg_file == '../../data/EEG/jesica_6_7_2019/Jesica_session10.vhdr':
        recode_events[:,-1] = decode_events[:,-1] * 100 + respond_events[1:,-1] * 10 + visible_events[2:,-1]
        df_temp = df_temp.iloc[2:,:]
        time_diff = time_diff[1:]
    elif eeg_file == '../../data/EEG/alba_6_10_2019/Alba_session10.vhdr':
        recode_events[:,-1] = decode_events[:,-1] * 100 + respond_events[1:,-1] * 10 + visible_events[2:,-1]
        df_temp = df_temp.iloc[2:,:]
        time_diff = time_diff[1:]
    else:
        recode_events[:,-1] = decode_events[:,-1] * 100 + respond_events[:,-1] * 10 + visible_events[:,-1]
    
    recode_events_adjust = recode_events.copy()
    recode_events_adjust[:,0] = recode_events_adjust[:,0] - (time_diff - 0.01) * raw.info['sfreq'] 
    

    event_ids = {}
    decode_ids = {'living':2,'nonliving':3}
    respond_ids = {'living':4,'nonliving':5}
    visible_ids = {'unconscious':6,
                   'glimpse':7,
                   'conscious':8,}
    unique_event_id = {f'{a_key} {b_key} {c_key}':100*a+10*b+c for a_key,a in decode_ids.items() for b_key,b in respond_ids.items() for c_key,c in visible_ids.items()}
    unique_recode_events = np.unique(recode_events_adjust[:,-1])
    unique_event_id = {a:b for a,b in unique_event_id.items() if (b in unique_recode_events)}
    print()
    epochs = preprocessing_conscious(raw,recode_events_adjust,idx,
                                     event_id = unique_event_id,
                                     tmin = - (20 * (1/100)),
                                     tmax = 1,
                                     baseline = (-20 * (1/100), 0),
                                     perform_ICA = True)
    epochs.filter(None,30,)
                                     
    
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
    try:
        evoked = epochs[key].average()
        fig = evoked.plot_joint(title = key)
        fig.savefig(os.path.join(log_dir,f'{key}.png'))
    except:
        pass
plt.close('all')


epochs_concat.save(f'../../data/{folder_name}/{subject}/clean-epo.fif')
df_concat.to_csv(f'../../data/clean behavioral/{subject}/concat.csv')




























