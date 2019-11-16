#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:11:16 2019

@author: nmei
"""
import os
import re
from glob import glob

import numpy as np
import pandas as pd


all_subjects = ['aingere_5_16_2019',
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
df = dict(sub = [],
          diff_mask_probe_mean = [],
          diff_mask_probe_std = [],
          diff_probe_trigger_mean = [],
          diff_probe_trigger_std = []
          )
for folder_name in all_subjects:
    folder_name
    csv_files = glob(os.path.join(folder_name, '*trials.csv'))
    log_files = [item.replace('trials.csv','.log') for item in csv_files]
    for f in log_files:
        temp = dict(time = [],
                    desp = [],)
        with open(f,'r') as logs:
            pre_mask_count = 1
            for line in logs:
                if ("premask_1" in line) and\
                ("premask_1: autoDraw = False" not in line) and\
                ("premask_1: phase" not in line) and\
                ("premask_1: autoDraw = True" not in line) and\
                ("Created" not in line):
                    present_time = re.findall('\d+.\d+',line)
                    temp['time'].append(float(present_time[0]))
                    temp['desp'].append(f'premask_{pre_mask_count}')
                    pre_mask_count += 1
                elif "probe: autoDraw = True" in line:
                    present_time = re.findall('\d+.\d+',line)
                    temp['time'].append(float(present_time[0]))
                    temp['desp'].append(f'probe')
                    pre_mask_count = 1
                elif ("trigger 3" in line) or ("trigger 2" in line):
                    persent_time = re.findall('\d+.\d+',line)
                    temp['time'].append(float(present_time[0]))
                    temp['desp'].append(f'trigger')
#                    print(line)
                    pre_mask_count = 1
        logs.close()
        temp = pd.DataFrame(temp)
        
        temp['pick'] = temp['desp'].apply(lambda x: ("probe" in x) or ("_20" in x) or ("trigger" in x))
        temp = temp[temp['pick'] == True]
        last_mask = temp[temp['desp'] == "premask_20"]
        probe = temp[temp['desp'] == 'probe']
        trigger = temp[temp['desp'] == 'trigger']
        time_diff = - (last_mask['time'].values - probe['time'].values)
        df['sub'].append(folder_name)
        df['diff_mask_probe_mean'].append(time_diff.mean())
        df['diff_mask_probe_std'].append(time_diff.std())
        time_diff = probe['time'].values - trigger['time'].values
        df['diff_probe_trigger_mean'].append(time_diff.mean())
        df['diff_probe_trigger_std'].append(time_diff.std())
df = pd.DataFrame(df)
