#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:17:50 2019

@author: nmei
"""

import os
from glob import glob
import numpy as np
import pandas as pd
from shutil import copyfile
import csv
from nibabel import load as load_mri


copyfile('../utils.py','utils.py')
import utils

sub             = 'sub-01'
MRI_dir         = '../../data/MRI/{}/func/'.format(sub)
MRI_data        = glob(os.path.join(MRI_dir,'session-*',"*"))

sub_behavorial  = 'ning_4_6_2019'
behavorial_dir  = '../../data/behavioral/{}'.format(sub_behavorial)
behavorial_data = glob(os.path.join(behavorial_dir,'*trials.csv'))
behavorial_data = np.sort(behavorial_data)


visible_map = {1:'unconscious',
               2:'glimpse',
               3:'conscious',
               99:'missing data'}

preloaded_behavorial    = [[utils.preload(f),utils.read(f),f] for f in behavorial_data]
behavorial_files_order  = dict(
        file_name       = [],
        session         = [],
        block           = [],)
for preloaded_behavorial_file,_,file_name in preloaded_behavorial:
    n_session   = preloaded_behavorial_file['index'][preloaded_behavorial_file['category'] == 'session'].values
    n_block     = preloaded_behavorial_file['index'][preloaded_behavorial_file['category'] == 'block'].values
    behavorial_files_order['file_name'  ].append(file_name)
    behavorial_files_order['session'    ].append(int(n_session))
    behavorial_files_order['block'      ].append(int(n_block))
behavorial_files_order = pd.DataFrame(behavorial_files_order)
behavorial_files_order = behavorial_files_order.sort_values(['session','block'])
behavorial_files_order['MRI_session'    ] = np.repeat(np.arange(2,8),9)
behavorial_files_order['MRI_run'        ] = np.tile(np.arange(1,10),6)


for (ii,row),MRI_file in zip(behavorial_files_order.iterrows(),
                             MRI_data):
    behavorial_file     = row['file_name']
    print(MRI_file,row[['MRI_session','MRI_run']])
    n_session           = row['MRI_session']
    n_run               = row['MRI_run']
    BOLD                = load_mri(os.path.join(
                            MRI_file,
                            'FEAT.session{}.run{}.feat'.format(n_session,n_run),
                            'filtered_func_data.nii.gz'))
    df                  = utils.read(behavorial_file)
    numerical_columns   = ['probe_Frames_raw',
                           'response.keys_raw',
                           'visible.keys_raw',]
    for col_name in numerical_columns:
        df[col_name]    = df[col_name].apply(utils.extract)
    
    df                  = df.sort_values(['order'])
    col_of_interest     = ['image_onset_time_raw',
                       ]
    for col in col_of_interest:
        df[col] = df[col] - 0.85 * 10
    df['start'] = df['image_onset_time_raw'] - 0.4 - 0.5 - 0.5
    df['t1']    = df['image_onset_time_raw'] + 4
    df['t2']    = df['image_onset_time_raw'] + 7
    
    total_volumes   = BOLD.shape[-1]
    time_coor       = np.arange(0,total_volumes * 0.85,0.85)
    tsv_outputs     = {'time_coor':time_coor}
    
    trials          = np.zeros(time_coor.shape)
    visibility      = trials.copy()
    correctAns      = trials.copy()
    response        = trials.copy()
    correct         = trials.copy()
    RT_response     = trials.copy()
    RT_visibility   = trials.copy()
    targets         = np.array(['AAAAAAAAAAAAAAAAAAAA'] * time_coor.shape[0])
    subcategory     = targets.copy()
    options         = targets.copy()
    labels          = targets.copy()
    
    for ii,row in df.iterrows():
        idx                 = np.where(time_coor >= row['start'])
        trials[idx]         = row['order'] + 1
        targets[idx]        = row['category']
        subcategory[idx]    = row['subcategory']
        labels[idx]         = row['label']
        visibility[idx]     = row['visible.keys_raw']
        correctAns[idx]     = row['correctAns_raw']
        correct[idx]        = row['response.corr_raw']
        response[idx]       = row['response.keys_raw']
        options[idx]        = row['response_window_raw']
        RT_response[idx]    = row['response.rt_raw']
        RT_visibility[idx]  = row['visible.rt_raw']
    to_tsv = pd.DataFrame(dict(
            time_coor       = time_coor,
            trials          = trials,
            targets         = targets,
            subcategory     = subcategory,
            labels          = labels,
            visibility      = visibility,
            correctAns      = correctAns,
            correct         = correct,
            response        = response,
            options         = options,
            RT_response     = RT_response,
            RT_visibility   = RT_visibility,)
        )
    temp        = []
    for ii,row in to_tsv.iterrows():
        time    = row['time_coor']
        if any([np.logical_and(interval[0] < time,
                               time < interval[1]) for interval in df[['t1','t2']].values]):
            temp.append(1)
        else:
            temp.append(0)
    to_tsv['volume_interest']   = temp
    to_tsv['visibility']        = to_tsv['visibility'].map(visible_map)
    to_tsv.to_csv(os.path.join(MRI_file,
                               '{}_session_{}_run_{}.csv'.format(sub,n_session,n_run)))
    f_csv = os.path.join(MRI_file,
                         '{}_session_{}_run_{}.csv'.format(sub,n_session,n_run))
    f_tsv = os.path.join(MRI_file,
                         '{}_session_{}_run_{}.tsv'.format(sub,n_session,n_run))
    
    with open(f_csv,'r') as csvin,open(f_tsv,'w') as tsvout:
        csvin   = csv.reader(csvin)
        tsvout  = csv.writer(tsvout,delimiter='\t')
        
        for row in csvin:
            tsvout.writerow(row)


















