#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:00:16 2020

@author: nmei
"""

import os
import gc
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy  as np

from glob                    import glob
from tqdm                    import tqdm
from sklearn.utils           import shuffle
from shutil                  import copyfile
copyfile('../../../utils.py','utils.py')
import utils
from joblib                  import Parallel, delayed
from collections             import OrderedDict,Counter
from scipy.spatial           import distance as D
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import make_pipeline
from matplotlib              import pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

def normalize(data,axis = 1):
    return data - data.mean(axis).reshape(-1,1)

sub                 = 'sub-01'
experiment          = 'stability'
RDM_folder          = 'RDM_plots'
stacked_data_dir    = '../../../../data/BOLD_average/{}/'.format(sub)
output_dir          = '../../../../results/MRI/nilearn/{}/{}'.format(sub,experiment)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
RDM_sav_dir         = '../../../../figures/MRI/nilearn/{}/{}'.format(sub,RDM_folder)
if not os.path.exists(RDM_sav_dir):
    os.mkdir(RDM_sav_dir)
BOLD_data           = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))
event_data          = np.sort(glob(os.path.join(stacked_data_dir,'*.csv')))

label_map           = {'Nonliving_Things':[0,1],
                       'Living_Things':   [1,0]}
average             = True
n_jobs              = -1
verbose             = 1
n_sampling          = [int(1e2),int(1e3),int(1e5)]

idx = 9
np.random.seed(12345)
BOLD_name,df_name   = BOLD_data[idx],event_data[idx]
BOLD                = np.load(BOLD_name)
df_event            = pd.read_csv(df_name)
roi_name            = df_name.split('/')[-1].split('_events')[0]

#results             = dict(
#                roi_name = [],
#                sub_name = [],
#                conscious_state = [],
#                RSA = [],
#                RSA_chance = [],
#                pval = [],
#                )

#for conscious_state in ['unconscious','glimpse','conscious']:
conscious_state     = 'unconscious'
if True:
    idx_unconscious = df_event['visibility'] == conscious_state
    data            = BOLD[idx_unconscious]
    df_data         = df_event[idx_unconscious].reset_index(drop=True)
    df_data         = df_data[df_data.columns[2:]]
    df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
    targets         = np.array([label_map[item] for item in df_data['targets'].values])[:,-1]
    
    def _proc(df_data,data):
        df_picked = df_data.groupby('labels').apply(lambda x: x.sample(n = 1).drop('labels',axis = 1)).reset_index()
        df_picked = df_picked.sort_values(['targets','subcategory','labels'])
        BOLD = data[df_picked['level_1']]
        BOLD = normalize(BOLD,axis = 1)
        RDM = D.pdist(BOLD,'cosine')
        return RDM
    RDMs = Parallel(n_jobs = n_jobs, verbose = verbose)(delayed(_proc)(**{
            'df_data':df_data,
            'data':data}) for _ in range(n_sampling[1]))
    RDMs = np.array(RDMs)
    RDM_of_RDMs = D.pdist(normalize(RDMs,axis = 1,),'cosine')
    gc.collect()
    
    print('plotting')
    plt.close('all')
    fig,axes = plt.subplots(figsize = (50,50),nrows = 25,ncols = 40)
    for ax,RDM in zip(axes.flatten(),RDMs):
        to_plot = D.squareform(RDM)
        np.fill_diagonal(to_plot,np.nan)
        ax.imshow(to_plot,cmap = plt.cm.coolwarm)
        ax.axvline(96/2 - .5,linestyle = '--',color = 'black',alpha = 1.)
        ax.axhline(96/2 - .5,linestyle = '--',color = 'black',alpha = 1.)
        ax.axis('off')
    fig.suptitle(f"{sub}_{conscious_state}_{roi_name}",y = 0.9,fontsize = 48)
    fig.savefig(os.path.join(RDM_sav_dir,f'{sub}_{conscious_state}_{roi_name}.jpeg'),
#                dpi = 300,
                bbox_inches = 'tight')
    plt.close('all')
    
    
    def _chance(df_data,data):
        random_signal = np.vstack([shuffle(row) for row in data])
        null = Parallel(n_jobs = 1, verbose = 0)(delayed(_proc)(**{
                'df_data':df_data,
                'data':random_signal}) for _ in range(n_sampling[0]))
        null = np.array(null)
        gc.collect()
        return D.pdist(normalize(null,axis = 1),'cosine').mean()
    null_dist = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(_chance)(**{
            'df_data':df_data,
            'data':data}) for _ in range(n_sampling[2]))
    
    pval = (np.sum(null_dist <= RDM_of_RDMs.mean()) + 1) / (len(null_dist) + 1)
    
    results = OrderedDict()
    n_rows = RDM_of_RDMs.shape[0]
    results['RDMs'] = RDM_of_RDMs
    idx_ = np.random.choice(n_sampling[2],n_sampling[0],replace = False)
    results['RMDs_chance'] = null_dist[idx_]
    results_to_save = pd.DataFrame(results)
    
    results_to_save['roi_name'] = roi_name
    results_to_save['sub_name'] = sub
    results_to_save['conscious_state'] = conscious_state
    results_to_save['pval'] = pval
    results_to_save.to_csv(os.path.join(output_dir,
                                        f'{sub}_{conscious_state}_{roi_name}.csv'),index = False)
    print(f'{roi_name},{conscious_state},stability = {RDM_of_RDMs.mean():.4f}+/-{RDM_of_RDMs.std():.4f},chance = {np.mean(null_dist):.4f}+/-{np.std(null_dist):.4f},p = {pval:.4f}')

