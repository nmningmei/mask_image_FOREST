#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:25:15 2019

@author: nmei
"""

import os
import re
from glob import glob
import pandas as pd
import numpy as np

sub             = 'pilot'
saving_folder   = 'pilot_maxfilterd'
working_dir     = f'../../data/MEG/{sub}'
working_data    = glob(os.path.join(working_dir,'*.fif'))
#bad_working_data = glob(os.path.join(working_dir,'60 Hz','*.fif'))
if not os.path.exists(f'../../data/MEG/{saving_folder}'):
    os.mkdir(f'../../data/MEG/{saving_folder}')
key1,key2 = '_s4.','_s15.'
day_1_file      = os.path.abspath([item for item in working_data if (key1 in item)][0])
#day_2_file      = os.path.abspath([item for item in bad_working_data if (key2 in item)][0])

#bad_grads = pd.read_csv(os.path.join(working_dir,'bad_grad_report.txt'),header = None)
#bad_mags = pd.read_csv(os.path.join(working_dir,'bad_mag_report.txt'),header = None)

command         = """nice /neuro/bin/util/maxfilter
-v
-f {in_file}
-o {out_file}
-origin 0.0 0.0 40.0
-frame head
-autobad off
-ctc /neuro/databases/ctc/ct_sparse.fif
-cal /neuro/databases/sss/sss_cal_3049_UPR_180927.dat
-st
-trans {head_file}
-movecomp
-hp {head_ps}
-linefreq 50""".replace('\n',' ')
# -bad {bad_list_in_str}
for file_name in [day_1_file]:#,day_2_file]:
    session = int(re.findall(r'\d+',file_name)[0])
#    bad_grads_row = bad_grads[bad_grads[0] == f'session {session}']
#    bad_mags_row = bad_mags[bad_mags[0] == f'session {session}']
    
#    idx_grad = [item.split(' ')[-1] for item in bad_grads_row[1].values]
#    idx_mags = [item.split(' ')[-1] for item in bad_mags_row[1].values]
#    idxs = np.concatenate([idx_grad,idx_mags])
#    idxs = [str(int(item)) for item in idxs]
#    bad_list_in_str = ' '.join(idxs)
    
    
    command_execute = command.format(in_file    = os.path.abspath(file_name),
                                     out_file   = os.path.abspath(file_name.replace('.fif','_tsss_mc-raw.fif').replace(sub,saving_folder)),
                                     head_file  = os.path.abspath(file_name),
                                     head_ps    = os.path.abspath(file_name.replace('.fif','_head.pos').replace(sub,saving_folder)),
#                                     bad_list_in_str = bad_list_in_str,
                                     )
    print(file_name)
    print(command_execute)
    print()
    os.system(command_execute + ' >> {}'.format(file_name.replace('fif','log').replace(sub,saving_folder))
                )

for file_name in working_data:
    if (key1 not in file_name) and (key2 not in file_name):
        align_to = [int(re.findall(r'\d+',item)[0]) for item in [day_1_file]]#,day_2_file]]
        
        session = int(re.findall(r'\d+',file_name)[-1])
        
#        if session < align_to[-1]:
        align_to_pick = day_1_file
#        else:
#            align_to_pick = day_2_file
        
#        bad_grads_row = bad_grads[bad_grads[0] == f'session {session}']
#        bad_mags_row = bad_mags[bad_mags[0] == f'session {session}']
#        
#        idx_grad = [item.split(' ')[-1] for item in bad_grads_row[1].values]
#        idx_mags = [item.split(' ')[-1] for item in bad_mags_row[1].values]
#        idxs = np.concatenate([idx_grad,idx_mags])
#        idxs = [str(int(item)) for item in idxs]
#        bad_list_in_str = ' '.join(idxs)
        
        command_execute = command.format(in_file    = os.path.abspath(file_name),
                                         out_file   = os.path.abspath(file_name.replace('.fif','_tsss_mc-raw.fif').replace(sub,saving_folder)),
                                         head_file  = os.path.abspath(align_to_pick),
                                         head_ps    = os.path.abspath(file_name.replace('.fif','_head.pos').replace(sub,saving_folder)),
#                                         bad_list_in_str = bad_list_in_str,
                                         )
        print(file_name)
        print(command_execute)
        print()
        os.system(command_execute + ' >> {}'.format(file_name.replace('fif','log').replace(sub,saving_folder))
                )
