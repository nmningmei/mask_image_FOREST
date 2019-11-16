#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 13:10:38 2019

@author: nmei
"""
import os
import numpy as np
template = 'RNN_tf_decoding.py'

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
all_subjects = np.sort(all_subjects)
nodes               = 1
cores               = 16
mem                 = 8 * cores * nodes
time_               = 60 * cores * nodes


for ii,subject in enumerate(all_subjects):
    with open(f'RNN_tf_decoding_{ii+1}.py','w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "subject = all_subjects" in line:
                    line = f"subject = '{subject}'\n"
                elif "sample_weight            = np.array(sample_weight), # this is the key !" in line:
                    line = "                           sample_weight            = np.array(sample_weight), # this is the key !\n                           verbose = 0,\n"
                new_file.write(line)
            old_file.close()
        new_file.close()
    if not os.path.exists('bash'):
        os.mkdir('bash')
    content = f"""
#!/bin/bash
#PBS -q bcbl
#PBS -l nodes=1:ppn={cores}
#PBS -l mem={mem}gb
#PBS -l cput={time_}:00:00
#PBS -N RNN{ii+1}
#PBS -o bash/out_{subject}.txt
#PBS -e bash/err_{subject}.txt
#source .bashrc

cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-1.14.0
pwd
python RNN_tf_decoding_{ii+1}.py
    """
    with open(f'RNN_qsub_{subject}','w') as f:
        f.write(content)
        f.close()
    


























