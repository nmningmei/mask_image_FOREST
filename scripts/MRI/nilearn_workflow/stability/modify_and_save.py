#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 16:24:03 2020

@author: nmei
"""

import os
from glob import glob
import numpy as np

sub                 = 'sub-01'
nodes               = 1
cores               = 16
mem                 = 4 * cores * nodes
time_               = 72 * cores * nodes
stacked_data_dir    = '../../../../data/BOLD_average/{}/'.format(sub)
BOLD_data           = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))
template            = 'stability across runs.py'
output_dir          = '../stability_bash'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for idx,BOLD_file in enumerate(BOLD_data):
    for conscious_state in ['unconscious','glimpse','conscious']:
        roi_name = BOLD_file.split('/')[-1].split('_BOLD')[0]
        with open(os.path.join(output_dir,'STP_{}_{}.py'.format(conscious_state,idx+1)),'w') as new_file:
            with open(template,'r') as old_file:
                for line in old_file:
                    if "sub                 = 'sub-" in line:
                        line = f"sub                 = '{sub}'\n"
                    elif "idx = 0" in line:
                        line = line.replace("idx = 0",
                                            "idx = {}".format(idx))
                    elif "conscious_state     = " in line:
                        line = line.replace('unconscious',conscious_state)
                    new_file.write(line)
                old_file.close()
            new_file.close()

if not os.path.exists(f'{output_dir}/bash'):
    os.mkdir(f'{output_dir}/bash')
else:
    [os.remove(f'{output_dir}/bash/'+f) for f in os.listdir(f'{output_dir}/bash') if (f"S{sub[-1]}" in f)]

for ii,BOLD_file in enumerate(BOLD_data):
    for conscious_state in ['unconscious','glimpse','conscious']:
        roi_name = BOLD_file.split('/')[-1].split('_BOLD')[0]
        content = f"""
#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={nodes}:ppn={cores}
#PBS -l mem={mem}gb
#PBS -l cput={time_}:00:00
#PBS -N S{sub[-1]}SPT{ii+1}{conscious_state}
#PBS -o bash/out_{sub[-1]}{conscious_state}{ii+1}.txt
#PBS -e bash/err_{sub[-1]}{conscious_state}{ii+1}.txt
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:$PATH"
pwd
echo "{roi_name} {conscious_state}"

python "STP_{conscious_state}_{ii + 1}.py"
"""
        print(content)
        with open(f'{output_dir}/RSA_{conscious_state}_{ii+1}_q','w') as f:
            f.write(content)

time = 1
content = '''
import os
import time
'''
with open(f'{output_dir}/qsub_jobs_RSA.py','w') as f:
    f.write(content)
    f.close()

with open(f'{output_dir}/qsub_jobs_RSA.py','a') as f:
    for ii, BOLD_data_file in enumerate(BOLD_data):
        for conscious_state in ['unconscious','glimpse','conscious']:
            if ii == 0:
                f.write('\nos.system("qsub RSA_{}_{}_q")\n'.format(conscious_state,ii+1))
            else:
                f.write('time.sleep({})\nos.system("qsub RSA_{}_{}_q")\n'.format(time,conscious_state,ii+1))
    f.close()






















