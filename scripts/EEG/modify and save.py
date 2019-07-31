#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:19:32 2019

@author: nmei
"""
import os
from glob import glob
import numpy as np

all_subjects = ['aingere_5_16_2019',
                'alba_6_10_2019',
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

preprocessing_template = 'preprocessing EEG and get match events details.py'
TD_template = 'temporal decoding (finalized).py'
TD_resposne_template = 'temporal decoding (response).py'

bash_dir = 'bash'
if not os.path.exists(bash_dir):
    os.mkdir(bash_dir)
    if not os.path.exists(os.path.join(bash_dir,'outputs')):
        os.mkdir(os.path.join(bash_dir,'outputs'))
else:
    try:
        [os.remove(item) for item in glob(os.path.join(bash_dir,'outputs','*.txt'))]
    except:
        print('it is empty')

for ii,subject in enumerate(all_subjects):
    with open(os.path.join(bash_dir,preprocessing_template.replace('.py',f' ({subject}).py')),
              'w') as new_file:
        with open(preprocessing_template,'r') as old_file:
            for line in old_file:
                if "subject = " in line:
                    line = f"subject = '{subject}'"
                elif "'../../" in line:
#                    print(line)
                    line = line.replace("../../","../../../")
                elif "copyfile('../utils.py','utils.py')" in line:
                    line = "copyfile('../../utils.py','utils.py')\n"
                new_file.write(line)
            old_file.close()
        new_file.close()
    
    with open(os.path.join(bash_dir,TD_template.replace('.py',f' ({subject}).py')),
              'w') as new_file:
        with open(TD_template,'r') as old_file:
            for line in old_file:
                if "subject             = " in line:
                    line = f"subject = '{subject}'"
                elif "'../../" in line:
                    line = line.replace("../../","../../../")
                elif "verbose             = True" in line:
                    line = line.replace("verbose             = True","verbose             = False")
                elif "copyfile('../utils.py','utils.py')" in line:
                    line = "copyfile('../../utils.py','utils.py')\n"
                new_file.write(line)
            old_file.close()
        new_file.close()
    
    with open(os.path.join(bash_dir,TD_resposne_template.replace('.py',f' ({subject}).py')),
              'w') as new_file:
        with open(TD_resposne_template,'r') as old_file:
            for line in old_file:
                if "subject             = " in line:
                    line = f"subject = '{subject}'"
                elif "'../../" in line:
                    line = line.replace("../../","../../../")
                elif "verbose             = True" in line:
                    line = line.replace("verbose             = True","verbose             = False")
                elif "copyfile('../utils.py','utils.py')" in line:
                    line = "copyfile('../../utils.py','utils.py')\n"
                new_file.write(line)
            old_file.close()
        new_file.close()
    
    with open(os.path.join(bash_dir,f'process_{subject}'),'w') as f:
        content = f"""
#!/bin/bash
# This is a script to send 2 processing scrips as a batch job.
        
#$ -cwd
#$ -o outputs/out_{subject}.txt
#$ -e outputs/err_{subject}.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "EEG{ii}"
#$ -S /bin/bash

module load rocks-python-3.6
# python "preprocessing EEG and get match events details ({subject}).py"
python "temporal decoding (finalized) ({subject}).py"
# python "temporal decoding (response) ({subject}).py"
"""
#
        f.write(content)
        f.close()

content = '''import os
import time
'''
with open(f'{bash_dir}/qsub_jobs.py','w') as f:
    f.write(content)

with open(f'{bash_dir}/qsub_jobs.py','a') as f:
    for ii, subject in enumerate(all_subjects):
        if ii == 0:
            f.write(f'\nos.system("qsub process_{subject}")\n')
        else:
            f.write(f'time.sleep(30)\nos.system("qsub process_{subject}")\n')
    f.close()










































