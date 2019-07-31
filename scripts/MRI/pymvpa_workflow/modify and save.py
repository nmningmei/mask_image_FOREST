#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 15:37:10 2019

@author: nmei
"""

import os
from glob import glob
import numpy as np
import pandas as pd


working_dir = '../../../data/BOLD_stacked/'
working_data = np.sort(glob(os.path.join(working_dir,'*.npy')))

template = 'decoding (single subject).py'

for ii,BOLD_data_file in enumerate(working_data):
    with open(os.path.join('decoding ({}).py'.format(ii+1)),'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "idx_ = 0" in line:
                    line = line.replace("idx_ = 0","idx_ = {}".format(ii))
                new_file.write(line)
            old_file.close()
        new_file.close()

if not os.path.exists('bash'):
    os.mkdir('bash')
else:
    [os.remove('bash/'+f) for f in os.listdir('bash')]

for ii, BOLD_data_file in enumerate(working_data):
    mask_name = BOLD_data_file.split('/')[-1].split('.')[0]
    content = """
#!/bin/bash

# This is a script to send "decoding ({}).py" as a batch job.


#$ -cwd
#$ -o bash/out_{}.txt
#$ -e bash/err_{}.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "CP{}"
#$ -S /bin/bash

module load rocks-python-3.6
python "decoding ({}).py"
"""
    print(content.format(ii+1,mask_name,mask_name,ii+1,ii+1))
    with open('decode_{}_q'.format(mask_name),'w') as f:
        f.write(content.format(ii+1,ii+1,ii+1,ii+1,ii+1))

content = '''
import os
import time
'''
with open('qsub_jobs_decode.py','w') as f:
    f.write(content)
    f.close()

with open('qsub_jobs_decode.py','a') as f:
    for ii, BOLD_data_file in enumerate(working_data):
        mask_name = BOLD_data_file.split('/')[-1].split('.')[0]
        if ii == 0:
            f.write('\nos.system("qsub decode_{}_q")\n'.format(mask_name))
        else:
            f.write('time.sleep(30)\nos.system("qsub decode_{}_q")\n'.format(mask_name))
    f.close()
content = '''
#!/bin/bash

# This is a script to send qsub_jobs_decode.py as a batch job.

#$ -cwd
#$ -o bash/out_q.txt
#$ -e bash/err_q.txt
#$ -m be
#$ -M nmei@bcbl.eu
#$ -N "qsubjobs"
#$ -S /bin/bash

module load rocks-python-3.6
python "qsub_jobs_decode.py"
'''
with open('qsub_jobs_decode','w') as f:
    print(content)
    f.write(content)
    f.close()













