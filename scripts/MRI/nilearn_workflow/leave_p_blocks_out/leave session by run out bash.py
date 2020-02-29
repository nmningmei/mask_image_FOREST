#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:47:50 2019

@author: nmei
"""

import os
import numpy as np
from glob import glob
key_word = 'LSRO'
template = 'leave session by run out.py'
output_dir = f'../{key_word}_bash'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

sub                 = 'sub-01'
node                = 1
cores               = 16
mem                 = 4 * cores * node
time_               = 54 * cores * node
stacked_data_dir    = '../../../../data/BOLD_average/{}/'.format(sub)
BOLD_data           = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))

for ii,BOLD_data_file in enumerate(BOLD_data):
    with open(os.path.join(output_dir,f'{key_word}_{ii+1}.py'),'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "../../../data/" in line:
                    line = line.replace("../../../data/",
                                        "../../../../data/")
                elif "sub                 = 'sub-" in line:
                    line = f"sub                 = '{sub}'\n"
                elif "os.chdir('..')" in line:
                    line = 'print(os.getcwd())\n'
                elif "../../../results/" in line:
                    line = line.replace("../../../results/",
                                        "../../../../results/")
                elif "idx = 0" in line:
                    line = line.replace("idx = 0",
                                        "idx = {}".format(ii))
                elif "../../utils.py" in line:
                    line = line.replace("../../utils.py",
                                        "../../../utils.py")
                new_file.write(line)
            old_file.close()
        new_file.close()
if not os.path.exists(f'{output_dir}/bash'):
    os.mkdir(f'{output_dir}/bash')
else:
    [os.remove(f'{output_dir}/bash/'+f) for f in os.listdir(f'{output_dir}/bash') if (f"S{sub[-1]}" in f)]



for ii, BOLD_data_file in enumerate(BOLD_data):
    mask_name = BOLD_data_file.split('/')[-1].replace('_BOLD.npy','').replace('ctx-','')
    content = f"""
#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={node}:ppn={cores}
#PBS -l mem={mem}gb
#PBS -l cput={time_}:00:00
#PBS -N S{sub[-1]}{key_word}{ii+1}
#PBS -o bash/out_{sub[-1]}{ii+1}.txt
#PBS -e bash/err_{sub[-1]}{ii+1}.txt
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:$PATH"
pwd
echo "{mask_name}"

python "{key_word}_{ii + 1}.py"
"""
    print(content)
    with open(f'{output_dir}/decode_{ii+1}_q','w') as f:
        f.write(content)


time = 1
content = '''
import os
import time
'''
with open(f'{output_dir}/qsub_jobs_decode_{key_word}.py','w') as f:
    f.write(content)
    f.close()

counter = 0
with open(f'{output_dir}/qsub_jobs_decode_{key_word}.py','a') as f:
    for ii, BOLD_data_file in enumerate(BOLD_data):
        if counter == 0:
            f.write(f'\nos.system("qsub decode_{ii+1}_q")\n')
        else:
            f.write(f'time.sleep({time})\nos.system("qsub decode_{ii+1}_q")\n')
        counter += 1
    f.close()

"""
#$ -cwd
#$ -o bash/out_{sub[-1]}{ii+1}.txt
#$ -e bash/err_{sub[-1]}{ii+1}.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "S{sub[-1]}R{ii+1}"
#$ -S /bin/bash

module load rocks-python-3.6

#$ -cwd
#$ -o bash/out_q.txt
#$ -e bash/err_q.txt
#$ -m be
#$ -M nmei@bcbl.eu
#$ -N "qsubjobs"
#$ -S /bin/bash

module load rocks-python-3.6
"""







