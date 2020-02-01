#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:54:31 2019

@author: nmei

remember to check the output_dir!!!!!!!!


"""

import os
import re
import numpy as np
from glob import glob
template = 'encoding model.py'
output_dir = '../encoding_model_bash'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
sub                 = 'sub-01'
nodes               = 2
cores               = 16
mem                 = int(3.5 * nodes * cores)
time_               = 16 * nodes * cores
stacked_data_dir    = '../../../../data/BOLD_average/{}/'.format(sub)
BOLD_data           = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))

if not os.path.exists(f'{output_dir}/bash'):
    os.mkdir(f'{output_dir}/bash')
else:
    [os.remove(f'{output_dir}/bash/'+f) for f in os.listdir(f'{output_dir}/bash') if (f"S{sub[-1]}" in f)]

for ii,BOLD_data_file in enumerate(BOLD_data):
    target_file = 'encoding_model_{}.py'.format(ii+1)
    with open(os.path.join(output_dir,target_file),'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "sub                 = 'sub-" in line:
                    line = line.replace('sub-01',f'{sub}')
                elif "idx = 0" in line:
                    line = line.replace("0","{}".format(ii))
                new_file.write(line)
            old_file.close()
        new_file.close()

    
    mask_name = BOLD_data_file.split('/')[-1].replace('_BOLD.npy','').replace('ctx-','')
    
    content = f"""
#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={nodes}:ppn={cores}
#pBS -l ncpus={cores * nodes}
#PBS -l mem={mem}gb
#PBS -l cput={time_}:00:00
#PBS -N S{sub[-1]}R{ii+1}EN
#PBS -o bash/out_{sub[-1]}{ii+1}.txt
#PBS -e bash/err_{sub[-1]}{ii+1}.txt
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:$PATH"
pwd
echo "{mask_name}"

python {target_file}

"""
    print(content)
    bash_file_name = f'encode_{ii+1}_q'
    with open(f'{output_dir}/{bash_file_name}','w') as f:
        f.write(content)
time = 3
content = '''
import os
import time
'''
with open(f'{output_dir}/qsub_jobs_encode_LOO.py','w') as f:
    f.write(content)
    f.close()

with open(f'{output_dir}/qsub_jobs_encode_LOO.py','a') as f:
    for ii, BOLD_data_file in enumerate(BOLD_data):
        bash_file_name = f'encode_{ii+1}_q'
        if ii == 0:
            f.write('\nos.system("qsub {}")\n'.format(bash_file_name))
        else:
            f.write('time.sleep({})\nos.system("qsub {}")\n'.format(time,bash_file_name))
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