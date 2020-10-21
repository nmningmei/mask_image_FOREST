#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 09:11:23 2020

@author: nmei
"""

import os
import re
import numpy as np
from glob import glob
template = 'ROI_based_RSA.py'
output_dir = '../ROI_RSA_bash'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    

nodes               = 1
cores               = 16
mem                 = int(3 * nodes * cores)
time_               = 24 * nodes * cores


if not os.path.exists(f'{output_dir}/bash'):
    os.mkdir(f'{output_dir}/bash')

    
from shutil                    import copyfile
copyfile('../../../utils.py',f'{output_dir}/utils.py')

for ii in range(7):
    target_file = 'RSA_{}.py'.format(ii+1)
    with open(os.path.join(output_dir,target_file),'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "sub                     = " in line:
                    line = line.replace('sub-01',f'sub-0{ii+1}')
                new_file.write(line)
            old_file.close()
        new_file.close()

    content = f"""
#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={nodes}:ppn={cores}
#pBS -l ncpus={cores * nodes}
#PBS -l mem={mem}gb
#PBS -l cput={time_}:00:00
#PBS -N S{ii+1}-RSA
#PBS -o bash/out_sub{ii+1}.txt
#PBS -e bash/err_sub{ii+1}.txt
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd
python {target_file}
"""
    print(content)
    bash_file_name = f'RSA_{ii+1}_q'
    with open(f'{output_dir}/{bash_file_name}','w') as f:
        f.write(content)
time = .1
content = '''
import os
import time
'''
with open(f'{output_dir}/qsub_jobs_RAS_LOO.py','w') as f:
    f.write(content)
    f.close()

count = 0
with open(f'{output_dir}/qsub_jobs_RAS_LOO.py','a') as f:
    for ii in range(7):
        bash_file_name = f'RSA_{ii+1}_q'
        if count == 0:
            f.write('\nos.system("qsub {}")\n'.format(bash_file_name))
        else:
            f.write('time.sleep({})\nos.system("qsub {}")\n'.format(time,bash_file_name))
        count += 1
    f.close()