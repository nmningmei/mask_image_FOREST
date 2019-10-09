#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:47:50 2019

@author: nmei
"""

import os
import numpy as np
from glob import glob
template = 'customized partition.py'
output_dir = '../customized_partition_bash'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

sub                 = 'sub-04'
cores               = 16
mem                 = 2 * cores
time_               = 48 * cores
stacked_data_dir    = '../../../../data/BOLD_average/{}/'.format(sub)
BOLD_data           = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))

for ii,BOLD_data_file in enumerate(BOLD_data):
    roi_name = BOLD_data_file.split('/')[-1].replace("_BOLD.npy","")
    with open(os.path.join(output_dir,'CP_{}.py'.format(roi_name)),'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "../../../data/" in line:
                    line = line.replace("../../../data/","../../../../data/")
                elif "sub                 = 'sub-01'" in line:
                    line = line.replace("sub-01",f"{sub}")
                elif "os.chdir('..')" in line:
                    line = 'print(os.getcwd())\n'
                elif "../../../results/" in line:
                    line = line.replace("../../../results/","../../../../results/")
                elif "idx = 0" in line:
                    line = line.replace("idx = 0","idx = {}".format(ii))
                elif "../../utils.py" in line:
                    line = line.replace("../../utils.py","../../../utils.py")
                elif "n_jobs = " in line:
                    line = f"n_jobs = {cores}\n"
                new_file.write(line)
            old_file.close()
        new_file.close()
if not os.path.exists(f'{output_dir}/bash'):
    os.mkdir(f'{output_dir}/bash')
else:
    [os.remove(f'{output_dir}/bash/'+f) for f in os.listdir(f'{output_dir}/bash')]
    
for ii, BOLD_data_file in enumerate(BOLD_data):
    roi_name = BOLD_data_file.split('/')[-1].replace("_BOLD.npy","")
    content = f"""
#!/bin/bash
#PBS -q bcbl
#PBS -l nodes=1:ppn={cores}
#PBS -l mem={mem}gb
#PBS -l cput={time_}:00:00
#PBS -N S{sub[-1]}R{ii+1}
#PBS -o bash/out_{sub[-1]}{ii+1}.txt
#PBS -e bash/err_{sub[-1]}{ii+1}.txt
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:$PATH"
pwd
echo "{roi_name}"
python "CP_{roi_name}.py"
"""
    print(content)
    with open(f'{output_dir}/decode_{roi_name}_q','w') as f:
        f.write(content)

content = '''
import os
import time
'''
with open(f'{output_dir}/qsub_jobs_decode_CP.py','w') as f:
    f.write(content)
    f.close()

with open(f'{output_dir}/qsub_jobs_decode_CP.py','a') as f:
    for ii, BOLD_data_file in enumerate(BOLD_data):
        roi_name = BOLD_data_file.split('/')[-1].replace("_BOLD.npy","")
        if ii == 0:
            f.write('\nos.system("qsub decode_{}_q")\n'.format(roi_name))
        else:
            f.write('time.sleep(3)\nos.system("qsub decode_{}_q")\n'.format(roi_name))
    f.close()
content = f'''
#!/bin/bash

# This is a script to send qsub_jobs_decode (CP).py as a batch job.

#PBS -q bcbl
#PBS -l nodes=1:ppn=1
#PBS -l mem=50mb
#PBS -l cput=1:00:00
#PBS -N "qsubjobs"
#PBS -o bash/out_q.txt
#PBS -e bash/err_q.txt
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:$PATH"
pwd
python "{output_dir}/qsub_jobs_decode_CP.py"
'''
with open(f'{output_dir}/qsub_jobs_decode_CP','w') as f:
    print(content)
    f.write(content)
    f.close()



"""
#$ -cwd
#$ -o bash/out_{ii+1}.txt
#$ -e bash/err_{ii+1}.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "CP{ii+1}"
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







