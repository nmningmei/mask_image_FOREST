#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:47:50 2019

@author: nmei
"""

import os
import numpy as np
import pandas as pd
from glob import glob
from nibabel import load as load_fmri
from utils import LOO_partition
template = 'LOO_searchlight_LOSO.py'
output_dir = '../decode_searchlight_LOSO_bash'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

content = '''
import os
import time
'''
with open(f'{output_dir}/qsub_jobs_decode_LOO.py','w') as f:
    f.write(content)
    f.close()

if not os.path.exists(f'{output_dir}/bash'):
    os.mkdir(f'{output_dir}/bash')


nodes               = 1
cores               = 16
mem                 = 4 * cores * nodes
time_               = 36 * cores * nodes
label_map           = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}

for conscious_state_source in ['unconscious','glimpse','conscious']:
    for conscious_state_target in ['unconscious','glimpse','conscious']:
        for sub in [1,2,3,4,5,6,7]:
                created_file_name = 'LOO_{}_{}_{}.py'.format(
                                                    conscious_state_source,
                                                    conscious_state_target,
                                                    sub,)
                with open(os.path.join(output_dir,created_file_name),'w') as new_file:
                    with open(template,'r') as old_file:
                        for line in old_file:
                            if "sub                 = 'sub-" in line:
                                line = f"    sub                 = 'sub-0{sub}'\n"
                            elif 'conscious_source    = ' in line:
                                line = f"    conscious_source  = '{conscious_state_source}'\n"
                            elif 'conscious_target    = ' in line:
                                line = f"    conscious_target  = '{conscious_state_target}'\n"
                            new_file.write(line)
                        old_file.close()
                    new_file.close()
                content = f"""
#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={nodes}:ppn={cores}
#PBS -l mem={mem}gb
#PBS -l cput={time_}:00:00
#PBS -N S{sub}_{conscious_state_source[0]}_{conscious_state_target[0]}
#PBS -o bash/out_sub{sub}_{conscious_state_source}_{conscious_state_target}.txt
#PBS -e bash/err_sub{sub}_{conscious_state_source}_{conscious_state_target}.txt
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd
echo "{sub} {conscious_state_source} --> {conscious_state_target}"

python "{created_file_name}"
"""
                print(content)
                with open(f'{output_dir}/decode_{sub}_{conscious_state_source}_{conscious_state_target}_q','w') as f:
                    f.write(content)
                    f.close()
                with open(f'{output_dir}/qsub_jobs_decode_LOO.py','a') as f:
                    line = f'\nos.system("qsub decode_{sub}_{conscious_state_source}_{conscious_state_target}_q")\n'
                    f.write(line)
                    f.close()

# content = f'''
# #!/bin/bash

# # This is a script to send qsub_jobs_decode_LOO.py as a batch job.
# #PBS -q bcbl
# #PBS -l nodes=1:ppn=1
# #PBS -l mem=50mb
# #PBS -l cput=1:00:00
# #PBS -N qsubjobs
# #PBS -o bash/out_q.txt
# #PBS -e bash/err_q.txt
# cd $PBS_O_WORKDIR
# export PATH="/scratch/ningmei/anaconda3/bin:$PATH"
# pwd
# python "{output_dir}/qsub_jobs_decode_LOO.py"
# '''
# with open(f'{output_dir}/qsub_jobs_decode_LOO','w') as f:
#     print(content)
#     f.write(content)
#     f.close()



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







