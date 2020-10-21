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
template = 'search_light_correlation_univariate_t_test.py'
output_dir = '../RSA_stats_bash'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    

nodes               = 1
cores               = 16
mem                 = int(5 * nodes * cores)
time_               = 26 * nodes * cores


if not os.path.exists(f'{output_dir}/bash'):
    os.mkdir(f'{output_dir}/bash')


from shutil                    import copyfile
copyfile('../../../utils.py',f'{output_dir}/utils.py')

time = .1
content = '''
import os
import time
'''
with open(f'{output_dir}/qsub_jobs_RAS_LOO.py','w') as f:
    f.write(content)
    f.close()

count = 0
with open(f'{output_dir}/qsub_jobs_RAS_LOO.py','a') as bash_file:
    for model_name in os.listdir('../../../../results/MRI/nilearn/RSA_searchlight'):
        target_file = 'RSA_{}.py'.format(model_name)
        with open(os.path.join(output_dir,target_file),'w') as new_file:
            with open(template,'r') as old_file:
                for line in old_file:
                    if "model_name = " in line:
                        line = f"model_name = '{model_name}'"
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
#PBS -N {model_name}-RSA
#PBS -o bash/out_{model_name}.txt
#PBS -e bash/err_{model_name}.txt
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd
python {target_file}
"""
        print(content)
        bash_file_name = f'RSA_{model_name}_q'
        with open(f'{output_dir}/{bash_file_name}','w') as f:
            f.write(content)
            f.close()
            
        if count == 0:
            bash_file.write('\nos.system("qsub {}")\n'.format(bash_file_name))
        else:
            bash_file.write('time.sleep({})\nos.system("qsub {}")\n'.format(time,bash_file_name))
        count += 1
    bash_file.close()