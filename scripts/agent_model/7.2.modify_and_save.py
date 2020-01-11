#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 13:09:50 2019

@author: nmei
"""
import os
from glob import glob
import numpy as np
from tqdm import tqdm
from shutil import rmtree

verbose = 1
node = 2
core = 16
mem = 2 * node * core
cput = 36 * node * core

working_dir = '../../results/agent_models'
working_data = np.sort(glob(os.path.join(working_dir,
                                 '*',
                                 '*',
                                 '*.npy')))
working_data = working_data.reshape(-1,2)

template = '8.simulation_experiment_decode_hidden_reprs.py'
scripts_folder = 'decode_bash'
if not os.path.exists(scripts_folder):
    os.mkdir(scripts_folder)
else:
    rmtree(scripts_folder)
    os.mkdir(scripts_folder)
os.mkdir(f'{scripts_folder}/outputs')

add_on = """from shutil import copyfile
copyfile('../utils_deep.py','utils_deep.py')

"""
collections = []
for ii,(features_,labels_) in tqdm(enumerate(working_data)):
    features_,labels_
    model = features_.split('/')[-3]
    noise_level = features_.split('/')[-2]
    new_scripts_name = os.path.join(scripts_folder,template.replace('.py',f'{model}_{noise_level}.py').replace('8.simulation','simulation'))
    with open(new_scripts_name,'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "../../" in line:
                    line = line.replace("../../","../../../")
                elif "import utils_deep" in line:
                    line = "{}\n{}".format(add_on,line)
                elif "verbose             = " in line:
                    line = f"    verbose             = {verbose}\n"
                elif "idx = 0" in line:
                    line = f'idx = {ii}\n'
                new_file.write(line)
            old_file.close()
        new_file.close()
    new_batch_script_name = os.path.join(scripts_folder,f'DSIM{ii+1}')
    content = f"""#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={node}:ppn={core}
#PBS -l mem={mem}gb
#PBS -l cput={cput}:00:00
#PBS -N DSIM{ii+1}
#PBS -o outputs/out_{ii+1}.txt
#PBS -e outputs/err_{ii+1}.txt

cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd
echo {new_scripts_name.split('/')[-1]}

python "{new_scripts_name.split('/')[-1]}"
    """
    with open(new_batch_script_name,'w') as f:
        f.write(content)
        f.close()
    collections.append(f"qsub SIM{ii+1}")

with open(f'{scripts_folder}/qsub_jobs.py','w') as f:
    f.write("""import os\nimport time""")

with open(f'{scripts_folder}/qsub_jobs.py','a') as f:
    for ii,line in enumerate(collections):
        if ii == 0:
            f.write(f'\nos.system("{line}")\n')
        else:
            f.write(f'time.sleep(1)\nos.system("{line}")\n')
    f.close()



















