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
core = 12
mem = 1 * node * core
cput = 36 * node * core
batch_show_name = 'DSIM'
groupby = 100

working_dir = '../../results/agent_models'
working_data = np.sort(glob(os.path.join(working_dir,
                                 '*',
                                 '*',
                                 '*.npy')))
working_data = working_data.reshape(-1,2)

template = '8.1.simulation_experiment_decode_hidden_reprs.py'
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
script_names = []
for ii,(features_,labels_) in tqdm(enumerate(working_data)):
    features_,labels_
    model = features_.split('/')[-3]
    noise_level = features_.split('/')[-2]
    new_scripts_name = os.path.join(scripts_folder,template.replace('.py',f'_{model}_{noise_level}.py').replace('8.1.simulation','simulation'))
    script_names.append(new_scripts_name)
    results_csv_name = os.path.join(
        '/'.join(features_.split('/')[:-1]),
        'scores as a function of decoder and noise.csv')
    with open(new_scripts_name,'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "../../" in line:
                    line = line.replace("../../","../../../")
                elif "import utils_deep" in line:
                    line = "{}\n{}".format(add_on,line)
                elif "verbose             = " in line:
                    line = f"verbose             = {verbose}\n"
                elif "# change index" in line:
                    line = line.replace('0',f'{ii}')
                new_file.write(line)
            old_file.close()
        new_file.close()
    new_batch_script_name = os.path.join(scripts_folder,f'{batch_show_name}{ii+1}')
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
    if not os.path.exists(results_csv_name):
        collections.append(f"qsub {batch_show_name}{ii+1}")
    else:
        print(results_csv_name)
    


collections = np.array_split(collections,len(collections) // groupby)
script_names = np.array_split(script_names,len(script_names) // groupby)


for jj,(script_names_block,collections_block) in enumerate(zip(
                script_names,collections)):
    script_names_block,collections_block
    with open(f'{scripts_folder}/qsub_jobs_{jj}.py','w') as f:
        f.write("""import os\nimport time\n""")
        f.close()
    with open(f'{scripts_folder}/qsub_jobs_{jj}.py','a') as f:
        for ii,line in enumerate(collections_block):
            if ii == 0:
                f.write(f'os.system("{line}")\nprint("{line}")\n')
            elif ii == len(collections_block) - 1:
                f.write(f'time.sleep(1)\nos.system("{line}")\nprint("{line}")\ntime.sleep({30*60})\n')
            else:
                f.write(f'time.sleep(1)\nos.system("{line}")\n')
        f.close()

    content = f"""#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={1}:ppn={1}
#PBS -l mem={1}gb
#PBS -l cput={1000}:00:00
#PBS -N qsub
#PBS -o outputs/out_qsub.txt
#PBS -e outputs/err_qsub.txt

cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd
echo qsub

python "qsub_jobs.py"
        """
    with open(f'{scripts_folder}/qsub_jobs','w') as f:
        f.write(content)
        f.close()













