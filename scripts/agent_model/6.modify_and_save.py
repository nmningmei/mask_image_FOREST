#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:02:48 2019

@author: nmei
"""

import os
import itertools
import numpy as np
import pandas as pd
from shutil import rmtree

verbose = 1
batch_size = 16
node = 2
core = 16
mem = 2 * node * core
cput = 36 * node * core
units = [5,10,20,50,100,300]
dropouts = [0,0.1,0.2]
models = ['DenseNet169',           # 1664
          'InceptionV3',           # 2048
          'MobileNetV2',           # 1280
          'ResNet50',              # 1536
          'VGG19',                 # 2048
          'Xception',              # 1280
          ]
activations = ['elu',
               'relu',
               'selu',
               'sigmoid',
               'tanh',
               'linear',
               ]
output_activations = ['softmax','sigmoid',]


temp = np.array(list(itertools.product(*[units,dropouts,models,activations,output_activations])))
df = pd.DataFrame(temp,columns = ['hidden_units','dropouts','model_names','hidden_activations','output_activations'])
df['hidden_units'] = df['hidden_units'].astype(int)
df['dropouts'] = df['dropouts'].astype(float)

preproc_func = {'DenseNet169':'densenet',
                'InceptionV3':'inception_v3',
                'MobileNetV2':'mobilenet_v2',
                'ResNet50':'resnet50',
                'VGG19':'vgg19',
                'Xception':'xception',
                }
df['preprocess_input'] = df['model_names'].map(preproc_func)

template = '7.simulation_experiment_get_hidden_reprs.py'
scripts_folder = 'bash'
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
for ii,row in df.iterrows():
    src = '_{}_{}_{}_{}_{}'.format(*list(row.to_dict().values()))
    
    new_scripts_name = os.path.join(scripts_folder,template.replace('.py',f'{src}.py').replace('7.simulation','simulation'))
    
    with open(new_scripts_name,'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "../../" in line:
                    line = line.replace("../../","../../../")
                elif "import utils_deep" in line:
                    line = "{}\n{}".format(add_on,line)
                elif "hidden_units        = " in line:
                    line = f"hidden_units        = {row['hidden_units']}" + "\n"
                elif "drop_rate           = " in line:
                    line = f"drop_rate           = {row['dropouts']}" + "\n"
                elif "model_name          = " in line:
                    line = f"model_name          = '{row['model_names']}'" + "\n"
                elif "model_pretrained    = " in line:
                    line = f"model_pretrained    = applications.{row['model_names']}" + "\n"
                elif "hidden_activation   = " in line:
                    line = f"hidden_activation   = '{row['hidden_activations']}'" + "\n"
                elif "output_activation   = " in line:
                    line = f"output_activation   = '{row['output_activations']}'" + "\n"
                elif "preprocess_input           = " in line:
                    line = f"preprocess_input           = applications.{row['preprocess_input']}.preprocess_input" + "\n"
                elif "verbose             = " in line:
                    line = f"verbose             = {verbose}\n"
                elif "batch_size          = " in line:
                    line = f"batch_size          = {batch_size}\n"
                new_file.write(line)
            old_file.close()
        new_file.close()
    new_batch_script_name = os.path.join(scripts_folder,f'SIM{ii+1}')
    content = f"""#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={node}:ppn={core}
#PBS -l mem={mem}gb
#PBS -l cput={cput}:00:00
#PBS -N SIM{ii+1}
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
            f.write(f'time.sleep(3)\nos.system("{line}")\n')
    f.close()

from glob import glob
all_scripts = glob(os.path.join(scripts_folder,'simulation*.py'))
with open(os.path.join(scripts_folder,'run_all.py'),'w') as f:
    f.write('import os\n')
    for files in all_scripts:
        file_name = files.split('bash/')[-1]
        f.write(f'os.system("python {file_name}")\n')
    f.close()





































