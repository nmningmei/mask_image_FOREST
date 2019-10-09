#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:42:25 2019

@author: nmei
"""

import os
import numpy as np

if not os.path.exists('bash'):
    os.mkdir('bash')
else:
    [os.remove(os.path.join('bash',f)) for f in os.listdir('bash')]
cores = 16
mem = cores * 5
ctime = 48 * cores
template = '../spaceNet.py'
for sub in np.arange(1,8):
    with open(f'../spaceNet_{sub}.py','w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "sub                 = 'sub-01'" in line:
                    line = f"sub                 = 'sub-0{sub}'\n"
                if "first_session       = 2" in line and sub != 1:
                    line = "first_session       = 1\n"
                new_file.write(line)
            old_file.close()
        new_file.close()
    content = f"""
#!/bin/bash
#PBS -q bcbl
#PBS -l nodes=1:ppn={cores}
#PBS -l mem={mem}gb
#PBS -l cput={ctime}:00:00
#PBS -N SN{sub}
#PBS -o spaceNet_outputs/out_spaceNet{sub}
#PBS -e spaceNet_outputs/err_spaceNet{sub}
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:$PATH"
pwd
python "spaceNet_{sub}.py"
    """
    with open(f'../spaceNet_DIPC{sub}','w') as f:
        print(content)
        f.write(content)
        f.close()

template = '../spaceNet post stats.py'
for sub in np.arange(1,8):
    with open(f'spaceNet post stats {sub}.py','w') as new_file:
        with open(template, 'r') as old_file:
            for line in old_file:
                if "../../../" in line:
                    line = line.replace("../../../","../../../../")
                elif "sub                     = 'sub-01'" in line:
                    line = f"sub                     = 'sub-0{sub}'\n"
                if "first_session       = 2" in line and sub == 1:
                    line = "first_session       = 1"
                new_file.write(line)
            old_file.close()
        new_file.close()
    
    content = f"""
#!/bin/bash
#$ -cwd
#$ -o bash/out_{sub}.txt
#$ -e bash/err_{sub}.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "SNP{sub}"
#$ -S /bin/bash
module load rocks-python-3.6 rocks-fsl-5.0.10 rocks-freesurfer-6.0.0
python "spaceNet post stats {sub}.py"
    """
    with open(f'spaceNet_post_stats_{sub}','w') as f:
        f.write(content)
        f.close()