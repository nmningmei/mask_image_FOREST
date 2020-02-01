#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:39:25 2020

@author: nmei
"""

content = f"""#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={2}:ppn={16}
#PBS -l mem={90}gb
#PBS -l cput={int(1e3)}:00:00
#PBS -N hidden
#PBS -o out_run_all.txt
#PBS -e err_run_all.txt

cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd

python "8.1.simulation_experiment_decode_hidden_reprs.py"
    """
with open('run_all','w') as f:
    f.write(content)
    f.close()