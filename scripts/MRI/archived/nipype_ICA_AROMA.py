#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:22:04 2019

@author: nmei
"""

import os
from glob import glob
from nipype.interfaces.fsl import ICA_AROMA
import re

sub = 'sub-01'
MRI_dir = '../../data/MRI/{}/func'.format(sub)
parent_dir = os.path.join(MRI_dir,'session-*','*','FEAT*',)
preprocessed_BOLD_files = glob(os.path.join(parent_dir,
                                            'filtered_func_data.nii.gz'))
to_work_dir = 'ICA_AROMA_parallel'
if not os.path.exists(to_work_dir):
    os.mkdir(to_work_dir)
template = """
#!/bin/bash
#$ -cwd
#$ -o out_{}.txt
#$ -e err_{}.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "s{}r{}"
#$ -S /bin/bash

module load rocks-fsl-5.0.10
module load rocks-python-2.7
{}
"""
for sample in preprocessed_BOLD_files:
    AROMA_obj = ICA_AROMA()
    parent_dir = '/'.join(sample.split('/')[:-1])
    n_session = int(re.findall('\d+',re.findall('session[\d+]',sample)[0])[0])
    n_run = int(re.findall('\d+',re.findall('run[\d+]',sample)[0])[0])
    print(n_session,n_run)
    func_to_struct = os.path.join(parent_dir,
                                  'reg',
                                  'example_func2highres.mat')
    warpfield = os.path.join(parent_dir,
                             'reg',
                             'highres2standard_warp.nii.gz')
    fsl_mcflirt_movpar = os.path.join(parent_dir,
                                      'mc',
                                      'prefiltered_func_data_mcf.par')
    mask = os.path.join(parent_dir,
                        'mask.nii.gz')
    output_dir = os.path.join(parent_dir,
                              'ICA_AROMA')
    AROMA_obj.inputs.in_file = os.path.abspath(sample)
    AROMA_obj.inputs.mat_file = os.path.abspath(func_to_struct)
    AROMA_obj.inputs.fnirt_warp_file = os.path.abspath(warpfield)
    AROMA_obj.inputs.motion_parameters = os.path.abspath(fsl_mcflirt_movpar)
    AROMA_obj.inputs.mask = os.path.abspath(mask)
    AROMA_obj.inputs.denoise_type = 'nonaggr'
    AROMA_obj.inputs.out_dir = os.path.abspath(output_dir)
    cmdline = 'python ../' + AROMA_obj.cmdline + ' -ow'
    qsub = template.format(10*n_session+n_run,10*n_session+n_run,n_session,n_run,cmdline)
    with open(os.path.join(to_work_dir,'session{}.run{}_qs'.format(n_session,n_run)),'w') as f:
        f.write(qsub)

to_qsub = """
import os
from time import sleep
"""
with open('{}/ICA_qsub_jobs.py'.format(to_work_dir),'w') as f:
    f.write(to_qsub)
    f.close()
for sample in preprocessed_BOLD_files:
    n_session = int(re.findall('\d+',re.findall('session[\d+]',sample)[0])[0])
    n_run = int(re.findall('\d+',re.findall('run[\d+]',sample)[0])[0])
    with open('{}/ICA_qsub_jobs.py'.format(to_work_dir),'a') as f:
        f.write('\nos.system("qsub session{}.run{}_qs")\nsleep(150)\n'.format(n_session,n_run))
        f.close()

qsub_qsub_jobs = """
#!/bin/bash
#$ -cwd
#$ -o out_q.txt
#$ -e err_q.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "qjobs"
#$ -S /bin/bash

module load rocks-python-2.7
python ICA_qsub_jobs.py
"""
with open('{}/qsub_ICA_qsub_jobs'.format(to_work_dir), 'w') as f:
    f.write(qsub_qsub_jobs)
    f.close()











