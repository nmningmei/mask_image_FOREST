#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 20:43:20 2019

@author: nmei
"""

import os
from glob import glob
import re
from time import sleep
import numpy as np


working_dir = ''
sub = 'sub-01'
data_dir = '../../../data/MRI/{}/func'.format(sub)
fsf_file_session_2_run1 = glob(os.path.join(working_dir,'session2.run1','*.fsf'))[0]
fsf_file_session_2_run2 = glob(os.path.join(working_dir,'session2.run2','*.fsf'))[0]


session2_run1 = """
#!/bin/bash
#$ -cwd
#$ -o out.txt
#$ -e err.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "s2r1"
#$ -S /bin/bash

module load rocks-fsl-5.0.9
feat "run01.fsf"
"""
with open('session2.run1/session2.run1q','w') as f:
    f.write(session2_run1)

fMRI_files = glob(os.path.join(data_dir,"session-*","*","*.nii"))
fMRI_files = [item for item in fMRI_files if ('wrong' not in item)]
fMRI_files = fMRI_files[2:]
for fMRI_file in fMRI_files:
    session = re.findall('session-0\d',fMRI_file)[0]
    run = re.findall('_run-0\d_',fMRI_file)[0]
    n_session = int(re.findall('\d+',session)[0])
    n_run = int(re.findall('\d+',run)[0])
    
    if not os.path.exists(f'session{n_session}.run{n_run}'):
        os.mkdir(f'session{n_session}.run{n_run}')
    with open(f'session{n_session}.run{n_run}/session{n_session}.run{n_run}.fsf','w') as new_fsf:
        with open('session2.run2/session2.run2.fsf','r') as template_fsf:
            for line in template_fsf:
                if "set fmri(outputdir)" in line:
                    line = line.replace(f'set fmri(outputdir) "/bcbl/home/public/Consciousness/uncon_feat/data/MRI/{sub}/func/session-02/{sub}_unfeat_run-02/FEAT.session2.run2"',
                                 f'set fmri(outputdir) "/bcbl/home/public/Consciousness/uncon_feat/data/MRI/{sub}/func/session-0{n_session}/{sub}_unfeat_run-0{n_run}/FEAT.session{n_session}.run{n_run}"')
                elif "set feat_files(1)" in line:
                    line = line.replace(f'set feat_files(1) "/bcbl/home/public/Consciousness/uncon_feat/data/MRI/{sub}/func/session-02/{sub}_unfeat_run-02/{sub}_unfeat_run-02_bold"',
                                        f'set feat_files(1) "/bcbl/home/public/Consciousness/uncon_feat/data/MRI/{sub}/func/session-0{n_session}/{sub}_unfeat_run-0{n_run}/{sub}_unfeat_run-0{n_run}_bold"')
                elif "set alt_ex_func(1)" in line:
                    line = line.replace(f'set alt_ex_func(1) "/bcbl/home/public/Consciousness/uncon_feat/data/MRI/{sub}/func/session-02/{sub}_unfeat_run-01/run1.FEAT+.feat/example_func"',
                                        f'set alt_ex_func(1) "/bcbl/home/public/Consciousness/uncon_feat/data/MRI/{sub}/func/session-02/{sub}_unfeat_run-01/run1.FEAT+.feat/example_func"')
                new_fsf.write(line)
        template_fsf.close()
    new_fsf.close()
    
    
    
    
fMRI_files = glob(os.path.join(data_dir,"*","*","*.nii"))
fMRI_files = [item for item in fMRI_files if ('wrong' not in item)]
fMRI_files = fMRI_files[1:]
for fMRI_file in fMRI_files:
    session = re.findall('session-0\d',fMRI_file)[0]
    run = re.findall('_run-0\d_',fMRI_file)[0]
    n_session = int(re.findall('\d+',session)[0])
    n_run = int(re.findall('\d+',run)[0])
    runs = f"""
#!/bin/bash
#$ -cwd
#$ -o out.txt
#$ -e err.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "s{n_session}r{n_run}"
#$ -S /bin/bash

module load rocks-fsl-5.0.9
feat "session{n_session}.run{n_run}.fsf"

"""
    with open(f'session{n_session}.run{n_run}/session{n_session}.run{n_run}q','w') as f:
        f.write(runs)
        f.close()

content = """
import os
from time import sleep
from glob import glob
import numpy as np

working_dir = ''

timing = 30
bashes = np.sort(glob(os.path.join(working_dir,"*","*q")))
for each in bashes[1:]:
    directory_to_go = os.path.join('/bcbl/home/public/Consciousness/uncon_feat/scripts/MRI/standard/',
                                   each.split('/')[0])
    os.chdir(directory_to_go)
    os.system("qsub {}".format(each.split('/')[-1]))
    sleep(timing)
"""
with open('qsub_jobs.py','w') as f:
    f.write(content)
    f.close()

content = """
#!/bin/bash
#$ -cwd
#$ -o out.txt
#$ -e err.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "qsub_jobs"
#$ -S /bin/bash

module load rocks-python-3.6
python qsub_jobs.py
"""
with open('qbash','w') as f:
    f.write(content)
    f.close()















