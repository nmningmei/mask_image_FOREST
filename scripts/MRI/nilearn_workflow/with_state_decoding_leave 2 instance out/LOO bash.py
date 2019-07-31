#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:47:50 2019

@author: nmei
"""

import os
from glob import glob
template = 'LOO.py'
output_dir = '../LOO bash'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

sub                 = 'sub-02'
stacked_data_dir    = '../../../../data/BOLD_no_average/{}/'.format(sub)
BOLD_data           = glob(os.path.join(stacked_data_dir,'*BOLD.npy'))

for ii,BOLD_data_file in enumerate(BOLD_data):
    with open(os.path.join(output_dir,'LOO ({}).py'.format(ii+1)),'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "../../../data/" in line:
                    line = line.replace("../../../data/",
                                        "../../../../data/")
                elif "os.chdir('..')" in line:
                    line = 'print(os.getcwd())\n'
                elif "../../../results/" in line:
                    line = line.replace("../../../results/",
                                        "../../../../results/")
                elif "idx = 0" in line:
                    line = line.replace("idx = 0",
                                        "idx = {}".format(ii))
                elif "../../utils.py" in line:
                    line = line.replace("../../utils.py",
                                        "../../../utils.py")
                new_file.write(line)
            old_file.close()
        new_file.close()
if not os.path.exists(f'{output_dir}/bash'):
    os.mkdir(f'{output_dir}/bash')
else:
    [os.remove(f'{output_dir}/bash/'+f) for f in os.listdir(f'{output_dir}/bash') if (f"S{sub[-1]}" in f)]
    
for ii, BOLD_data_file in enumerate(BOLD_data):
    mask_name = BOLD_data_file.split('/')[-1].replace('_BOLD.npy','').replace('ctx-','')
    content = f"""
#!/bin/bash

# This is a script to send "LOO ({ii+1}).py" as a batch job.


#$ -cwd
#$ -o bash/out_{sub[-1]}{ii+1}.txt
#$ -e bash/err_{sub[-1]}{ii+1}.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "S{sub[-1]}R{ii+1}"
#$ -S /bin/bash

module load rocks-python-3.6
python "LOO ({ii+1}).py"
"""
    print(content)
    with open(f'{output_dir}/decode_{ii+1}_q','w') as f:
        f.write(content)
time = 60 * 60
content = '''
import os
import time
'''
with open(f'{output_dir}/qsub_jobs_decode (LOO).py','w') as f:
    f.write(content)
    f.close()

with open(f'{output_dir}/qsub_jobs_decode (LOO).py','a') as f:
    for ii, BOLD_data_file in enumerate(BOLD_data):
        if ii == 0:
            f.write('\nos.system("qsub decode_{}_q")\n'.format(ii+1))
        else:
            f.write('time.sleep({})\nos.system("qsub decode_{}_q")\n'.format(time,ii+1))
    f.close()
content = f'''
#!/bin/bash

# This is a script to send qsub_jobs_decode (LOO).py as a batch job.

#$ -cwd
#$ -o bash/out_q.txt
#$ -e bash/err_q.txt
#$ -m be
#$ -M nmei@bcbl.eu
#$ -N "qsubjobs"
#$ -S /bin/bash

module load rocks-python-3.6
python "{output_dir}/qsub_jobs_decode (LOO).py"
'''
with open(f'{output_dir}/qsub_jobs_decode_LOO','w') as f:
    print(content)
    f.write(content)
    f.close()











