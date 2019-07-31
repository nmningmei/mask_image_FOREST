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

template = '../spaceNet post stats.py'
for sub in np.arange(1,5):
    with open(f'spaceNet post stats {sub}.py','w') as new_file:
        with open(template, 'r') as old_file:
            for line in old_file:
                if "../../../" in line:
                    line = line.replace("../../../","../../../../")
                elif "sub                     = 'sub-01'" in line:
                    line = f"sub                     = 'sub-0{sub}'\n"
                new_file.write(line)
            old_file.close()
        new_file.close()
    
    content = f"""
#!/bin/bash

# This is a script to send "spaceNet post stats {sub}.py" as a batch job.


#$ -cwd
#$ -o bash/out_{sub}.txt
#$ -e bash/err_{sub}.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "SN{sub}"
#$ -S /bin/bash

module load rocks-python-3.6 rocks-fsl-5.0.10
python "spaceNet post stats {sub}.py"
    """
    with open(f'spaceNet_post_stats_{sub}','w') as f:
        f.write(content)
        f.close()