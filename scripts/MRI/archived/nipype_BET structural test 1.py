#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:09:32 2019

@author: nmei
"""

import os
from glob import glob
structural_dir = '../../data/MRI/sub-{}/anat'
sub = '01'
#from nilearn.plotting import plot_anat

# use BET to extract the brain
from nipype.interfaces.fsl import BET
skullstrip = BET()
in_file = glob(os.path.join(structural_dir.format(sub),'*.nii'))[1]
skullstrip.inputs.in_file = os.path.abspath(in_file)
skullstrip.inputs.out_file = os.path.abspath(
                            in_file.replace('.nii',
                                            '_brain.nii.gz')
                            )
skullstrip.inputs.frac = 0.40
skullstrip.inputs.robust = True
skullstrip.run()

#plot_anat(skullstrip.inputs.out_file)












