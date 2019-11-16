#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:43:59 2019

@author: nmei
"""

from nipype.interfaces import freesurfer
import os
from glob import glob

freesurfer_result_dir = '../../data/MRI/{}/anat/{}/mri'
sub = 'sub-01'

freesurfer_data = os.path.abspath(freesurfer_result_dir.format(sub,sub))


ROI_names = """
fusif infpar superiorparietal inftemp latoccip lingual phipp pericalc precun sfrontal parsoper parsorbi parstri middlefrontal
"""

ROI_names = ROI_names.replace('\n','').split(' ')



























