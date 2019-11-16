#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:14:55 2019

@author: nmei
"""

from nipype.interfaces.freesurfer import ReconAll
#from nipype.workflows.smri.freesurfer import create_reconall_workflow
import os
sub = 'sub-01'
working_dir = '../../data/MRI/{}/anat'.format(sub)
reconall = ReconAll()
reconall.inputs.subject_id = sub
reconall.inputs.directive = 'all'
reconall.inputs.subjects_dir = os.path.abspath(working_dir)
reconall.inputs.T1_files = os.path.abspath(os.path.join(working_dir,
                                                        'sub-01-T1W_mprage_sag_p2_1iso_MGH_day_6.nii'))
reconall.cmdline
reconall.run()

#reconall = create_reconall_workflow()
#reconall.inputs.subject_id = sub
#reconall.inputs.directive = 'all'
#reconall.inputs.subjects_dir = '.'
#reconall.inputs.T1_files = os.path.abspath(os.path.join(working_dir,
#                                                        'sub-01-T1W_mprage_sag_p2_1iso_MGH_day_1.nii'))
#reconall.cmdline