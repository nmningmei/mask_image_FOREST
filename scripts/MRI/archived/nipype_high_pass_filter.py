#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:35:01 2019

@author: nmei
"""

import os
from glob import glob
from nipype.workflows.fmri.fsl    import preprocess
from nipype.interfaces            import fsl
from nipype.pipeline              import engine as pe
from nipype.interfaces            import utility as util
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
getthreshop         = preprocess.getthreshop
getmeanscale        = preprocess.getmeanscale
sub = 'sub-01'
data_dir = '../../data/MRI/{}/func/*/*/outputs/func/ICA_AROMA'.format(sub)
ICAed_data = glob(os.path.join(
        data_dir,
        'denoised_func_data_nonaggr.nii.gz'))
HP_freq = 60
TR = 0.85

for file_name in ICAed_data[:1]:
    file_name
    highpass_workflow = pe.Workflow(name = 'highpassfiler')
    
    inputnode               = pe.Node(interface = util.IdentityInterface(
                                      fields    = ['ICAed_file',]),
                                      name      = 'inputspec')
    outputnode              = pe.Node(interface = util.IdentityInterface(
                                      fields    = ['filtered_file']),
                                      name      = 'outputspec')
    
    img2float = pe.MapNode(interface = fsl.ImageMaths(out_data_type = 'float',op_string = '',suffix = '_dtype'),
                           iterfield = ['in_file'],
                           name = 'img2float')
    highpass_workflow.connect(inputnode,'ICAed_file',
                              img2float,'in_file')
    
    getthreshold = pe.MapNode(interface = fsl.ImageStats(op_string = '-p 2 -p 98'),
                              iterfield = ['in_file'],
                              name = 'getthreshold')
    highpass_workflow.connect(img2float,'out_file',
                              getthreshold,'in_file')
    thresholding = pe.MapNode(interface = fsl.ImageMaths(out_data_type = 'char',suffix = '_thresh',
                                                         op_string = '-Tmin -bin'),
                                iterfield = ['in_file','op_string'],
                                name = 'thresholding')
    highpass_workflow.connect(img2float,'out_file',
                              thresholding,'in_file')
    highpass_workflow.connect(getthreshold,('out_stat',getthreshop),
                              thresholding,'op_string')
    
    dilatemask = pe.MapNode(interface = fsl.ImageMaths(suffix = '_dil',op_string = '-dilF'),
                            iterfield = ['in_file'],
                            name = 'dilatemask')
    highpass_workflow.connect(thresholding,'out_file',
                              dilatemask,'in_file')
    
    maskfunc = pe.MapNode(interface = fsl.ImageMaths(suffix = '_mask',op_string = '-mas'),
                          iterfield = ['in_file','in_file2'],
                          name = 'apply_dilatemask')
    highpass_workflow.connect(img2float,'out_file',
                              maskfunc,'in_file')
    highpass_workflow.connect(dilatemask,'out_file',
                              maskfunc,'in_file2')
    
    medianval = pe.MapNode(interface = fsl.ImageStats(op_string = '-k %s -p 50'),
                           iterfield = ['in_file','mask_file'],
                           name = 'cal_intensity_scale_factor')
    highpass_workflow.connect(img2float,'out_file',
                              medianval,'in_file')
    highpass_workflow.connect(thresholding,'out_file',
                              medianval,'mask_file')
    
    meanscale = pe.MapNode(interface = fsl.ImageMaths(suffix = '_intnorm'),
                           iterfield = ['in_file','op_string'],
                           name = 'meanscale')
    highpass_workflow.connect(maskfunc,'out_file',
                              meanscale,'in_file')
    highpass_workflow.connect(medianval,('out_stat',getmeanscale),
                              meanscale,'op_string')
    
    meanfunc = pe.MapNode(interface = fsl.ImageMaths(suffix = '_mean',
                                                     op_string = '-Tmean'),
                           iterfield = ['in_file'],
                           name = 'meanfunc')
    highpass_workflow.connect(meanscale,'out_file',
                              meanfunc,'in_file')
    
    
    hpf = pe.MapNode(interface = fsl.ImageMaths(suffix = '_tempfilt',
                                                op_string = '-bptf %.10f -1' % (60/2/0.85)),
                     iterfield = ['in_file'],
                     name = 'highpass_filering')
    highpass_workflow.connect(meanscale,'out_file',
                              hpf,'in_file',)
    
    addMean = pe.MapNode(interface = fsl.BinaryMaths(operation = 'add'),
                         iterfield = ['in_file','operand_file'],
                         name = 'addmean')
    highpass_workflow.connect(hpf,'out_file',
                              addMean,'in_file')
    highpass_workflow.connect(meanfunc,'out_file',
                              addMean,'operand_file')
    
    
    
    highpass_workflow.base_dir = 'hpf'
    highpass_workflow.write_graph()
    
    highpass_workflow.inputs.inputspec.ICAed_file = os.path.abspath(file_name)
    highpass_workflow.run()
    
    
