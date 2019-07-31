#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:09:21 2019

@author: nmei
"""

from autoreject import (AutoReject,get_rejection_threshold)
import numpy as np
import pandas as pd
import mne
from glob import glob
import re
import os


def preprocessing_conscious(
                  raw,events,
                  n_interpolates = np.arange(1,32,4),
                  consensus_pers = np.linspace(0,1.0,11),
                  event_id = {'living':1,'nonliving':2},
                  tmin = -0.15,
                  tmax = 0.15 * 6,
                  high_pass = 0.1,
                  low_pass = 50,
                  notch_filter = 50):
    """
    Preprocessing pipeline for conscious trials
    
    Inputs
    -------------------
    raw: MNE Raw object, contineous EEG raw data
    events: Numpy array with 3 columns, where the first column indicates time and the last column indicates event code
    n_interpolates: list of values 1 <= N <= max number of channels
    consensus_pers: ?? autoreject hyperparameter search grid
    event_id: MNE argument, to control for epochs
    tmin: first time stamp of the epoch
    tmax: last time stamp of the epoch
    high_pass: low cutoff of the bandpass filter
    low_pass: high cutoff of the bandpass filter
    notch_filter: frequency of the notch filter, 60 in US and 50 in Europe
    
    Output
    -------------------
    Epochs: MNE Epochs object, segmented and cleaned EEG data (n_trials x n_channels x n_times)
    """
    # preprocessing: https://www.biorxiv.org/content/10.1101/518662v1
    """
    0. re-reference - explicitly
    """
    raw_ref ,_  = mne.set_eeg_reference(raw,
                                       ref_channels     = 'average',
                                       projection       = True,)
    raw_ref.apply_proj() # it might tell you it already has been re-referenced, but do it anyway
    try: # cut the portion of the EEG data that is not very useful: before and after the experiment
        raw_ref = raw_ref.crop(events[0][0]     - 10,
                               events[-1][0]    + 10)
    except:
        pass
    """
    1. bandpass filter between 0.1 Hz and 50 Hz 
        by a 4th order zero-phase Butterworth filter
    """
    # everytime before filtering, explicitly pick the type of channels you want
    # to perform the filters
    picks = mne.pick_types(raw_ref.info,
                           meg = False, # No MEG
                           eeg = True,  # YES EEG
                           eog = True,  # YES EOG
                           )
    # regardless the bandpass filtering later, we should always filter
    # for wire artifacts and their oscillations
    raw_ref.notch_filter(np.arange(notch_filter,241,notch_filter),
                         picks = picks)
    # bandpass filtering
    picks = mne.pick_types(raw_ref.info,
                           meg = False, # No MEG
                           eeg = True,  # YES EEG
                           eog = False, # No EOG
                           )
    raw_ref.filter(high_pass,
                   low_pass,
                   picks            = picks,
                   filter_length    = 'auto',    # the filter length is chosen based on the size of the transition regions (6.6 times the reciprocal of the shortest transition band for fir_window=’hamming’ and fir_design=”firwin2”, and half that for “firwin”)
                   method           = 'fir',     # overlap-add FIR filtering
                   phase            = 'zero',    # the delay of this filter is compensated for
                   fir_window       = 'hamming', # The window to use in FIR design
                   fir_design       = 'firwin',  # a time-domain design technique that generally gives improved attenuation using fewer samples than “firwin2”
                   )
    
    """
    2. epoch the data
    """
    picks       = mne.pick_types(raw_ref.info,
                           eeg      = True, # YES EEG
                           eog      = True, # YES EOG
                           )
    epochs      = mne.Epochs(raw_ref,
                             events,    # numpy array
                             event_id,  # dictionary
                        tmin        = tmin,
                        tmax        = tmax,
                        baseline    = (tmin,0), # range of time for computing the mean references for each channel and subtract these values from all the time points per channel
                        picks       = picks,
                        detrend     = 1, # linear detrend
                        preload     = True # must be true if we want to do further processing
                        )
    
    """
    3. ica on epoch data
    """
    # calculate the noise covariance of the epochs
    noise_cov   = mne.compute_covariance(epochs,
                                         tmin                   = tmin,
                                         tmax                   = tmax,
                                         method                 = 'empirical',
                                         rank                   = None,)
    # define an ica function
    ica         = mne.preprocessing.ICA(n_components            = .99,
                                        n_pca_components        = .99,
                                        max_pca_components      = None,
                                        method                  = 'extended-infomax',
                                        max_iter                = int(3e3),
                                        noise_cov               = noise_cov,
                                        random_state            = 12345,)
    # search for a global rejection threshold globally
    reject      = get_rejection_threshold(epochs)
    picks       = mne.pick_types(epochs.info,
                                 eeg = True, # YES EEG
                                 eog = False # NO EOG
                                 ) 
    ica.fit(epochs,
            picks   = picks,
            start   = tmin,
            stop    = tmax,
            reject  = reject, # if some data in a window has values that exceed the rejection threshold, this window will be ignored when computing the ICA
            decim   = 3,
            tstep   = 1. # Length of data chunks for artifact rejection in seconds. It only applies if inst is of type Raw.
            )
    # search for artificial ICAs automatically
    # most of these hyperparameters were used in a unrelated published study
    ica.detect_artifacts(epochs,
                         eog_ch         = ['FT9','FT10','TP9','TP10'],
                         eog_criterion  = 0.4, # arbitary choice
                         skew_criterion = 2,   # arbitary choice
                         kurt_criterion = 2,   # arbitary choice
                         var_criterion  = 2,   # arbitary choice
                         )
    # explicitly search for eog ICAs 
    eog_idx,scores = ica.find_bads_eog(raw_ref,
                            start       = tmin,
                            stop        = tmax,
                            l_freq      = 1,
                            h_freq      = 50,
                            )
    ica.exclude += eog_idx
    
    ica_epochs  = ica.apply(epochs,
                             exclude    = ica.exclude,
                             )
    """
    4. apply autoreject
    """
    picks       = mne.pick_types(ica_epochs.info,
                           eeg          = True, # YES EEG
                           eog          = False # NO EOG
                           )
    ar          = AutoReject(
                    n_interpolate       = n_interpolates,
                    consensus           = consensus_pers,
                    thresh_method       = 'bayesian_optimization',
                    picks               = picks,
                    random_state        = 12345
                    )
    epochs_clean = ar.fit_transform(ica_epochs)
    
    epochs_clean.pick_types(eeg = True,eog = False) # filter out EOG channels
    
    return epochs_clean
def str2num(x):
    return float(re.findall('\d+',x)[0])

def get_frames(directory,new = True,):
    files = glob(os.path.join(directory,'*trials.csv'))
    for ii,f in enumerate(files):
        df = pd.read_csv(f).dropna()
        for vis,df_sub in df.groupby(['visible.keys_raw']):
            try:
                print(f'session {ii+1}, vis = {vis}, n_trials = {df_sub.shape[0]}')
            except:
                print('session {}, vis = {}, n_trials = {}'.format(ii+1,
                      vis,df_sub.shape[0]))
                
    df = pd.concat([pd.read_csv(f).dropna() for f in files])
    
    df['probeFrames_raw'] = df['probeFrames_raw'].apply(str2num)
    for vis,df_sub in df.groupby(['visible.keys_raw']):
        try:
            print(f"vis = {vis},mean frames = {np.median(df_sub['probeFrames_raw']):.5f}")
        except:
            print("vis = {},mean frames = {:.5f}".format(
                    vis,np.median(df_sub['probeFrames_raw'])))
    if new:
        df = []
        for f in files:
            temp = pd.read_csv(f).dropna()
            temp[['probeFrames_raw','visible.keys_raw']]
            probeFrame = []
            for ii,row in temp.iterrows():
                if int(re.findall('\d',row['visible.keys_raw'])[0]) == 1:
                    probeFrame.append(row['probeFrames_raw'])
                elif int(re.findall('\d',row['visible.keys_raw'])[0]) == 2:
                    probeFrame.append(row['probeFrames_raw'])
                elif int(re.findall('\d',row['visible.keys_raw'])[0]) == 3:
                    probeFrame.append(row['probeFrames_raw'])
                elif int(re.findall('\d',row['visible.keys_raw'])[0]) == 4:
                    probeFrame.append(row['probeFrames_raw'])
            temp['probeFrames'] = probeFrame
            df.append(temp)
        df = pd.concat(df)
    else:
        df = []
        for f in files:
            temp = pd.read_csv(f).dropna()
            temp[['probeFrames_raw','visible.keys_raw']]
            probeFrame = []
            for ii,row in temp.iterrows():
                if int(re.findall('\d',row['visible.keys_raw'])[0]) == 1:
                    probeFrame.append(row['probeFrames_raw'] - 2)
                elif int(re.findall('\d',row['visible.keys_raw'])[0]) == 2:
                    probeFrame.append(row['probeFrames_raw'] - 1)
                elif int(re.findall('\d',row['visible.keys_raw'])[0]) == 3:
                    probeFrame.append(row['probeFrames_raw'] + 1)
                elif int(re.findall('\d',row['visible.keys_raw'])[0]) == 4:
                    probeFrame.append(row['probeFrames_raw'] + 2)
            temp['probeFrames'] = probeFrame
            df.append(temp)
        df = pd.concat(df)
    df['probeFrames'] = df['probeFrames'].apply(str2num)
    results = []
    for vis,df_sub in df.groupby(['visible.keys_raw']):
        try:
            print(f"vis = {vis},mean frames = {np.mean(df_sub['probeFrames']):.2f} +/- {np.std(df_sub['probeFrames']):.2f}")
        except:
            print("vis = {},mean frames = {:.2f} +/- {:.2f}".format(
                    vis,np.mean(df_sub['probeFrames']),np.std(df_sub['probeFrames'])))
        results.append([vis,np.mean(df_sub['probeFrames']),np.std(df_sub['probeFrames'])])
    return results

def read(f):
    temp = pd.read_csv(f).iloc[:-12,:]
    return temp

def preload(f):
    temp = pd.read_csv(f).iloc[-12:,:2]
    return temp

def extract(x):
    try:
        return int(re.findall('\d',x)[0])
    except:
        return int(99)

def groupy_average(fmri,df,groupby = ['trials']):
    return np.array([np.mean(fmri[df_sub.index],0) for _,df_sub in df.groupby(groupby)])

def get_brightness_threshold(thresh):
    return [0.5 * (val[1] - val[1] * 0.1) for val in thresh]

def get_brightness_threshold_double(thresh):
    return [1 * (val[1] - val[1] * 0.1) for val in thresh]

def cartesian_product(fwhms, in_files, usans, btthresh):
    from nipype.utils.filemanip import ensure_list
    # ensure all inputs are lists
    in_files                = ensure_list(in_files)
    fwhms                   = [fwhms] if isinstance(fwhms, (int, float)) else fwhms
    # create cartesian product lists (s_<name> = single element of list)
    cart_in_file            = [
            s_in_file for s_in_file in in_files for s_fwhm in fwhms
                                ]
    cart_fwhm               = [
            s_fwhm for s_in_file in in_files for s_fwhm in fwhms
                                ]
    cart_usans              = [
            s_usans for s_usans in usans for s_fwhm in fwhms
                                ]
    cart_btthresh           = [
            s_btthresh for s_btthresh in btthresh for s_fwhm in fwhms
                                ]
    return cart_in_file, cart_fwhm, cart_usans, cart_btthresh

def getusans(x):
    return [[tuple([val[0], 0.5 * val[1]])] for val in x]

def create_fsl_FEAT_workflow_func(whichrun          = 0,
                                  whichvol          = 'middle',
                                  workflow_name     = 'nipype_mimic_FEAT',
                                  first_run         = True,
                                  func_data_file    = 'temp',
                                  fwhm              = 3):
    from nipype.workflows.fmri.fsl             import preprocess
    from nipype.interfaces                     import fsl
    from nipype.interfaces                     import utility as util
    from nipype.pipeline                       import engine as pe
    """
    Setup some functions and hyperparameters
    """
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    pickrun             = preprocess.pickrun
    pickvol             = preprocess.pickvol
    getthreshop         = preprocess.getthreshop
    getmeanscale        = preprocess.getmeanscale
#    chooseindex         = preprocess.chooseindex
    
    """
    Start constructing the workflow graph
    """
    preproc             = pe.Workflow(name = workflow_name)
    """
    Initialize the input and output spaces
    """
    inputnode           = pe.Node(
                        interface   = util.IdentityInterface(fields = ['func',
                                                                       'fwhm',
                                                                       'anat']),
                        name        = 'inputspec')
    outputnode          = pe.Node(
                        interface   = util.IdentityInterface(fields = ['reference',
                                                                       'motion_parameters',
                                                                       'realigned_files',
                                                                       'motion_plots',
                                                                       'mask',
                                                                       'smoothed_files',
                                                                       'mean']),
                        name        = 'outputspec')
    """
    first step: convert Images to float values
    """
    img2float           = pe.MapNode(
                        interface   = fsl.ImageMaths(
                                        out_data_type   = 'float',
                                        op_string       = '',
                                        suffix          = '_dtype'),
                        iterfield   = ['in_file'],
                        name        = 'img2float')
    preproc.connect(inputnode,'func',
                    img2float,'in_file')
    """
    delete first 10 volumes
    """
    develVolume         = pe.MapNode(
                        interface   = fsl.ExtractROI(t_min  = 10,
                                                     t_size = 508),
                        iterfield   = ['in_file'],
                        name        = 'remove_volumes')
    preproc.connect(img2float,      'out_file',
                    develVolume,    'in_file')
    if first_run == True:
        """ 
        extract example fMRI volume: middle one
        """
        extract_ref     = pe.MapNode(
                        interface   = fsl.ExtractROI(t_size = 1,),
                        iterfield   = ['in_file'],
                        name        = 'extractref')
        # connect to the deleteVolume node to get the data
        preproc.connect(develVolume,'roi_file',
                        extract_ref,'in_file')
        # connect to the deleteVolume node again to perform the extraction
        preproc.connect(develVolume,('roi_file',pickvol,0,whichvol),
                        extract_ref,'t_min')
        # connect to the output node to save the reference volume
        preproc.connect(extract_ref,'roi_file',
                        outputnode, 'reference')
    if first_run == True:
        """
        Realign the functional runs to the reference (`whichvol` volume of first run)
        """
        motion_correct  = pe.MapNode(
                        interface   = fsl.MCFLIRT(save_mats     = True,
                                                  save_plots    = True,
                                                  save_rms      = True,
                                                  stats_imgs    = True,
                                                  interpolation = 'spline'),
                        iterfield   = ['in_file','ref_file'],
                        name        = 'MCFlirt',
                                                  )
        # connect to the develVolume node to get the input data
        preproc.connect(develVolume,    'roi_file',
                        motion_correct, 'in_file',)
        ######################################################################################
        #################  the part where we replace the actual reference image if exists ####
        ######################################################################################
        # connect to the develVolume node to get the reference
        preproc.connect(extract_ref,    'roi_file', 
                        motion_correct, 'ref_file')
        ######################################################################################
        # connect to the output node to save the motion correction parameters
        preproc.connect(motion_correct, 'par_file',
                        outputnode,     'motion_parameters')
        # connect to the output node to save the other files
        preproc.connect(motion_correct, 'out_file',
                        outputnode,     'realigned_files')
    else:
        """
        Realign the functional runs to the reference (`whichvol` volume of first run)
        """
        motion_correct      = pe.MapNode(
                            interface   = fsl.MCFLIRT(ref_file      = first_run,
                                                      save_mats     = True,
                                                      save_plots    = True,
                                                      save_rms      = True,
                                                      stats_imgs    = True,
                                                      interpolation = 'spline'),
                            iterfield   = ['in_file','ref_file'],
                            name        = 'MCFlirt',
                        )
        # connect to the develVolume node to get the input data
        preproc.connect(develVolume,    'roi_file',
                        motion_correct, 'in_file',)
        # connect to the output node to save the motion correction parameters
        preproc.connect(motion_correct, 'par_file',
                        outputnode,     'motion_parameters')
        # connect to the output node to save the other files
        preproc.connect(motion_correct, 'out_file',
                        outputnode,     'realigned_files')
    """
    plot the estimated motion parameters
    """
    plot_motion             = pe.MapNode(
                            interface   = fsl.PlotMotionParams(in_source = 'fsl'),
                            iterfield   = ['in_file'],
                            name        = 'plot_motion',
            )
    plot_motion.iterables = ('plot_type',['rotations',
                                          'translations',
                                          'displacement'])
    preproc.connect(motion_correct, 'par_file',
                    plot_motion,    'in_file')
    preproc.connect(plot_motion,    'out_file',
                    outputnode,     'motion_plots')
    """
    extract the mean volume of the first functional run
    """
    meanfunc                = pe.Node(
                            interface  = fsl.ImageMaths(op_string   = '-Tmean',
                                                        suffix      = '_mean',),
                            name        = 'meanfunc')
    preproc.connect(motion_correct, ('out_file',pickrun,whichrun),
                    meanfunc,       'in_file')
    """
    strip the skull from the mean functional to generate a mask
    """
    meanfuncmask            = pe.Node(
                            interface   = fsl.BET(mask        = True,
                                                  no_output   = True,
                                                  frac        = 0.3,
                                                  surfaces    = True,),
                            name        = 'bet2_mean_func')
    preproc.connect(meanfunc,       'out_file',
                    meanfuncmask,   'in_file')
    """
    Mask the motion corrected functional data with the mask to create the masked (bet) motion corrected functional data
    """
    maskfunc                = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix = '_bet',
                                                         op_string = '-mas'),
                            iterfield   = ['in_file'],
                            name        = 'maskfunc')
    preproc.connect(motion_correct, 'out_file',
                    maskfunc,       'in_file')
    preproc.connect(meanfuncmask,   'mask_file',
                    maskfunc,       'in_file2')
    """
    determine the 2nd and 98th percentiles of each functional run
    """
    getthreshold            = pe.MapNode(
                            interface   = fsl.ImageStats(op_string = '-p 2 -p 98'),
                            iterfield   = ['in_file'],
                            name        = 'getthreshold')
    preproc.connect(maskfunc,       'out_file',
                    getthreshold,   'in_file')
    """
    threshold the functional data at 10% of the 98th percentile
    """
    threshold               = pe.MapNode(
                            interface   = fsl.ImageMaths(out_data_type  = 'char',
                                                         suffix         = '_thresh',
                                                         op_string      = '-Tmin -bin'),
                            iterfield   = ['in_file','op_string'],
                            name        = 'tresholding')
    preproc.connect(maskfunc, 'out_file',
                    threshold,'in_file')
    """
    define a function to get 10% of the intensity
    """
    preproc.connect(getthreshold,('out_stat',getthreshop),
                    threshold,    'op_string')
    """
    Determine the median value of the functional runs using the mask
    """
    medianval               = pe.MapNode(
                            interface   = fsl.ImageStats(op_string = '-k %s -p 50'),
                            iterfield   = ['in_file','mask_file'],
                            name        = 'cal_intensity_scale_factor')
    preproc.connect(motion_correct,     'out_file',
                    medianval,          'in_file')
    preproc.connect(threshold,          'out_file',
                    medianval,          'mask_file')
    """
    dilate the mask
    """
    dilatemask              = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix = '_dil',
                                                         op_string = '-dilF'),
                            iterfield   = ['in_file'],
                            name        = 'dilatemask')
    preproc.connect(threshold,  'out_file',
                    dilatemask, 'in_file')
    preproc.connect(dilatemask, 'out_file',
                    outputnode, 'mask')
    """
    mask the motion corrected functional runs with the dilated mask
    """
    dilateMask_MCed         = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix     = '_mask',
                                                         op_string  = '-mas'),
                            iterfield   = ['in_file','in_file2'],
                            name        = 'dilateMask_MCed')
    preproc.connect(motion_correct,     'out_file',
                    dilateMask_MCed,    'in_file',)
    preproc.connect(dilatemask,         'out_file',
                    dilateMask_MCed,    'in_file2')
    """
    We now take this functional data that is motion corrected, high pass filtered, and
    create a "mean_func" image that is the mean across time (Tmean)
    """
    meanfunc2               = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix     = '_mean',
                                                         op_string  = '-Tmean',),
                            iterfield   = ['in_file'],
                            name        = 'meanfunc2')
    preproc.connect(dilateMask_MCed,    'out_file',
                    meanfunc2,          'in_file')
    """
    smooth each run using SUSAN with the brightness threshold set to 
    50% of the median value for each run and a mask constituing the 
    mean functional
    """
    merge                   = pe.Node(
                            interface   = util.Merge(2, axis = 'hstack'), 
                            name        = 'merge')
    preproc.connect(meanfunc2,  'out_file', 
                    merge,      'in1')
    preproc.connect(getthreshold,('out_stat',get_brightness_threshold_double), 
                    merge,      'in2')
    smooth                  = pe.MapNode(
                            interface   = fsl.SUSAN(dimension   = 3,
                                                    use_median  = True),
                            iterfield   = ['in_file',
                                           'brightness_threshold',
                                           'fwhm',
                                           'usans'],
                            name        = 'susan_smooth')
    preproc.connect(dilateMask_MCed,    'out_file', 
                    smooth,             'in_file')
    preproc.connect(getthreshold,       ('out_stat',get_brightness_threshold),
                    smooth,             'brightness_threshold')
    preproc.connect(inputnode,          'fwhm', 
                    smooth,             'fwhm')
    preproc.connect(merge,              ('out',getusans),
                    smooth,             'usans')
    """
    mask the smoothed data with the dilated mask
    """
    maskfunc3               = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix     = '_mask',
                                                         op_string  = '-mas'),
                            iterfield   = ['in_file','in_file2'],
                            name        = 'dilateMask_smoothed')
    # connect the output of the susam smooth component to the maskfunc3 node
    preproc.connect(smooth,     'smoothed_file',
                    maskfunc3,  'in_file')
    # connect the output of the dilated mask to the maskfunc3 node
    preproc.connect(dilatemask, 'out_file',
                    maskfunc3,  'in_file2')
    """
    scale the median value of the run is set to 10000
    """
    meanscale               = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix = '_intnorm'),
                            iterfield   = ['in_file','op_string'],
                            name        = 'meanscale')
    preproc.connect(maskfunc3, 'out_file',
                    meanscale, 'in_file')
    preproc.connect(meanscale, 'out_file',
                    outputnode,'smoothed_files')
    """
    define a function to get the scaling factor for intensity normalization
    """
    preproc.connect(medianval,('out_stat',getmeanscale),
                    meanscale,'op_string')
    """
    generate a mean functional image from the first run
    should this be the 'mean.nii.gz' we will use in the future?
    """
    meanfunc3               = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix     = '_mean',
                                                         op_string  = '-Tmean',),
                            iterfield   = ['in_file'],
                            name        = 'gen_mean_func_img')
    preproc.connect(meanscale, 'out_file',
                    meanfunc3, 'in_file')
    preproc.connect(meanfunc3, 'out_file',
                    outputnode,'mean')
    
    
    # initialize some of the input files
    preproc.inputs.inputspec.func       = os.path.abspath(func_data_file)
    preproc.inputs.inputspec.fwhm       = 3
    preproc.base_dir                    = os.path.abspath('/'.join(
                                        func_data_file.split('/')[:-1]))
    
    output_dir                          = os.path.abspath(os.path.join(
                                        preproc.base_dir,
                                        'outputs',
                                        'func'))
    MC_dir                              = os.path.join(output_dir,'MC')
    for directories in [output_dir,MC_dir]:
        if not os.path.exists(directories):
            os.makedirs(directories)
    
    # initialize all the output files
    if first_run == True:
        preproc.inputs.extractref.roi_file      = os.path.abspath(os.path.join(
                output_dir,'example_func.nii.gz'))
    
    preproc.inputs.dilatemask.out_file          = os.path.abspath(os.path.join(
                output_dir,'mask.nii.gz'))
    preproc.inputs.meanscale.out_file           = os.path.abspath(os.path.join(
                output_dir,'prefiltered_func.nii.gz'))
    preproc.inputs.gen_mean_func_img.out_file   = os.path.abspath(os.path.join(
                output_dir,'mean_func.nii.gz'))
    
    return preproc,MC_dir,output_dir


def create_registration_workflow(
                                 anat_brain,
                                 anat_head,
                                 func_ref,
                                 standard_brain,
                                 standard_head,
                                 standard_mask,
                                 workflow_name = 'registration',
                                 output_dir = 'temp'):
    from nipype.interfaces          import fsl
    from nipype.interfaces         import utility as util
    from nipype.pipeline           import engine as pe
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    """
    Start constructing the workflow graph
    """
    registration                    = pe.Workflow(name = 'registration')
    """
    Initialize the input and output spaces
    """
    inputnode                       = pe.Node(
                                    interface   = util.IdentityInterface(
                                    fields      = [
                                            'anat_brain',
                                            'anat_head',
                                            'func_ref',
                                            'standard_brain',
                                            'standard_head',
                                            'standard_mask'
                                            ]),
                                    name        = 'inputspec')
    outputnode                      = pe.Node(
                                    interface   = util.IdentityInterface(
                                    fields      = [
                                            'example2highres_FLIRT_mat',
                                            'example2highres_FLIRT_log',
                                            'example2highres_FLIRT_out_file',
                                            'highres2example_func_mat',
                                            'highres2standard_FLIRT_out_file',
                                            'highres2standard_FLIRT_mat',
                                            'highres2standard_FLIRT_log',
                                            'highres2standard_warp',
                                            'highres2standard_gz',
                                            'highres2highres_jac',
                                            'highres2highres_log',
                                            'highres2standard_mat',
                                            'example_func2standard_mat',
                                            'example_func2standard_warp',
                                            'example_func2standard',
                                            'standard2example_func_mat',
                                            ]),
                                    name        = 'outputspec')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/flirt 
        -in example_func 
        -ref highres 
        -out example_func2highres 
        -omat example_func2highres.mat 
        -cost corratio 
        -dof 7 
        -searchrx -180 180 
        -searchry -180 180 
        -searchrz -180 180 
        -interp trilinear 
    """
    example2highres_flirt           = pe.MapNode(
                                    interface   = fsl.FLIRT(cost          = 'corratio',
                                                            interp        = 'trilinear',
                                                            dof           = 7,
                                                            save_log      = True,
                                                            searchr_x     = [180, 180],
                                                            searchr_y     = [180, 180],
                                                            searchr_z     = [180, 180],),
                                    iterfield   = ['in_file','reference'],
                                    name        = 'example2highres_flirt')
    registration.connect(inputnode,             'func_ref',
                         example2highres_flirt, 'in_file')
    registration.connect(inputnode,             'anat_brain',
                         example2highres_flirt, 'reference')
    registration.connect(example2highres_flirt, 'out_file',
                         outputnode,            'example2highres_FLIRT_out_file')
    registration.connect(example2highres_flirt, 'out_matrix_file',
                         outputnode,            'example2highres_FLIRT_mat')
    registration.connect(example2highres_flirt, 'out_log',
                         outputnode,            'example2highres_FLIRT_log')
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat highres2example_func.mat example_func2highres.mat
    """
    inverse_example2highres         = pe.MapNode(
                                    interface   = fsl.ConvertXFM(invert_xfm = True),
                                    iterfield   = ['in_file',],
                                    name        = 'inverse_example2highres')
    registration.connect(example2highres_flirt,  'out_matrix_file',
                         inverse_example2highres,'in_file')
    registration.connect(inverse_example2highres,'out_file',
                         outputnode,             'highres2example_func_mat')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/flirt 
        -in highres 
        -ref standard 
        -out highres2standard 
        -omat highres2standard.mat 
        -cost corratio 
        -dof 12 
        -searchrx -180 180 
        -searchry -180 180 
        -searchrz -180 180 
        -interp trilinear 
    """
    highres2standard_flirt                  = pe.MapNode(
                                            interface   = fsl.FLIRT(cost        = 'corratio',
                                                                    interp      = 'trilinear',
                                                                    dof         = 12,
                                                                    save_log    = True,
                                                                    searchr_x   = [180, 180],
                                                                    searchr_y   = [180, 180],
                                                                    searchr_z   = [180, 180],),
                                            iterfield   = ['in_file','reference'],
                                            name        = 'highres2standard_flirt')
    registration.connect(inputnode,                 'anat_brain',
                         highres2standard_flirt,    'in_file')
    registration.connect(inputnode,                 'standard_brain',
                         highres2standard_flirt,    'reference')
    registration.connect(highres2standard_flirt,    'out_file',
                         outputnode,                'highres2standard_FLIRT_out_file')
    registration.connect(highres2standard_flirt,    'out_matrix_file',
                         outputnode,                'highres2standard_FLIRT_mat')
    registration.connect(highres2standard_flirt,    'out_log',
                         outputnode,                'highres2standard_FLIRT_log')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/fnirt 
        --iout=highres2standard_head 
        --in=highres_head 
        --aff=highres2standard.mat 
        --cout=highres2standard_warp 
        --iout=highres2standard 
        --jout=highres2highres_jac 
        --config=T1_2_MNI152_2mm 
        --ref=standard_head 
        --refmask=standard_mask 
        --warpres=10,10,10
    """
    highres2standard_fnirt                  = pe.MapNode(
                                            interface   = fsl.FNIRT(warp_resolution = (10,10,10),
                                                                    config_file     = 'T1_2_MNI152_2mm'),
                                            iterfield   = ['in_file','affine_file',],
                                            name        = 'highres2standard_fnirt')
    registration.connect(inputnode,             'anat_head',
                         highres2standard_fnirt,'in_file') # <- nonlinear
    registration.connect(inputnode,             'standard_head',
                         highres2standard_fnirt,'ref_file',) # <- nonlinear
    registration.connect(highres2standard_flirt,'out_matrix_file', # <- linear
                         highres2standard_fnirt,'affine_file') # <- nonlinear
    registration.connect(highres2standard_fnirt,'fieldcoeff_file', # <- nonlinear
                         outputnode,            'highres2standard_warp')
    registration.connect(highres2standard_fnirt,'warped_file', # <- nonlinear
                         outputnode,            'highres2standard_gz_flirt')
    registration.connect(highres2standard_fnirt,'jacobian_file', # <- nonlinear
                         outputnode,            'highres2highres_jac')
    registration.connect(inputnode,             'standard_mask',
                         highres2standard_fnirt,'refmask_file') # <- nonlinear
    registration.connect(highres2standard_fnirt,'log_file', # <- nonlinear
                         outputnode,            'highres2standard_log')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/applywarp 
        -i highres 
        -r standard 
        -o highres2standard 
        -w highres2standard_warp
    """
    warp_anat_brain                         = pe.MapNode(
                                            interface   = fsl.ApplyWarp(),
                                            iterfield   = ['in_file',
                                                           'ref_file',
                                                           'field_file'],
                                            name        = 'warp_anat_brain')
    registration.connect(inputnode,             'anat_brain',
                         warp_anat_brain,       'in_file')
    registration.connect(inputnode,             'standard_brain',
                         warp_anat_brain,       'ref_file',)
    registration.connect(highres2standard_fnirt,'fieldcoeff_file',
                         warp_anat_brain,       'field_file')
    registration.connect(warp_anat_brain,       'out_file',
                         outputnode,            'highres2standard_gz_warp')
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat standard2highres.mat highres2standard.mat
    """
    inverse_highres2standard            = pe.MapNode(
                                        interface   = fsl.ConvertXFM(invert_xfm = True),
                                        iterfield   = ['in_file',],
                                        name        = 'inverse_highres2standard')
    registration.connect(highres2standard_flirt,    'out_matrix_file',
                         inverse_highres2standard,  'in_file')
    registration.connect(inverse_example2highres,   'out_file',
                         outputnode,                'highres2standard_mat')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -omat example_func2standard.mat -concat highres2standard.mat example_func2highres.mat
    """
    example2standard                    = pe.MapNode(
                                        interface   = fsl.ConvertXFM(concat_xfm = True),
                                        iterfield   = ['in_file','in_file2',],
                                        name        = 'example2standard')
    registration.connect(highres2standard_flirt,    'out_matrix_file',
                         example2standard,          'in_file')
    registration.connect(example2highres_flirt,     'out_matrix_file',
                         example2standard,          'in_file2')
    registration.connect(example2standard,          'out_file',
                         outputnode,                'example_func2standard_mat')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convertwarp 
        --ref=standard 
        --premat=example_func2highres.mat 
        --warp1=highres2standard_warp 
        --out=example_func2standard_warp
    """
    convertwarp_standard_brain          = pe.MapNode(
                                        interface   = fsl.ConvertWarp(),
                                        iterfield   = ['reference',
                                                       'premat',
                                                       'warp1'],
                                        name        = 'convertwarp_standard_brain')
    registration.connect(inputnode,                 'standard_brain',
                         convertwarp_standard_brain,'reference')
    registration.connect(example2highres_flirt,     'out_matrix_file',
                         convertwarp_standard_brain,'premat')
    registration.connect(highres2standard_fnirt,    'fieldcoeff_file',
                         convertwarp_standard_brain,'warp1')
    registration.connect(convertwarp_standard_brain,'out_file',
                         outputnode,                'example_func2standard_warp')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/applywarp 
        --ref=standard 
        --in=example_func 
        --out=example_func2standard 
        --warp=example_func2standard_warp
    """
    warp_standard_brain                 = pe.MapNode(
                                        interface   = fsl.ApplyWarp(),
                                        iterfield   = ['in_file','ref_file','field_file'],
                                        name        = 'warp_standard_brain')
    registration.connect(inputnode,                 'standard_brain',
                         warp_standard_brain,       'ref_file')
    registration.connect(inputnode,                 'func_ref',
                         warp_standard_brain,       'in_file')
    registration.connect(warp_standard_brain,       'out_file',
                         outputnode,                'example_func2standard')
    registration.connect(convertwarp_standard_brain,'out_file',
                         warp_standard_brain,       'field_file')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat standard2example_func.mat example_func2standard.mat
    """
    inverse_example2standard            = pe.MapNode(
                                        interface   = fsl.ConvertXFM(invert_xfm = True),
                                        iterfield   = ['in_file',],
                                        name        = 'inverse_example2standard')
    registration.connect(example2standard,          'out_file',
                         inverse_example2standard,  'in_file')
    registration.connect(inverse_example2standard,  'out_file',
                         outputnode,                'standard2example_func_mat')
    
    # initialize some of the input files with the directory
    registration.base_dir                                       = os.path.abspath(output_dir)
    registration.inputs.inputspec.anat_brain                    = anat_brain
    registration.inputs.inputspec.anat_head                     = anat_head
    registration.inputs.inputspec.func_ref                      = func_ref
    registration.inputs.inputspec.standard_brain                = standard_brain
    registration.inputs.inputspec.standard_head                 = standard_head
    registration.inputs.inputspec.standard_mask                 = standard_mask
    
    # define all the oupput file names with the directory
    registration.inputs.example2highres_flirt.out_file          = os.path.abspath(os.path.join(output_dir,'example_func2highres.nii.gz'))
    registration.inputs.example2highres_flirt.out_matrix_file   = os.path.abspath(os.path.join(output_dir,'example_func2highres.mat'))
    registration.inputs.example2highres_flirt.out_log           = os.path.abspath(os.path.join(output_dir,'example_func2highres.log'))
    registration.inputs.inverse_example2highres.out_file        = os.path.abspath(os.path.join(output_dir,'highres2example_func.mat'))
    registration.inputs.highres2standard_flirt.out_file         = os.path.abspath(os.path.join(output_dir,'highres2standard.nii.gz'))
    registration.inputs.highres2standard_flirt.out_matrix_file  = os.path.abspath(os.path.join(output_dir,'highres2standard.mat'))
    registration.inputs.highres2standard_flirt.out_log          = os.path.abspath(os.path.join(output_dir,'highres2standard.log'))
    registration.inputs.highres2standard_fnirt.fieldcoeff_file  = os.path.abspath(os.path.join(output_dir,'highres2standard_warp.nii.gz'))
    registration.inputs.highres2standard_fnirt.jacobian_file    = os.path.abspath(os.path.join(output_dir,'highres2highres_jac.nii.gz'))
    registration.inputs.warp_anat_brain.out_file                = os.path.abspath(os.path.join(output_dir,'highres2standard.nii.gz'))
    registration.inputs.inverse_highres2standard.out_file       = os.path.abspath(os.path.join(output_dir,'standard2highres.mat'))
    registration.inputs.example2standard.out_file               = os.path.abspath(os.path.join(output_dir,'example_func2standard.mat'))
    registration.inputs.convertwarp_standard_brain.out_file     = os.path.abspath(os.path.join(output_dir,'example_func2standard_warp.nii.gz'))
    registration.inputs.inverse_example2standard.out_file       = os.path.abspath(os.path.join(output_dir,'standard2example_func.mat'))
    #registration.inputs.highres2standard_fnirt.warped_file = os.path.abspath(os.path.join(output_dir,
    #                                                               'highres2standard.nii.gz'))
    #registration.inputs.warp_standard_brain.out_file = os.path.abspath(os.path.join(output_dir,
    #                                                  "example_func2standard.nii.gz"))
    return registration

def create_simple_struc2BOLD(roi,
                             roi_name,
                             preprocessed_functional_dir,
                             output_dir):
    from nipype.interfaces            import fsl
    from nipype.pipeline              import engine as pe
    from nipype.interfaces            import utility as util
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    
    simple_workflow         = pe.Workflow(name  = 'struc2BOLD')
    
    inputnode               = pe.Node(interface = util.IdentityInterface(
                                      fields    = ['flt_in_file',
                                                   'flt_in_matrix',
                                                   'flt_reference',
                                                   'mask']),
                                      name      = 'inputspec')
    outputnode              = pe.Node(interface = util.IdentityInterface(
                                      fields    = ['BODL_mask']),
                                      name      = 'outputspec')
    """
     flirt 
 -in /export/home/dsoto/dsoto/fmri/$s/sess2/label/$i 
 -ref /export/home/dsoto/dsoto/fmri/$s/sess2/run1_prepro1.feat/example_func.nii.gz  
 -applyxfm 
 -init /export/home/dsoto/dsoto/fmri/$s/sess2/run1_prepro1.feat/reg/highres2example_func.mat 
 -out  /export/home/dsoto/dsoto/fmri/$s/label/BOLD${i}
    """
    flirt_convert           = pe.MapNode(
                                    interface   = fsl.FLIRT(apply_xfm = True),
                                    iterfield   = ['in_file',
                                                   'reference',
                                                   'in_matrix_file'],
                                    name        = 'flirt_convert')
    simple_workflow.connect(inputnode,      'flt_in_file',
                            flirt_convert,  'in_file')
    simple_workflow.connect(inputnode,      'flt_reference',
                            flirt_convert,  'reference')
    simple_workflow.connect(inputnode,      'flt_in_matrix',
                            flirt_convert,  'in_matrix_file')
    
    """
     fslmaths /export/home/dsoto/dsoto/fmri/$s/label/BOLD${i} -mul 2 
     -thr `fslstats /export/home/dsoto/dsoto/fmri/$s/label/BOLD${i} -p 99.6` 
    -bin /export/home/dsoto/dsoto/fmri/$s/label/BOLD${i}
    """
    def getthreshop(thresh):
        return ['-mul 2 -thr %.10f -bin' % (val) for val in thresh]
    getthreshold            = pe.MapNode(
                                    interface   = fsl.ImageStats(op_string='-p 99.6'),
                                    iterfield   = ['in_file','mask_file'],
                                    name        = 'getthreshold')
    simple_workflow.connect(flirt_convert,  'out_file',
                            getthreshold,   'in_file')
    simple_workflow.connect(inputnode,      'mask',
                            getthreshold,   'mask_file')
    
    threshold               = pe.MapNode(
                                    interface   = fsl.ImageMaths(
                                            suffix      = '_thresh',
                                            op_string   = '-mul 2 -bin'),
                                    iterfield   = ['in_file','op_string'],
                                    name        = 'thresholding')
    simple_workflow.connect(flirt_convert,  'out_file',
                            threshold,      'in_file')
    simple_workflow.connect(getthreshold,   ('out_stat',getthreshop),
                            threshold,      'op_string')
#    simple_workflow.connect(threshold,'out_file',outputnode,'BOLD_mask')
    
    bound_by_mask           = pe.MapNode(
                                    interface   = fsl.ImageMaths(
                                            suffix      = '_mask',
                                            op_string   = '-mas'),
                                    iterfield   = ['in_file','in_file2'],
                                    name        = 'bound_by_mask')
    simple_workflow.connect(threshold,      'out_file',
                            bound_by_mask,  'in_file')
    simple_workflow.connect(inputnode,      'mask',
                            bound_by_mask,  'in_file2')
    simple_workflow.connect(bound_by_mask,  'out_file',
                            outputnode,     'BOLD_mask')
    
    # setup inputspecs 
    simple_workflow.inputs.inputspec.flt_in_file    = roi
    simple_workflow.inputs.inputspec.flt_in_matrix  = os.path.abspath(os.path.join(preprocessed_functional_dir,
                                                        'reg',
                                                        'highres2example_func.mat'))
    simple_workflow.inputs.inputspec.flt_reference  = os.path.abspath(os.path.join(preprocessed_functional_dir,
                                                        'func',
                                                        'example_func.nii.gz'))
    simple_workflow.inputs.inputspec.mask           = os.path.abspath(os.path.join(preprocessed_functional_dir,
                                                        'func',
                                                        'mask.nii.gz'))
    simple_workflow.inputs.bound_by_mask.out_file   = os.path.abspath(os.path.join(output_dir,
                                                         roi_name.replace('_fsl.nii.gz',
                                                                          '_BOLD.nii.gz')))
    return simple_workflow

def registration_plotting(output_dir,
                          anat_brain,
                          standard_brain):
    ######################
    ###### plotting ######
    try:
        example_func2highres    = os.path.abspath(os.path.join(output_dir,
                                                'example_func2highres'))
        example_func2standard   = os.path.abspath(os.path.join(output_dir,
                                                 'example_func2standard_warp'))
        highres2standard        = os.path.abspath(os.path.join(output_dir,
                                                 'highres2standard'))
        highres                 = os.path.abspath(anat_brain)
        standard                = os.path.abspath(standard_brain)
        
        plot_example_func2highres   = f"""
        /opt/fsl/fsl-5.0.10/fsl/bin/slicer {example_func2highres} {highres} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2highres}1.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/slicer {highres} {example_func2highres} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2highres}2.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend {example_func2highres}1.png - {example_func2highres}2.png {example_func2highres}.png; 
        /bin/rm -f sl?.png {example_func2highres}2.png
        /bin/rm {example_func2highres}1.png
        """
        
        plot_highres2standard       = f"""
        /opt/fsl/fsl-5.0.10/fsl/bin/slicer {highres2standard} {standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {highres2standard}1.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/slicer {standard} {highres2standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {highres2standard}2.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend {highres2standard}1.png - {highres2standard}2.png {highres2standard}.png; 
        /bin/rm -f sl?.png {highres2standard}2.png
        /bin/rm {highres2standard}1.png
        """
        
        plot_example_func2standard  = f"""
        /opt/fsl/fsl-5.0.10/fsl/bin/slicer {example_func2standard} {standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2standard}1.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/slicer {standard} {example_func2standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2standard}2.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend {example_func2standard}1.png - {example_func2standard}2.png {example_func2standard}.png; 
        /bin/rm -f sl?.png {example_func2standard}2.png
        """
        for cmdline in [plot_example_func2highres,
                        plot_example_func2standard,
                        plot_highres2standard]:
            os.system(cmdline)
    except:
        print('you should not use python 2.7, update your python!!')

def create_highpass_filter_workflow(workflow_name = 'highpassfiler',
                                    HP_freq = 60,
                                    TR = 0.85):
    from nipype.workflows.fmri.fsl    import preprocess
    from nipype.interfaces            import fsl
    from nipype.pipeline              import engine as pe
    from nipype.interfaces            import utility as util
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    getthreshop         = preprocess.getthreshop
    getmeanscale        = preprocess.getmeanscale
    highpass_workflow = pe.Workflow(name = workflow_name)
    
    inputnode               = pe.Node(interface = util.IdentityInterface(
                                      fields    = ['ICAed_file',]),
                                      name      = 'inputspec')
    outputnode              = pe.Node(interface = util.IdentityInterface(
                                      fields    = ['filtered_file']),
                                      name      = 'outputspec')
    
    img2float = pe.MapNode(interface    = fsl.ImageMaths(out_data_type     = 'float',
                                                         op_string         = '',
                                                         suffix            = '_dtype'),
                           iterfield    = ['in_file'],
                           name         = 'img2float')
    highpass_workflow.connect(inputnode,'ICAed_file',
                              img2float,'in_file')
    
    getthreshold = pe.MapNode(interface     = fsl.ImageStats(op_string = '-p 2 -p 98'),
                              iterfield     = ['in_file'],
                              name          = 'getthreshold')
    highpass_workflow.connect(img2float,    'out_file',
                              getthreshold, 'in_file')
    thresholding = pe.MapNode(interface     = fsl.ImageMaths(out_data_type  = 'char',
                                                             suffix         = '_thresh',
                                                             op_string      = '-Tmin -bin'),
                                iterfield   = ['in_file','op_string'],
                                name        = 'thresholding')
    highpass_workflow.connect(img2float,    'out_file',
                              thresholding, 'in_file')
    highpass_workflow.connect(getthreshold,('out_stat',getthreshop),
                              thresholding,'op_string')
    
    dilatemask = pe.MapNode(interface   = fsl.ImageMaths(suffix     = '_dil',
                                                         op_string  = '-dilF'),
                            iterfield   = ['in_file'],
                            name        = 'dilatemask')
    highpass_workflow.connect(thresholding,'out_file',
                              dilatemask,'in_file')
    
    maskfunc = pe.MapNode(interface     = fsl.ImageMaths(suffix     = '_mask',
                                                         op_string  = '-mas'),
                          iterfield     = ['in_file','in_file2'],
                          name          = 'apply_dilatemask')
    highpass_workflow.connect(img2float,    'out_file',
                              maskfunc,     'in_file')
    highpass_workflow.connect(dilatemask,   'out_file',
                              maskfunc,     'in_file2')
    
    medianval = pe.MapNode(interface    = fsl.ImageStats(op_string = '-k %s -p 50'),
                           iterfield    = ['in_file','mask_file'],
                           name         = 'cal_intensity_scale_factor')
    highpass_workflow.connect(img2float,    'out_file',
                              medianval,    'in_file')
    highpass_workflow.connect(thresholding, 'out_file',
                              medianval,    'mask_file')
    
    meanscale = pe.MapNode(interface    = fsl.ImageMaths(suffix = '_intnorm'),
                           iterfield    = ['in_file','op_string'],
                           name         = 'meanscale')
    highpass_workflow.connect(maskfunc,     'out_file',
                              meanscale,    'in_file')
    highpass_workflow.connect(medianval,    ('out_stat',getmeanscale),
                              meanscale,    'op_string')
    
    meanfunc = pe.MapNode(interface     = fsl.ImageMaths(suffix     = '_mean',
                                                         op_string  = '-Tmean'),
                           iterfield    = ['in_file'],
                           name         = 'meanfunc')
    highpass_workflow.connect(meanscale, 'out_file',
                              meanfunc,  'in_file')
    
    
    hpf = pe.MapNode(interface  = fsl.ImageMaths(suffix     = '_tempfilt',
                                                 op_string  = '-bptf %.10f -1' % (HP_freq/2/TR)),
                     iterfield  = ['in_file'],
                     name       = 'highpass_filering')
    highpass_workflow.connect(meanscale,'out_file',
                              hpf,      'in_file',)
    
    addMean = pe.MapNode(interface  = fsl.BinaryMaths(operation = 'add'),
                         iterfield  = ['in_file','operand_file'],
                         name       = 'addmean')
    highpass_workflow.connect(hpf,      'out_file',
                              addMean,  'in_file')
    highpass_workflow.connect(meanfunc, 'out_file',
                              addMean,  'operand_file')
    
    highpass_workflow.connect(addMean,      'out_file',
                              outputnode,   'filtered_file')
    
    return highpass_workflow































#def _create_fsl_FEAT_workflow_func(whichrun = 0,
#                                   whichvol = 'middle',
#                                   workflow_name = 'nipype_mimic_FEAT',
#                                   first_run = True):
#    from nipype.workflows.fmri.fsl import preprocess
#    from nipype.interfaces import fsl
#    from nipype.interfaces import utility as util
#    from nipype.pipeline import engine as pe
#    
#    """
#    Setup some functions and hyperparameters
#    """
#    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
#    pickrun = preprocess.pickrun
#    pickvol = preprocess.pickvol
#    getthreshop = preprocess.getthreshop
#    getmeanscale = preprocess.getmeanscale
#    create_susan_smooth = preprocess.create_susan_smooth
##    chooseindex = preprocess.chooseindex
#    
#    """
#    Start constructing the workflow graph
#    """
#    preproc = pe.Workflow(name = workflow_name)
#    """
#    Initialize the input and output spaces
#    """
#    inputnode = pe.Node(
#            interface = util.IdentityInterface(
#                    fields=['func','fwhm','anat']),
#                    name = 'inputspec')
#    outputnode = pe.Node(
#            interface = util.IdentityInterface(
#                    fields = [
#                            'reference',
#                            'motion_parameters',
#                            'realigned_files',
#                            'motion_plots',
#                            'mask',
#                            'smoothed_files',
#                            'mean']),
#                    name = 'outputspec')
#    """
#    first step: convert Images to float values
#    """
#    img2float = pe.MapNode(
#            interface=fsl.ImageMaths(
#                    out_data_type='float',op_string='',suffix='_dtype'),
#                    iterfield=['in_file'],
#                    name = 'img2float')
#    preproc.connect(inputnode,'func',img2float,'in_file')
#    """
#    delete first 10 volumes
#    """
#    develVolume = pe.MapNode(
#            interface = fsl.ExtractROI(t_min = 10,t_size = 508),
#            iterfield = ['in_file'],
#            name = 'remove_volumes')
#    preproc.connect(img2float,'out_file',develVolume,'in_file')
#    """ 
#    extract example fMRI volume: middle one
#    """
#    extract_ref = pe.MapNode(interface=fsl.ExtractROI(t_size=1,),
#                          iterfield=['in_file'],
#                          name = 'extractref')
#    # connect to the deleteVolume node to get the data
#    preproc.connect(develVolume,'roi_file',
#                    extract_ref,'in_file')
#    # connect to the deleteVolume node again to perform the extraction
#    preproc.connect(develVolume,('roi_file',pickvol,0,whichvol),
#                    extract_ref,'t_min')
#    # connect to the output node to save the reference volume
#    preproc.connect(extract_ref,'roi_file',
#                    outputnode,'reference')
#    if first_run == True:
#        """
#        Realign the functional runs to the reference (`whichvol` volume of first run)
#        """
#        motion_correct = pe.MapNode(
#                interface = fsl.MCFLIRT(
#                        save_mats = True,
#                        save_plots = True,
#                        save_rms = True,
#                        stats_imgs = True,
#                        interpolation = 'spline'),
#                        iterfield = ['in_file','ref_file'],
#                        name = 'MCFlirt',
#                        )
#        # connect to the develVolume node to get the input data
#        preproc.connect(develVolume,'roi_file',
#                        motion_correct,'in_file',)
#        ######################################################################################
#        #################  the part where we replace the actual reference image if exists ####
#        ######################################################################################
#        # connect to the develVolume node to get the reference
#        preproc.connect(extract_ref, 'roi_file', 
#                        motion_correct,'ref_file')
#        ######################################################################################
#        # connect to the output node to save the motion correction parameters
#        preproc.connect(motion_correct,'par_file',
#                        outputnode,'motion_parameters')
#        # connect to the output node to save the other files
#        preproc.connect(motion_correct,'out_file',
#                        outputnode,'realigned_files')
#    else:
#        """
#        Realign the functional runs to the reference (`whichvol` volume of first run)
#        """
#        motion_correct = pe.MapNode(
#                interface = fsl.MCFLIRT(
#                        ref_file = first_run,
#                        save_mats = True,
#                        save_plots = True,
#                        save_rms = True,
#                        stats_imgs = True,
#                        interpolation = 'spline'),
#                        iterfield = ['in_file','ref_file'],
#                        name = 'MCFlirt',
#                        )
#        # connect to the develVolume node to get the input data
#        preproc.connect(develVolume,'roi_file',
#                        motion_correct,'in_file',)
#        # connect to the output node to save the motion correction parameters
#        preproc.connect(motion_correct,'par_file',
#                        outputnode,'motion_parameters')
#        # connect to the output node to save the other files
#        preproc.connect(motion_correct,'out_file',
#                        outputnode,'realigned_files')
#    """
#    plot the estimated motion parameters
#    """
#    plot_motion = pe.MapNode(
#            interface=fsl.PlotMotionParams(in_source='fsl'),
#            iterfield = ['in_file'],
#            name = 'plot_motion',
#            )
#    plot_motion.iterables = ('plot_type',['rotations','translations'])
#    preproc.connect(motion_correct,'par_file',
#                    plot_motion,'in_file')
#    preproc.connect(plot_motion,'out_file',
#                    outputnode,'motion_plots')
#    """
#    extract the mean volume of the first functional run
#    """
#    meanfunc = pe.Node(
#            interface=fsl.ImageMaths(op_string = '-Tmean',
#                                     suffix='_mean',
#                                     ),
#            name = 'meanfunc')
#    preproc.connect(motion_correct,('out_file',pickrun,whichrun),
#                    meanfunc,'in_file')
#    """
#    strip the skull from the mean functional to generate a mask
#    """
#    meanfuncmask = pe.Node(
#            interface=fsl.BET(mask=True,
#                              no_output=True,
#                              frac=0.3,
##                              Robust=True,
#                              ),
#            name='bet2_mean_func')
#    preproc.connect(meanfunc,'out_file',
#                    meanfuncmask,'in_file')
#    """
#    mask the motion corrected functional runs with the extracted mask
#    """
#    maskfunc = pe.MapNode(
#            interface=fsl.ImageMaths(suffix='_bet',op_string='-mas'),
#            iterfield=['in_file'],
#            name='maskfunc')
#    preproc.connect(motion_correct,'out_file',
#                    maskfunc,'in_file')
#    preproc.connect(meanfuncmask,'mask_file',
#                    maskfunc,'in_file2')
#    """
#    determine the 2nd and 98th percentiles of each functional run
#    """
#    getthreshold = pe.MapNode(
#            interface=fsl.ImageStats(op_string='-p 2 -p 98'),
#            iterfield = ['in_file'],
#            name='getthreshold')
#    preproc.connect(maskfunc,'out_file',getthreshold,'in_file')
#    """
#    threshold the first run of the functional data at 10% of the 98th percentile
#    """
#    threshold = pe.MapNode(
#            interface=fsl.ImageMaths(out_data_type='char',suffix='_thresh',
#                                     op_string = '-Tmin -bin'),
#            iterfield=['in_file','op_string'],
#            name='tresholding')
#    preproc.connect(maskfunc,'out_file',threshold,'in_file')
#    """
#    define a function to get 10% of the intensity
#    """
#    preproc.connect(getthreshold,('out_stat',getthreshop),threshold,
#                    'op_string')
#    """
#    Determine the median value of the functional runs using the mask
#    """
#    medianval = pe.MapNode(
#            interface = fsl.ImageStats(op_string = '-k %s -p 50'),
#            iterfield = ['in_file','mask_file'],
#            name='cal_intensity_scale_factor')
#    preproc.connect(motion_correct,'out_file',
#                    medianval,'in_file')
#    preproc.connect(maskfunc,'out_file',
#                    medianval,'mask_file')
#    """
#    dilate the mask
#    """
#    dilatemask = pe.MapNode(
#            interface = fsl.ImageMaths(suffix='_dil',op_string='-dilF'),
#            iterfield=['in_file'],
#            name = 'dilatemask')
#    preproc.connect(threshold,'out_file',dilatemask,'in_file')
#    preproc.connect(dilatemask,'out_file',outputnode,'mask')
#    """
#    mask the motion corrected functional runs with the dilated mask
#    """
#    maskfunc2 = pe.MapNode(
#            interface = fsl.ImageMaths(suffix='_mask',op_string='-mas'),
#            iterfield=['in_file','in_file2'],
#            name='dilateMask_MCed')
#    preproc.connect(motion_correct,'out_file',maskfunc2,'in_file',)
#    preproc.connect(dilatemask,'out_file',maskfunc2,'in_file2')
#    """
#    smooth each run using SUSAN with the brightness threshold set to 
#    75% of the median value for each run and a mask constituing the 
#    mean functional
#    """
#    smooth = create_susan_smooth()
#    preproc.connect(inputnode,'fwhm',smooth,'inputnode.fwhm')
#    preproc.connect(maskfunc2,'out_file',smooth,'inputnode.in_files')
#    preproc.connect(dilatemask,'out_file',smooth,'inputnode.mask_file')
#    """
#    mask the smoothed data with the dilated mask
#    """
#    maskfunc3 = pe.MapNode(
#            interface = fsl.ImageMaths(suffix='_mask',op_string='-mas'),
#            iterfield = ['in_file','in_file2'],
#            name='dilateMask_smoothed')
#    # connect the output of the susam smooth component to the maskfunc3 node
#    preproc.connect(smooth,'outputnode.smoothed_files',
#                    maskfunc3,'in_file')
#    # connect the output of the dilated mask to the maskfunc3 node
#    preproc.connect(dilatemask,'out_file',
#                    maskfunc3,'in_file2')
#    """
#    scale the median value of the run is set to 10000
#    """
#    meanscale = pe.MapNode(
#            interface = fsl.ImageMaths(suffix='_gms'),
#            iterfield = ['in_file','op_string'],
#            name = 'meanscale')
#    preproc.connect(maskfunc3,'out_file',
#                    meanscale,'in_file')
#    """
#    define a function to get the scaling factor for intensity normalization
#    """
#    preproc.connect(medianval,('out_stat',getmeanscale),
#                    meanscale,'op_string')
#    """
#    generate a mean functional image from the first run
#    should this be the 'mask.nii.gz' we will use in the future?
#    """
#    meanfunc3 = pe.MapNode(
#            interface = fsl.ImageMaths(suffix='_mean',
#                                       op_string='-Tmean',),
#            iterfield = ['in_file'],
#            name='gen_mean_func_img')
#    preproc.connect(meanscale,'out_file',meanfunc3,'in_file')
#    preproc.connect(meanfunc3,'out_file',outputnode,'mean')
#    
#    return preproc
