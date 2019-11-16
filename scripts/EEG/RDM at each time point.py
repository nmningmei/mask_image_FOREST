#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:14:39 2019

@author: nmei
"""

import mne
import os
print(os.getcwd())
import re
import numpy as np
import pandas as pd
from glob                    import glob
from datetime                import datetime


from mne.decoding            import (
                                        Scaler,
                                        Vectorizer,
                                        SlidingEstimator,
                                        cross_val_multiscore,
                                        GeneralizingEstimator
                                        )
from sklearn.preprocessing   import StandardScaler
#from sklearn.calibration     import CalibratedClassifierCV
#from sklearn.svm             import LinearSVC
from sklearn.linear_model    import (
#                                        LogisticRegressionCV,
                                        LogisticRegression,
#                                        SGDClassifier
                                        )
from sklearn.pipeline        import make_pipeline
from sklearn.model_selection import (
                                        StratifiedShuffleSplit,
                                        cross_val_score
                                        )
from sklearn.utils           import shuffle
from sklearn.base            import clone
from sklearn.metrics         import make_scorer,roc_auc_score
from matplotlib              import pyplot as plt
from scipy                   import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools               import partial
from shutil                  import copyfile
copyfile(os.path.abspath('../utils.py'),'utils.py')
from utils                   import get_frames

n_jobs = 8
func   = partial(roc_auc_score,average = 'micro')
func.__name__ = 'micro_AUC'
scorer = make_scorer(func,needs_proba = True)
speed  = True

subject             = 'aingere_5_16_2019' 
# there was a bug in the csv file, so the early behavioral is treated differently
date                = '/'.join(re.findall('\d+',subject))
date                = datetime.strptime(date,'%m/%d/%Y')
breakPoint          = datetime(2019,3,10)
if date > breakPoint:
    new             = True
else:
    new             = False
folder_name = "clean_EEG_detrend"
working_dir         = os.path.abspath(f'../../data/{folder_name}/{subject}')
working_data        = glob(os.path.join(working_dir,'*-epo.fif'))
frames,_            = get_frames(directory = os.path.abspath(f'../../data/behavioral/{subject}'),new = new)
# create the directories for figure and decoding results (numpy arrays)
figure_dir          = os.path.abspath(f'../../figures/EEG/decode/{subject}')
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
array_dir           = os.path.abspath(f'../../results/EEG/decode/{subject}')
if not os.path.exists(array_dir):
    os.makedirs(array_dir)


for epoch_file in working_data:
    epochs  = mne.read_epochs(epoch_file)
    
    
    conscious   = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if (' conscious' in name)])
    see_maybe   = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('glimpse' in name)])
    unconscious = mne.concatenate_epochs([epochs[name] for name in epochs.event_id.keys() if ('unconscious' in name)])
    del epochs
    
    for ii,(epochs,conscious_state) in enumerate(zip([unconscious.copy(),
                                                      see_maybe.copy(),
                                                      conscious.copy()],
                                                     ['unconscious',
                                                      'glimpse',
                                                      'conscious'])):
        epochs
        epochs_ = epochs.resample(50)
        from scipy.spatial import distance
        X,y = epochs_.get_data(),epochs_.events[:,-1]
        X = mne.decoding.Scaler(epochs_.info).fit_transform(X)
        RDMs = []
        idx_sort = np.argsort(epochs_.ch_names)
        sort_ch_names = np.array(epochs_.ch_names)[idx_sort]
        for idx_time in range(len(epochs_.times)):
            features = X[:,idx_sort,idx_time].copy()
            features_norm = features - features.mean(1).reshape(-1,1)
            RDM = distance.squareform(distance.pdist(features_norm.T,'cosine'))
            np.fill_diagonal(RDM,0)
            RDMs.append(RDM)
        RDMs = np.array(RDMs)
        
        def plot_RDM(idx):
            RDM = RDMs[idx]
            time = epochs_.times[idx]
            df = pd.DataFrame(RDM,columns = np.sort(epochs_.ch_names),
                              index = np.sort(epochs_.ch_names))
            fig,ax = plt.subplots()
            im = ax.imshow(df,
                           origin = 'lower',
                           cmap = plt.cm.RdBu_r,
                           vmin = 0,
                           vmax = 2,
                           animated = True)
            ax.set(xticks = np.arange(60)[::10],xticklabels = sort_ch_names[::10],
                   yticks = np.arange(60)[::10],yticklabels = sort_ch_names[::10],
                   title = f'time = {time:.2f}')
            # Used to return the plot as an image rray
            fig.canvas.draw()       # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close('all')
            return image
        import imageio
        imageio.mimsave('some.gif', [plot_RDM(ii) for ii in range(len(epochs_.times))], fps=10)
        sdaf
        
    

































