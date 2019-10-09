#!/usr/bin/env python3 -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:59:34 2019

@author: nmei

decoding pipeline with multiple models and multiple rois, using customized partition
cross validation method

"""
import os
import warnings
warnings.filterwarnings('ignore') 
os.chdir('..')
import pandas  as pd
import numpy   as np
import seaborn as sns

from shutil import copyfile
copyfile('../../utils.py','utils.py')
from utils import get_label_category_mapping

from matplotlib                  import pyplot as plt
from glob                        import glob
from sklearn.utils               import shuffle
from sklearn.model_selection     import StratifiedKFold
from sklearn.feature_selection   import VarianceThreshold
from sklearn.pipeline            import make_pipeline
from sklearn.preprocessing       import MinMaxScaler
from sklearn.metrics             import roc_auc_score

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

sub                 = 'sub-01'
stacked_data_dir    = '../../../data/BOLD_average/{}/'.format(sub)
output_dir          = '../../../results/MRI/nilearn/{}/L2O'.format(sub)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
BOLD_data           = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))
event_data          = np.sort(glob(os.path.join(stacked_data_dir,'*.csv')))

label_map           = {'Nonliving_Things':[0,1],
                       'Living_Things':   [1,0]}
target_map          = {key:ii for ii,key in enumerate(get_label_category_mapping().keys())}
average             = True
n_jobs              = 12

idx = 0
np.random.seed(12345)
BOLD_name,df_name   = BOLD_data[idx],event_data[idx]
BOLD                = np.load(BOLD_name)
df_event            = pd.read_csv(df_name)
roi_name            = df_name.split('/')[-1].split('_events')[0]
for conscious_state in ['unconscious','glimpse','conscious']:
    idx_unconscious = df_event['visibility'] == conscious_state
    data            = BOLD[idx_unconscious]
    df_data         = df_event[idx_unconscious].reset_index(drop=True)
    df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
    targets         = np.array([label_map[item] for item in df_data['targets'].values])[:,-1]
    label_picked    = 'labels'
    groups          = df_data[label_picked].values
    
    model_name  = 'DNN'
    figure_name = f'RDM_{sub}_{roi_name}_{conscious_state}_{model_name}).png'
    
    features    = data.copy()
    labels      = groups.copy()
    
    n_splits = 5
    cv = StratifiedKFold(n_splits               = n_splits,
                         shuffle                = True,
                         random_state           = 12345)
#    from sklearn.model_selection import LeaveOneOut
#    cv = LeaveOneOut()
    
    transformer = make_pipeline(VarianceThreshold(),MinMaxScaler())
    features_trans = transformer.fit_transform(features)
    
    def initialize_clf(features,labels):
        from tensorflow.keras import layers,regularizers,Model,optimizers,losses
        from tensorflow.keras.callbacks import EarlyStopping
        import tensorflow.keras.backend as K
        K.clear_session()
        inputs = layers.Input(
                               shape        = (features.shape[1],),
                               batch_size   = 1,
                               name         = 'inputs')
        hidden = inputs
        hidden = layers.Dropout(rate = 0.5,seed = 12345,name = 'drop_hidden')(hidden)
        outputs = layers.Dense(np.unique(labels).shape[0],
                               activation = 'softmax',
                               kernel_regularizer = regularizers.l1_l2(),
                               activity_regularizer = regularizers.l1(),
                               name = 'outputs')(hidden)
        model = Model(inputs,outputs,name = 'clf')
        
        model.compile(optimizer = optimizers.Adam(lr = 1e-4,), 
                      loss = losses.categorical_crossentropy)
        
        earlyStop = EarlyStopping(monitor          = 'val_loss',
                                  min_delta        = 1e-4,
                                  patience         = 5,
                                  verbose          = 0, 
                                  mode             = 'min',
                                  restore_best_weights = True,
                                 )
        return model,[earlyStop]
    
    
    classes = set(labels)
    sparse = np.zeros((len(classes),len(classes)))
    sparse_map = {}
    for ii,(item,row) in enumerate(zip(classes,sparse)):
        row[ii] = 1
        sparse_map[item] = row
    o = np.array([sparse_map[item] for item in labels])
    
    # Compute confusion matrix for each cross-validation fold
    y_pred = np.zeros((len(labels), len(classes)))
    
    for train,test in cv.split(features_trans,labels):
        clf,callback = initialize_clf(features_trans,labels)
        x_,y_ = shuffle(features_trans[train],o[train])
        clf.fit(x_,y_,
                epochs = 100,
                callbacks = callback,
                validation_split = 0.2,
                )
        y_pred[test] = clf.predict(features_trans[test])
        del clf,callback
    
    confusion = np.zeros((len(classes), len(classes)))
    for ii, train_class in enumerate(classes):
        for jj in range(ii, len(classes)):
            confusion[ii, jj] = roc_auc_score(labels == train_class, y_pred[:, jj])
            confusion[jj, ii] = confusion[ii, jj]
    
    df_for_sort = pd.DataFrame(np.vstack([df_data[label_picked].values,df_data['targets'].values]).T,columns = ['labels','targets'])
    temp = dict(labels=[],targets=[])
    for (label,target),df_sub in df_for_sort.groupby(['labels','targets']):
        temp['labels'].append(label)
        temp['targets'].append(target)
    df_for_sort = pd.DataFrame(temp)
    df_for_sort = df_for_sort.sort_values(['targets','labels'])
    
    df_plot = pd.DataFrame(confusion,columns = list(classes),index = list(classes))
    df_plot = df_plot.loc[df_for_sort['labels'].values,df_for_sort['labels'].values]
    
#    fig, ax = plt.subplots(figsize = (16,16))
#    ax = sns.heatmap(df_plot,cmap = plt.cm.RdBu_r,
#                     vmin = 0.1,
#                     vmax = 0.9,
#                     ax = ax,
#                     xticklabels = True,
#                     yticklabels = True,
#                     )
#    ax.axvline(len(classes) / 2 - 1,linestyle = '--',color = 'black')
#    ax.axhline(len(classes) / 2 - 1,linestyle = '--',color = 'black')
#    ax.set(title = f'{conscious_state} {roi_name}')
    g = sns.clustermap(df_plot,xticklabels = True, yticklabels = True,figsize = (16,16),
                       cmap = plt.cm.RdBu_r,
                       method = 'centroid',
                       )
    g.fig.suptitle(f'{conscious_state} {roi_name}')
    










