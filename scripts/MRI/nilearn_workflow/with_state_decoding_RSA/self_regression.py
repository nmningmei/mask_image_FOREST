#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:18:08 2019

@author: nmei
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
from tqdm                        import tqdm
from sklearn.model_selection     import KFold,GridSearchCV
from sklearn.feature_selection   import VarianceThreshold
from sklearn.pipeline            import make_pipeline
from sklearn.preprocessing       import MinMaxScaler
from sklearn.metrics             import r2_score
from sklearn.linear_model        import Ridge


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

idx = 7
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
    
    df = pd.DataFrame(index = set(groups),
                      columns = set(groups))
    
    for label_1 in set(groups):
        for label_2 in set(groups):
            if label_1 != label_2:
                idx_rows_1 = np.where(df_data['labels'] == label_1)
                idx_rows_2 = np.where(df_data['labels'] == label_2)
                
                BOLD_1 = data[idx_rows_1]
                BOLD_2 = data[idx_rows_2]
                
                BOLD_1_stacked = np.repeat(BOLD_1,BOLD_2.shape[0],axis = 0)
                BOLD_2_stacked = np.concatenate([BOLD_2 for _ in range(BOLD_1.shape[0])])
                
                transformer = make_pipeline(VarianceThreshold(),
                                            MinMaxScaler())
                X = transformer.fit_transform(BOLD_1_stacked)
                y = transformer.fit_transform(BOLD_2_stacked)
                cv = KFold(n_splits = 3,shuffle = True,random_state = 12345)
                reg = Ridge(alpha = 1,
                            normalize = True,
                            random_state = 12345,
                            solver = "sparse_cg",
                            tol = 1e-1,
                            max_iter = int(1e3),)
                
                
                from joblib import Parallel,delayed
                from sklearn.base import clone
                import gc
                gc.collect()
#                def fit_predict(X,y,train,test,reg):
#                    reg.fit(X[train],y[train])
#                    return reg.predict(X[test])
#                scores = Parallel(n_jobs = 8,
#                         verbose = 2)(delayed(fit_predict)(**{
#                    'X':X,'y':y,'reg':clone(reg),'train':train,
#                    'test':test,}) for train,test in cv.split(X,y))
                scores = []
                for train,test in tqdm(cv.split(X,y)):
                    reg_ = GridSearchCV(clone(reg),
                                        {"alpha":np.logspace(2,12,11)},
                                        cv = cv,
                                        n_jobs = 8,
                                        scoring = 'explained_variance')
                    reg_.fit(X[train],y[train])
                    preds = reg_.predict(X[test])
                    score = r2_score(y[test],preds,multioutput='raw_values')
                    scores.append(score)
                scores = np.array(scores)
                
                scores[scores <= 0] = np.nan
                df.loc[label_1,label_2] = np.nanmean(scores)
                df.loc[label_2,label_1] = np.nanmean(scores)
    
    import seaborn as sns
    sns.clustermap(df)
    cxv


















