#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:28:44 2019

@author: nmei
"""

import os
from glob import glob
import pymc3 as pm
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')

working_dir = '../../../../results/MRI/decoding'

def get_fs(x):
    return x.split(' + ')[0]
def get_clf(x):
    return x.split(' + ')[1]
def rename_roi(x):
    return x.split('-')[-1] + '-' + x.split('-')[1]

working_data = glob(os.path.join(working_dir,'*.csv'))
df = pd.concat([pd.read_csv(f) for f in working_data])
#df = df.groupby(['conscious_state','model_name','roi','sub']).mean().reset_index()
df['feature_selector'] = df['model_name'].apply(get_fs)
df['estimator'] = df['model_name'].apply(get_clf)
if 'score' in df.columns:
    df['roc_auc'] = df['score']

conscious_state_map = {'unconscious':0,'glimpse':1,'conscious':2}
feature_selector_map = {'None':0,'PCA':1,'Mutual':2,'RandomForest':3}
roi_map = {name:ii for ii,name in enumerate(pd.unique(df['roi']))}
estimator_map = {name:ii for ii,name in enumerate(pd.unique(df['estimator']))}


df['idx_conscious'] = df['conscious_state'].map(conscious_state_map)
idx_conscious = df['idx_conscious'].values
df['feature_selector'] = df['feature_selector'].map(feature_selector_map)
df['roi'] = df['roi'].map(roi_map)
df['estimator'] = df['estimator'].map(estimator_map)

with pm.Model() as hierarchical_model:
    # Hyperpriors for group nodes
#    mu_state = pm.Normal('mu_state', mu=0., sd=100**2)
#    sigma_state = pm.HalfCauchy('sigma_state', 5)
    mu_a = pm.Normal('mu_a', mu=0., sd=100**2)
    sigma_a = pm.HalfCauchy('sigma_a', 5)
    mu_fs = pm.Normal('mu_fs', mu=0., sd=100**2)
    sigma_fs = pm.HalfCauchy('sigma_fs', 5)
    mu_est = pm.Normal('mu_est',mu = 0, sd = 100**2)
    sigma_est = pm.HalfCauchy('sigma_est',5)
    mu_roi = pm.Normal('mu_roi',mu=0,sd=100**2)
    sigma_roi = pm.HalfCauchy('sigma_roi',5)
    
    a = pm.Normal('a',mu=mu_a,sd = sigma_a,shape=3)
    fs = pm.Normal('fs', mu=mu_fs, sd=sigma_fs, shape=3)
    est = pm.Normal('est',mu=mu_est,sd=sigma_est,shape=3)
    roi = pm.Normal('roi',mu=mu_roi,sd=sigma_roi,shape=3)
    
    eps = pm.HalfCauchy('eps', 5)
    
    roc = a[idx_conscious] + fs[idx_conscious]*df['feature_selector'].values + est[idx_conscious]*df['estimator'].values + roi[idx_conscious]*df['roi'].values
    
    # Data likelihood
    roc_like = pm.Normal('roc_like', mu=roc, sd=eps, observed=df['roc_auc'].values)


with hierarchical_model:
    hierarchical_trace = pm.sample(draws=2000, n_init=1000)

pm.traceplot(hierarchical_trace)
pm.summary(hierarchical_trace)























