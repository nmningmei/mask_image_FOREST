#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:07:00 2019

@author: nmei
"""

import numpy as np
import pandas as pd
import pickle

from sklearn.metrics                               import roc_auc_score,roc_curve
from sklearn.metrics                               import (
                                                           classification_report,
                                                           matthews_corrcoef,
                                                           confusion_matrix,
                                                           f1_score,
                                                           log_loss,
                                                           r2_score
                                                           )

from sklearn.preprocessing                         import (MinMaxScaler,
                                                           OneHotEncoder,
                                                           FunctionTransformer,
                                                           StandardScaler)

from sklearn.pipeline                              import make_pipeline
from sklearn.ensemble.forest                       import _generate_unsampled_indices
from sklearn.utils                                 import shuffle
from sklearn.svm                                   import SVC,LinearSVC
from sklearn.calibration                           import CalibratedClassifierCV
from sklearn.decomposition                         import PCA
from sklearn.dummy                                 import DummyClassifier
from sklearn.feature_selection                     import (SelectFromModel,
                                                           SelectPercentile,
                                                           VarianceThreshold,
                                                           mutual_info_classif,
                                                           f_classif,
                                                           chi2,
                                                           f_regression,
                                                           GenericUnivariateSelect)
from sklearn.model_selection                       import (StratifiedShuffleSplit,
                                                           cross_val_score)
from sklearn.ensemble                              import RandomForestClassifier,BaggingClassifier,VotingClassifier
from sklearn.neural_network                        import MLPClassifier
from sklearn.exceptions                            import ConvergenceWarning
from sklearn.utils.testing                         import ignore_warnings
from xgboost                                       import XGBClassifier
from itertools                                     import product,combinations
from sklearn.base                                  import clone
from sklearn.neighbors                             import KNeighborsClassifier
from sklearn.tree                                  import DecisionTreeClassifier
from collections                                   import OrderedDict

from scipy                                         import stats
from collections                                   import Counter

try:
    #from mvpa2.datasets.base                           import Dataset
    from mvpa2.mappers.fx                              import mean_group_sample
    #from mvpa2.measures                                import rsa
    #from mvpa2.measures.searchlight                    import sphere_searchlight
    #from mvpa2.base.learner                            import ChainLearner
    #from mvpa2.mappers.shape                           import TransposeMapper
    #from mvpa2.generators.partition                    import NFoldPartitioner
except:
    print('pymvpa is not installed')
try:
    from tqdm import tqdm
except:
    print('why is tqdm not installed?')
def resample_ttest(x,baseline = 0.5,n_ps = 100,n_permutation = 5000,one_tail = False):
    """
    http://www.stat.ucla.edu/~rgould/110as02/bshypothesis.pdf
    Inputs:
    ----------
    x: numpy array vector, the data that is to be compared
    baseline: the single point that we compare the data with
    n_ps: number of p values we want to estimate
    n_permutation: number of permutation we want to perform, the more the further it could detect the strong effects, but it is so unnecessary
    one_tail: whether to perform one-tailed comparison
    """
    import numpy as np
    experiment      = np.mean(x) # the mean of the observations in the experiment
    experiment_diff = x - np.mean(x) + baseline # shift the mean to the baseline but keep the distribution
    # newexperiment = np.mean(experiment_diff) # just look at the new mean and make sure it is at the baseline
    # simulate/bootstrap null hypothesis distribution
    # 1st-D := number of sample same as the experiment
    # 2nd-D := within one permutation resamping, we perform resampling same as the experimental samples,
    # but also repeat this one sampling n_permutation times
    # 3rd-D := repeat 2nd-D n_ps times to obtain a distribution of p values later
    temp            = np.random.choice(experiment_diff,size=(x.shape[0],n_permutation,n_ps),replace=True)
    temp            = temp.mean(0)# take the mean over the sames because we only care about the mean of the null distribution
    # along each row of the matrix (n_row = n_permutation), we count instances that are greater than the observed mean of the experiment
    # compute the proportion, and we get our p values
    
    if one_tail:
        ps = (np.sum(temp >= experiment,axis=0)+1.) / (n_permutation + 1.)
    else:
        ps = (np.sum(np.abs(temp) >= np.abs(experiment),axis=0)+1.) / (n_permutation + 1.)
    return ps
def resample_ttest_2sample(a,b,n_ps=100,n_permutation=5000,one_tail=False,match_sample_size = True,):
    # when the N is matched just simply test the pairwise difference against 0
    # which is a one sample comparison problem
    if match_sample_size:
        difference  = a - b
        ps          = resample_ttest(difference,baseline=0,n_ps=n_ps,n_permutation=n_permutation,one_tail=one_tail)
        return ps
    else: # when the N is not matched
        difference              = np.mean(a) - np.mean(b)
        concatenated            = np.concatenate([a,b])
        np.random.shuffle(concatenated)
        temp                    = np.zeros((n_permutation,n_ps))
        # the next part of the code is to estimate the "randomized situation" under the given data's distribution
        # by randomized the items in each group (a and b), we can compute the chance level differences
        # and then we estimate the probability of the chance level exceeds the true difference 
        # as to represent the "p value"
        try:
            iterator            = tqdm(range(n_ps),desc='ps')
        except:
            iterator            = range(n_ps)
        for n_p in iterator:
            for n_permu in range(n_permutation):
                idx_a           = np.random.choice(a    = [0,1],
                                                   size = (len(a)+len(b)),
                                                   p    = [float(len(a))/(len(a)+len(b)),
                                                           float(len(b))/(len(a)+len(b))]
                                                   ).astype(np.bool)
                idx_b           = np.logical_not(idx_a)
                d               = np.mean(concatenated[idx_a]) - np.mean(concatenated[idx_b])
                if np.isnan(d):
                    idx_a       = np.random.choice(a        = [0,1],
                                                   size     = (len(a)+len(b)),
                                                   p        = [float(len(a))/(len(a)+len(b)),
                                                               float(len(b))/(len(a)+len(b))]
                                                   ).astype(np.bool)
                    idx_b       = np.logical_not(idx_a)
                    d           = np.mean(concatenated[idx_a]) - np.mean(concatenated[idx_b])
                temp[n_permu,n_p] = d
        if one_tail:
            ps = (np.sum(temp >= difference,axis=0)+1.) / (n_permutation + 1.)
        else:
            ps = (np.sum(np.abs(temp) >= np.abs(difference),axis=0)+1.) / (n_permutation + 1.)
        return ps

class MCPConverter(object):
    import statsmodels as sms
    """
    https://gist.github.com/naturale0/3915e2def589553e91dce99e69d138cc
    https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
    input: array of p-values.
    * convert p-value into adjusted p-value (or q-value)
    """
    def __init__(self, pvals, zscores = None):
        self.pvals                    = pvals
        self.zscores                  = zscores
        self.len                      = len(pvals)
        if zscores is not None:
            srted                     = np.array(sorted(zip(pvals.copy(), zscores.copy())))
            self.sorted_pvals         = srted[:, 0]
            self.sorted_zscores       = srted[:, 1]
        else:
            self.sorted_pvals         = np.array(sorted(pvals.copy()))
        self.order                    = sorted(range(len(pvals)), key=lambda x: pvals[x])
    
    def adjust(self, method           = "holm"):
        import statsmodels as sms
        """
        methods = ["bonferroni", "holm", "bh", "lfdr"]
         (local FDR method needs 'statsmodels' package)
        """
        if method is "bonferroni":
            return [np.min([1, i]) for i in self.sorted_pvals * self.len]
        elif method is "holm":
            return [np.min([1, i]) for i in (self.sorted_pvals * (self.len - np.arange(1, self.len+1) + 1))]
        elif method is "bh":
            p_times_m_i = self.sorted_pvals * self.len / np.arange(1, self.len+1)
            return [np.min([p, p_times_m_i[i+1]]) if i < self.len-1 else p for i, p in enumerate(p_times_m_i)]
        elif method is "lfdr":
            if self.zscores is None:
                raise ValueError("Z-scores were not provided.")
            return sms.stats.multitest.local_fdr(abs(self.sorted_zscores))
        else:
            raise ValueError("invalid method entered: '{}'".format(method))
            
    def adjust_many(self, methods = ["bonferroni", "holm", "bh", "lfdr"]):
        if self.zscores is not None:
            df = pd.DataFrame(np.c_[self.sorted_pvals, self.sorted_zscores], columns=["p_values", "z_scores"])
            for method in methods:
                df[method] = self.adjust(method)
        else:
            df = pd.DataFrame(self.sorted_pvals, columns=["p_values"])
            for method in methods:
                if method is not "lfdr":
                    df[method] = self.adjust(method)
        return df
def binarize(labels):
    """
    By Usman
    """
    try:
        if len(np.unique(labels)) > 2: raise ValueError
        return (labels == 1).astype(int)
    except:
        exit("more than 2 classes, fix it!")
def load_preprocessed(pre_fil):
    """
    By Usman
    
    Load all preprocessed data for a specific roi (stored @ pre_fil).

    Inputs:
    -----------
    pre_fil: str

    Returns:
    --------
    feat_ds: pymvpa2 Dataset
    """
    import gc
    # disable garbage collection to speed up pickle
    gc.disable()
    with open(pre_fil, 'rb') as f:
        feat_ds = pickle.load(f)

#    feat_ds.sa['id'] = feat_ds.sa.id.astype(int)
#    feat_ds.sa['targets'] = binarize(feat_ds.sa.targets.astype(float).astype(int))

    return feat_ds

def get_blocks(dataset__,label_map,key_type='labels'):
    """
    # use ids, chunks,and labels to make unique blocks of the pre-average dataset, because I don't want to 
    # average the dataset until I actually want to, but at the same time, I want to balance the data for 
    # both the training and test set.
    """
    ids                     = dataset__.sa.id.astype(int)
    chunks                  = dataset__.sa.chunks
    words                   = dataset__.sa.labels
    if key_type == 'labels':
        try: # in metasema
            labels              = np.array([label_map[item] for item in dataset__.sa.targets])[:,-1]
        except:# not in metasema
            labels              = np.array([label_map[item] for item in dataset__.sa.targets])
        
    elif key_type == 'words':
        labels              = np.array([label_map[item] for item in dataset__.sa.labels])
    sample_indecies         = np.arange(len(labels))
    blocks                  = [np.array([ids[ids             == target],
                                         chunks[ids          == target],
                                         words[ids           == target],
                                         labels[ids          == target],
                                         sample_indecies[ids == target]
                                         ]) for target in np.unique(ids)]
    block_labels            = np.array([np.unique(ll[-2]) for ll in blocks]).ravel()
    return blocks,block_labels
def customized_partition(df_data,groupby_column = 'labels',n_splits = 100,):
    idx_object      = {label:df_sub.index.tolist() for label,df_sub in df_data.groupby([groupby_column])}
    idxs_test       = []
    np.random.seed(12345)
    for counter in range(int(1e4)):
        idx_test = [np.random.choice(idx_object[item]) for item in idx_object.keys()]
        if counter >= n_splits:
            return idxs_test
            break
        if counter > 0:
            temp = []
            for used in idxs_test:
                a = set(used)
                b = set(idx_test)
                temp.append(len(a.intersection(b)) != len(idx_test))
            if all(temp) == True:
                idxs_test.append(idx_test)
        else:
            idxs_test.append(idx_test)
def get_train_test_splits(dataset,label_map,n_splits):
    idxs_train,idxs_test = [],[]
    np.random.seed(12345)
    used_test = []
    fold = -1
    for abc in range(int(1e3)):
#        print('paritioning ...')
        idx_train,idx_test = customized_partition(dataset,label_map,)
        current_sample = np.sort(idx_test)
        candidates = [np.sort(item) for item in used_test if (len(item) == len(idx_test))]
        if any([np.sum(current_sample == item) == len(current_sample) for item in candidates]):
            pass
        else:
            fold += 1
            used_test.append(idx_test)
            idxs_train.append(idx_train)
            idxs_test.append(idx_test)
#            print('done, get fold {}'.format(fold))
            if fold == n_splits - 1:
                break
    return idxs_train,idxs_test
def check_train_test_splits(idxs_test):
    temp = []
    for ii,item1 in enumerate(idxs_test):
        for jj,item2 in enumerate(idxs_test):
            if not ii == jj:
                if len(item1) == len(item2):
                    sample1 = np.sort(item1)
                    sample2 = np.sort(item2)
                    
                    temp.append(np.sum(sample1 == sample2) == len(sample1))
    temp = np.array(temp)
    return any(temp)
def check_train_balance(df,idx_train,keys):
    Counts = dict(Counter(df.iloc[idx_train]['targets'].values))
    if np.abs(Counts[keys[0]] - Counts[keys[1]]) > 2:
        if Counts[keys[0]] > Counts[keys[1]]:
            key_major = keys[0]
            key_minor = keys[1]
        else:
            key_major = keys[1]
            key_minor = keys[0]
            
        ids_major = df.iloc[idx_train]['id'][df.iloc[idx_train]['targets'] == key_major]
        
        idx_train_new = idx_train.copy()
        for n in range(len(idx_train_new)):
            random_pick = np.random.choice(np.unique(ids_major),size = 1)[0]
            # print(random_pick,np.unique(ids_major))
            idx_train_new = np.array([item for item,id_temp in zip(idx_train_new,df.iloc[idx_train_new]['id']) if (id_temp != random_pick)])
            ids_major = np.array([item for item in ids_major if (item != random_pick)])
            new_counts = dict(Counter(df.iloc[idx_train_new]['targets']))
            if np.abs(new_counts[keys[0]] - new_counts[keys[1]]) > 3:
                if new_counts[keys[0]] > new_counts[keys[1]]:
                    key_major = keys[0]
                    key_minor = keys[1]
                else:
                    key_major = keys[1]
                    key_minor = keys[0]
                
                ids_major = df.iloc[idx_train_new]['id'][df.iloc[idx_train_new]['targets'] == key_major]
            elif np.abs(new_counts[keys[0]] - new_counts[keys[1]]) < 3:
                break
        return idx_train_new
    else:
        return idx_train
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold         = roc_curve(target, predicted)
    i                           = np.arange(len(tpr)) 
    roc                         = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t                       = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def cross_CV( source_set,           # pymvpa dataset - source dataset
              target_set,           # pymvpa dataset - target dataset 
              idx_train_source,     # indecies of the training set in the source dataset
              idx_train_target,     # indecies of the training set in the target dataset
              idx_test_target,      # indecies of the test set in the target dataset
              pipeline,             # pipeline that contains a scaler, a feature processor, and an estimator
              label_map,            # one hot label map used in deep learning models
              results,              # the dictionary for storing the results
              sub_name,             # subject name
              fold,                 # cross validation fold
              model_name,           # name of the pipeline
              language,             # language of the stimuli
              condition_target,
              condition_source,
              roi_name = None,      # not None if we work on selected ROI data
              average = True,       # averaging the trainig data
              transfer = False,     # do I do domain adaptation
              print_train = False,
              concatenate = False,
              flip = False,
              to_save = True,
              MultiOutput = False,
              ):
    """
    unified pipeline for decoding across many experiments and conditions 
    
    Inputs:
    --------
    source_set:         pymvpa dataset, used to get the primary training dataset
    target_set:         pymvpa dataset, used to get the secondary training dataset and/or the testing dataset
    idx_train_source:   indecies of the training set in the source dataset
    idx_train_target:   indecies of the training set in the target dataset
    idx_test_target:    indecies of the test set in the target dataset
    pipeline:           scitkit learn pipeline/estimator, predict_proba required
    label_map:          a dictionary for creating the one hot labels for cross validation
    results:            a dictionary for storing the results and other attributes
    sub_name:           subject's coded name
    fold:               a counting unit
    model_name:         name of the pipeline
    language:           language in which the stimuli is
    condition_target:   condition/conscious in which is the subject is for the target data
    condition_source:   condition/conscious in which is the subject is for the source data
    roi_name:           name of the ROI
    average:            control whether to average the training set/s
    transfer:           indicator of whether this pipeline is in transfer learning/domain adaptation
    print_train:        to print out the training informations
    concatenate:        domain adaptation specific, whether to combine the primary and secondary training dataset
    flip:               doubling the size of the training dataset by flipping the training set matrix left-right, and concatenate them
    to_save:            control whether to save the outputs
    ----------------------------------------------------------------------------------------------------------
    Returns:
    --------
    results:            a dictionary for storing the results and other attributes
    score_baseline1:    roc auc
    score_baseline2:    matthews correlation coefficient
    score_baseline3:    f1 score
    score_baseline4:    log loss value
    """
    # select the training and testing split again to make sure we do the right cv
    tr_source           = source_set[idx_train_source]
    tr_target           = target_set[idx_train_target]
    te                  = target_set[idx_test_target]
    # average the test data for the baseline model
    try:
        te              = te.get_mapped(mean_group_sample(['chunks', 'id'],order = 'occurrence'))
    except:
        print('it is socialcon, and it is already been averaged')
    # average the train data for baseline to improve the signal to noise ratio
    if average:
        tr_source       = tr_source.get_mapped(mean_group_sample(['chunks', 'id'],order = 'occurrence'))
        tr_target       = tr_target.get_mapped(mean_group_sample(['chunks', 'id'],order = 'occurrence'))
    # get the numpy arrays
    X_train_source      = tr_source.samples.astype('float32')
    X_train_target      = tr_target.samples.astype('float32')
    X_test              = te.samples.astype('float32')
    # transfer the string labels to integers
    y_train_source      = np.array([label_map[item] for item in tr_source.targets])
    y_train_target      = np.array([label_map[item] for item in tr_target.targets])
    y_test              = np.array([label_map[item] for item in te.targets])
    
    
    # whether to combine the primary and secondary datasets
    if concatenate:
        X_train         = np.concatenate([X_train_source,X_train_target])
        y_train         = np.concatenate([y_train_source,y_train_target])
    else:
        X_train         = X_train_source
        y_train         = y_train_source
    # whether to double the size of training dataset
    if flip:
        X_train_flip    = np.fliplr(X_train)
        X_train         = np.concatenate([X_train,X_train_flip])
        y_train         = np.concatenate([y_train,y_train])
    # check the size of the labels, if it does NOT have 2 columns, make it so
    if y_train.shape[-1]== 2:
        labels_train    = y_train
        labels_test     = y_test
    else:
        labels_train    = OneHotEncoder().fit_transform(y_train.reshape(-1,1)).toarray()
        labels_test     = OneHotEncoder().fit_transform(y_test.reshape(-1,1)).toarray()
    
    if print_train:
        print('train on {} samples, test on {} samples by {}'.format(
                X_train.shape[0],
                te.shape[0],
                model_name))
    # shuffle the order of trials for both features and targets
    np.random.seed(12345)
    X_train,labels_train    = shuffle(X_train,labels_train)
    # train the classification pipeline
    if MultiOutput:
        pipeline.fit(X_train,labels_train)
        
        pred_ = np.array(pipeline.predict_proba(X_test))[:,:,-1]
        if print_train:
            print('test labels',labels_test,'prediction',pred_)
        score_baseline1         = np.array([roc_auc_score(a,b) for a,b in zip(labels_test.T,pred_)])
        threshold_              = np.array([Find_Optimal_Cutoff(a,b) for a,b in zip(labels_test.T,pred_)])
        score_baseline2         = np.array([matthews_corrcoef(a,b>t) for a,b,t in zip(labels_test.T,pred_,threshold_)])
        score_baseline3         = np.array([f1_score(a,b>t) for a,b,t in zip(labels_test.T,pred_,threshold_)])
        score_baseline4         = np.array([log_loss(a,b) for a,b in zip(labels_test.T,pred_)])
        cm                      = np.array([confusion_matrix(a,b>t) for a,b,t in zip(labels_test.T,pred_,threshold_)])
        tn, fp, fn, tp          = cm.mean(0).flatten()
    else:
        with ignore_warnings(category=ConvergenceWarning):
            pipeline.fit(X_train,labels_train[:,-1])
        if print_train:
            print('training score = {:.2}'.format(pipeline.score(X_train,labels_train[:,-1])))
        # provide probabilistic predictions on the test data
        pred_                   = pipeline.predict_proba(X_test)
        if print_train:
            print('test labels',labels_test[:,-1],'prediction',pred_[:,-1])
        """"
        Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
         For binary y_true, y_score is supposed to be the score of the class with greater label.
        """
        score_baseline1         = roc_auc_score(labels_test[:,-1],pred_[:,-1])
        threshold_              = np.array([Find_Optimal_Cutoff(labels_test[:,ii],pred_[:,ii])[0] for ii in range(labels_test.shape[-1])])[-1]
        score_baseline2         = matthews_corrcoef(labels_test[:,-1],pred_[:,-1]>threshold_)
        score_baseline3         = f1_score(labels_test[:,-1],pred_[:,-1]>threshold_, average='weighted',labels=np.unique(pred_[:,-1]>threshold_))
        score_baseline4         = log_loss(labels_test[:,-1],pred_[:,-1])
        cm                      = confusion_matrix(labels_test[:,-1],pred_[:,-1]>threshold_).ravel()
        tn, fp, fn, tp          = cm
        if print_train:
            print(classification_report(labels_test[:,-1],pred_[:,-1]>threshold_))
    if to_save:
        if MultiOutput:
            results['tn'                    ].append(tn)
            results['fp'                    ].append(fp)
            results['fn'                    ].append(fn)
            results['tp'                    ].append(tp)
            results['fold'                  ].append(fold+1)
            results['sub'                   ].append(sub_name)
            if roi_name is not None:
                results['roi'               ].append(roi_name)
            results['model'                 ].append(model_name)
            results['roc_auc'               ].append(score_baseline1.mean())
            results['matthews_correcoef'    ].append(score_baseline2.mean())
            results['f1_score'              ].append(score_baseline3.mean())
            results['log_loss'              ].append(score_baseline4.mean())
            results['language'              ].append(language)
            results['transfer'              ].append(transfer)
            results['condition_target'      ].append(condition_target)
            results['condition_source'      ].append(condition_source)
            results['concatenate'           ].append(concatenate)
            results['flip'                  ].append(flip)
        else:
            results['tn'                    ].append(tn)
            results['fp'                    ].append(fp)
            results['fn'                    ].append(fn)
            results['tp'                    ].append(tp)
            results['fold'                  ].append(fold+1)
            results['sub'                   ].append(sub_name)
            if roi_name is not None:
                results['roi'               ].append(roi_name)
            results['model'                 ].append(model_name)
            results['roc_auc'               ].append(score_baseline1)
            results['matthews_correcoef'    ].append(score_baseline2)
            results['f1_score'              ].append(score_baseline3)
            results['log_loss'              ].append(score_baseline4)
            results['language'              ].append(language)
            results['transfer'              ].append(transfer)
            results['condition_target'      ].append(condition_target)
            results['condition_source'      ].append(condition_source)
            results['concatenate'           ].append(concatenate)
            results['flip'                  ].append(flip)
        
        return results,(score_baseline1,score_baseline2,score_baseline3,score_baseline4)
    else:
        return (score_baseline1,score_baseline2,score_baseline3,score_baseline4)

def build_feature_selector_dictionary(print_train = False,class_weight = 'balanced',n_jobs = 1):
    xgb = XGBClassifier(
                        learning_rate                           = 1e-3, # not default
                        max_depth                               = 100, # not default
                        n_estimators                            = 300, # not default
                        objective                               = 'binary:logistic', # default
                        booster                                 = 'gbtree', # default
                        subsample                               = 0.9, # not default
                        colsample_bytree                        = 0.9, # not default
                        reg_alpha                               = 0, # default
                        reg_lambda                              = 1, # default
                        random_state                            = 12345, # not default
                        importance_type                         = 'gain', # default
                        n_jobs                                  = n_jobs,# default to be 1
                                              )
    RF = SelectFromModel(xgb,
                        prefit                                  = False,
                        threshold                               = '1.96*mean' # induce sparsity
                        )
    uni = SelectPercentile(mutual_info_classif) # so annoying that I cannot control the random state
    
    return {'RandomForest':make_pipeline(MinMaxScaler(),
                                         RF,),
            'MutualInfo': make_pipeline(MinMaxScaler(),
                                        uni,)
            }
    
def build_model_dictionary(print_train = False,class_weight = 'balanced',n_jobs = 1):
    svm = LinearSVC(penalty = 'l2', # default
                    dual = True, # default
                    tol = 1e-3, # not default
                    random_state = 12345, # not default
                    max_iter = int(1e3), # default
                    class_weight = class_weight, # not default
                    )
    svm = CalibratedClassifierCV(base_estimator = svm,
                                 method = 'sigmoid',
                                 cv = 3)
    xgb = XGBClassifier(
                        learning_rate                           = 1e-3, # not default
                        max_depth                               = 100, # not default
                        n_estimators                            = 200, # not default
                        objective                               = 'binary:logistic', # default
                        booster                                 = 'gbtree', # default
                        subsample                               = 0.9, # not default
                        colsample_bytree                        = 0.9, # not default
                        reg_alpha                               = 0, # default
                        reg_lambda                              = 1, # default
                        random_state                            = 12345, # not default
                        importance_type                         = 'gain', # default
                        n_jobs                                  = n_jobs,# default to be 1
                                              )
    bagging = BaggingClassifier(base_estimator                  = svm,
                                 n_estimators                   = 30, # not default
                                 max_features                   = 0.9, # not default
                                 max_samples                    = 0.9, # not default
                                 bootstrap                      = True, # default
                                 bootstrap_features             = True, # default
                                 random_state                   = 12345, # not default
                                                 )
    RF = SelectFromModel(xgb,
                        prefit                                  = False,
                        threshold                               = 'median' # induce sparsity
                        )
    uni = SelectPercentile(mutual_info_classif,50) # so annoying that I cannot control the random state
    knn = KNeighborsClassifier()
    tree = DecisionTreeClassifier(random_state = 12345,
                                  class_weight = class_weight)
    dummy = DummyClassifier(strategy = 'uniform',random_state = 12345,)
    models = OrderedDict([
            ['None + Dummy',                     make_pipeline(MinMaxScaler(),
                                                               dummy,)],
            ['None + Linear-SVM',                make_pipeline(MinMaxScaler(),
                                                              svm,)],
            ['None + Ensemble-SVMs',             make_pipeline(MinMaxScaler(),
                                                              bagging,)],
            ['None + KNN',                       make_pipeline(MinMaxScaler(),
                                                              knn,)],
            ['None + Tree',                      make_pipeline(MinMaxScaler(),
                                                              tree,)],
            ['PCA + Dummy',                      make_pipeline(MinMaxScaler(),
                                                               PCA(),
                                                               dummy,)],
            ['PCA + Linear-SVM',                 make_pipeline(MinMaxScaler(),
                                                              PCA(),
                                                              svm,)],
            ['PCA + Ensemble-SVMs',              make_pipeline(MinMaxScaler(),
                                                              PCA(),
                                                              bagging,)],
            ['PCA + KNN',                        make_pipeline(MinMaxScaler(),
                                                              PCA(),
                                                              knn,)],
            ['PCA + Tree',                       make_pipeline(MinMaxScaler(),
                                                              PCA(),
                                                              tree,)],
            ['Mutual + Dummy',                   make_pipeline(MinMaxScaler(),
                                                               uni,
                                                               dummy,)],
            ['Mutual + Linear-SVM',              make_pipeline(MinMaxScaler(),
                                                              uni,
                                                              svm,)],
            ['Mutual + Ensemble-SVMs',           make_pipeline(MinMaxScaler(),
                                                              uni,
                                                              bagging,)],
            ['Mutual + KNN',                     make_pipeline(MinMaxScaler(),
                                                              uni,
                                                              knn,)],
            ['Mutual + Tree',                    make_pipeline(MinMaxScaler(),
                                                              uni,
                                                              tree,)],
            ['RandomForest + Dummy',             make_pipeline(MinMaxScaler(),
                                                               RF,
                                                               dummy,)],
            ['RandomForest + Linear-SVM',        make_pipeline(MinMaxScaler(),
                                                              RF,
                                                              svm,)],
            ['RandomForest + Ensemble-SVMs',     make_pipeline(MinMaxScaler(),
                                                              RF,
                                                              bagging,)],
            ['RandomForest + KNN',               make_pipeline(MinMaxScaler(),
                                                              RF,
                                                              knn,)],
            ['RandomForest + Tree',              make_pipeline(MinMaxScaler(),
                                                              RF,
                                                              tree,)],]
            )
    return models

def get_roi_dict():
    roi_dict ={'lh_fusif': 'lh-fusiform',
           'lh_infpar': 'lh-inferiorparietal',
           'lh_inftemp': 'lh-inferiortemporal',
           'lh_latoccip': 'lh-lateraloccipital',
           'lh_lingual': 'lh-lingual',
           'lh_middlefrontal': 'lh-rostralmiddlefrontal',
           'lh_phipp':'lh-parahippocampal',
           'lh_precun': 'lh-precuneus',
           'lh_sfrontal':'lh-superiorfrontal',
           'lh_superiorparietal': 'lh-superiorparietal',
           'lh_ventrollateralPFC': 'lh-ventrolateralPFC',
           'lh_pericalc':'lh-pericalcarine',
           'rh_fusif': 'rh-fusiform',
           'rh_infpar': 'rh-inferiorparietal',
           'rh_inftemp': 'rh-inferiortemporal',
           'rh_latoccip': 'rh-lateraloccipital',
           'rh_lingual': 'rh-lingual',
           'rh_middlefrontal': 'rh-rostralmiddlefrontal',
           'rh_phipp': 'rh-parahippocampal',
           'rh_precun': 'rh-precuneus',
           'rh_sfrontal':'rh-superiorfrontal',
           'rh_superiorparietal':'rh-superiorparietal',
           'rh_ventrollateralPFC': 'rh-ventrolateralPFC',
           'rh_pericalc':'rh-pericalcarine'}
    return roi_dict

def map_labels():
    temp = {
           'Chest-of':'Chest-of-drawers', 
           'armadill':'armadillo', 
           'armchair':'armchair', 
           'axe':'axe', 
           'barn-owl':'barn-owl', 
           'bed':'bed',
           'bedside-':'bedside-table', 
           'boat':'boat', 
           'bookcase':'bookcase', 
           'bus':'bus', 
           'butterfl':'butterfly', 
           'car':'car', 
           'castle':'castle',
           'cat':'cat', 
           'cathedra':'cathedral', 
           'chair':'chair', 
           'cheetah':'cheetah', 
           'church':'church', 
           'coking-p':'coking-pot',
           'coking-pot':'coking-pot',
           'couch':'couch', 
           'cow':'cow', 
           'crab':'crab', 
           'cup':'cup', 
           'dolphin':'dolphin', 
           'dragonfl':'dragonfly', 
           'drum':'drum',
           'duck':'duck', 
           'elephant':'elephant', 
           'factory':'factory', 
           'filling-':'filling-cabinet', 
           'fondue':'fondue', 
           'frying-p':'frying-pan',
           'giraffe':'giraffe', 
           'goldfinc':'goldfinch', 
           'goose':'goose', 
           'granary':'granary', 
           'guitar':'guitar', 
           'hammer':'hammer',
           'hen':'hen', 
           'hippopot':'hippopotamus', 
           'horse':'horse', 
           'house':'house', 
           'hummingb':'hummingbird', 
           'killer-w':'killer-whale',
           'kiwi':'kiwi', 
           'ladybird':'ladybird', 
           'lamp':'lamp', 
           'lectern':'lectern', 
           'lioness':'lioness', 
           'lobster':'lobster',
           'lynx':'lynx', 
           'magpie':'magpie', 
           'manatee':'manatee', 
           'mill':'mill', 
           'motorbik':'motorbike', 
           'narwhal':'narwhal',
           'ostrich':'ostrich', 
           'owl':'owl', 
           'palace':'palace', 
           'partridg':'partridge', 
           'pelican':'pelican', 
           'penguin':'penguin',
           'piano':'piano_', 
           'pigeon':'pigeon', 
           'plane':'plane', 
           'pomfret':'pomfret', 
           'pot':'pot', 
           'raven':'raven', 
           'rhino':'rhino',
           'rocking-':'rocking-chair', 
           'rooster':'rooster', 
           'saucepan':'saucepan', 
           'saxophon':'saxophone', 
           'scorpion':'scorpion',
           'seagull':'seagull', 
           'shark':'shark', 
           'ship':'ship', 
           'small-sa':'small-saucepan', 
           'sofa':'sofa', 
           'sparrow':'sparrow',
           'sperm-wh':'sperm-whale', 
           'table':'table', 
           'tapir':'tapir', 
           'teapot':'teapot', 
           'tiger':'tiger', 
           'toucan':'toucan',
           'tractor':'tractor', 
           'train':'train', 
           'trumpet':'trumpet', 
           'tuba':'tuba', 
           'turtle':'turtle', 
           'van':'van', 
           'violin':'violin',
           'wardrobe':'wardrobe', 
           'whale':'whale', 
           'zebra':'zebra',
            }
    return temp

def define_roi_category():
    roi_dict = {'fusiform':'Visual',
                'parahippocampal':'Visual',
                'pericalcarine':'Visual',
                'precuneus':'Visual',
                'superiorparietal':'Visual',
                'inferiortemporal':'Working Memory',
                'inferiortemporal':'Working Memory',
                'lateraloccipital':'Working Memory',
                'lingual':'Working Memory',
                'rostralmiddlefrontal':'Working Memory',
                'superiorfrontal':'Working Memory',
                'ventrolateralPFC':'Working Memory',
                }
    
    return roi_dict
































