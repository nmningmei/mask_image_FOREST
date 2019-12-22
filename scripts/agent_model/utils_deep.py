#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:41:06 2019

@author: nmei
"""

import os
import gc
gc.collect()

import numpy      as np
import tensorflow as tf


from tensorflow.keras                           import applications,layers,models,optimizers,losses,regularizers
from tensorflow.keras.preprocessing.image       import ImageDataGenerator

from sklearn.metrics                            import roc_auc_score
from sklearn.svm                                import LinearSVC,SVC
from sklearn.preprocessing                      import MinMaxScaler
from sklearn.pipeline                           import make_pipeline
from sklearn.model_selection                    import StratifiedShuffleSplit,cross_val_score,permutation_test_score
from sklearn.calibration                        import CalibratedClassifierCV
from sklearn.utils                              import shuffle as sk_shuffle
from sklearn.base                               import clone
from sklearn.linear_model                       import LogisticRegression
from sklearn.exceptions                         import ConvergenceWarning
from sklearn.utils.testing                      import ignore_warnings
from sklearn.ensemble                           import RandomForestClassifier
from sklearn.neighbors                          import KNeighborsClassifier

from scipy                                      import stats

from tqdm                                       import tqdm

from joblib                                     import Parallel,delayed


def resample_ttest(x,
                   baseline         = 0.5,
                   n_ps             = 100,
                   n_permutation    = 10000,
                   one_tail         = False,
                   n_jobs           = 12, 
                   verbose          = 0,
                   ):
    """
    http://www.stat.ucla.edu/~rgould/110as02/bshypothesis.pdf
    https://www.tau.ac.il/~saharon/StatisticsSeminar_files/Hypothesis.pdf
    Inputs:
    ----------
    x: numpy array vector, the data that is to be compared
    baseline: the single point that we compare the data with
    n_ps: number of p values we want to estimate
    one_tail: whether to perform one-tailed comparison
    """
    import gc
    import numpy as np
    
    # t statistics with the original data distribution
    t_experiment    = (np.mean(x) - baseline) / (np.std(x) / np.sqrt(x.shape[0]))
    null            = x - np.mean(x) + baseline # shift the mean to the baseline but keep the distribution
    gc.collect()
    def t_statistics(null,size,):
        """
        null: shifted data distribution
        size: tuple of 2 integers (n_for_averaging,n_permutation)
        """
        null_dist   = np.random.choice(null,size = size,replace = True)
        t_null      = (np.mean(null_dist,0) - baseline) / (np.std(null_dist,0) / np.sqrt(null_dist.shape[0]))
        if one_tail:
            return ((np.sum(t_null >= t_experiment)) + 1) / (size[1] + 1)
        else:
            return ((np.sum(np.abs(t_null) >= np.abs(t_experiment))) + 1) / (size[1] + 1) /2
    ps = Parallel(n_jobs = n_jobs,
                  verbose = verbose
                  )(delayed(t_statistics)(**{
                    'null':null,
                    'size':(null.shape[0],
                            int(n_permutation)),
                                            }) for i in range(n_ps))
    
    return np.array(ps)

def resample_ttest_2sample(a,b,
                           n_ps                 = 100,
                           n_permutation        = 10000,
                           one_tail             = False,
                           match_sample_size    = True,
                           n_jobs               = -1,
                           verbose              = 0,
                           ):
    # when the samples are dependent just simply test the pairwise difference against 0
    # which is a one sample comparison problem
    if match_sample_size:
        difference  = a - b
        ps          = resample_ttest(difference,
                                     baseline       = 0,
                                     n_ps           = n_ps,
                                     n_permutation  = n_permutation,
                                     one_tail       = one_tail,
                                     n_jobs         = n_jobs,
                                     verbose        = verbose,)
        return ps
    else: # when the samples are independent
        t_experiment,_ = stats.ttest_ind(a,b,equal_var = False)
        def t_statistics(a,b):
            group = np.random.choice(np.concatenate([a,b]),size = int(len(a) + len(b)),replace = True)
            new_a = group[:a.shape[0]]
            new_b = group[a.shape[0]:]
            t_null,_ = stats.ttest_ind(new_a,new_b,equal_var = False)
            return t_null
        from joblib import Parallel,delayed
        import gc
        gc.collect()
        ps = np.zeros(n_ps)
        for ii in range(n_ps):
            t_null_null = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(t_statistics)(**{
                            'a':a,
                            'b':b}) for i in range(n_permutation))
            if one_tail:
                ps[ii] = ((np.sum(t_null_null >= t_experiment)) + 1) / (n_permutation + 1)
            else:
                ps[ii] = ((np.sum(np.abs(t_null_null) >= np.abs(t_experiment))) + 1) / (n_permutation + 1) / 2
        return ps

# the most important helper function: early stopping and model saving
def make_CallBackList(model_name,monitor='val_loss',mode='min',verbose=0,min_delta=1e-4,patience=50,frequency = 1):
    from tensorflow.keras.callbacks             import ModelCheckpoint,EarlyStopping
    """
    Make call back function lists for the keras models
    
    Inputs
    -------------------------
    model_name: directory of where we want to save the model and its name
    monitor:    the criterion we used for saving or stopping the model
    mode:       min --> lower the better, max --> higher the better
    verboser:   printout the monitoring messages
    min_delta:  minimum change for early stopping
    patience:   temporal windows of the minimum change monitoring
    frequency:  temporal window steps of the minimum change monitoring
    
    Return
    --------------------------
    CheckPoint:     saving the best model
    EarlyStopping:  early stoppi
    """
    checkPoint = ModelCheckpoint(model_name,# saving path
                                 monitor          = monitor,# saving criterion
                                 save_best_only   = True,# save only the best model
                                 mode             = mode,# saving criterion
#                                 save_freq        = 'epoch',# frequency of check the update 
                                 verbose          = verbose,# print out (>1) or not (0)
#                                 load_weights_on_restart = True,
                                 )
    earlyStop = EarlyStopping(   monitor          = monitor,
                                 min_delta        = min_delta,
                                 patience         = patience,
                                 verbose          = verbose, 
                                 mode             = mode,
#                                 restore_best_weights = True,
                                 )
    return [checkPoint,earlyStop]

def train_validation_generator(preprocess_input,
                               working_dir = '',
                               sub_folder = ['101_ObjectCategories','101_ObjectCategories'],
                               image_resize = 128,
                               batch_size = 32,
                               class_mode = 'categorical',
                               shuffle = [True,True],
                               less_intense_validation = False,
                               ):
    """
    Inputs 
    ------------------------
    preprocess_input: keras object, the preprocessing function used by the generator
    working_dir: string, working directory for finding the image directory
    sub_folder: list of strings, list of image directories
    image_resize: int, resize the image to N x N pixels
    batch_size: int, batch size
    shuffle: list of boolean, whether to shuffle the train and validation set
    less_intense_validation: boolean, customized control parameter
    
    Outputs
    ------------------------
    image generators: train and validation
    """
    gen             = ImageDataGenerator(rotation_range         = 45,               # allow rotation
                                         width_shift_range      = 0.1,              # horizontal schetch
                                         height_shift_range     = 0.1,              # vertical schetch
                                         zoom_range             = 0.1,              # zoom in
                                         horizontal_flip        = True,             # 
                                         vertical_flip          = True,             # 
                                         preprocessing_function = preprocess_input, # scaling function (-1,1)
                                         validation_split       = 0.1,              # validation split raio
                                         )
    gen_train       = gen.flow_from_directory(os.path.join(working_dir,sub_folder[0]), # train
                                              target_size       = (image_resize,image_resize),  # resize the image
                                              batch_size        = batch_size,                   # batch size
                                              class_mode        = class_mode,                   # get the labels from the folders
                                              shuffle           = shuffle[0],                   # shuffle for different epochs
                                              seed              = 12345,                        # replication purpose
                                              subset            = 'training'
                                              )
    if less_intense_validation:
        gen_            = ImageDataGenerator(preprocessing_function = preprocess_input,
                                             rotation_range         = 25,               # allow rotation
                                             width_shift_range      = 0.01,              # horizontal schetch
                                             height_shift_range     = 0.01,              # vertical schetch
                                             zoom_range             = 0.01,              # zoom in
                                             horizontal_flip        = True,             # 
                                             vertical_flip          = True,             # 
                                             )
        gen_valid       = gen_.flow_from_directory(os.path.join(working_dir,sub_folder[1]), # validate
                                                   target_size      = (image_resize,image_resize),  # resize the image
                                                   batch_size       = batch_size,                   # batch size
                                                   class_mode       = class_mode,                   # get the labels from the folders
                                                   shuffle          = shuffle[1],                   # shuffle for different epochs
                                                   seed             = 12345,                        # replication purpose
                                                   )
    else:
        gen_valid       = gen.flow_from_directory(os.path.join(working_dir,sub_folder[1]), # validate
                                                   target_size      = (image_resize,image_resize),  # resize the image
                                                   batch_size       = batch_size,                   # batch size
                                                   class_mode       = class_mode,                   # get the labels from the folders
                                                   shuffle          = shuffle[1],                   # shuffle for different epochs
                                                   seed             = 12345,                        # replication purpose
                                                   subset           = 'validation',
                                                   )
    return gen_train,gen_valid

def build_computer_vision_model(model_pretrained,
                                image_resize = 128,
                                hidden_units = 300,
                                output_size = 2,
                                model_name = '',
                                hidden_activation = 'selu',
                                drop_rate = None
                                ):
    """
    Inputs
    ----------------
    model_pretrained: keras object, a pretrained model function that is callable
    image_resize: int, since the images are resized during the data flowing pipeline, we need to specify it here
    hidden_units: int, the number of artificial neurons in the hidden layer
    output_size: int, the number of classes of the predictions
    model_name: string, the name of the model, optional
    drop_rate: float, between 0 and 1., the dropout rate between the hidden layer and the classification layer
    
    Outputs
    ----------------
    clf: keras object, a keras high-level API of classification model
    """
    np.random.seed(12345)
    try:
        tf.random.set_random_seed(12345)
    except:
        tf.random.set_seed(12345)
    model_loaded    = model_pretrained(weights      = 'imagenet',
                                       include_top  = False,
                                       input_shape  = (image_resize,image_resize,3),
                                       pooling      = 'max',
                                       )
    for layer in model_loaded.layers:
        layer.trainable = False
    
    # now, adding 2 more layers: CNN --> hidden_units --> discriminative prediction
    fine_tune_model = model_loaded.output
    #fine_tune_model = layers.Dropout(drop_rate,name = 'feature_drop')(fine_tune_model)
    fine_tune_model = layers.Dense(hidden_units,
                                   activation                       = hidden_activation,
                                   kernel_initializer               = 'lecun_normal', # seggested in documentation
                                   kernel_regularizer               = regularizers.l2(),
                                   name                             = 'feature'
                                   )(fine_tune_model)
    if drop_rate > 0:
        fine_tune_model = layers.AlphaDropout(rate = drop_rate,
                                              seed = 12345,
                                              name = 'predict_drop')(fine_tune_model)
    fine_tune_model = layers.Dense(output_size,
                                   activation                       = 'softmax',
                                   activity_regularizer             = regularizers.l1(),
                                   name                             = 'predict'
                                   )(fine_tune_model)
    clf             = models.Model(model_loaded.inputs,fine_tune_model,name = model_name)
    
    return clf

def process_func(x,preprocess_input,var = 1e2):
    """
    Inputs
    ----------------
    x: numpy array, the image that is converted into numpy array
    preprocess_input: keras object, the preprocessing function of the proposed pre-trained model
    var: int or float, to control the noise level
    
    Outputs:
    ---------------
    noise: numpy array, the image array that is added Gaussian noise
    """
    row,col,ch  = x.shape
    mean        = 0
    var         = var
    sigma       = var**0.5
    gauss       = np.random.normal(mean,sigma,(row,col,ch))
    gauss       = gauss.reshape(row,col,ch)
    noise       = x + gauss
    noise       = preprocess_input(noise)
    return noise

def decoder_dict(name = 'linear-svm',
                 knn_n_neighbors = 5,):
    if name == 'linear-svm':
        np.random.seed(12345)
        svm = LinearSVC(penalty         = 'l2',         # default
                        dual            = True,         # default
                        tol             = 1e-3,         # not default
                        random_state    = 12345,        # not default
                        max_iter        = int(1e3),     # default
                        class_weight    = 'balanced',   # not default
                        )
        decoder = CalibratedClassifierCV(
                        base_estimator  = svm,
                        method          = 'sigmoid',
                        cv              = 8)
    elif name == 'rbf-svm':
        svm = SVC(kernel                = 'rbf',
                  class_weight          = 'balanced',
                  max_iter              = int(1e3),
                  tol                   = 1e-3,
                  random_state          = 12345,
                  gamma                 = 'scale',
                  )
        decoder = CalibratedClassifierCV(
                        base_estimator  = svm,
                        method          = 'sigmoid',
                        cv              = 8)
    elif name == 'randomforest':
        rf = RandomForestClassifier(n_estimators = 100,
                                    max_depth = 3,
                                    criterion = 'entropy',
                                    n_jobs = 4,
                                    oob_score = True,
                                    class_weight = 'balanced',
                                    )
        decoder = clone(rf)
    elif name == 'logistic':
        logistic = LogisticRegression(
                    class_weight        = 'balanced',
                    random_state        = 12345,
                    solver              = 'liblinear',
                                      )
        decoder = clone(logistic)
    elif name == 'knn':
        knn = KNeighborsClassifier(n_neighbors = knn_n_neighbors)
        decoder = clone(knn)
    return decoder
    
    
def decode_hidden(decoder_name,
                  hidden_features,
                  labels,
                  n_splits = 100,
                  verbose = 1,
                  n_permutations = int(2e2),
                  knn_n_neighbors = 5,
                  ):
    """
    Inputs:
    ------------------
    decoder_name: string, name of the decoder, call decoder_dict to generate the sckikit-learn classifier
    hidden_features: numpy array, n_instance by n_features
    labels: numpy array, n_instance by n_classes, one-hot encoded
    n_splits: int, the number of cross validation to perform
    verbose:
    n_permutations: int, number of permutation cycles, higher --> better estimate of p val
    
    Outputs:
    -------------------
    res: dictionary of scikit-learn output, the results of the cross-validation
    chance: dictionary of scikit-learn output, the empirical chances of the cross-validation 
    """
    np.random.seed(12345)
    reps,labels     = sk_shuffle(hidden_features,labels)
    
    decoder         = decoder_dict(name             = decoder_name,
                                   knn_n_neighbors  = knn_n_neighbors)
    pipeline        = make_pipeline(MinMaxScaler(),
                                    decoder)
    cv              = StratifiedShuffleSplit(n_splits        = n_splits,
                                             test_size       = 0.2,
                                             random_state    = 12345)
    with ignore_warnings(category = ConvergenceWarning):
        if verbose == 0: print('permutation')
        res,permu_scores,pval = permutation_test_score(pipeline,reps,labels,
                                                       cv               = cv,
                                                       n_permutations   = n_permutations,
                                                       n_jobs           = -1,
                                                       random_state     = 12345,
                                                       verbose          = verbose,
                                                       scoring          = 'roc_auc',)
        if verbose == 0: print('cross validation')
        res = cross_val_score(pipeline,
                              reps,
                              labels,
                              cv        = cv,
                              n_jobs    = -1,
                              verbose   = verbose,
                              )
    return res,permu_scores,pval

def performance_of_CNN_and_get_hidden_features(
               classifier,
               gen,
               working_dir          = '',
               folder_name          = '',
               image_resize         = 128,
               n_sessions           = int(5e2),
               batch_size           = 32,
               get_hidden           = False,
               hidden_model         = None,
               save_agumentations   = False,
               saving_dir           = '',
               n_jobs               = 12,
               verbose              = 1,
               n_permutation        = int(1e3),
               ):
    """
    Inputs:
    -------------------------------
    classifier: keras object, the trained keras high-level API model
    gen: keras object, the pre-defined image generator with preprocessing function specified #TODO how to catch this?
    working_dir: string, mother directory of the image directories
    folder_name: string, image directory
    image_resize: int, the image size used in training the model, N x N pixels
    n_sessions: int, number of times passing the image data flow to the classifier
    batch_size: int, batch size
    get_hidden: boolean, whether to get the hidden layer representations of the images
    hidden_model: keras object, the trained keras high-level API model
    save_agumentations: boolean, whether to save the images generated by the image generator
    saving_dir: string, directory to save the agumented images
    Outputs:
    -----------------------------
    behavioral: numpy array, (n_sessions,), performance of the classifier with permutation estimation
    beha_chance: numpy array, (n_sessions,), empirical chance performance of the classifier with permutation estimation
    ps: p value of comparing the performance against to its empirical chance performance
    hidden_features: numpy array, (n_session x batch_size, n_features), outputs of the hidden layer
    y_true: numpy array, (n_session x batch_size, n_classes), real labels, one-hot encoded
    y_pred: numpy array, (n_session x batch_size, n_classes), predictions generated by the classifier
    
    """
    if save_agumentations:
        images_flow = gen.flow_from_directory(os.path.join(working_dir,folder_name),
                                              target_size       = (image_resize,image_resize),  # resize the image
                                              batch_size        = batch_size,                   # batch size
                                              class_mode        = 'categorical',                # get the labels from the folders
                                              shuffle           = True,                        # shuffle for different epochs
                                              seed              = 12345,                        # replication purpose
                                              save_to_dir       = saving_dir,
                                              save_format       = 'png',
                                              )
    else:
        images_flow = gen.flow_from_directory(os.path.join(working_dir,folder_name),
                                              target_size       = (image_resize,image_resize),  # resize the image
                                              batch_size        = batch_size,                   # batch size
                                              class_mode        = 'categorical',                # get the labels from the folders
                                              shuffle           = True,
                                              seed              = 12345,
                                              )
    desc = 'feature' if get_hidden else 'CNN'
    y_true          = []
    y_pred          = []
    hidden_features = []
    for n_ in tqdm(range(n_sessions),desc=desc):
        image_arrays,labels     = images_flow.next()
        preds                   = classifier.predict_on_batch(image_arrays,)
        if get_hidden and (hidden_model is not None):
            hidden_features.append(hidden_model.predict_on_batch(image_arrays,))
        y_true.append(labels)
        y_pred.append(preds)
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    if y_pred.shape[1] < 2:
        y_true = y_true[:,-1]
    
    if get_hidden and (hidden_model is not None):
        hidden_features = np.concatenate(hidden_features)
    
    behavioral = np.array(
            [roc_auc_score(y_true[idx],y_pred[idx]) for idx in np.random.choice(np.arange(y_true.shape[0]),
                                                                                size    = (100,y_true.shape[0] * 2),
                                                                                replace = True)])
    from joblib import Parallel,delayed
    import gc
    gc.collect()
    def randomized():
        return np.mean(
            [roc_auc_score(y_true[idx],sk_shuffle(y_pred[idx])) for idx in np.random.choice(np.arange(y_true.shape[0]),
                                                                                            size    = (100,y_true.shape[0] * 2),
                                                                                            replace = True)
            ]
            )
    beha_chance = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(randomized)(**{}) for i in range(n_permutation))
    mean_behavioral = behavioral.mean(0)
    mean_chance = np.array(beha_chance).copy()
    ps = ((np.sum(mean_chance > mean_behavioral)) + 1) / (n_permutation + 1)
    if get_hidden and (hidden_model is not None):
        return behavioral,beha_chance,ps,hidden_features,y_true,y_pred
    else:
        return behavioral,beha_chance,ps