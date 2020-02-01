#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:43:24 2019

@author: nmei
"""

import os
import gc

from glob import glob
from tqdm import tqdm

import numpy      as np
import pandas     as pd
import tensorflow as tf
import seaborn    as sns

from tensorflow.keras                           import applications,layers,models,optimizers,losses,regularizers
from tensorflow.keras.preprocessing.image       import ImageDataGenerator, img_to_array, load_img

from scipy.spatial import distance
from matplotlib import pyplot as plt

sns.set_context('poster')

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
                                 save_weights_only= True,
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

# define tons of directories
saving_dir  = '../../../data/computer_vision_features'
untun_dir   = '../../../data/computer_vision_raw'
untun_back_d= '../../../data/computer_vision_raw_background'
chan_dir    = '../../../data/computer_vision_chance'
back_dir    = '../../../data/computer_vision_background'
model_dir   = '../../../data/computer_vision_weights'
figure_dir  = '../../../figures/computer_vision_features'
report_dir  = '../../../results/computer_vision_features'
for d in [
          saving_dir,
          untun_dir,
          model_dir,
          figure_dir,
          report_dir,
          chan_dir,
          back_dir,
          untun_back_d,
          ]:
    if not os.path.exists(d):
        os.mkdir(d)
# where are the images we will use for passing through the trained models
working_dir     = '../../../data/'
fine_tune_at    = '101_ObjectCategories_grayscaled'
working_fold    = 'bw_bc_bl'
background_fold = 'experiment_background'
working_data    = np.sort(glob(os.path.join(working_dir,
                                            working_fold,
                                            "*",
                                            "*",
                                            "*.jpg")))
working_background = np.sort(glob(os.path.join(working_dir,
                                            background_fold,
                                            "*",
                                            "*",
                                            "*.jpg")))

# candidate models from the pretrained repository
model_names     = ['DenseNet169',           # 1024
                   'InceptionResNetV2',     # 1536
                   'InceptionV3',           # 2048
                   'MobileNetV2',           # 1280
#                   'NASNetMobile',          # 1024
                   'ResNet50',              # 1536
                   'VGG19',                 # 2048
                   'Xception',              # 1280
                   ]

pretrained_models = [applications.DenseNet169,
                     applications.InceptionResNetV2,
                     applications.InceptionV3,
                     applications.MobileNetV2,
#                     applications.NASNetMobile,
                     applications.ResNet50,
                     applications.VGG19,
                     applications.Xception,
                     ]

preprcessing_funcs = [applications.densenet.preprocess_input,
                      applications.inception_resnet_v2.preprocess_input,
                      applications.inception_v3.preprocess_input,
                      applications.mobilenet_v2.preprocess_input,
#                      applications.nasnet.preprocess_input,
                      applications.resnet50.preprocess_input,
                      applications.vgg19.preprocess_input,
                      applications.xception.preprocess_input,
                      ]

# define some hyperparameters for training
batch_size      = 8
image_resize    = 128
dropout         = False


for model_name,model,preprocess_input in zip(model_names,pretrained_models,preprcessing_funcs):
    # define the training/validation data pipeline. The pipeline will load a subset
    # of all the images in memory whenever they are required, so that saving some 
    # memory
    ## define the augmentation procedures
    gen             = ImageDataGenerator(rotation_range         = 90,               # allow rotation
                                         width_shift_range      = 0.1,              # horizontal schetch
                                         height_shift_range     = 0.1,              # vertical schetch
                                         zoom_range             = 0.1,              # zoom in
                                         horizontal_flip        = True,             # 
                                         vertical_flip          = True,             # 
                                         preprocessing_function = preprocess_input, # scaling function (-1,1)
                                         validation_split       = 0.1,
                                         )
    gen_train       = gen.flow_from_directory(os.path.join(working_dir,fine_tune_at),
                                              target_size       = (image_resize,image_resize),  # resize the image
                                              batch_size        = batch_size,                   # batch size
                                              class_mode        = 'categorical',                # get the labels from the folders
                                              shuffle           = True,                         # shuffle for different epochs
                                              seed              = 12345,                        # replication purpose
                                              subset            = 'training',
                                              )
    
    gen_valid       = gen.flow_from_directory(os.path.join(working_dir,fine_tune_at),
                                               target_size       = (image_resize,image_resize),  # resize the image
                                               batch_size        = batch_size,                   # batch size
                                               class_mode        = 'categorical',                # get the labels from the folders
                                               shuffle           = True,                         # shuffle for different epochs
                                               seed              = 12345,                        # replication purpose
                                               subset            = 'validation',
                                               )
    tf.keras.backend.clear_session()
    
    # after loading the model from the pretrained repository, freeze the parameters
    print(f'loading {model_name} ...')
    model_loaded    = model(weights     = 'imagenet',
                            include_top = False,
                            input_shape = (image_resize,image_resize,3),
                            )
    for layer in model_loaded.layers:
        layer.trainable = False
    
    # now, adding 2 more layers: CNN --> 300 --> discriminative prediction
    drop_rate = 0.5
    fine_tune_model = model_loaded.output
    fine_tune_model = layers.GlobalAveragePooling2D(name = 'Globalave')(fine_tune_model)
    r = models.Model(model_loaded.inputs,fine_tune_model)
    
#    # get outputs from the last convolutional layer
#    for image_name in tqdm(working_data,desc='raw features'):
#        label_              = image_name.split('/')[-1].split('_')[0]
#        image_save_name     = image_name.split('/')[-1].replace('.jpg','.npy')
#        image_loaded        = load_img(image_name,target_size = (image_resize,image_resize,3))
#        image_data          = preprocess_input(img_to_array(image_loaded)[np.newaxis,])
#        feature_            = r.predict(image_data)
#        if not os.path.exists(os.path.join(untun_dir,model_name)):
#            os.mkdir(os.path.join(untun_dir,model_name))
#        np.save(os.path.join(untun_dir,model_name,image_save_name),np.squeeze(feature_))
#    # get outputs of the backgrounds from the last convolutional layer
#    for image_name in tqdm(working_background,desc='raw background'):
#        label_              = image_name.split('/')[-1].split('_')[0]
#        image_save_name     = image_name.split('/')[-1].replace('.jpg','.npy')
#        image_loaded        = load_img(image_name,target_size = (image_resize,image_resize,3))
#        image_data          = preprocess_input(img_to_array(image_loaded)[np.newaxis,])
#        feature_            = r.predict(image_data)
#        if not os.path.exists(os.path.join(untun_back_d,model_name)):
#            os.mkdir(os.path.join(untun_back_d,model_name))
#        np.save(os.path.join(untun_back_d,model_name,image_save_name),np.squeeze(feature_))
    if dropout:
        fine_tune_model = layers.Dropout(drop_rate,
                                         seed                       = 12345,
                                         name                       = 'drop_c2d'
                                         )(fine_tune_model)
    hidden_layer = layers.Dense(300,
                                   activation                       = tf.keras.activations.selu, # SOTA activation function
                                   kernel_initializer               = 'lecun_normal', # seggested in documentation
                                   kernel_regularizer               = regularizers.l2(),
                                   activity_regularizer             = regularizers.l1(),
                                   name                             = 'feature'
                                   )(fine_tune_model)
    if dropout:
        hidden_layer = layers.AlphaDropout(drop_rate,
                                              seed                  = 12345,
                                              name                  = 'drop_d2o'
                                              )(hidden_layer) # suggested in documentation
    fine_tune_model = layers.Dense(len(gen_train.class_indices),
                                   activation                       = 'softmax',
                                   kernel_regularizer               = regularizers.l2(),
                                   activity_regularizer             = regularizers.l1(),
                                   name                             = 'predict'
                                   )(hidden_layer)
    clf             = models.Model(model_loaded.inputs,fine_tune_model)
    # compile the model with an optimizer, a loss function
    clf.compile(optimizers.Adam(lr = 1e-3,),
                losses.categorical_crossentropy,
                metrics = ['categorical_accuracy'])
    # make the output of the new layer as the embedding features
    print(clf.layers[-2].output)
    feature_extractor = models.Model(model_loaded.inputs,clf.layers[-2].output)
    for image_name in tqdm(working_data,desc='before fine-tuning'):
        label_      = image_name.split('/')[-1].split('_')[0]
        image_save_name     = image_name.split('/')[-1].replace('.jpg','.npy')
        image_loaded        = load_img(image_name,target_size = (image_resize,image_resize,3))
        image_data          = preprocess_input(img_to_array(image_loaded)[np.newaxis,])
        feature_            = feature_extractor.predict(image_data)
        if not os.path.exists(os.path.join(chan_dir,model_name)):
            os.mkdir(os.path.join(chan_dir,model_name))
        np.save(os.path.join(chan_dir,model_name,image_save_name),np.squeeze(feature_))
        
    saving_model_name   = os.path.join(model_dir,f'{model_name}_fine_tune.h5')
    callbacks           = make_CallBackList(saving_model_name,
                                            monitor                 = 'val_{}'.format(clf.metrics_names[-2]),
                                            mode                    = 'min',
                                            verbose                 = 0,
                                            min_delta               = 1e-4,
                                            patience                = 2,
                                            frequency               = 1)
    print(f'training {model_name} ...')
    if not os.path.exists(saving_model_name):
        ###############################################################################################################
        ################################# here is where the code is raising the error #################################
        ###############################################################################################################
        clf.fit_generator(gen_train,
                          steps_per_epoch                           = np.ceil(gen_train.n / batch_size),
                          epochs                                    = 1000, # arbitrary choice
                          validation_data                           = gen_valid,
                          callbacks                                 = callbacks,
                          )
    
    if tf.__version__ == "2.0.0":# in tf 2.0
        try:
            clf.load_weights(saving_model_name)
        except:
            del clf
            clf = tf.keras.models.load_model(saving_model_name)
        print(clf.summary())
    else: # in tf 1.0
        clf.load_weights(saving_model_name)
    # make the output of the new layer as the embedding features
    feature_extractor = models.Model(model_loaded.inputs,clf.layers[-2].output)
    for image_name in tqdm(working_background,desc='background'):
        label_      = image_name.split('/')[-1].split('_')[0]
        image_save_name     = image_name.split('/')[-1].replace('.jpg','.npy')
        image_loaded        = load_img(image_name,target_size = (image_resize,image_resize,3))
        image_data          = preprocess_input(img_to_array(image_loaded)[np.newaxis,])
        feature_            = feature_extractor.predict(image_data)
        if not os.path.exists(os.path.join(back_dir,model_name)):
            os.mkdir(os.path.join(back_dir,model_name))
        np.save(os.path.join(back_dir,model_name,image_save_name),np.squeeze(feature_))
        
    # get the features of all the images
    features        = []
    target_labels   = []
    categories      = []
    subcate         = []
    for image_name in tqdm(working_data,desc='fine-tuned'):
        label_      = image_name.split('/')[-1].split('_')[0]
        target_labels.append(label_)
        categories.append(image_name.split('/')[5])
        subcate.append(image_name.split('/')[6])
        image_save_name     = image_name.split('/')[-1].replace('.jpg','.npy')
        image_loaded        = load_img(image_name,target_size = (image_resize,image_resize,3))
        image_data          = preprocess_input(img_to_array(image_loaded)[np.newaxis,])
        feature_            = feature_extractor.predict(image_data)
        if not os.path.exists(os.path.join(saving_dir,model_name)):
            os.mkdir(os.path.join(saving_dir,model_name))
        np.save(os.path.join(saving_dir,model_name,image_save_name),np.squeeze(feature_))
        features.append(feature_)
        
    features        = np.squeeze(np.array(features),1)
    target_labels   = np.array(target_labels)
    
    target_labels = pd.DataFrame(target_labels,columns = ['labels'])
    target_labels['targets'] = categories
    target_labels['subcategory'] = subcate
    target_labels = target_labels.sort_values(['targets','subcategory','labels'])
    
    temp = []
    labels = []
    liv = []
    sub = []
    for (name,cate,subcate),df_sub in target_labels.groupby(['labels','targets','subcategory']):
        idx_ = list(df_sub.index)
        picked = features[idx_].mean(0)
        temp.append(picked)
        labels.append(name)
        liv.append(cate)
        sub.append(subcate)
    temp = np.array(temp)
    temp = temp - temp.mean(1).reshape(-1,1)
    labels = np.array(labels)
    liv = np.array(liv)
    sub = np.array(sub)
    
    df_temp = pd.DataFrame(labels,columns = ['labels'])
    df_temp['targets'] = liv
    df_temp['subcategory'] = sub
    df_temp = df_temp.sort_values(['targets','subcategory','labels'])
    idx_sort = list(df_temp.index)
    temp = temp[idx_sort]
    labels = labels[idx_sort]
    liv = liv[idx_sort]
    
    RDM = distance.squareform(distance.pdist(temp,'cosine'))
    np.fill_diagonal(RDM,np.nan)
    
    fig,ax = plt.subplots(figsize = (30,30))
    im = ax.imshow(RDM,
                   cmap = plt.cm.RdBu_r,
                   vmin = 0,
                   origin = 'lower',
                   alpha = 0.9)
    ax.set(xticks   = np.arange(96),
           yticks   = np.arange(96),
           title    = f'Representational Dissimilarity Matrix\n96 unique items\nimage resize to {image_resize} by {image_resize}, pretrained model: {model_name}')
    ax.set_xticklabels(labels,rotation = 90)
    ax.set_yticklabels(labels)
    ax.axhline(95/2,linestyle='--',alpha=1.,color='black')
    ax.axvline(95/2,linestyle='--',alpha=1.,color='black')
    plt.colorbar(im)
    fig.savefig(os.path.join(figure_dir,f'RDM {model_name}.jpeg'),
                dpi = 500,
                bbox_inches = 'tight',)
    
    from shutil import copyfile
    copyfile('../../utils.py','utils.py')
    from utils import LOO_partition
    idxs_train,idxs_test = LOO_partition(target_labels)
    
    targets_ = np.array([{'Living_Things':0,'Nonliving_Things':1}[item] for item in categories])
    
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import roc_auc_score
    
    cv = zip(idxs_train,idxs_test)#StratifiedKFold(20,shuffle = True,random_state = 12345)
    clf = CalibratedClassifierCV(LinearSVC(class_weight = 'balanced',random_state = 12345),
                                 cv = 3,)
    clf = make_pipeline(StandardScaler(),clf)
    res = cross_validate(clf,features,targets_,scoring = 'roc_auc',cv = cv,
                         n_jobs = 10,return_estimator = True,verbose = 2)
    scores = np.array([roc_auc_score(targets_[idx_test],est.predict_proba(features[idx_test])[:,-1]) for est,idx_test in tqdm(zip(res['estimator'],idxs_test))])#.split(features,labels))])
    
    idx_wrong, = np.where(scores < 1)
    a = f"""{model_name}, leave 2 objects out cross validation (folds = {len(idxs_train)}),
scores = {scores.mean():.4f} +/- {scores.std():.4f}
pairs of living-nonliving objects incorrectly decoded:
"""
    for ii,idx_test,iii in zip([np.unique(item) for item in target_labels['labels'].values[np.array(idxs_test)[idx_wrong]]],
                            np.array(idxs_test)[idx_wrong],
                            idx_wrong):
        item1,item2 = ii
        temp_score = roc_auc_score(targets_[idx_test],res['estimator'][iii].predict_proba(features[idx_test])[:,-1]) 
        a += f'{str(item1):25s},{str(item2):25} ROC_AUC = {temp_score:.4f}\n'
    if not os.path.exists(os.path.join(report_dir,model_name)):
        os.mkdir(os.path.join(report_dir,model_name))
    with open(os.path.join(report_dir,model_name,'report.txt'),'w') as f:
        f.write(a)
        f.close()
    try:
        del model_loaded
    except:
        pass
    for _ in range(5):
        gc.collect()
    






















