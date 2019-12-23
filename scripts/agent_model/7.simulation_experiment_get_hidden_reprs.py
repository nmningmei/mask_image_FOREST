#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:52:57 2019

@author: nmei
"""

import os
import gc
gc.collect()

import multiprocessing
print(f'ncpu = {multiprocessing.cpu_count()}')

import numpy      as np
import pandas     as pd
import tensorflow as tf
#import seaborn    as sns

from tensorflow.keras                           import applications,layers,models,optimizers,losses,regularizers
from tensorflow.keras.preprocessing.image       import ImageDataGenerator

import utils_deep

from functools  import partial
#from matplotlib import pyplot as plt
#from matplotlib import ticker as mtick
#sns.set_style('whitegrid')
#sns.set_context('talk')

with tf.device('/device:GPU:0'):
    working_dir         = '../../data'
    # define some hyperparameters for training
    batch_size          = 16
    image_resize        = 128
    drop_rate           = 0
    hidden_activation   = 'relu'
    output_activation   = 'sigmoid'
    model_name          = 'DenseNet169'
    model_pretrained    = applications.DenseNet169
    preprocess_input    = applications.densenet.preprocess_input
    patience            = 5
    n_splits            = 50 # n_split for decoding the hidden layer
    n_permutations      = 200 
    n_sessions          = int(2e2) # n_permutations for CNN performance
    loss_func           = losses.binary_crossentropy
    hidden_units        = 10
    verbose             = 1
    max_epochs          = int(1e4) # arbitrary choice
    model_dir           = f'../../results/agent_models/{model_name}_{drop_rate}_{hidden_units}_{hidden_activation}_{output_activation}'
    #figure_dir          = f'../../figures/agent_models/{model_name}_{drop_rate}_{hidden_units}_{hidden_activation}_{output_activation}'
    #saving_dir          = f'../../results/agent_models/{model_name}_{drop_rate}_{hidden_units}_{hidden_activation}_{output_activation}'
    
    colorful            = 'experiment_images_tilted'
    black_and_white     = 'greyscaled'
    noisy               = 'bw_bc_bl'
    caltech101          = '101_ObjectCategories'
    experiment96        = 'experiment_images_grayscaled'
    #noisy_im_examples   = os.path.join(figure_dir,f'{hidden_units}units')
    
    #for d in [model_dir,figure_dir,saving_dir]:
    #    if not os.path.exists(d):
    #        os.makedirs(d)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    
    class_mode_dict     = {'softmax':'categorical',
                           'sigmoid':'binary',}
    
    # first - fine tune on celtech101 data
    gen_train,gen_valid = utils_deep.train_validation_generator(preprocess_input,
                                                                working_dir             = working_dir,
                                                                sub_folder              = [caltech101] * 2,
                                                                image_resize            = image_resize,
                                                                batch_size              = batch_size,
                                                                class_mode              = 'categorical',
                                                                shuffle                 = [True,True],
                                                                less_intense_validation = False,)
    tf.keras.backend.clear_session()
    print(f'\nbuild pre-trained {model_name} ...\n')
    clf = utils_deep.build_computer_vision_model(model_pretrained,
                                                 image_resize       = image_resize,
                                                 hidden_units       = hidden_units,
                                                 output_size        = len(gen_train.class_indices),
                                                 model_name         = model_name + '_classifier',
                                                 drop_rate          = drop_rate,
                                                 hidden_activation  = hidden_activation,
                                                 )
    #print(clf.trainable_weights)
    clf.compile(optimizers.Adam(lr = 1e-4,),
                loss_func,
                metrics = ['categorical_accuracy'])
    saving_model_name   = os.path.join(model_dir,f'{model_name}_{hidden_units}_caltech101.h5')
    callbacks           = utils_deep.make_CallBackList(saving_model_name,
                                                       monitor                 = 'val_{}'.format(clf.metrics_names[-2]),
                                                       mode                    = 'min',
                                                       verbose                 = 0,
                                                       min_delta               = 1e-4,
                                                       patience                = patience,
                                                       frequency               = 1)
    
    if not os.path.exists(saving_model_name):
        print(f'training {model_name} ...')
        clf.fit_generator(gen_train,
                          steps_per_epoch                           = np.ceil(gen_train.n / batch_size),
                          epochs                                    = max_epochs,
                          validation_data                           = gen_valid,
                          callbacks                                 = callbacks,
                          validation_steps                          = np.ceil(gen_valid.n / batch_size),
                          verbose                                   = verbose,)
    else:
        print(f'loading {model_name}')
        if tf.__version__ == "2.0.0":# in tf 2.0
            del clf
            clf = tf.keras.models.load_model(saving_model_name)
        else: # in tf 1.0
            clf.load_weights(saving_model_name)
    # then - train on the experiment data
    gen_train,gen_valid = utils_deep.train_validation_generator(preprocess_input,
                                                                working_dir             = working_dir,
                                                                sub_folder              = [noisy,black_and_white],
                                                                image_resize            = image_resize,
                                                                batch_size              = batch_size,
                                                                class_mode              = class_mode_dict[output_activation],
                                                                shuffle                 = [True,True],
                                                                less_intense_validation = True,)
    try:
        tf.random.set_random_seed(12345)
    except:
        tf.random.set_seed(12345)
    print('build CNN binary classifier')
    units = 2 if output_activation == 'softmax' else 1
    binary_classifier   = layers.Dense(units,
                                       activation                       = output_activation,
                                       activity_regularizer             = regularizers.l1(),
                                       name                             = 'predict'
                                       )(clf.layers[-2].output)
    classifier          = models.Model(clf.inputs,binary_classifier)
    
    #print(classifier.trainable_weights)
    # compile the model with an optimizer, a loss function
    classifier.compile(optimizers.Adam(lr = 1e-4,),
                       loss_func,
                       metrics = ['categorical_accuracy'])
    saving_model_name   = os.path.join(model_dir,f'{model_name}_{hidden_units}_binary.h5')
    callbacks           = utils_deep.make_CallBackList(saving_model_name,
                                                       monitor                 = 'val_{}'.format(clf.metrics_names[-2]),
                                                       mode                    = 'min',
                                                       verbose                 = 0,
                                                       min_delta               = 1e-4,
                                                       patience                = patience,
                                                       frequency               = 1)
    
    if not os.path.exists(saving_model_name):
        print(f'training {model_name} ...')
        classifier.fit_generator(gen_train,
                                 steps_per_epoch                            = np.ceil(gen_train.n / batch_size),
                                 epochs                                     = max_epochs,
                                 validation_data                            = gen_valid,
                                 callbacks                                  = callbacks,
                                 validation_steps                           = np.ceil(gen_valid.n / batch_size),
                                 verbose                                    = verbose,
                                 )
    else:
        print(f'loading {model_name}')
        if tf.__version__ == "2.0.0":# in tf 2.0
            del classifier
            classifier = tf.keras.models.load_model(saving_model_name)
        else: # in tf 1.0
            classifier.load_weights(saving_model_name)
    classifier.trainable = False
    for layer in classifier.layers:
        layer.trainable = False
    
    # test on clear images to estimate the "best performance"
    gen = ImageDataGenerator(
                            featurewise_center              = False,
                            samplewise_center               = False,
                            featurewise_std_normalization   = False,
                            samplewise_std_normalization    = False,
                            zca_whitening                   = False,
                            zca_epsilon                     = 1e-06,
                            rotation_range                  = 45,
                            width_shift_range               = 0.2,
                            height_shift_range              = 0.2,
                            brightness_range                = None,
                            shear_range                     = 0.01,
                            zoom_range                      = 0.02,
                            channel_shift_range             = 0.0,
                            fill_mode                       = 'nearest',
                            cval                            = 0.0,
                            horizontal_flip                 = True,
                            vertical_flip                   = True, # change from experiment
                            preprocessing_function          = preprocess_input,
                            data_format                     = None,
                            validation_split                = 0.0
                        )
    
    behavioral,beha_chance,ps = utils_deep.performance_of_CNN_and_get_hidden_features(
                   classifier,
                   gen,
                   working_dir  = working_dir,
                   folder_name  = experiment96,
                   image_resize = image_resize,
                   n_sessions   = n_sessions,
                   batch_size   = batch_size,
                   hidden_model = None,
                   verbose      = verbose,
                   n_jobs       = -1,
                   )
    print(f'test on clear images, performance = {behavioral.mean():.3f}+/-{behavioral.std():.0e} vs {np.mean(beha_chance):.3f}+/-{np.std(beha_chance):.0e},p = {ps:.3f}')
    
    hidden_model = models.Model(classifier.input,classifier.layers[-2].output,name = 'hidden')
    
    
    simulation_saving_name = os.path.join(model_dir,'scores as a function of decoder and noise.csv')
    
    # add noise to the images
    noise_levels = np.concatenate([[a * 10 ** b for a in np.arange(1,10) for b in np.arange(4,9)]]) #[1e2,1e3],
    noise_levels = np.sort(noise_levels)
    if not os.path.exists(simulation_saving_name):
        df_temp = dict(noise_level      = [],
                       decoder          = [],
                       performance_mean = [],
                       performance_std  = [],
                       chance_mean      = [],
                       chance_std       = [],
                       pval             = [],
                       n_permute        = [],
                       )
        beha_chance = np.array(beha_chance)
        df_temp['noise_level'       ].append(0)
        df_temp['decoder'           ].append('CNN')
        df_temp['performance_mean'  ].append(behavioral.mean())
        df_temp['performance_std'   ].append(behavioral.std())
        df_temp['chance_mean'       ].append(beha_chance.mean())
        df_temp['chance_std'        ].append(beha_chance.std())
        df_temp['pval'              ].append(ps.mean())
        df_temp['n_permute'         ].append(n_sessions)
    else:
        temp = pd.read_csv(simulation_saving_name)
        df_temp = {col_name:list(temp[col_name]) for col_name in temp.columns}
        
    for var in (noise_levels):
        if var not in df_temp['noise_level']:
            noise_folder = os.path.join(model_dir,f'{var:.0e}')
            if not os.path.exists(noise_folder):
                os.mkdir(noise_folder)
            if not os.path.exists(os.path.join(noise_folder,'features.npy')):
                noise_func = partial(utils_deep.process_func,preprocess_input = preprocess_input,var = var)
                gen = ImageDataGenerator(
                                        featurewise_center              = False,
                                        samplewise_center               = False,
                                        featurewise_std_normalization   = False,
                                        samplewise_std_normalization    = False,
                                        zca_whitening                   = False,
                                        zca_epsilon                     = 1e-06,
                                        rotation_range                  = 45,
                                        width_shift_range               = 0.2,
                                        height_shift_range              = 0.2,
                                        brightness_range                = None,
                                        shear_range                     = 0.01,
                                        zoom_range                      = 0.02,
                                        channel_shift_range             = 0.0,
                                        fill_mode                       = 'nearest',
                                        cval                            = 0.0,
                                        horizontal_flip                 = True,
                                        vertical_flip                   = True, # change from experiment
                                        preprocessing_function          = noise_func, #
                                        data_format                     = None,
                                        validation_split                = 0.0
                                        )
                #behavioral,beha_chance,ps = utils_deep.performance_of_CNN_and_get_hidden_features(
                #               classifier,
                #               gen,
                #               working_dir = working_dir,
                #               folder_name = experiment96,
                #               image_resize = image_resize,
                #               n_sessions = 100,
                #               batch_size = batch_size,
                #               get_hidden = False, #
                #               hidden_model = hidden_model,
                #               save_agumentations = False,
                #               )
                #print(f'test on noisy images, performance = {behavioral.mean():.3f}+/-{behavioral.std():.2f} vs {beha_chance.mean():.3f}+/-{beha_chance.std():.2f},p = {ps.mean():.3f}+/-{ps.std():.3f}')
                
                behavioral,beha_chance,ps,hidden_features,y_true,y_pred = utils_deep.performance_of_CNN_and_get_hidden_features(
                               classifier,
                               gen,
                               working_dir          = working_dir,
                               folder_name          = experiment96,
                               image_resize         = image_resize,
                               n_sessions           = n_sessions,
                               batch_size           = batch_size,
                               get_hidden           = True,
                               hidden_model         = hidden_model,
                               save_agumentations   = False, #
                               saving_dir           = '',
                               verbose              = verbose,
                               n_jobs               = -1,
                               )
                beha_chance = np.array(beha_chance)
                df_temp['noise_level'       ].append(var)
                df_temp['decoder'           ].append('CNN')
                df_temp['performance_mean'  ].append(behavioral.mean())
                df_temp['performance_std'   ].append(behavioral.std())
                df_temp['chance_mean'       ].append(beha_chance.mean())
                df_temp['chance_std'        ].append(beha_chance.std())
                df_temp['pval'              ].append(ps.mean())
                df_temp['n_permute'         ].append(n_sessions)
                print(f'test on {var:.0e}, performance = {behavioral.mean():.3f}+/-{behavioral.std():.0e} vs {np.mean(beha_chance):.3f}+/-{np.std(beha_chance):.0e},p = {ps:.3f}')
                
                try:
                    decode_targets = y_true[:,-1]
                except:
                    decode_targets = y_true.copy()
                
                np.save(os.path.join(noise_folder,'features.npy'),hidden_features)
                np.save(os.path.join(noise_folder,'labels.npy'),decode_targets,)
                
                df_to_save = pd.DataFrame(df_temp)
                df_to_save.to_csv(simulation_saving_name,index = False)

















