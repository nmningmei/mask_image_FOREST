#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:42:40 2019

@author: nmei
"""

import os
import gc
gc.collect()

import numpy      as np
import tensorflow as tf


from tensorflow.keras                           import applications,layers,models,optimizers,losses,regularizers
from tensorflow.keras.preprocessing.image       import ImageDataGenerator


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

working_dir         = '../../data'
# define some hyperparameters for training
batch_size          = 8
image_resize        = 128
drop_rate           = 0.25
model_name          = 'VGG19'
model_pretrained    = applications.VGG19
preprocess_input    = applications.vgg19.preprocess_input
model_dir           = ''
patience            = 5
loss_func           = losses.binary_crossentropy

# define the training/validation data pipeline. The pipeline will load a subset
# of all the images in memory whenever they are required, so that saving some 
# memory
## define the augmentation procedures
colorful            = 'experiment_images_tilted'
black_and_white     = 'greyscaled'
noisy               = 'bw_bc_bl'
caltech101          = '101_ObjectCategories'

gen             = ImageDataGenerator(rotation_range         = 45,               # allow rotation
                                     width_shift_range      = 0.1,              # horizontal schetch
                                     height_shift_range     = 0.1,              # vertical schetch
                                     zoom_range             = 0.1,              # zoom in
                                     horizontal_flip        = True,             # 
                                     vertical_flip          = True,             # 
                                     preprocessing_function = preprocess_input, # scaling function (-1,1)
                                     validation_split       = 0.1,              # validation split raio
                                     )
gen_train       = gen.flow_from_directory(os.path.join(working_dir,caltech101), # train
                                          target_size       = (image_resize,image_resize),  # resize the image
                                          batch_size        = batch_size,                   # batch size
                                          class_mode        = 'categorical',                # get the labels from the folders
                                          shuffle           = True,                         # shuffle for different epochs
                                          seed              = 12345,                        # replication purpose
                                          subset            = 'training'
                                          )
gen_valid       = gen.flow_from_directory(os.path.join(working_dir,caltech101), # validate
                                           target_size      = (image_resize,image_resize),  # resize the image
                                           batch_size       = batch_size,                   # batch size
                                           class_mode       = 'categorical',                # get the labels from the folders
                                           shuffle          = True,                         # shuffle for different epochs
                                           seed             = 12345,                        # replication purpose
                                           subset           = 'validation',
                                           )

tf.keras.backend.clear_session()

# after loading the model from the pretrained repository, freeze the parameters
print(f'\nloading {model_name} ...\n')
np.random.seed(12345)
tf.random.set_random_seed(12345)
model_loaded    = model_pretrained(weights      = 'imagenet',
                                   include_top  = False,
                                   input_shape  = (image_resize,image_resize,3),
                                   pooling      = 'max',
                                   )
for layer in model_loaded.layers:
    layer.trainable = False

# now, adding 2 more layers: CNN --> 300 --> discriminative prediction
fine_tune_model = model_loaded.output
#fine_tune_model = layers.Dropout(drop_rate,name = 'feature_drop')(fine_tune_model)
fine_tune_model = layers.Dense(300,
                               activation                       = tf.keras.activations.selu, # SOTA activation function
                               kernel_initializer               = 'lecun_normal', # seggested in documentation
#                               kernel_regularizer               = regularizers.l2(),
#                               activity_regularizer             = regularizers.l1(),
                               name                             = 'feature'
                               )(fine_tune_model)
#fine_tune_model = layers.Dropout(drop_rate,name = 'predict_drop')(fine_tune_model)
fine_tune_model = layers.Dense(len(gen_train.class_indices),
                               activation                       = 'softmax',
#                               kernel_regularizer               = regularizers.l2(),
                               activity_regularizer             = regularizers.l1(),
                               name                             = 'predict'
                               )(fine_tune_model)
clf             = models.Model(model_loaded.inputs,fine_tune_model)
print(clf.trainable_weights)
# compile the model with an optimizer, a loss function
clf.compile(optimizers.Adam(lr = 1e-4,),
            loss_func,
            metrics = ['categorical_accuracy'])
saving_model_name   = os.path.join(model_dir,f'{model_name}_caltech101.h5')
callbacks           = make_CallBackList(saving_model_name,
                                        monitor                 = 'val_{}'.format(clf.metrics_names[-1]),
                                        mode                    = 'max',
                                        verbose                 = 0,
                                        min_delta               = 1e-4,
                                        patience                = patience,
                                        frequency               = 1)
print(f'training {model_name} ...')
if not os.path.exists(saving_model_name):
    clf.fit_generator(gen_train,
                      steps_per_epoch                           = np.ceil(gen_train.n / batch_size),
                      epochs                                    = 1000, # arbitrary choice
                      validation_data                           = gen_valid,
                      callbacks                                 = callbacks,
                      )

# train one the experimental images

## define the augmentation procedures
gen             = ImageDataGenerator(rotation_range         = 45,               # allow rotation
                                     width_shift_range      = 0.1,              # horizontal schetch
                                     height_shift_range     = 0.1,              # vertical schetch
                                     zoom_range             = 0.1,              # zoom in
                                     horizontal_flip        = True,             # 
                                     vertical_flip          = True,             # 
                                     preprocessing_function = preprocess_input, # scaling function (-1,1)
                                     )
gen_train       = gen.flow_from_directory(os.path.join(working_dir,noisy + '_sub'), # train
                                          target_size       = (image_resize,image_resize),  # resize the image
                                          batch_size        = batch_size,                   # batch size
                                          class_mode        = 'categorical',                # get the labels from the folders
                                          shuffle           = True,                         # shuffle for different epochs
                                          seed              = 12345,                        # replication purpose
                                          )

gen_            = ImageDataGenerator(preprocessing_function = preprocess_input,
                                     rotation_range         = 25,               # allow rotation
                                     width_shift_range      = 0.01,              # horizontal schetch
                                     height_shift_range     = 0.01,              # vertical schetch
                                     zoom_range             = 0.01,              # zoom in
                                     horizontal_flip        = True,             # 
                                     vertical_flip          = True,             # 
                                     )
gen_valid       = gen_.flow_from_directory(os.path.join(working_dir,black_and_white + '_sub'), # validate
                                           target_size       = (image_resize,image_resize),  # resize the image
                                           batch_size        = batch_size,                   # batch size
                                           class_mode        = 'categorical',                # get the labels from the folders
                                           shuffle           = True,                         # shuffle for different epochs
                                           seed              = 12345,                        # replication purpose
                                           )

clf.load_weights(saving_model_name)
tf.random.set_random_seed(12345)
binary_classifier   = layers.Dense(len(gen_train.class_indices),
                                   activation                       = 'softmax',
#                                   kernel_regularizer               = regularizers.l2(),
                                   activity_regularizer             = regularizers.l1(),
                                   name                             = 'predict'
                                   )(clf.layers[-2].output)
classifier          = models.Model(model_loaded.inputs,binary_classifier)

print(classifier.trainable_weights)

# compile the model with an optimizer, a loss function
classifier.compile(optimizers.Adam(lr = 1e-4,),
                   loss_func,
                   metrics = ['categorical_accuracy'])
saving_model_name   = os.path.join(model_dir,f'{model_name}_experimental.h5')
callbacks           = make_CallBackList(saving_model_name,
                                        monitor                 = 'val_{}'.format(clf.metrics_names[-1]),
                                        mode                    = 'max',
                                        verbose                 = 0,
                                        min_delta               = 1e-4,
                                        patience                = patience,
                                        frequency               = 1)
print(f'training {model_name} ...')
if not os.path.exists(saving_model_name):
    classifier.fit_generator(gen_train,
                             steps_per_epoch                           = np.ceil(gen_train.n / batch_size),
                             epochs                                    = 1000, # arbitrary choice
                             validation_data                           = gen_valid,
                             callbacks                                 = callbacks,
                             )

#########################################################################################################################
## define the augmentation procedures
gen             = ImageDataGenerator(rotation_range         = 45,               # allow rotation
                                     width_shift_range      = 0.1,              # horizontal schetch
                                     height_shift_range     = 0.1,              # vertical schetch
                                     zoom_range             = 0.1,              # zoom in
                                     horizontal_flip        = True,             # 
                                     vertical_flip          = True,             # 
                                     preprocessing_function = preprocess_input, # scaling function (-1,1)
                                     )
gen_train       = gen.flow_from_directory(os.path.join(working_dir,noisy), # train
                                          target_size       = (image_resize,image_resize),  # resize the image
                                          batch_size        = batch_size,                   # batch size
                                          class_mode        = 'categorical',                # get the labels from the folders
                                          shuffle           = True,                         # shuffle for different epochs
                                          seed              = 12345,                        # replication purpose
                                          )

gen_            = ImageDataGenerator(preprocessing_function = preprocess_input,
                                     rotation_range         = 25,               # allow rotation
                                     width_shift_range      = 0.01,              # horizontal schetch
                                     height_shift_range     = 0.01,              # vertical schetch
                                     zoom_range             = 0.01,              # zoom in
                                     horizontal_flip        = True,             # 
                                     vertical_flip          = True,             # 
                                     )
gen_valid       = gen_.flow_from_directory(os.path.join(working_dir,black_and_white), # validate
                                           target_size       = (image_resize,image_resize),  # resize the image
                                           batch_size        = batch_size,                   # batch size
                                           class_mode        = 'categorical',                # get the labels from the folders
                                           shuffle           = True,                         # shuffle for different epochs
                                           seed              = 12345,                        # replication purpose
                                           )

classifier.load_weights(saving_model_name)
classifier.trianable = False
for layer in classifier.layers:
    layer.trianable = False

tf.random.set_random_seed(12345)
classification_layer   = layers.Dense(2,
                                   activation                       = 'softmax',
#                                   kernel_regularizer               = regularizers.l2(),
                                   activity_regularizer             = regularizers.l1(),
                                   name                             = 'predict'
                                   )(classifier.layers[-2].output)
binary_model          = models.Model(model_loaded.inputs,classification_layer)
binary_model.layers[-2].trainable = False
print(binary_model.trainable_weights)

# compile the model with an optimizer, a loss function
binary_model.compile(optimizers.Adam(lr = 1e-4,),
                   loss_func,
                   metrics = ['categorical_accuracy'])
saving_model_name   = os.path.join(model_dir,f'{model_name}_binary.h5')
callbacks           = make_CallBackList(saving_model_name,
                                        monitor                 = 'val_{}'.format(clf.metrics_names[-1]),
                                        mode                    = 'max',
                                        verbose                 = 0,
                                        min_delta               = 1e-4,
                                        patience                = patience,
                                        frequency               = 1)
print(f'training {model_name} ...')
if not os.path.exists(saving_model_name):
    binary_model.fit_generator(gen_train,
                             steps_per_epoch                           = np.ceil(gen_train.n / batch_size),
                             epochs                                    = 1000, # arbitrary choice
                             validation_data                           = gen_valid,
                             callbacks                                 = callbacks,
                             )


















