#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:14:27 2019

@author: nmei
"""

import os
import gc
gc.collect()

import numpy      as np
import pandas     as pd
import tensorflow as tf

from glob import glob
from tqdm import tqdm

from tensorflow.keras                           import applications,layers,models,optimizers,losses,regularizers
from tensorflow.keras.preprocessing.image       import ImageDataGenerator,load_img,img_to_array
from tensorflow.image                           import resize

from sklearn.metrics                            import roc_auc_score

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
drop_rate           = 0.5
model_name          = 'VGG19'
model_pretrained    = applications.VGG19
preprocess_input    = applications.vgg19.preprocess_input
model_dir           = ''
loss_func           = losses.binary_crossentropy
colorful            = 'experiment_images_tilted'
black_and_white     = 'greyscaled'
noisy               = 'bw_bc_bl'

model_loaded    = model_pretrained(weights      = 'imagenet',
                                   include_top  = False,
                                   input_shape  = (image_resize,image_resize,3),
                                   pooling      = 'max',
                                   )
# freeze the pre-trained model weights
model_loaded.trainable = False
# now, adding 2 more layers: CNN --> 300 --> discriminative prediction
fine_tune_model = model_loaded.output
fine_tune_model = layers.Dense(300,
                               activation                       = tf.keras.activations.selu, # SOTA activation function
                               kernel_initializer               = 'lecun_normal', # seggested in documentation
                               kernel_regularizer               = regularizers.l2(),
#                               activity_regularizer             = regularizers.l1(),
                               name                             = 'feature'
                               )(fine_tune_model)
binary_classifier   = layers.Dense(2,
                                   activation                       = 'softmax',
                                   kernel_regularizer               = regularizers.l2(),
                                   activity_regularizer             = regularizers.l1(),
                                   name                             = 'predict'
                                   )(fine_tune_model)
classifier          = models.Model(model_loaded.inputs,binary_classifier)
# freeze the classifier weights
classifier.trainable = False
# load the best weights we fine-tuned
saving_model_name   = os.path.join(model_dir,f'{model_name}_binary.h5')
classifier.load_weights(saving_model_name)
print(classifier.summary())
# compile to check the weights being the best
classifier.compile(optimizers.Adam(lr = 1e-4,),
                   loss_func,
                   metrics = ['categorical_accuracy'])

gen             = ImageDataGenerator(
                                     preprocessing_function = preprocess_input, # scaling function (-1,1)
                                     )
gen_train       = gen.flow_from_directory(os.path.join(working_dir,noisy), 
                                          target_size       = (image_resize,image_resize),  # resize the image
                                          batch_size        = batch_size,                   # batch size
                                          class_mode        = 'categorical',                # get the labels from the folders
                                          shuffle           = False,                         # shuffle for different epochs
                                          seed              = 12345,                        # replication purpose
                                          )
print(classifier.evaluate_generator(gen_train,
                                     steps = np.ceil(gen_train.n/batch_size),
                                     verbose = 1))

results = dict(true_label = [],
               class_1 = [],
               class_2 = [])

predictions = classifier.predict_generator(gen_train,steps = np.ceil(gen_train.n/batch_size),verbose = 1)
targets = gen_train.classes


perfect_score = roc_auc_score(targets,predictions[:,-1])
print(f'score = {perfect_score:.4f}')

# train the errors






