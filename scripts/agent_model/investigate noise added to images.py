#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:04:28 2019

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
from sklearn.svm                                import LinearSVC
from sklearn.preprocessing                      import MinMaxScaler
from sklearn.pipeline                           import make_pipeline
from sklearn.model_selection                    import StratifiedShuffleSplit,cross_validate
from sklearn.calibration                        import CalibratedClassifierCV
from sklearn.utils                              import shuffle

from matplotlib                                 import pyplot as plt

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
batch_size          = 96
image_resize        = 128
drop_rate           = 0.25
model_name          = 'VGG19'
model_pretrained    = applications.VGG19
preprocess_input    = applications.vgg19.preprocess_input
model_dir           = ''
loss_func           = losses.binary_crossentropy
colorful            = 'experiment_images_tilted'
black_and_white     = 'greyscaled'
noisy               = 'bw_bc_bl'
grayscaled          = 'experiment_images_grayscaled'

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
#                               kernel_regularizer               = regularizers.l2(),
#                               activity_regularizer             = regularizers.l1(),
                               name                             = 'feature'
                               )(fine_tune_model)
binary_classifier   = layers.Dense(2,
                                   activation                       = 'softmax',
#                                   kernel_regularizer               = regularizers.l2(),
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
def process_func(x,var = 7e5):
    row,col,ch= x.shape
    mean = 0
    var = var
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noise = x + gauss
    noise = preprocess_input(noise)
    return noise
gen = ImageDataGenerator(
                        featurewise_center=False,
                        samplewise_center=False,
                        featurewise_std_normalization=False,
                        samplewise_std_normalization=False,
                        zca_whitening=False,
                        zca_epsilon=1e-06,
                        rotation_range=45,
                        width_shift_range=0,
                        height_shift_range=0,
                        brightness_range=None,
                        shear_range=0.01,
                        zoom_range=-0.02,
                        channel_shift_range=0.0,
                        fill_mode='nearest',
                        cval=0.0,
                        horizontal_flip=True,
                        vertical_flip=True, # change from experiment
                        preprocessing_function=process_func,
                        data_format=None,
                        validation_split=0.0
                    )
images_flow = gen.flow_from_directory(os.path.join(working_dir,grayscaled),
                                              target_size       = (image_resize,image_resize),  # resize the image
                                              batch_size        = batch_size,                   # batch size
                                              class_mode        = 'categorical',                # get the labels from the folders
                                              shuffle           = True,                        # shuffle for different epochs
                                              seed              = 12345,                        # replication purpose
                                              )
results = dict(session = [],
               score = [],
               )
for session in range(100):
    
    image_array,labels = images_flow.next()
    predictions = classifier.predict(image_array,batch_size = 32,verbose = 1)
    score = roc_auc_score(labels,predictions)
    results['session'].append(session + 1)
    results['score'].append(score)
#    classifier.evaluate_generator(images_flow,steps = np.ceil(images_flow.n / batch_size),verbose = 1)

results = pd.DataFrame(results)
print(results['score'].describe())
























