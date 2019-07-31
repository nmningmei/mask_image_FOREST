#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:33:01 2019

@author: nmei
"""

import os
from glob import glob
import numpy as np
import tensorflow as tf
from keras import layers,applications,optimizers,losses,models,regularizers,callbacks
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn import metrics

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

K.clear_session()
batch_size = 64
n_epochs = 100
size = 100
mother_path = 'data/cross-val'
train_dir = '../{}/train'.format(mother_path)
validate_dir = '../{}/validate'.format(mother_path)
model_name = 'supervised.hdf5'

label_map = {'Living_Things':[0,1],
             'Nonliving_Things':[1,0]}
def preprocessing_fun(im):
    sigma = 1#np.random.uniform(0.5,1.5)
    return ndimage.gaussian_filter(im, sigma=sigma)

train_paths = glob(os.path.join(train_dir,'*','*.jpg'))
valid_paths = glob(os.path.join(validate_dir,'*','*.jpg'))
if len(train_paths) == 0:
    train_paths = glob(os.path.join(train_dir,'*','*','*.jpg'))
    valid_paths = glob(os.path.join(validate_dir,'*','*','*.jpg'))

data_length = len(train_paths)
data_length_val = len(valid_paths)

traingen = ImageDataGenerator(      featurewise_center              = False, 
                                    samplewise_center               = False, 
                                    featurewise_std_normalization   = False, 
                                    samplewise_std_normalization    = False, 
                                    zca_whitening                   = False, 
                                    zca_epsilon                     = 1e-06, 
                                    rotation_range                  = 45, 
                                    width_shift_range               = 0., 
                                    height_shift_range              = 0.2, 
                                    brightness_range                = None, 
                                    shear_range                     = 0.2, 
                                    zoom_range                      = 0.2, 
                                    channel_shift_range             = 0.0, 
                                    fill_mode                       = 'nearest', 
                                    cval                            = 0.0, 
                                    horizontal_flip                 = True, 
                                    vertical_flip                   = True, 
                                    rescale                         = 1/255., 
                                    preprocessing_function          = None,#preprocessing_fun, 
                                    data_format                     = None, 
                                    validation_split                = 0.0
                                )
#traingen = ImageDataGenerator(rescale=1/255.)
train_generator = traingen.flow_from_directory(
        train_dir,
        target_size = (size,size),
        batch_size = batch_size,
        shuffle = True,
        class_mode = 'categorical')

validgen = ImageDataGenerator(rescale=1/255.)
validation_generator = validgen.flow_from_directory(
        validate_dir,
        target_size = (size,size),
        batch_size = batch_size,
        class_mode = 'categorical')

encoder = applications.VGG19(input_shape=(size,size,3),include_top=False,pooling='avg')
for layer in encoder.layers:
    layer.trainable = False

model = models.Sequential([
        encoder,
#        layers.Dense(units = 128,
#                     activation = 'selu',
#                     use_bias = True,
#                     kernel_initializer = 'he_normal',
#                     kernel_regularizer = regularizers.l2(),
#                     activity_regularizer = regularizers.l1(),
#                     ),
#        layers.BatchNormalization(),
#        layers.Dropout(0.5,),
#        layers.Dense(units = 32,
#                     activation = 'selu',
#                     use_bias = True,
#                     kernel_initializer = 'he_normal',
#                     kernel_regularizer = regularizers.l2(),
#                     activity_regularizer = regularizers.l1(),
#                     ),
#        layers.BatchNormalization(),
#        layers.Dropout(0.5,),
        layers.Dense(units = 8,
                     activation = 'selu',
                     use_bias = True,
                     kernel_initializer = 'he_normal',
                     kernel_regularizer = regularizers.l2(),
                     activity_regularizer = regularizers.l1(),
                     ),
        layers.BatchNormalization(),
        layers.Dropout(0.5,),
        layers.Dense(units = 2,
                     activation = 'softmax',
                     activity_regularizer = regularizers.l1(1e2),
                     )
                            ])

model.compile(optimizer = optimizers.Adam(lr = 1e-4),
              loss = losses.categorical_crossentropy,
              metrics = ['binary_accuracy'],)

preds = model.predict_generator(validation_generator,
                                steps = np.ceil(data_length_val/batch_size),
                                max_queue_size = 1, 
                                pickle_safe = False,)
a = metrics.roc_auc_score(validation_generator.classes,preds[:,-1]);print(a)

callbackList = []
callbackList.append(callbacks.EarlyStopping(monitor = 'val_{}'.format(model.metrics_names[-1]),
                                            mode = 'max',
                                            min_delta = 1e-5,
                                            patience = 20,
                                            verbose = 0))
callbackList.append(callbacks.ModelCheckpoint('../{}/{}'.format(mother_path,model_name),
                                              monitor = 'val_{}'.format(model.metrics_names[-1]),
                                              mode = 'max',
                                              verbose = 0,
                                              save_best_only = True,))

model.fit_generator(
            train_generator,
            epochs = n_epochs,
            steps_per_epoch = np.ceil(data_length/batch_size),
            validation_data = validation_generator,
            validation_steps = np.ceil(data_length/batch_size),
            callbacks = callbackList,
            max_queue_size = 1,
            pickle_safe = False,
            shuffle = True,)


preds = model.predict_generator(validation_generator,
                                steps = np.ceil(data_length_val/batch_size),
                                max_queue_size = 1, 
                                pickle_safe = False,)
print(model.summary())
a = metrics.roc_auc_score(validation_generator.classes,preds[:,-1]);print(a)
print(metrics.classification_report(validation_generator.classes,preds[:,-1]>0.5))

plot_confusion_matrix(validation_generator.classes,preds[:,-1]>0.5, classes=['living','nonliving'], normalize=True,
                      title='Normalized confusion matrix')


#mother_path = 'data/experiment_cropped_2'
#train_dir = '../{}/train'.format(mother_path)
#validate_dir = '../{}/validate'.format(mother_path)
#model_name = 'supervised.hdf5'
#
#label_map = {'Living_Things':[0,1],
#             'Nonliving_Things':[1,0]}
#
#train_paths = glob(os.path.join(train_dir,'*','*.jpg'))
#valid_paths = glob(os.path.join(validate_dir,'*','*.jpg'))
#if len(train_paths) == 0:
#    train_paths = glob(os.path.join(train_dir,'*','*','*.jpg'))
#    valid_paths = glob(os.path.join(validate_dir,'*','*','*.jpg'))
#
#data_length = len(train_paths)
#
#traingen = ImageDataGenerator(      featurewise_center              = False, 
#                                    samplewise_center               = False, 
#                                    featurewise_std_normalization   = False, 
#                                    samplewise_std_normalization    = False, 
#                                    zca_whitening                   = False, 
#                                    zca_epsilon                     = 1e-06, 
#                                    rotation_range                  = 0, 
#                                    width_shift_range               = 0., 
#                                    height_shift_range              = 0., 
#                                    brightness_range                = None, 
#                                    shear_range                     = 0., 
#                                    zoom_range                      = 0., 
#                                    channel_shift_range             = 0.0, 
#                                    fill_mode                       = 'nearest', 
#                                    cval                            = 0.0, 
#                                    horizontal_flip                 = False, 
#                                    vertical_flip                   = False, 
#                                    rescale                         = 1/255., 
#                                    preprocessing_function          = None,#preprocessing_fun, 
#                                    data_format                     = None, 
#                                    validation_split                = 0.0
#                                )
#train_generator = traingen.flow_from_directory(
#        train_dir,
#        target_size = (size,size),
#        batch_size = batch_size,
#        shuffle = True,
#        class_mode = 'categorical')
#
#validgen = ImageDataGenerator(rescale=1/255.)
#validation_generator = validgen.flow_from_directory(
#        validate_dir,
#        target_size = (size,size),
#        batch_size = batch_size,
#        class_mode = 'categorical')
#
#preds = model.predict_generator(validation_generator,
#                                steps = np.ceil(data_length/batch_size),
#                                max_queue_size = 1, 
#                                pickle_safe = False,)
#
#a = metrics.roc_auc_score(validation_generator.classes,preds[:,0])
#print(f'pretrained = {a:.4f}')
#
#model.fit_generator(
#            train_generator,
#            epochs = n_epochs,
#            steps_per_epoch = np.ceil(data_length/batch_size),
#            validation_data = validation_generator,
#            validation_steps = np.ceil(data_length/batch_size),
#            callbacks = callbackList,
#            max_queue_size = 1,
#            pickle_safe = False,
#            shuffle = True,)
#
#a = metrics.roc_auc_score(validation_generator.classes,preds[:,0]>0.5)
#print(f'posttrained = {a:.4f}')











