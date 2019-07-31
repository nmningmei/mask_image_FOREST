#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:43:24 2019

@author: nmei
"""

import os
from glob import glob
from tqdm import tqdm
import torch
from torchvision import transforms,models,datasets
from torch.utils.data import DataLoader
from torch import nn
import numpy as np

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

working_dir = '../../../grayscaled'
working_data = glob(os.path.join(working_dir,
                                 "*",
                                 "*",
                                 "*.jpg"))


transform = transforms.Compose([            #[1]
# transforms.Resize(256),                    #[2]
# transforms.CenterCrop(224),                #[3]
# transforms.Grayscale(),
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
                         mean=[0.485],                #[6]
                         std=[0.229]                  #[7]
 )])


picked_models = [
        models.densenet121,
        models.mobilenet_v2,
        models.vgg19_bn,
        ]
model_names = ['DenseNet121','MobileNetV2','VGG19_bn']
def process(model_loaded,image,paths):
    temp = model_loaded.features(image.cuda())
    temp = torch.squeeze(nn.AdaptiveAvgPool2d(1)(temp))
    _,_,_,_,category,subcategory,image_name = paths[0].split('/')
    return temp.cpu().numpy(),image_name.split('.')[0]

for model_,model_name in zip(picked_models,model_names):
    if not os.path.exists(f'../../../data/computer vision features/{model_name}'):
        os.mkdir(f'../../../data/computer vision features/{model_name}')
    model_loaded = model_(pretrained = True)
    model_loaded.eval()
    model_loaded.cuda()
    for param in model_loaded.parameters():
        param.requires_grad = False
    
    ImageLoader = ImageFolderWithPaths(working_dir,transform=transform,)
    data = DataLoader(ImageLoader,batch_size=1,)
    data_iterator = iter(data)
    
    for image,_,paths in tqdm(data_iterator,desc=f'{model_name}'):
        feature,saving_name = process(model_loaded,image,paths)
        
        np.save(f'../../../data/computer vision features/{model_name}/{saving_name}.npy',
                feature)


























