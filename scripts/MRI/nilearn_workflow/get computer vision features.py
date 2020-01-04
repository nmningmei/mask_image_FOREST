#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:43:24 2019

@author: nmei
"""

import os
from glob import glob
from tqdm import tqdm
from copy import deepcopy
import torch
from torchvision import transforms,models,datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score

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

working_dirs = ['../../../data/greyscaled','../../../data/experiment_images_grayscaled']
working_data = glob(os.path.join(working_dirs[0],
                                 "*",
                                 "*",
                                 "*.jpg"))


transform = transforms.Compose([
 transforms.Resize(128),
 transforms.RandomRotation(45),
 transforms.RandomHorizontalFlip(),
 transforms.RandomVerticalFlip(),
 transforms.ToTensor(),
 transforms.Normalize(
                         mean=[0.485],
                         std=[0.229]
 )])




picked_models = dict(
#        resnet18 = models.resnet18(pretrained=True),
#        alexnet = models.alexnet(pretrained=True),
#        squeezenet = models.squeezenet1_1(pretrained=True),
#        vgg19_bn = models.vgg19_bn(pretrained=True),
        densenet169 = models.densenet169(pretrained=True),
#        inception = models.inception_v3(pretrained=True),
#        googlenet = models.googlenet(pretrained=True),
#        shufflenet = models.shufflenet_v2_x0_5(pretrained=True),
#        mobilenet = models.mobilenet_v2(pretrained=True),
#        resnext50_32x4d = models.resnext50_32x4d(pretrained=True),
        )
def load_dataset(working_dir,transform,batch_size = 1,num_workers = 1,shuffle = True):
    dataset_from_folder = datasets.ImageFolder(
        root=working_dir,
        transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset_from_folder,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    return data_loader
def train_model(model_working,working_dirs,transform,n_epoch,device = 'cpu',batch_size = 8):
    
    ImageLoader_train = load_dataset(working_dirs[0],transform,batch_size = batch_size,num_workers = 2,shuffle = True)
    ImageLoader_valid = load_dataset(working_dirs[1],transform,batch_size = batch_size,num_workers = 2,shuffle = True)
    train_iterator = iter(ImageLoader_train)
    valid_iterator = iter(ImageLoader_valid)
    
    model_working = model_working.to(device)
    criterion = nn.BCEWithLogitsLoss()
#    print(list(model_working.parameters()))
    optimizer = torch.optim.Adam(model_working.parameters(),
                                 lr = 0.001, 
                                 weight_decay = 1e-4)
    torch.manual_seed(12345)
    np.random.seed(12345)
    model_working.train()
    for ii, (batch_in,batch_label) in enumerate(train_iterator):
#        print(batch_label)
        running_loss = 0.
        batch_in = Variable(batch_in.float()).to(device)
        batch_label = torch.stack([1 - batch_label,batch_label]).T
        batch_label = Variable(batch_label).to(device)
        
        optimizer.zero_grad()
        
        preds = model_working(batch_in)
        loss = criterion(preds,Variable(batch_label.float()))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        print(f'step {n_epoch}-{(ii+1)/len(train_iterator):.3f},loss = {running_loss/(ii+1):.4f}')
    valid_loss = 0.
    y_pred = []
    y_true = []
    for jj, (batch_in,batch_label) in tqdm(enumerate(valid_iterator),desc='validation'):
        with torch.no_grad():
            model_working.eval()
            batch_in = Variable(batch_in.float()).to(device)
            batch_label = torch.stack([1 - batch_label,batch_label]).T
            batch_label = Variable(batch_label).to(device)
            preds = model_working(batch_in)
            y_pred.append(preds)
            y_true.append(batch_label)
            loss = criterion(preds,Variable(batch_label.float()))
            valid_loss += loss.item()
    y_pred,y_true = torch.cat(y_pred).detach().cpu().numpy(),torch.cat(y_true).detach().cpu().numpy()
    score_ = roc_auc_score(y_true.astype(int),y_pred.astype('float32'))
    print(f'epoch {n_epoch}, validation loss = {valid_loss/(jj+1):.4f}, score = {score_:.4f}')
    
    return model_working,score_
def feature_extract(feature_extractor,model_name,working_dir,transform,batch_size = 8):
    ImageLoader = load_dataset(working_dir,transform,batch_size = batch_size,num_workers = 2,shuffle = True)
    data_iterator = iter(ImageLoader)
    
    for batch_in,_ ,paths in tqdm(data_iterator,desc='{}'.format(model_name)):
        is_train = False
        with torch.set_grad_enabled(is_train):
            batch_out = torch.squeeze(feature_extractor(batch_in#.cuda()
            ))
            
            feature = batch_out.cpu().numpy()
            _,_,_,_,_,category,subcategory,image_name = paths[0].split('/')
            np.save(f'../../../data/computer vision features/{model_name}/{image_name.split(".")[0]}.npy',
                    feature)

for model_name,model_loaded in picked_models.items():
    torch.cuda.empty_cache()
    
    if not os.path.exists(f'../../../data/computer vision features/{model_name}'):
        os.makedirs(f'../../../data/computer vision features/{model_name}')
    model_working = deepcopy(model_loaded)
    for param in model_working.parameters():
        param.requires_grad = False
    print(model_name)
    # training
    model_working.classifier = nn.Sequential(*[
                        nn.Linear(in_features = model_loaded.classifier.in_features,
                                  out_features = 300,bias = True),
                        nn.ReLU(inplace = True,),
                        nn.Linear(in_features = 300,
                                  out_features = 2,bias = True,),
                        nn.Softmax(dim = 1),
            ])
    batch_size = 8
    for n_epochs in range(10):
        model_trained,score_ = train_model(model_working,working_dirs,transform,n_epochs,batch_size = batch_size)
    adf
    feature_extract(feature_extractor,model_name,working_dirs[0],transform)
























