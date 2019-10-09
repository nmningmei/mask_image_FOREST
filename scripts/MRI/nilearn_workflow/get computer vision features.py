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

working_dir = '../../../data/bw_bc_bl'
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




picked_models = dict(
        resnet18 = models.resnet18(pretrained=True),
        alexnet = models.alexnet(pretrained=True),
        squeezenet = models.squeezenet1_1(pretrained=True),
        vgg19_bn = models.vgg19_bn(pretrained=True),
        densenet121 = models.densenet121(pretrained=True),
#        inception = models.inception_v3(pretrained=True),
        googlenet = models.googlenet(pretrained=True),
        shufflenet = models.shufflenet_v2_x0_5(pretrained=True),
        mobilenet = models.mobilenet_v2(pretrained=True),
        resnext50_32x4d = models.resnext50_32x4d(pretrained=True),
        )
def train_model(model_working,working_dir,transform,batch_size = 8):
    
    ImageLoader = ImageFolderWithPaths(working_dir,transform=transform,)
    training_iterator = iter(DataLoader(ImageLoader,
                                    batch_size = batch_size,
                                    num_workers = 1,
                                    shuffle = True))
    model_working.eval()
    model_working = model_working#.cuda()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model_loaded.parameters(),
                                 lr = 0.001, 
                                 weight_decay = 1e-4)
    torch.manual_seed(12345)
    np.random.seed(12345)
    for ii, (batch_in,batch_label,_) in enumerate(training_iterator):
#        print(batch_label)
        running_loss = 0.
        batch_in = Variable(batch_in,requires_grad = False)#.cuda()
        batch_label = Variable(batch_label,requires_grad = False)#.cuda()
        
        optimizer.zero_grad()
        
        preds = model_working(batch_in)
        loss = criterion(preds,batch_label)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if ii % 50 == 0:
            print('step {:3},loss = {:.4f}'.format(ii+1,running_loss/(ii+1)))
    print('finish training, loss = {:.4f}'.format(running_loss/(ii+1)))
    
    return model_working
def feature_extract(feature_extractor,model_name,working_dir,transform):
    ImageLoader = ImageFolderWithPaths(working_dir,transform=transform,)
    data = DataLoader(ImageLoader,batch_size=1,shuffle = False)
    data_iterator = iter(data)
    
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
    if (model_name == 'resnet18') or (model_name == 'googlenet') or (model_name == 'shufflenet') or (model_name == 'resnext50_32x4d'):
        model_working.fc = nn.Sequential(*[
                            nn.Dropout(),
                            nn.Linear(in_features = model_loaded.fc.in_features,
                                      out_features = 4096,bias = True),
                            nn.ReLU(inplace = True,),
                            nn.Dropout(),
                            nn.Linear(in_features = 4096,
                                      out_features = 1024,bias = True),
                            nn.ReLU(inplace = True,),
                            nn.Dropout(),
                            nn.Linear(in_features = 1024,
                                      out_features = 512,bias = True),
                            nn.ReLU(inplace = True,),
                            nn.Linear(in_features = 512,
                                      out_features = 2,bias = True,),
                            nn.LogSoftmax(dim = 1),
                ])
    elif (model_name == 'alexnet') or (model_name == 'mobilenet') :
        model_working.classifier = nn.Sequential(*[
                            nn.Dropout(),
                            nn.Linear(in_features = model_loaded.classifier[1].in_features,
                                      out_features = 4096,bias = True),
                            nn.ReLU(inplace = True,),
                            nn.Dropout(),
                            nn.Linear(in_features = 4096,
                                      out_features = 1024,bias = True),
                            nn.ReLU(inplace = True,),
                            nn.Dropout(),
                            nn.Linear(in_features = 1024,
                                      out_features = 512,bias = True),
                            nn.ReLU(inplace = True,),
                            nn.Linear(in_features = 512,
                                      out_features = 2,bias = True,),
                            nn.LogSoftmax(dim = 1),
                ])
    elif model_name == 'vgg19_bn':
        model_working.classifier = nn.Sequential(*[
                            nn.Dropout(),
                            nn.Linear(in_features = model_loaded.classifier[0].in_features,
                                      out_features = 4096,bias = True),
                            nn.ReLU(inplace = True,),
                            nn.Dropout(),
                            nn.Linear(in_features = 4096,
                                      out_features = 1024,bias = True),
                            nn.ReLU(inplace = True,),
                            nn.Dropout(),
                            nn.Linear(in_features = 1024,
                                      out_features = 512,bias = True),
                            nn.ReLU(inplace = True,),
                            nn.Linear(in_features = 512,
                                      out_features = 2,bias = True,),
                            nn.LogSoftmax(dim = 1),
                ])
    elif model_name == 'densenet121':
        model_working.classifier = nn.Sequential(*[
                            nn.Dropout(),
                            nn.Linear(in_features = model_loaded.classifier.in_features,
                                      out_features = 4096,bias = True),
                            nn.ReLU(inplace = True,),
                            nn.Dropout(),
                            nn.Linear(in_features = 4096,
                                      out_features = 1024,bias = True),
                            nn.ReLU(inplace = True,),
                            nn.Dropout(),
                            nn.Linear(in_features = 1024,
                                      out_features = 512,bias = True),
                            nn.ReLU(inplace = True,),
                            nn.Linear(in_features = 512,
                                      out_features = 2,bias = True,),
                            nn.LogSoftmax(dim = 1),
                ])
    elif model_name == 'squeezenet':
        model_working.classifier = nn.Sequential(*[
                nn.Dropout(p=0.5, inplace=False),
                nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.LogSoftmax(dim = 1)])
    model_trained = train_model(model_working,working_dir,transform,batch_size = 8)
    
    # feature extraction
    
    if (model_name == 'resnet18') or (model_name == 'googlenet')\
     or (model_name == 'shufflenet') or (model_name == 'resnext50_32x4d'):
        feature_extractor = deepcopy(model_trained)
        seqs = list(model_trained.fc.children())[:-3]
        seqs.append(nn.Sigmoid())
        feature_extractor.fc = nn.Sequential(*seqs)
    elif (model_name == 'alexnet') or (model_name == 'mobilenet')\
    or (model_name == 'vgg19_bn') or(model_name == 'densenet121'):
        feature_extractor = deepcopy(model_trained)
        seqs = list(model_trained.classifier.children())[:-3]
        seqs.append(nn.Sigmoid())
        feature_extractor.classifier = nn.Sequential(*seqs)
    elif model_name == 'squeezenet':
        feature_extractor = nn.Sequential(*[model_trained.features,
                                            nn.AdaptiveAvgPool2d(output_size = (1,1))])
    feature_extract(feature_extractor,model_name,working_dir,transform)
























