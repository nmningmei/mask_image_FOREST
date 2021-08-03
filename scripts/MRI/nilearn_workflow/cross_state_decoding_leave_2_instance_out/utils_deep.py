#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:02:43 2020

@author: nmei
"""

import torch
from torch.autograd import Variable
from torch import nn

def conv_block(in_channels,
               out_channels,
               kernel_size = (3,3),
               *args,**kwargs):
    return nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                          out_channels = out_channels,
                          kernel_size = kernel_size,
                          stride = 1,
                          padding = 0,
                          padding_mode = 'zeros',
                          bias = True,
                          ),
            nn.Conv2d(in_channels = out_channels,
                      out_channels = out_channels,
                      kernel_size = kernel_size,
                      stride = 1,
                      padding = 0,
                      padding_mode = 'zeros',
                      bias = True,
                      ),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(kernel_size = kernel_size,
                         stride = 1,),
            nn.Conv2d(in_channels = out_channels,
                      out_channels = out_channels,
                      kernel_size = kernel_size,
                      stride = 1,
                      padding = 0,
                      padding_mode = 'zeros',
                      bias = True,
                      ),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(kernel_size = kernel_size,
                         stride = 1,),
            nn.ReLU(),
            )

class feature_extractor(nn.Module):
    def __init__(self,
                 batch_size = 8,
                 device = 'cpu',
                 input_channels = 88,# or 66
                 out_channels = [64,64,64,128,128,256,256,512],
                 kernel_size = (3,3),
                 ):
        super(feature_extractor,self).__init__()
        torch.manual_seed(12345)
        self.batch_size = batch_size
        self.device = device
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.conv_block1 = conv_block(input_channels,
                                      out_channels[0],
                                      kernel_size = self.kernel_size)
        self.conv_block2 = conv_block(out_channels[0],
                                      out_channels[1],
                                      kernel_size = self.kernel_size)
        self.conv_block3 = conv_block(out_channels[1],
                                      out_channels[2],
                                      kernel_size = self.kernel_size)
        self.conv_block4 = conv_block(out_channels[2],
                                      out_channels[3],
                                      kernel_size = self.kernel_size)
        self.conv_block5 = conv_block(out_channels[3],
                                      out_channels[4],
                                      kernel_size = (3,3))
        self.conv_block6 = conv_block(out_channels[4],
                                      out_channels[5],
                                      kernel_size = (3,3))
        self.conv_block7 = conv_block(out_channels[5],
                                      out_channels[6],
                                      kernel_size = (3,3))
        self.conv_block8 = conv_block(out_channels[6],
                                      out_channels[7],
                                      kernel_size = (3,3))
    
    def forward(self,x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        out = self.conv_block5(out)
        out = self.conv_block6(out)
        out = self.conv_block7(out)
        out = self.conv_block8(out)
        return out

class CNN2D_model(nn.Module):
    def __init__(self,
                 batch_size = 8,
                 device = 'cpu',
                 kernel_sizes = [(3,3),(3,2),(3,3)],
                 ):
        super(CNN2D_model,self).__init__()
        torch.manual_seed(12345)
        self.batch_size = batch_size
        self.device = device
        self.kernel_sizes = kernel_sizes
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1,1))
        
        
    def forward(self,x):
        # x0 is x
        # x1 is x.permute
        x1 = x.permute(0,2,3,1)
        # x2 is x.permute
        x2 = x.permute(0,1,3,2)
        
        feature_extractors = {ii:feature_extractor(batch_size = self.batch_size,
                                                   device = self.device,
                                                   input_channels = input_x.size()[1],
                                                   kernel_size = kernel_size) for ii,(input_x,
                                                                        kernel_size) in enumerate(
                                                                                zip([x,x1,x2],
                                                                                    self.kernel_sizes))}
        
        out0,out1,out2 = [feature_extractors[ii](input_x) for ii,input_x in enumerate([x,x1,x2])]
        out0 = self.AdaptiveAvgPool(out0).view(-1,1,out0.size()[1])
        out1 = self.AdaptiveAvgPool(out1).view(-1,1,out1.size()[1])
        out2 = self.AdaptiveAvgPool(out2).view(-1,1,out2.size()[1])
        return out0,out1,out2

class rnn_classifier(nn.Module):
    def __init__(self,
                 batch_size = 8,
                 device = 'cpu',
                 num_classes = 2,
                 feature_extractor = None):
        super(rnn_classifier,self,).__init__()
        torch.manual_seed(12345)
        self.batch_size = batch_size
        self.device = device
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = CNN2D_model(batch_size = self.batch_size)
        self.rnn = nn.GRU(input_size = 512,
                          hidden_size = 5,
                          num_layers = 1,
                          batch_first = True,
                          bidirectional = True,)
        self.linear = nn.Linear(10,num_classes)
        self.out_activation = nn.Softmax(dim = -1)
    def forward(self,x):
        feature_extractor = self.feature_extractor
        outs = feature_extractor(x)
        out0,out1,out2 = outs
        features = torch.cat([out0,out1,out2],dim = 1)
        out,hidden = self.rnn(features)
        out = self.linear(out)
        out = torch.mean(out,dim = 1)
        out = self.out_activation(out)
        return out,hidden