#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn
import torch.nn.functional as F
from utils_DANN import ReverseLayerF

class DANN(nn.Module):

    def __init__(self, input_size=1177, hidden_size=512, hidden_layer_num=1, 
                 feature_layer_size=512, hidden_size_2nd=256, hidden_layer_2nd_num=1,
                 pre_output_size=32, output_size=1, dropout=0.1):
        super(DANN, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.hidden_layer_num = hidden_layer_num
        self.hidden_layer_2nd_num = hidden_layer_2nd_num
        self.feature_extractor_layer = nn.ModuleList()
        self.feature_extractor_layer.append(nn.Linear(input_size, hidden_size))
        for layer in range(hidden_layer_num):
            self.feature_extractor_layer.append(nn.Linear(hidden_size, hidden_size))
        self.feature_extractor_layer.append(nn.Linear(hidden_size, feature_layer_size))
        
        self.class_classifier_layer = nn.ModuleList()
        self.class_classifier_layer.append(nn.Linear(feature_layer_size, hidden_size_2nd))
        for layer in range(hidden_layer_2nd_num):
            self.class_classifier_layer.append(nn.Linear(hidden_size_2nd, hidden_size_2nd))
        self.class_classifier_layer.append(nn.Linear(hidden_size_2nd, pre_output_size))
        self.class_classifier_layer.append(nn.Linear(pre_output_size, output_size))
        
        self.domain_classifier_layer = nn.ModuleList()
        self.domain_classifier_layer.append(nn.Linear(feature_layer_size, hidden_size_2nd))
        for layer in range(hidden_layer_2nd_num):
            self.domain_classifier_layer.append(nn.Linear(hidden_size_2nd, hidden_size_2nd))
        self.domain_classifier_layer.append(nn.Linear(hidden_size_2nd, pre_output_size))
        self.domain_classifier_layer.append(nn.Linear(pre_output_size, output_size))       
        # Define proportion of neurons to dropout
        self.dropout = nn.Dropout(dropout)
        
    def feature_extractor(self, x):       
        for layer in self.feature_extractor_layer[:-1]:
            x = F.relu(layer(x))
            # Apply dropout in the hidden layer
            x = self.dropout(x)
        features = F.relu(self.feature_extractor_layer[-1](x))
        return features
    
    def class_classifier(self, x):
        for layer in self.class_classifier_layer[:-1]:
            x = F.relu(layer(x))
            # Apply dropout in the hidden layer
            x = self.dropout(x)
        class_predict = self.class_classifier_layer[-1](x)
        return class_predict
    
    def domain_classifier(self, x):
        for layer in self.domain_classifier_layer[:-1]:
            x = F.relu(layer(x))
            # Apply dropout in the hidden layer
            x = self.dropout(x)
        dimain_predict = self.domain_classifier_layer[-1](x)
        return dimain_predict
    
    def forward(self, x, lambda_):
        features = self.feature_extractor(x)
        # Ref: Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
        # Forward pass is the identity function. In the backward pass,
        # the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
        reverse_feature = ReverseLayerF.apply(features, lambda_)
        class_predict = self.class_classifier(features)
        dimain_predict = self.domain_classifier(reverse_feature)
        return class_predict, dimain_predict

