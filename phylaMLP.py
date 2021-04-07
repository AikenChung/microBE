#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn


class MLP(nn.Module):
    """ Multi-Layer Perceptron for classifying IBD and Healthy microbiome data"""
    def __init__(self, input_dim=1177, hidden_dim=256, 
                 hidden_layer_num=1, 
                 pre_output_dim = 64, 
                 output_dim=1):        
        super(MLP, self).__init__()        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for layer in range(hidden_layer_num):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, pre_output_dim))
        self.layers.append(nn.Linear(pre_output_dim, output_dim))
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        out = self.layers[-1](x)
        return out
