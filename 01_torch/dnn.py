"""
********************************************************************************
deep neural network
********************************************************************************
"""

import os
import time
import numpy as np
import torch
from torch import nn

class DNN(nn.Module):
    def __init__(
        self, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr = 5e-4, opt = "Adam", f_scl = "minmax", 
        d_type = "float32", r_seed = 1234
    ):
        # init
        super().__init__()
        self.f_in   = f_in
        self.f_out  = f_out
        self.f_hid  = f_hid
        self.depth  = depth
        self.w_init = w_init
        self.b_init = b_init
        self.act    = act
        self.lr     = lr
        self.opt    = opt
        self.f_scl  = f_scl
        self.d_type = d_type
        self.r_seed = r_seed
        # self.device = device
        self.setup(r_seed, d_type)

    def setup(
        self, d_type, r_seed
    ):
        self.d_type
        self.r_seed

    def dnn_init(
        self, f_in, f_out, f_hid, depth
    ):
        arch = nn.Sequential(
            nn.Linear(f_in, f_hid)
        )
        return arch

    def forward_pass(
        self, x
    ):
        # scaling
        if self.f_scl == None:
            z = x
        elif self.f_scl == "minmax":
            z = x
        elif self.f_scl == "mean":
            z = x
        else:
            raise NotImplementedError(">>>>> forward_pass")
        
        y = 
        return y

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
