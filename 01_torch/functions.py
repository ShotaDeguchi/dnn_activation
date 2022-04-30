"""
********************************************************************************
functions to be learned
********************************************************************************
"""

import numpy as np
import torch

def func0(x):
    y = np.sin(np.pi * x)
    return y
def func0_torch(x):
    y = torch.sin(np.pi * x)
    return y

def func1(x):
    y = .7 * np.sin(np.pi * x) \
        + .3 * np.sin(4. * np.pi * x)
    return y
def func1_torch(x):
    y = .7 * torch.sin(np.pi * x) \
        + .3 * torch.sin(4. * np.pi * x)
    return y

def func2(x):
    y = np.exp(x) * np.sin(2. * np.pi * x)
    return y
def func2_torch(x):
    y = torch.exp(x) * torch.sin(2. * np.pi * x)
    return y

