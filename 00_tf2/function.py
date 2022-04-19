"""
********************************************************************************
functions to be learned
********************************************************************************
"""

import numpy as np

def func1(x):
    f = .6 * np.sin(np.pi * x) \
        + .3 * np.sin(2. * np.pi * x) \
        + .1 * np.cos(8. * np.pi * x)
    return f

def func2(x):
    f = .6 * np.sin(np.pi * x) 
    return f

def func3(x, y):
    f = .6 * np.sin(np.pi * x) 
    return f

