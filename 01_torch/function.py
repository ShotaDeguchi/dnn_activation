"""
********************************************************************************
functions to be learned
********************************************************************************
"""

import numpy as np

def func1(x):
    y = .6 * np.sin(np.pi * x) \
        + .3 * np.sin(2. * np.pi * x) \
        + .1 * np.cos(8. * np.pi * x)
    return y

def func2(x):
    if x <= 0:
        y = .2 * np.sin(6. * np.pi * x)
    else:
        y = 1. + .1 * x * np.cos(12. * np.pi * x)
    return y

def func3(x, y):
    z = np.sin(2. * np.pi * x) + np.exp(y) 
    return z

