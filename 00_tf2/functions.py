"""
********************************************************************************
functions to be learned
********************************************************************************
"""

import numpy as np
import tensorflow as tf

def func0(x):
    y = np.sin(np.pi * x)
    return y
def func0_tf(x):
    y = tf.sin(np.pi * x)
    return y

def func1(x):
    y = .6 * np.sin(np.pi * x) \
        + .4 * np.sin(4. * np.pi * x)
    return y
def func1_tf(x):
    y = .6 * tf.sin(np.pi * x) \
        + .4 * tf.sin(4. * np.pi * x)
    return y

def func2(x):
    y = x * np.cos(2. * np.pi * x)
    # y = np.exp(x) * np.sin(2. * np.pi * x)
    return y
def func2_tf(x):
    y = x * tf.cos(2. * np.pi * x)
    # y = tf.exp(x) * tf.sin(2. * np.pi * x)
    return y

