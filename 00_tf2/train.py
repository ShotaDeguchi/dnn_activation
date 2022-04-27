"""
********************************************************************************
training
********************************************************************************
"""

import os
import time
import numpy as np
import tensorflow as tf

from dnn import *

def __main__():
    # argparse

    # train
    model = dnn_1D
    model.train()

if __name__ == "__main__":
    __main__()
