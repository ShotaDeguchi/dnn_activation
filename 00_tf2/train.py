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

    # params
    f_in   = 1
    f_out  = 1
    f_hid  = 5
    depth  = 3
    w_init = "Glorot"
    b_init = "zeros"
    act    = "tanh"
    lr     = 5e-4
    opt    = "Adam"
    f_scl  = "minmax"
    d_type = "float32"
    r_seed = 1234
    n_epc  = int(5e4)
    n_btc  = -1
    c_tol  = 1e-6

    # train
    model_tanh = DNN(
        x_train, y_train, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )
    with tf.device("/device:GPU:0"):
        model_tanh.train(n_epc, n_btc, c_tol)
    # model_tanh.save("./saved_model/model_tanh")

if __name__ == "__main__":
    __main__()
