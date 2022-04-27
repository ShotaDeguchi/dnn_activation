"""
********************************************************************************
main file to execute your program
********************************************************************************
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from config_gpu import config_gpu
from functions import *
from dnn import *

def main():
    # gpu configuration
    gpu_flg = 1
    config_gpu(gpu_flg)

    # problem id
    p_id = 2
    if p_id == 0:
        x = np.linspace(-1, 1, 100)
        y = func0(x)
        plt.figure(figsize=(4, 4))
        plt.scatter(x, y, marker=".")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(alpha=.5)
        plt.savefig("./figures/problem" + str(p_id) + ".jpg")
    elif p_id == 1:
        x = np.linspace(-1, 1, 100)
        y = func1(x)
        plt.figure(figsize=(4, 4))
        plt.scatter(x, y, marker=".")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(alpha=.5)
        plt.savefig("./figures/problem" + str(p_id) + ".jpg")
    elif p_id == 2:
        x = np.linspace(-1, 1, 100)
        y = func2(x)
        plt.figure(figsize=(4, 4))
        plt.scatter(x, y, marker=".")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-2.5, 2.5)
        plt.grid(alpha=.5)
        plt.savefig("./figures/problem" + str(p_id) + ".jpg")
    else:
        raise NotImplementedError(">>>>> p_id")

    # params
    f_in   = 1
    f_out  = 1
    f_hid  = 2 ** 6
    depth  = 4
    w_init = "Glorot"
    b_init = "zeros"
    act    = "swish"
    lr     = 5e-4
    opt    = "Adam"
    f_scl  = "minmax"
    d_type = "float32"
    r_seed = 1234

    x = x.reshape(-1, 1)

    x_train = tf.cast(x, dtype=d_type)
    y_train = tf.cast(y, dtype=d_type)

    print(x.shape)
    print(x_train.shape)

    # define model
    model = DNN(
        x_train, y_train, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )
    # train
    # with tf.device("/device:CPU:0"):
    with tf.device("/device:GPU:0"):
        model.train(n_epc = int(1e3), n_btc = -1, c_tol = 1e-5)
    # infer
    model.infer(x)

if __name__ == "__main__":
    main()

