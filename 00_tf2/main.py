"""
********************************************************************************
main file to execute your program
********************************************************************************
"""

import numpy as np
import matplotlib.pyplot as plt

from config_gpu import config_gpu
from functions import *
from dnn import *

def main():
    # gpu configuration
    gpu_flg = 1
    config_gpu(gpu_flg)

    # problem setup
    xmin = -1.
    xmax =  1.
    nx   = 100
    p_id = 2

    # params
    f_in   = 1
    f_out  = 1
    f_hid  = 2 ** 4
    depth  = 3
    w_init = "Glorot"
    b_init = "zeros"
    lr     = 5e-4
    opt    = "Adam"
    f_scl  = "minmax"
    d_type = "float32"
    r_seed = 1234

    # prepare data
    x = np.linspace(xmin, xmax, nx)
    x_train = np.linspace(xmin, xmax, int(nx / 5)).reshape(-1, 1)
    x_train = tf.convert_to_tensor(x_train, dtype=d_type)
    x_infer = np.linspace(xmin, xmax, nx).reshape(-1, 1)
    x_infer = tf.convert_to_tensor(x_infer, dtype=d_type)

    if p_id == 0:
        y = func0(x)
        y_train = func0_tf(x_train)
    elif p_id == 1:
        y = func1(x)
        y_train = func1_tf(x_train)
    elif p_id == 2:
        y = func2(x)
        y_train = func2_tf(x_train)
    else:
        raise NotImplementedError(">>>>> p_id")

    # perpare dataset
    # x_train = np.linspace(xmin, xmax, int(nx / 5)).reshape(-1, 1)
    # x_train = tf.convert_to_tensor(x_train, dtype=d_type)
    y_train = tf.sin(np.pi * x_train)

    # define model
    act = "swish"
    model = DNN(
        x_train, y_train, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )
    # train
    with tf.device("/device:GPU:0"):
        model.train(n_epc = int(1e4), n_btc = -1, c_tol = 1e-5)
    # infer
    # x_infer = np.linspace(xmin, xmax, nx).reshape(-1, 1)
    # x_infer = tf.convert_to_tensor(x_infer, dtype=d_type)
    y_infer = model.infer(x_infer)

    # compare
    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, marker=".", label="function")
    plt.scatter(x_infer, y_infer, marker=".", label="dnn")
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.grid(alpha=.5)
    plt.legend(loc="upper left")
    plt.savefig("./figures/problem" + str(p_id) + ".jpg")

if __name__ == "__main__":
    main()

