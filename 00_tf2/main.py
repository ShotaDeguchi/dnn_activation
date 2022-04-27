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
    p_id = 1
    xmin = -1.
    xmax =  1.
    nx   = 100

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
    # y_train = tf.sin(np.pi * x_train)

    # define model
    act = "relu"
    model_relu = DNN(
        x_train, y_train, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )
    # train
    with tf.device("/device:GPU:0"):
        model_relu.train(n_epc = int(1e4), n_btc = -1, c_tol = 1e-5)

    # define model
    act = "tanh"
    model_tanh = DNN(
        x_train, y_train, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )
    # train
    with tf.device("/device:GPU:0"):
        model_tanh.train(n_epc = int(1e4), n_btc = -1, c_tol = 1e-5)

    # define model
    act = "swish"
    model_swish = DNN(
        x_train, y_train, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )
    # train
    with tf.device("/device:GPU:0"):
        model_swish.train(n_epc = int(1e4), n_btc = -1, c_tol = 1e-5)

    # infer
    y_relu = model_relu.infer(x_infer)
    y_tanh = model_tanh.infer(x_infer)
    y_swish = model_swish.infer(x_infer)

    # compare
    if p_id == 0:
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, label="function", alpha=.7, linestyle="-", lw = 3, c="k")
        plt.scatter(x_train, y_train, alpha=.7, marker="x", c="r")
        plt.plot(x_infer, y_relu, label="relu", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_tanh, label="tanh", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_swish, label="swish", alpha=.7, linestyle="--")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(alpha=.5)
        plt.legend(loc="upper left")
        plt.savefig("./figures/problem" + str(p_id) + ".jpg")
    elif p_id == 1:
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, label="function", alpha=.7, linestyle="-", lw = 3, c="k")
        plt.scatter(x_train, y_train, alpha=.7, marker="x", c="r")
        plt.plot(x_infer, y_relu, label="relu", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_tanh, label="tanh", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_swish, label="swish", alpha=.7, linestyle="--")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(alpha=.5)
        plt.legend(loc="upper left")
        plt.savefig("./figures/problem" + str(p_id) + ".jpg")
    elif p_id == 2:
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, label="function", alpha=.7, linestyle="-", lw = 3, c="k")
        plt.scatter(x_train, y_train, alpha=.7, marker="x", c="r")
        plt.plot(x_infer, y_relu, label="relu", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_tanh, label="tanh", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_swish, label="swish", alpha=.7, linestyle="--")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-2.5, 2.5)
        plt.grid(alpha=.5)
        plt.legend(loc="upper left")
        plt.savefig("./figures/problem" + str(p_id) + ".jpg")

if __name__ == "__main__":
    main()

