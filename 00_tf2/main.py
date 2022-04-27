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
    gpu_flg = 0
    config_gpu(gpu_flg)

    # problem setup
    p_id = 2
    xmin = -1.
    xmax =  1.
    nx   = 2 ** 8
    nx_  = 2 ** 4

    # params
    f_in   = 1
    f_out  = 1
    f_hid  = 5
    depth  = 3
    b_init = "zeros"
    lr     = 5e-4
    opt    = "Adam"
    f_scl  = "minmax"
    d_type = "float32"
    r_seed = 1234
    n_epc  = int(5e4)
    n_btc  = -1
    c_tol  = 1e-6

    # prepare data
    x = np.linspace(xmin, xmax, nx)
    x_train = np.linspace(xmin, xmax, nx_).reshape(-1, 1)
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

    # define, train, and infer with relu model
    w_init = "He"
    act = "relu"
    model_relu = DNN(
        x_train, y_train, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )
    with tf.device("/device:GPU:0"):
        model_relu.train(n_epc, n_btc, c_tol)
    # model_relu.save("./saved_model/model_relu")
    y_relu = model_relu.infer(x_infer)

    # tanh model
    w_init = "Glorot"
    act = "tanh"
    model_tanh = DNN(
        x_train, y_train, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )
    with tf.device("/device:GPU:0"):
        model_tanh.train(n_epc, n_btc, c_tol)
    y_tanh = model_tanh.infer(x_infer)

    # swish model
    w_init = "He"
    act = "swish"
    model_swish = DNN(
        x_train, y_train, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )
    with tf.device("/device:GPU:0"):
        model_swish.train(n_epc, n_btc, c_tol)
    y_swish = model_swish.infer(x_infer)

    # compare
    if p_id == 0:
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, label="function", alpha=.3, linestyle="-", lw = 5, c="k")
        plt.scatter(x_train, y_train, alpha=.7, marker="x", c="r")
        plt.plot(x_infer, y_relu, label="relu", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_tanh, label="tanh", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_swish, label="swish", alpha=.7, linestyle="--")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(alpha=.5)
        plt.legend(loc="upper left")
        plt.savefig("./figures/approx_problem" + str(p_id) + ".png")

        plt.figure(figsize=(8, 4))
        plt.plot(model_relu.loss_log, label="relu", alpha=.7, linestyle="--")
        plt.plot(model_tanh.loss_log, label="tanh", alpha=.7, linestyle="--")
        plt.plot(model_swish.loss_log, label="swish", alpha=.7, linestyle="--")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.grid(alpha=.5)
        plt.legend(loc="upper right")
        plt.savefig("./figures/loss_problem" + str(p_id) + ".png")

    elif p_id == 1:
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, label="function", alpha=.3, linestyle="-", lw = 5, c="k")
        plt.scatter(x_train, y_train, alpha=.7, marker="x", c="r")
        plt.plot(x_infer, y_relu, label="relu", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_tanh, label="tanh", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_swish, label="swish", alpha=.7, linestyle="--")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(alpha=.5)
        plt.legend(loc="upper left")
        plt.savefig("./figures/approx_problem" + str(p_id) + ".png")

        plt.figure(figsize=(8, 4))
        plt.plot(model_relu.loss_log, label="relu", alpha=.7, linestyle="--")
        plt.plot(model_tanh.loss_log, label="tanh", alpha=.7, linestyle="--")
        plt.plot(model_swish.loss_log, label="swish", alpha=.7, linestyle="--")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.grid(alpha=.5)
        plt.legend(loc="upper right")
        plt.savefig("./figures/loss_problem" + str(p_id) + ".png")

    elif p_id == 2:
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, label="function", alpha=.3, linestyle="-", lw = 5, c="k")
        plt.scatter(x_train, y_train, alpha=.7, marker="x", c="r")
        plt.plot(x_infer, y_relu, label="relu", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_tanh, label="tanh", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_swish, label="swish", alpha=.7, linestyle="--")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-2.5, 2.5)
        plt.grid(alpha=.5)
        plt.legend(loc="upper left")
        plt.savefig("./figures/approx_problem" + str(p_id) + ".png")

        plt.figure(figsize=(8, 4))
        plt.plot(model_relu.loss_log, label="relu", alpha=.7, linestyle="--")
        plt.plot(model_tanh.loss_log, label="tanh", alpha=.7, linestyle="--")
        plt.plot(model_swish.loss_log, label="swish", alpha=.7, linestyle="--")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.grid(alpha=.5)
        plt.legend(loc="upper right")
        plt.savefig("./figures/loss_problem" + str(p_id) + ".png")

if __name__ == "__main__":
    main()

