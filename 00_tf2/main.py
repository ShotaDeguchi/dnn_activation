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

    p_id = 0
    if p_id == 0:
        x = np.linspace(xmin, xmax, nx)
        y = func0(x)
        plt.figure(figsize=(4, 4))
        plt.scatter(x, y, marker=".")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(alpha=.5)
        plt.savefig("./figures/problem" + str(p_id) + ".jpg")
    elif p_id == 1:
        x = np.linspace(xmin, xmax, nx)
        y = func1(x)
        plt.figure(figsize=(4, 4))
        plt.scatter(x, y, marker=".")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(alpha=.5)
        plt.savefig("./figures/problem" + str(p_id) + ".jpg")
    elif p_id == 2:
        x = np.linspace(xmin, xmax, nx)
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
    depth  = 3
    w_init = "Glorot"
    b_init = "zeros"
    act    = "relu"
    lr     = 5e-4
    opt    = "Adam"
    f_scl  = "minmax"
    d_type = "float32"
    r_seed = 1234

    # perpare dataset
    x_train = tf.random.uniform(
        (10, 1), xmin, xmax, dtype=d_type
    )
    x_train = np.linspace(xmin, xmax, 10)
    x_train = x_train.reshape(-1, 1)
    x_train = tf.convert_to_tensor(x_train, dtype=d_type)
    y_train = tf.sin(np.pi * x_train)
    print("x_train.shape", x_train.shape)
    print("y_train.shape", y_train.shape)

    # define model
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
    x_infer = tf.cast(x, dtype=d_type)
    y_infer = model.infer(x_infer)
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

