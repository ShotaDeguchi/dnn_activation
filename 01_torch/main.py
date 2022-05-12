"""
********************************************************************************
main file to execute your program
********************************************************************************
"""

import numpy as np
import matplotlib.pyplot as plt

# from config_gpu import config_gpu
from functions import *
from dnn import *

def main():
    # gpu configuration
    # config_gpu(gpu_flg = 1)

    # problem setup
    p_id = 0
    xmin = -1.
    xmax =  1.
    nx   = 2 ** 8
    nx_  = 2 ** 4

    # params
    f_in   = 1
    f_out  = 1
    f_hid  = 5
    depth  = 3
    lr     = 5e-4
    opt    = "Adam"
    f_scl  = "minmax"
    d_type = torch.float32
    r_seed = 1234
    n_epc  = int(3e4)
    n_btc  = -1
    c_tol  = 1e-6
    es_pat = 100

    # prepare data
    x = np.linspace(xmin, xmax, nx)
    x_train = torch.linspace(
        xmin, xmax, nx_, dtype=d_type, device=None
    ).reshape(-1, 1)
    x_infer = torch.linspace(
        xmin, xmax, nx, dtype=d_type, device=None
    ).reshape(-1, 1)

    if p_id == 0:
        y = func0(x)
        y_train = func0_torch(x_train)
    elif p_id == 1:
        y = func1(x)
        y_train = func1_torch(x_train)
    elif p_id == 2:
        y = func2(x)
        y_train = func2_torch(x_train)
    else:
        raise NotImplementedError(">>>>> p_id")

    # tanh model
    # define, train, save
    w_init = "Glorot"
    b_init = "zeros"
    act    = "tanh"
    print("d_type", d_type)
    print("r_seed", r_seed)
    model_tanh = DNN(
        x_train, y_train, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )
    model_tanh.train()
    torch.save(model_tanh.state_dict(), "./saved_model/model_tanh.pth")
    # define, load, infer
    model_tanh = DNN(
        x_train, y_train, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )
    model_tanh.load_state_dict(torch.load("./saved_model/model_tanh.pth"))
    y_tanh = model_tanh.infer(x_infer)

    # relu
    # define, train, save
    w_init = "He"
    b_init = "zeros"
    act    = "relu"
    print("d_type", d_type)
    print("r_seed", r_seed)
    model_relu = DNN(
        x_train, y_train, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )
    model_relu.train()
    torch.save(model_relu.state_dict(), "./saved_model/model_relu.pth")
    # define, load, infer
    model_relu = DNN(
        x_train, y_train, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )
    model_relu.load_state_dict(torch.load("./saved_model/model_relu.pth"))
    y_relu = model_relu.infer(x_infer)

    # swish (silu)
    # define, train, save
    w_init = "He"
    b_init = "zeros"
    act    = "silu"
    print("d_type", d_type)
    print("r_seed", r_seed)
    model_silu = DNN(
        x_train, y_train, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )
    model_silu.train()
    torch.save(model_silu.state_dict(), "./saved_model/model_silu.pth")
    # define, load, infer
    model_silu = DNN(
        x_train, y_train, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )
    model_silu.load_state_dict(torch.load("./saved_model/model_silu.pth"))
    y_silu = model_silu.infer(x_infer)

    # compare
    if p_id == 0:
        plt.figure(figsize=(4, 4))
        plt.plot(x, y, label="function", alpha=.3, linestyle="-", lw = 5, c="k")
        plt.scatter(x_train, y_train, label="observation", alpha=.7, marker="x", c="r")
        plt.plot(x_infer, y_tanh, label="tanh", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_relu, label="relu", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_silu, label="silu", alpha=.7, linestyle="--")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(alpha=.5)
        plt.legend(loc="upper left")
        plt.savefig("./figures/approx_problem" + str(p_id) + ".png")

        plt.figure(figsize=(8, 4))
        plt.plot(model_tanh.loss_log, label="tanh", alpha=.7, linestyle="--")
        plt.plot(model_relu.loss_log, label="relu", alpha=.7, linestyle="--")
        plt.plot(model_silu.loss_log, label="silu", alpha=.7, linestyle="--")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.grid(alpha=.5)
        plt.legend(loc="upper right")
        plt.savefig("./figures/loss_problem" + str(p_id) + ".png")

    elif p_id == 1:
        plt.figure(figsize=(4, 4))
        plt.plot(x, y, label="function", alpha=.3, linestyle="-", lw = 5, c="k")
        plt.scatter(x_train, y_train, label="observation", alpha=.7, marker="x", c="r")
        plt.plot(x_infer, y_tanh, label="tanh", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_relu, label="relu", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_silu, label="silu", alpha=.7, linestyle="--")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(alpha=.5)
        plt.legend(loc="upper left")
        plt.savefig("./figures/approx_problem" + str(p_id) + ".png")

        plt.figure(figsize=(8, 4))
        plt.plot(model_tanh.loss_log, label="tanh", alpha=.7, linestyle="--")
        plt.plot(model_relu.loss_log, label="relu", alpha=.7, linestyle="--")
        plt.plot(model_silu.loss_log, label="silu", alpha=.7, linestyle="--")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.grid(alpha=.5)
        plt.legend(loc="upper right")
        plt.savefig("./figures/loss_problem" + str(p_id) + ".png")

    elif p_id == 2:
        plt.figure(figsize=(4, 4))
        plt.plot(x, y, label="function", alpha=.3, linestyle="-", lw = 5, c="k")
        plt.scatter(x_train, y_train, label="observation", alpha=.7, marker="x", c="r")
        plt.plot(x_infer, y_tanh, label="tanh", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_relu, label="relu", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_silu, label="silu", alpha=.7, linestyle="--")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-2.5, 2.5)
        plt.grid(alpha=.5)
        plt.legend(loc="upper left")
        plt.savefig("./figures/approx_problem" + str(p_id) + ".png")

        plt.figure(figsize=(8, 4))
        plt.plot(model_tanh.loss_log, label="tanh", alpha=.7, linestyle="--")
        plt.plot(model_relu.loss_log, label="relu", alpha=.7, linestyle="--")
        plt.plot(model_silu.loss_log, label="silu", alpha=.7, linestyle="--")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.grid(alpha=.5)
        plt.legend(loc="upper right")
        plt.savefig("./figures/loss_problem" + str(p_id) + ".png")

if __name__ == "__main__":
    main()
