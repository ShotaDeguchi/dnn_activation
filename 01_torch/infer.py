"""
********************************************************************************
define, load, and infer
********************************************************************************
"""

import torch
import matplotlib.pyplot as plt

def main():
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

    # define, load, infer
    w_init = "Glorot"
    b_init = "zeros"
    act    = "tanh"
    model_tanh = DNN(
        x_train, y_train, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )
    model_tanh.load_state_dict(torch.load("./saved_model/model_tanh.pth"))
    y_tanh = model_tanh.infer(x_infer)

    plt.figure(figsize=(4, 4))
    plt.plot(x, y, label="ground truth")
    plt.plot(x_infer, y_tanh,  label="tanh")
    plt.plot(x_infer, y_relu,  label="relu")
    plt.plot(x_infer, y_swish, label="swish")
    plt.legend(loc="upper right")
    plt.savefig()

if __name__ == "__main__":
    main()

