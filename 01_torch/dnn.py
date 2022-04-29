"""
********************************************************************************
deep neural network
********************************************************************************
"""

import os
import time
import numpy as np
import torch
from torch import nn

class DNN(nn.Module):
    def __init__(
        self, 
        x, y, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr = 5e-4, opt = "Adam", f_scl = "minmax", 
        d_type = "float32", r_seed = 1234
    ):
        # initialization
        super().__init__()
        self.f_in   = f_in
        self.f_out  = f_out
        self.f_hid  = f_hid
        self.depth  = depth
        self.w_init = w_init
        self.b_init = b_init
        self.act    = act
        self.lr     = lr
        self.opt    = opt
        self.f_scl  = f_scl
        self.d_type = d_type
        self.r_seed = r_seed
        # self.device = device
        self.setup(r_seed, d_type)

        print("\n************************************************************")
        print("********************     DELLO WORLD     *******************")
        print("************************************************************")

        # data
        self.x = x
        self.y = y

    def setup(
        self, d_type, r_seed
    ):
        os.environ["PYTHONHASHSEED"] = str(r_seed)
        np.random.seed(r_seed)
        torch.manual_seed(r_seed)
        torch.set_default_dtype(d_type)

    def weight_init(
        self, init, seed
    ):
        print(">>>>> weight_init")
        print("         initializer:", init)
        if init == "Glorot":
            weight = tf.keras.initializers.GlorotNormal(seed = seed)
        elif init == "He":
            weight = tf.keras.initializers.HeNormal(seed = seed)
        elif init == "LeCun":
            weight = tf.keras.initializers.LecunNormal(seed = seed)
        else:
            raise NotImplementedError(">>>>> weight_init")
        return weight

    def bias_init(
        self, init
    ):
        print(">>>>> bias_init")
        print("         initializer:", init)
        if init == "zeros":
            bias = tf.keras.initializers.Zeros()
        elif init == "ones":
            bias = tf.keras.initializers.Ones()
        else:
            raise NotImplementedError(">>>>> bias_init")
        return bias

    def act_func(
        self, act
    ):
        print(">>>>> act_func")
        print("         activation:", act)
        if act == "relu":
            activation = tf.keras.activations.relu()
        elif act == "elu":
            activation = tf.keras.activations.elu()
        elif act == "swish":
            activation = tf.keras.activations.swish()
        elif act == "tanh":
            activation = tf.keras.activations.tanh()
        elif act == "sin":
            activation = tf.math.sin()
        else:
            raise NotImplementedError(">>>>> act_func")
        return activation

    def dnn_init(
        self, f_in, f_out, f_hid, depth
    ):
        network = nn.Sequential(
            nn.Linear(f_in, f_hid)
        )
        return network

    def forward_pass(
        self, x
    ):
        # feature scaling
        if self.f_scl == None:
            z = x
        elif self.f_scl == "minmax":
            z = x
        elif self.f_scl == "mean":
            z = x
        else:
            raise NotImplementedError(">>>>> forward_pass")
        
        # y = 
        # return y

    def train(
        self, n_epc, n_btc, c_tol, es_pat
    ):
        print(">>>>> train")
        print("         n_epoch :", n_epc)
        print("         n_batch :", n_btc)
        print("         c_tlrnc :", c_tol)
        print("         patience:", es_pat)

        wait = 0
        loss_best = 9999
        t0 = time.time()
        # if n_btc == -1:
        #     print(">>>>> executing full-batch training")
        #     for epc in range(n_epc):
        #         loss_epc = 0.
        #         loss_epc = self.grad_desc(self.x, self.y)
        #         self.loss_log.append(loss_epc)

        #         # monitor 
        #         if epc % 100 == 0:
        #             elps = time.time() - t0
        #             print("epc: %d, loss: %.6e, elps: %.3f"
        #                 % (epc, loss_epc, elps))
        #             t0 = time.time()

        #         # # save 
        #         # if epc % 100 == 0:
        #         #     self.save(self.save_path + "model" + str(epc))

        #         # early stopping
        #         if loss_epc < loss_best:
        #             loss_best = loss_epc
        #             wait = 0
        #         else:
        #             if wait >= es_pat:
        #                 print(">>>>> early stopping")
        #                 break
        #             wait += 1

        #         # terminate if converged
        #         if loss_epc < c_tol:
        #             print(">>>>> converging to the tolerance")
        #             break

        # else:
        #     print("\n>>>>> executing mini-batch training")
        #     raise NotImplementedError(">>>>> train")

    def infer(
        self, x
    ):
        print(">>>>> infer")

