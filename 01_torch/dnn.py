"""
********************************************************************************
deep neural network
Author: Shota DEGUCHI
        Structural Analysis Laboratory, Kyushu Univ. (Feb. 3, 2022)
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
        self.setup(self.d_type, self.r_seed)

        print("\n************************************************************")
        print("********************     DELLO WORLD     *******************")
        print("************************************************************")

        # input - output pair
        self.x = x
        self.y = y

        # for feature scaling
        self.XY = torch.cat((x, y), dim=1)
        self.lower = torch.min (self.XY, dim=0)
        self.upper = torch.max (self.XY, dim=0)
        self.mean  = torch.mean(self.XY, dim=0)

        # # normalize?
        # self.XY = nn.functional.normalize(self.XY, p=2., dim=0)

        # build a deep neural network
        self.w_init = self.weight_init(self.w_init, self.r_seed)
        self.b_init = self.bias_init(self.b_init)
        self.act    = self.act_func(self.act)
        self.dnn = self.dnn_init(
            self.f_in, self.f_out, self.f_hid, self.depth, 
            self.w_init, self.b_init, self.act
        )
        self.params = self.dnn.parameters()

    def setup(
        self, d_type, r_seed
    ):
        os.environ["PYTHONHASHSEED"] = str(r_seed)
        np.random.seed(r_seed)
        torch.manual_seed(r_seed)
        torch.set_default_dtype(d_type)

    def weight_init(
        self, init, tnsr
    ):
        print(">>>>> weight_init")
        print("         initializer:", init)
        if init == "Glorot":
            weight = nn.init.xavier_normal_(tnsr, gain=1.)
        elif init == "He":
            weight = nn.init.kaiming_normal_(tnsr, a=0, mode="fan_in", nonlinearity="relu")
        else:
            raise NotImplementedError(">>>>> weight_init")
        return weight

    def bias_init(
        self, init, tnsr
    ):
        print(">>>>> bias_init")
        print("         initializer:", init)
        if init == "zeros":
            bias = nn.init.zeros_(tnsr)
        elif init == "ones":
            bias = nn.init.ones_(tnsr)
        else:
            raise NotImplementedError(">>>>> bias_init")
        return bias

    def act_func(
        self, act
    ):
        print(">>>>> act_func")
        print("         activation:", act)
        if act == "relu":
            activation = nn.ReLU(inplace=False)
        elif act == "elu":
            activation = nn.ELU(alpha=1.0, inplace=False)
        elif act == "swish" or act == "silu":
            activation = nn.SiLU(inplace=False)
        elif act == "tanh":
            activation = nn.Tanh()
        elif act == "sin":
            activation = torch.sin()
        else:
            raise NotImplementedError(">>>>> act_func")
        return activation

    def dnn_init(
        self, 
        f_in, f_out, f_hid, depth,
        w_init, b_init, act
    ):
        print(">>>>> dnn_init")
        print("         f_in :", f_in)
        print("         f_out:", f_out)
        print("         f_hid:", f_hid)
        print("         depth:", depth)
        layers = []
        layers.append(nn.Linear(f_in, f_hid))
        for l in range(depth - 1):
            layers.append(nn.Linear(f_hid, f_hid))
            layers.append(act)

            layers.append(nn.Linear(f_hid, f_hid, bias=True))
            w_init(nn.Linear(f_hid, f_hid).weight)

            self.weight_init(init, tnsr)

        layers.append(nn.Linear(f_hid, f_out))
        network = nn.Sequential(layers)
        print(network)
        return network

    def opt_alg(
        self, lr, opt, params
    ):
        print(">>>>> opt_alg")
        print("         learning rate:", lr)
        print("         optimizer    :", opt)
        if opt == "SGD":
            optimizer = torch.optim.SGD(
                params, lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, maximize=False
            )
        elif opt == "RMSprop":
            optimizer = torch.optim.RMSprop(
                params, lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False
            )
        elif opt == "Adam":
            optimizer = torch.optim.Adam(
                params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, maximize=False
            )
        elif opt == "Adamax":
            optimizer = torch.optim.Adamax(
                params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
            )
        elif opt == "Nadam":
            optimizer = torch.optim.NAdam(
                params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004
            )
        else:
            raise NotImplementedError(">>>>> opt_alg")
        return optimizer

    def forward_pass(
        self, x
    ):
        # feature scaling
        if self.f_scl == None:
            z = x
        elif self.f_scl == "minmax":
            z = 2. * (x - self.lower) / (self.upper - self.lower) - 1.            
        elif self.f_scl == "mean":
            z = (x - self.mean) / (self.upper - self.lower)
        else:
            raise NotImplementedError(">>>>> forward_pass")
        y_ = self.dnn(z)
        return y_

    def loss(
        self, x, y
    ):
        y_ = self.dnn(x)
        loss = nn.MSELoss(reduction='mean')
        loss = loss(y, y_)

    def train(
        self, n_epc, n_btc, c_tol, es_pat
    ):
        print(">>>>> train")
        print("         n_epoch:", n_epc)
        print("         n_batch:", n_btc)
        print("         c_tlrnc:", c_tol)
        print("         es_pat :", es_pat)

        wait = 0
        loss_best = 9999
        t0 = time.time()
        if n_btc == -1:
            print(">>>>> executing full-batch training")
            for epc in range(n_epc):
                loss_epc = 0.
        #         loss_epc = self.grad_desc(self.x, self.y)
        #         self.loss_log.append(loss_epc)

        #         # monitor 
        #         if epc % 100 == 0:
        #             elps = time.time() - t0
        #             print("epc: %d, loss: %.6e, elps: %.3f"
        #                 % (epc, loss_epc, elps))
        #             t0 = time.time()

                # early stopping
                if loss_epc < loss_best:
                    loss_best = loss_epc
                    wait = 0
                else:
                    if wait >= es_pat:
                        print(">>>>> early stopping")
                        break
                    wait += 1

                # terminate if converged
                if loss_epc < c_tol:
                    print(">>>>> converging to the tolerance")
                    break

        else:
            print(">>>>> executing mini-batch training")
            raise NotImplementedError(">>>>> train")

    def infer(
        self, x
    ):
        print(">>>>> infer")


